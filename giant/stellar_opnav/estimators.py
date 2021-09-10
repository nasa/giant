# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


r"""
This module provides the ability to find the rotation that best aligns 1 set of unit vectors with another set of unit
vectors.

Description of the Problem
__________________________

Mathematically we are trying to solve a minimization problem given by

.. math::
    \min_TJ(\mathbf{T})=\frac{1}{2}\sum_iw_i\left\|\mathbf{a}_i-\mathbf{T}\mathbf{b}_i\right\|^2

where :math:`\mathbf{T}` is the rotation matrix that aligns the vectors in frame :math:`b` (:math:`\mathbf{b}_i`) with
the vectors in frame :math:`a` (:math:`\mathbf{a}_i`) and :math:`w_i` is a weight for each pairing.  This is known as
Wahba's problem.

Wahba's problem has many different solutions, and GIANT currently provides one of those solutions, known as Davenport's
Q Method solution, which solves for the rotation quaternion representation of :math:`\mathbf{T}` using an
eigenvalue-eigenvector problem.  This implementation is given through the :class:`DavenportQMethod` class.
To implement your own solution to Wahba's problem, you should subclass the :class:`AttitudeEstimator` class (though this
is not required) and then tailor it to your method.
"""


from typing import Optional
from abc import ABCMeta, abstractmethod

import numpy as np

from giant.rotations import Rotation
from giant._typing import NONEARRAY, SCALAR_OR_ARRAY


class AttitudeEstimator(metaclass=ABCMeta):
    """
    This abstract base class (ABC) serves as a template for creating an attitude estimator that GIANT can use.

    While it is not required to subclass this ABC in user created estimators, it is encouraged as it will ensure
    that the appropriate methods and attributes are created to ensure a seamless integration.  In particular,
    the following attributes should be implemented:

    .. rubric:: Attributes

    * :attr:`target_frame_directions`
    * :attr:`base_frame_directions`
    * :attr:`weighted_estimation`
    * :attr:`weights`
    * :attr:`rotation`
    * :attr:`post_fit_covariance`

    These attributes will be accessed directly for reading or writing by the GIANT :class:`.StarID` and
    :class:`.StellarClass` classes (that is, they won't be specified during initialization).  In addition, the following
    method should be implemented with no arguments:

    .. rubric:: Methods

    * :meth:`estimate`

    This method will only be called after the :attr:`target_frame_directions`, :attr:`base_frame_directions`,
    :attr:`weighted_estimation`, and :attr:`weights` attributes have been set.

    See the :class:`DavenportQMethod` class for an example of how to make a working attitude estimator
    """

    __REQUIRED_ATTRIBUTES = ['target_frame_directions', 'base_frame_directions', 'weighted_estimation', 'weights',
                             'rotation']

    def __init_subclass__(cls, **kwargs):
        inst = cls()
        for attr in cls.__REQUIRED_ATTRIBUTES:
            if not hasattr(inst, attr):
                raise NotImplementedError('Attribute {} is required for Attitude estimators'.format(attr))

    @abstractmethod
    def estimate(self):
        """
        This method solves for the rotation matrix that best aligns the unit vectors in :attr:`base_frame_directions`
        with the unit vectors in :attr:`target_frame_directions` and stores the results in :attr:`rotation` attribute.

        The solved for rotation should represent the best fit rotation from the database frame to the camera frame.

        This method should respect the :attr:`weighted_estimation` flag and :attr:`weights` attribute, if applicable.
        """
        pass

    @property
    @abstractmethod
    def post_fit_covariance(self) -> np.ndarray:
        """
        The post-fit covariance from the attitude estimation as a 4x4 array
        """
        return np.zeros((4, 4))


class DavenportQMethod(AttitudeEstimator):
    r"""
    This class estimates the rotation quaternion that best aligns unit vectors from one frame with unit vectors in
    another frame using Davenport's Q-Method solution to Wahba's problem.

    This class is relatively easy to use.  When you initialize the class, simply specify the
    :attr:`target_frame_directions` unit vectors (:math:`\textbf{a}_i` from the :mod:`~.stellar_opnav.estimators`
    documentation) as a 3xn array of vectors (each column is a vector) and the :attr:`base_frame_directions` unit
    vectors (:math:`\textbf{b}_i` from the :mod:`~.stellar_opnav.estimators` documentation) as a 3xn array of
    vectors  (each column is a vector).  Here the :attr:`target_frame_directions` unit vectors are expressed in the
    end frame (the frame you want to rotate to) and the :attr:`base_frame_directions` unit vectors are expressed in
    the starting frame (the frame you want to rotate from).  You can also leave these inputs to be ``None`` and then
    set the attributes directly. Each column of :attr:`target_frame_directions` and :attr:`base_frame_directions` should
    correspond to each other as a pair (i.e. column 1 in :attr:`target_frame_directions` is paired with column ` in
    :attr:`base_frame_directions`.

    Optionally, either at initialization or by setting the attributes, you can set the :attr:`weighted_estimation` and
    :attr:`weights` values to specify whether to use weighted estimation or not, and what weights to use if you are
    using weighted estimation.  When performing weighted estimation you should set :attr:`weighted_estimation` to
    ``True`` and specify :attr:`weights` to be a length n array of the weights to apply to each unit vector pair.

    Once the appropriate values are set, the :meth:`estimate` method can be called to compute the attitude quaternion
    that best aligns the two frames.  When the :meth:`estimate` method completes, the solved for rotation can be found
    as an :class:`.Rotation` object in the :attr:`rotation` attribute of the class.  In addition, the formal post fit
    covariance matrix of the estimate can be found in the :attr:`post_fit_covariance` attribute.  Note that as will all
    attitude quaternions, the post fit covariance matrix will be rank deficient since there are only 3 true degrees of
    freedom.

    A description of the math behind the DavenportQMethod Solution can be found
    `here <https://math.stackexchange.com/a/2275087/202119>`_.
    """

    def __init__(self, target_frame_directions: NONEARRAY = None, base_frame_directions: NONEARRAY = None,
                 weighted_estimation: bool = False, weights: SCALAR_OR_ARRAY = 1):
        """
        :param target_frame_directions: A 3xn array of unit vectors expressed in the camera frame
        :param base_frame_directions: A 3xn array of unit vectors expressed in the catalogue frame corresponding the the
                                    ``target_frame_directions`` unit vectors
        :param weighted_estimation: A flag specifying whether to weight the estimation routine by unit vector pairs
        :param weights: The weights to apply to the unit vectors if the ``weighted_estimation`` flag is set to ``True``.
        """

        self.target_frame_directions = np.asarray(target_frame_directions)  # type: np.ndarray
        r"""
        The unit vectors in the target frame as a 3xn array (:math:`\mathbf{a}_i`).

        Each column should represent the pair of the corresponding column in :attr:`base_frame_directions`.
        """

        self.base_frame_directions = np.asarray(base_frame_directions)  # type: np.ndarray
        r"""
        The unit vectors in the base frame as a 3xn array (:math:`\mathbf{b}_i`).

        Each column should represent the pair of the corresponding column in :attr:`target_frame_directions`.
        """

        self.weights = np.asarray(weights)  # type: np.ndarray
        """
        A length n array of the weights to apply if weighted_estimation is True. (:math:`w_i`)

        Each element should represent the pair of the corresponding column in :attr:`target_frame_directions` and
        :attr:`base_frame_directions`.
        """

        self.weighted_estimation = weighted_estimation  # type: bool
        """
        A flag specifying whether to use weights in the estimation of the rotation.
        """

        self.rotation = None  # type: Optional[Rotation]
        """
        The solved for rotation that best aligns the :attr:`base_frame_directions` and :attr:`target_frame_directions` 
        after calling :meth:`estimate`.

        This rotation goes go from the base frame to the target frame.
        
        If :meth:`estimate` has not been called yet then this will be set to ``None``.
        """

        self._attitude_prof_mat = None
        """
        The attitude profile matrix.  
        
        This is used internally and is only set after a call to :meth:`estimate`.  It is stored for reuse in the 
        computation of the post fit covariance matrix.
        """

    def compute_residuals(self) -> np.ndarray:
        r"""
        This method computes the residuals between the aligned unit vectors according to Wahba's problem definitions.
        
        If the updated attitude has been estimated (:attr:`rotation` is not ``None``) then this method computes the 
        post-fit residuals.  If not then this method computes the pre-fit residuals.  The residuals are computed 
        according to
        
        .. math::
            r_i=\frac{1}{2}\left\|\mathbf{a}_i-\mathbf{T}\mathbf{b}_i\right\|^2
            
        where :math:`r_i` is the residual, :math:`\mathbf{a}_i` is the camera direction unit vector,
        :math:`\mathbf{b}_i` is the database direction unit vector, and :math:`\mathbf{T}` is the solved for rotation 
        matrix from the catalogue frame to the camera frame, or the identity matrix if the matrix hasn't been solved for
        yet.
        
        The output will be a length n array with each element representing the residual for the correspond unit vector
        pair.
        
        :return: The residuals between the aligned unit vectors
        """

        # apply the solved for rotation if it exists
        if self.rotation is not None:
            return ((self.target_frame_directions - np.matmul(self.rotation.matrix,
                                                              self.base_frame_directions)) ** 2).sum() / 2

        else:
            return ((self.target_frame_directions - self.base_frame_directions) ** 2).sum() / 2

    def estimate(self) -> None:
        """
        This method solves for the rotation matrix that best aligns the unit vectors in :attr:`base_frame_directions`
        with the unit vectors in :attr:`target_frame_directions` using Davenport's Q-Method solution to Wahba's Problem.

        Once the appropriate attributes have been set, simply call this method with no arguments and the solved for
        rotation will be stored in the :attr:`rotation` attribute as an :class:`.Rotation` object.
        """

        # apply the weights if required
        if self.weighted_estimation:
            target_frame_directions = self.weights.flatten() * self.target_frame_directions
        else:
            target_frame_directions = self.target_frame_directions

        # compute the attitude profile matrix (sum of the outer products of the vector sets)
        att_prof_mat = np.einsum('ij,jk->jik', self.base_frame_directions, target_frame_directions.T).sum(axis=0)

        self._attitude_prof_mat = att_prof_mat

        # for the S matrix
        s_mat = att_prof_mat + att_prof_mat.T

        # retrieve the z vector from its skew matrix
        temp = att_prof_mat - att_prof_mat.T
        z = np.array([temp[1, 2], -temp[0, 2], temp[0, 1]])

        # initialize the davenport matrix
        davenport_mat = np.zeros((4, 4))

        # compute the trace of the attitude profile matrix
        att_prof_trace = att_prof_mat.trace()

        # form the davenport matrix
        davenport_mat[0:3, 0:3] = (s_mat - np.eye(3) * att_prof_trace)
        davenport_mat[3, 0:3] = z.flatten()
        davenport_mat[0:3, 3] = z.flatten()
        davenport_mat[3, 3] = att_prof_trace

        # get the eigenvalues and corresponding eigenvectors
        vals, vecs = np.linalg.eigh(davenport_mat)

        # identify the maximum eigenvalue
        max_loc = np.argmax(vals)

        # retrieve the corresponding eigenvector and store it in an Rotation object as this is the new rotation.
        self.rotation = Rotation(vecs[:, max_loc])

    @property
    def post_fit_covariance(self) -> np.ndarray:
        """
        This returns the post-fit covariance after calling the ``estimate`` method as a 4x4 numpy array.

        This should be only be called after the estimate method has been called, otherwise it raises a ValueError
        """

        if self._attitude_prof_mat is None:
            raise ValueError('estimate method must be called before requesting the post-fit covariance')

        # compute rotation matrix times attitude profile matrix transposed
        tbt = self.rotation.matrix@self._attitude_prof_mat.T

        # compute the fisher information matrix
        ftt = np.trace(tbt)*np.eye(3) - (tbt+tbt.T)/2

        # form covariance matrix
        cov = np.zeros((4, 4), dtype=np.float64)
        cov[:3, :3] = np.linalg.inv(ftt)/4

        return cov
