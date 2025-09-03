
import numpy as np

from giant.rotations import Rotation
from giant._typing import DOUBLE_ARRAY
from giant.stellar_opnav.estimators.attitude_estimator import AttitudeEstimator, AttitudeEstimatorOptions


class DavenportQMethod(AttitudeEstimator):
    r"""
    This class estimates the rotation quaternion that best aligns unit vectors from one frame with unit vectors in
    another frame using Davenport's Q-Method solution to Wahba's problem.

    This class is relatively easy to use.  After you initialize the class, simply specify the
    `target_frame_directions` unit vectors (:math:`\textbf{a}_i` from the :mod:`~.stellar_opnav.estimators`
    documentation) as a 3xn array of vectors (each column is a vector) and the `base_frame_directions` unit
    vectors (:math:`\textbf{b}_i` from the :mod:`~.stellar_opnav.estimators` documentation) as a 3xn array of
    vectors  (each column is a vector) in the call to :meth:`.estimate`.  Here the `target_frame_directions` 
    unit vectors are expressed in the end frame (the frame you want to rotate to) and the 
    `base_frame_directions` unit vectors are expressed in the starting frame (the frame you want to rotate 
    from).  Each column of `target_frame_directions` and `base_frame_directions` should
    correspond to each other as a pair (i.e. column 1 in `target_frame_directions` is paired with column ` in
    `base_frame_directions`.

    Optionally, you can set the :attr:`weighted_estimation` value to `True` and then provide the
    `weights` input to :meth:`estimate` to specify whether to use weighted estimation or not, and what weights to 
    use if you are using weighted estimation.  The `weights` input should be a length n array of the weights to 
    apply to each unit vector pair.

    The :meth:`estimate` method can be called to compute the attitude quaternion that best aligns the two frames.  
    When the :meth:`estimate` method completes, the solved for rotation is returned as an :class:`.Rotation` object.  
    In addition, the formal post fit covariance matrix of the estimate can be found in the :attr:`post_fit_covariance` 
    attribute.  Note that as with all attitude quaternions, the post fit covariance matrix will be rank deficient 
    since there are only 3 true degrees of freedom.

    A description of the math behind the DavenportQMethod Solution can be found
    `here <https://math.stackexchange.com/a/2275087/202119>`_.
    """

    def __init__(self, options: AttitudeEstimatorOptions | None = None):
        """
        :param options: A dataclass specifying the options to set for this instance.
        """
        
        super().__init__(AttitudeEstimatorOptions, options=options)

        self._attitude_prof_mat = None
        """
        The attitude profile matrix.  
        
        This is used internally and is only set after a call to :meth:`estimate`.  It is stored for reuse in the 
        computation of the post fit covariance matrix.
        """
        
        self._solved_rotation: Rotation | None = None
        """
        the solved for rotation from the base to the target frame.
        
        This gets computed in the call to estimate.  Users shouldn't use it, rely on the retunr from estimate instead.
        """
        
    def estimate(self, target_frame_directions: DOUBLE_ARRAY, base_frame_directions: DOUBLE_ARRAY, weights: DOUBLE_ARRAY | None) -> Rotation:
        """
        This method solves for the rotation matrix that best aligns the unit vectors in :attr:`base_frame_directions`
        with the unit vectors in :attr:`target_frame_directions` using Davenport's Q-Method solution to Wahba's Problem.

        Once the appropriate attributes have been set, simply call this method with no arguments and the solved for
        rotation will be stored in the :attr:`rotation` attribute as an :class:`.Rotation` object.
        
        :param target_frame_directions: Matrix of observed unit vectors (3xN)
        :param base_frame_directions: Matrix of reference unit vectors (3xN)
        :param weights: Vector of weights for each observation (N,) or None
        :param n_iter: Number of iterations for lambda computation (default=5, use 0 for lam=1)
        
        :returns: Optimal attitude quaternion 
        
        """

        # apply the weights if required
        if self.weighted_estimation:
            assert weights is not None, "Weights miust not be None if doing weighted estimation"
            target_frame_directions = weights.squeeze() * target_frame_directions

        att_prof_mat = self.attitude_profile_matrix(base_frame_directions, target_frame_directions)
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
        self._solved_rotation = Rotation(vecs[:, max_loc])
        return self._solved_rotation.copy()

    @property
    def post_fit_covariance(self) -> DOUBLE_ARRAY:
        """
        This returns the post-fit covariance after calling the ``estimate`` method as a 4x4 numpy array.

        This should be only be called after the estimate method has been called, otherwise it raises a ValueError
        """

        if self._attitude_prof_mat is None or self._solved_rotation is None:
            raise ValueError('estimate method must be called before requesting the post-fit covariance')

        # compute rotation matrix times attitude profile matrix transposed
        tbt = self._solved_rotation.matrix@self._attitude_prof_mat.T

        # compute the fisher information matrix
        ftt = np.trace(tbt)*np.eye(3) - (tbt+tbt.T)/2

        # form covariance matrix
        cov = np.zeros((4, 4), dtype=np.float64)
        cov[:3, :3] = np.linalg.inv(ftt)/4

        return cov
