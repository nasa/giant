from dataclasses import dataclass
from abc import ABCMeta, abstractmethod

import numpy as np

from giant.utilities.options import UserOptions
from giant.utilities.mixin_classes import AttributeEqualityComparison, AttributePrinting, UserOptionConfigured
from giant.rotations import Rotation
from giant._typing import DOUBLE_ARRAY

@dataclass
class AttitudeEstimatorOptions(UserOptions):
    """
    Dataclass for configuring attitude estimator subclasses
    """
    
    weighted_estimation: bool = False
    """
    A flag specifying whether to use weights in the estimation of the rotation.
    """
   
    
class AttitudeEstimator(UserOptionConfigured[AttitudeEstimatorOptions], AttitudeEstimatorOptions, AttributePrinting, AttributeEqualityComparison,
                        metaclass=ABCMeta):
    """
    This abstract base class (ABC) serves as a template for creating an attitude estimator that GIANT can use.

    While it is not required to subclass this ABC in user created estimators, it is encouraged as it will ensure
    that the appropriate methods and attributes are created to ensure a seamless integration.  In particular,
    the following attributes should be implemented:

    .. rubric:: Attributes

    * :attr:`post_fit_covariance`

    These attributes will be accessed directly for reading or writing by the GIANT :class:`.StarID` and
    :class:`.StellarClass` classes (that is, they won't be specified during initialization).  In addition, the following
    method should be implemented with no arguments:

    .. rubric:: Methods

    * :meth:`estimate`

    See the :class:`DavenportQMethod` class for an example of how to make a working attitude estimator
    """
    
    def __init__(self, options_type: type[AttitudeEstimatorOptions], *args, options: AttitudeEstimatorOptions | None = None, **kwargs) -> None:
        super().__init__(options_type, *args, options=options, **kwargs)

    @abstractmethod
    def estimate(self, target_frame_directions: DOUBLE_ARRAY, base_frame_directions: DOUBLE_ARRAY, weights: DOUBLE_ARRAY | None) -> Rotation:
        """
        This method solves for the rotation matrix that best aligns the unit vectors in `base_frame_directions`
        with the unit vectors in `target_frame_directions` and returns the results

        The solved for rotation should represent the best fit rotation from the base frame to the target frame.

        This method should also prepare the :attr:`post_fit_covariance` attribute if applicable
        
        :param target_frame_directions: unit vectors in the target frame as a 3xn double array
        :param base_frame_directions: unit vectors in the base frame as a 3xn double array
        :param weights: the weights for each observation or None
        """
        pass

    @property
    @abstractmethod
    def post_fit_covariance(self) -> DOUBLE_ARRAY:
        """
        The post-fit covariance from the attitude estimation as a 4x4 array
        """
        return np.zeros((4, 4), dtype=np.float64)

    @staticmethod
    def compute_residuals(target_frame_directions: DOUBLE_ARRAY, base_frame_directions: DOUBLE_ARRAY, rotation: Rotation) -> float:
        r"""
        This method computes the residuals between the aligned unit vectors according to Wahba's problem definitions.
        
        The residuals are computed according to
        
        .. math::
            r_i=\frac{1}{2}\left\|\mathbf{a}_i-\mathbf{T}\mathbf{b}_i\right\|^2
            
        where :math:`r_i` is the residual, :math:`\mathbf{a}_i` is the camera direction unit vector,
        :math:`\mathbf{b}_i` is the database direction unit vector, and :math:`\mathbf{T}` is the solved for rotation 
        matrix from the base frame to the target frame.
        
        The output will be a length n array with each element representing the residual for the correspond unit vector
        pair.
        
        :return: The residuals between the aligned unit vectors
        """

        return ((target_frame_directions - np.matmul(rotation.matrix, base_frame_directions)) ** 2).sum(axis=0) / 2.0
    
    @staticmethod
    def attitude_profile_matrix(base_frame_directions: DOUBLE_ARRAY, target_frame_directions: DOUBLE_ARRAY) -> DOUBLE_ARRAY:
        """
        Computes the attitude profile matrix for the provided vector sets
        """

        # compute the attitude profile matrix (sum of the outer products of the vector sets)
        return np.einsum('ij,jk->jik', base_frame_directions, target_frame_directions.T).sum(axis=0)
