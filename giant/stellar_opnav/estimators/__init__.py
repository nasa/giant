


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


import giant.stellar_opnav.estimators.attitude_estimator as attitude_estimator
import giant.stellar_opnav.estimators.esoq2 as esoq2
import giant.stellar_opnav.estimators.davenport_q_method as davenport_q_method

from giant.stellar_opnav.estimators.attitude_estimator import AttitudeEstimator, AttitudeEstimatorOptions
from giant.stellar_opnav.estimators.davenport_q_method import DavenportQMethod
from giant.stellar_opnav.estimators.esoq2 import ESOQ2, ESOQ2Options

from enum import Enum, auto

class AttitudeEstimatorImplementations(Enum):
    """
    An enum specifying the available attitude estimator implementations.
    
    For a non-standard implementation, choose CUSTOM
    """
    
    DAVENPORT_Q_METHOD = auto()
    """
    Use Davenport's Q-Method solution to Wahba's problem to find the optimal quaternion
    """
    
    ESOQ2 = auto()
    """
    Use the Second Sequential Estimator of Quaternions to find the near-optimal quaternion
    """
    
    CUSTOM = auto()
    """
    A custom implementation which implements the AttitudeEstimator base class
    """
    
    
def get_estimator(type: AttitudeEstimatorImplementations, options: AttitudeEstimatorOptions | None = None) -> DavenportQMethod | ESOQ2:
    """
    Returns an instance of the appropriate estimator per the provided type.
    
    :returns: the requested estimator initialized with options
    :raises: ValueError if CUSTOM is chosen
    """
    
    match type:
        case AttitudeEstimatorImplementations.DAVENPORT_Q_METHOD:
            return DavenportQMethod(options)
        case AttitudeEstimatorImplementations.ESOQ2:
            assert isinstance(options, ESOQ2Options) or options is None, "Options must be ESOQ2Options or None"
            return ESOQ2(options)
        case _:
            raise ValueError('Cannot return a custom attitude implementation')


__all__ = ["AttitudeEstimator", "AttitudeEstimatorOptions", "DavenportQMethod", "ESOQ2", "ESOQ2Options", "AttitudeEstimatorImplementations"]
