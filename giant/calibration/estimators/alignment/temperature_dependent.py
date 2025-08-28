
from typing import Iterable, NamedTuple

import numpy as np

from giant.rotations import Rotation, euler_to_quaternion
from giant._typing import EULER_ORDERS


class TemperatureDependentResults(NamedTuple):
    """
    Named tuple to make the results clear
    """
    
    order: EULER_ORDERS
    """
    The order of the angles.
    
    This is the same as the argument provided by the user and is included for awareness.
    """
    
    angle_m_offset: float 
    """
    The estimated constant angle offset for the m (first) rotation axis in radians.
    """

    angle_m_slope: float
    """
    The estimated angle temperature slope for the m (first) rotation axis in radians.
    """

    angle_n_offset: float
    """
    The estimated constant angle offset for the n (second) rotation axis in radians.
    """

    angle_n_slope: float
    """
    The estimated angle temperature slope for the n (second) rotation axis in radians.
    """

    angle_p_offset: float
    """
    The estimated constant angle offset for the p (third) rotation axis in radians.
    """

    angle_p_slope: float
    """
    The estimated angle temperature slope for the p (third) rotation axis in radians.
    """


def temperature_dependent_alignment_estimator(frame_1_rotations: Iterable[Rotation], 
                                              frame_2_rotations: Iterable[Rotation], 
                                              temperatures: Iterable[float],
                                              order: EULER_ORDERS = 'xyz') -> TemperatureDependentResults:
    r"""
    This function estimates a temperature dependent attitude alignment between one frame and another.

    The temperature dependent alignment is found by fitting linear temperature dependent euler angles (or
    Tait-Bryan angles) to transform from the first frame to the second.  That is

    .. math::
        \mathbf{T}_B=\mathbf{R}_m(\theta_m(t))\mathbf{R}_n(\theta_n(t))\mathbf{R}_p(\theta_p(t))\mathbf{T}_A

    where :math:`\mathbf{T}_B` is the target frame, :math:`\mathbf{R}_i` is the rotation matrix about the :math:`i^{th}`
    axis, :math:`\mathbf{T}_A` is the base frame, and :math:`\theta_i(t)` are the linear angles.

    This fit is done in a least squares sense by computing the values for :math:`\theta_i(t)` across a range of
    temperatures (by estimating the attitude for multiple single images) and then solving the system

    .. math::
        \left[\begin{array}{cc} 1 & t_1 \\ 1 & t_2 \\ \vdots & \vdots \\ 1 & t_n \end{array}\right]
        \left[\begin{array}{ccc} \theta_{m0} & \theta_{n0} & \theta_{p0} \\
        \theta_{m1} & \theta_{n1} & \theta_{p1}\end{array}\right] =
        \left[\begin{array}{ccc}\vphantom{\theta}^0\theta_m &\vphantom{\theta}^0\theta_n &\vphantom{\theta}^0\theta_p\\
        \vdots & \vdots & \vdots \\
        \vphantom{\theta}^k\theta_m &\vphantom{\theta}^k\theta_n &\vphantom{\theta}^k\theta_p\end{array}\right]

    where :math:`\vphantom{\theta}^k\theta_i` is the measured Euler/Tait-Bryan angle for the :math:`k^{th}` image.
    
    Internally, we take each provided pair of rotations, compute the difference (`r2*r1.inv()`), convert the 
    difference to Euler/Tait-Bryan angles, and then perform the linear regression on the results (which yes, is 
    somewhat a case of castcading estimators...).

    In general a user should not use this function and instead the
    :meth:`.Calibration.estimate_temperature_dependent_alignment` should be used which handles the proper setup.
    
    :param frame_1_rotations: the base frame rotations (normally common to base frame)
    :param frame_2_rotations: the target frame rotations (normally common to target frame)
    :param temperatures: the temperature of the image for each corresponding rotation
    :param  order: the order of the Euler/Tait-Bryan angles you want to use
    """

    relative_euler_angles = []

    # get the independent euler angles
    for f1, f2 in zip(frame_1_rotations, frame_2_rotations):

        relative_euler_angles.append(list((f2*f1.inv()).as_euler_angles(order=order)))

    # make the coefficient matrix
    temperatures = list(temperatures)
    coef_mat = np.vstack([np.ones(len(temperatures)), temperatures]).T

    # solve for the solution
    solution = np.linalg.lstsq(coef_mat, relative_euler_angles)[0]

    # return the solution
    return TemperatureDependentResults(order, solution[0, 0], solution[1, 0], solution[0, 1], solution[1, 1], solution[0, 2], solution[1, 2])


def evaluate_temperature_dependent_alignment(alignment: TemperatureDependentResults, temperature: float) -> Rotation:
    """
    This function takes a fit temperature dependent alignment solution and evaluates what the 
    alignment rotation is at a specified temperature.
    
    Essentially, we compute the euler angles specified by the alignment for the requested temperature
    and then convert the angles into a :class:`.Rotation` object using the specified order.
    """
    
    euler_angles = (np.array(list(alignment)[1:]).reshape(3, 2) @ [1, temperature]).ravel()
    
    return Rotation(euler_to_quaternion(euler_angles, alignment.order))
    
    
