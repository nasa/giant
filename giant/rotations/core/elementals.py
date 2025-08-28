import numpy as np

from giant._typing import SCALAR_OR_ARRAY, ARRAY_LIKE, DOUBLE_ARRAY
from giant.rotations.core._helpers import _check_vector_array_and_shape


__all__ = ["rot_x", "rot_y", "rot_z", "skew"]


def rot_x(theta: SCALAR_OR_ARRAY) -> DOUBLE_ARRAY:
    r"""
    This function performs a right handed rotation about the x axis by angle theta.

    Mathematically this rotation is defined as:

    .. math::
        \mathbf{R}_x(\theta)=\left[\begin{array}{ccc} 1 & 0 & 0 \\
        0 & \text{cos}(\theta) & -\text{sin}(\theta) \\
        0 & \text{sin}(\theta) & \text{cos}(\theta) \end{array}\right]

    Theta should be in units of radians and can be a scalar or a vector.  If theta is a vector then each theta value
    will have a corresponding rotation vector down the first axis of the output.  For example::

        >>> from giant.rotations import rot_x
        >>> rot_x([2, 0.5])
        array([[[ 1.        ,  0.        ,  0.        ],
                [ 0.        , -0.41614684, -0.90929743],
                [ 0.        ,  0.90929743, -0.41614684]],
               [[ 1.        ,  0.        ,  0.        ],
                [ 0.        ,  0.87758256, -0.47942554],
                [ 0.        ,  0.47942554,  0.87758256]]])

    :param theta: The angles to form the rotation matrix(ces) for
    :return: The rotation matrix(ces) corresponding to the rotation angle(s)
    """

    # ensure we have an array of theta(s)
    theta = np.atleast_1d(np.asarray(theta)).flatten()

    # form an array of ones the same shape as theta
    ones = np.ones(theta.shape)

    # form an array of zeros the same shape as theta
    zeros = np.zeros(theta.shape)

    # compute the cosine of theta
    ctheta = np.cos(theta)

    # compute the sine of theta
    stheta = np.sin(theta)

    # form and return the matrix(ces)
    return np.vstack([ones, zeros, zeros, zeros, ctheta, -stheta, zeros, stheta, ctheta]).T.reshape(-1, 3, 3).squeeze()


def rot_y(theta: SCALAR_OR_ARRAY) -> DOUBLE_ARRAY:
    r"""
    This function performs a right handed rotation about the y axis by angle theta.

    This rotation is defined as:

    .. math::
        \mathbf{R}_y(\theta)=\left[\begin{array}{ccc} \text{cos}(\theta) & 0 & \text{sin}(\theta) \\
        0 & 1 & 0 \\
        -\text{sin}(\theta) & 0 & \text{cos}(\theta) \end{array}\right]

    Theta should be in units of radians and can be a scalar or a vector.  If theta is a vector then each theta value
    will have a corresponding rotation vector down the first axis of the output.  For example::

        >>> from giant.rotations import rot_y
        >>> rot_y([2, 0.5])
        array([[[-0.41614684,  0.        ,  0.90929743],
                [ 0.        ,  1.        ,  0.        ],
                [-0.90929743,  0.        , -0.41614684]],
               [[ 0.87758256,  0.        ,  0.47942554],
                [ 0.        ,  1.        ,  0.        ],
                [-0.47942554,  0.        ,  0.87758256]]])

    :param theta: The angles to form the rotation matrix(ces) for
    :return: The rotation matrix(ces) corresponding to the rotation angle(s)
    """

    # ensure we have an array of theta(s)
    theta = np.atleast_1d(np.asarray(theta)).flatten()

    # form an array of ones the same shape as theta
    ones = np.ones(theta.shape)

    # form an array of zeros the same shape as theta
    zeros = np.zeros(theta.shape)

    # compute the cosine of theta
    ctheta = np.cos(theta)

    # compute the sine of theta
    stheta = np.sin(theta)

    # form and return the matrix(ces)
    return np.vstack([ctheta, zeros, stheta, zeros, ones, zeros, -stheta, zeros, ctheta]).T.reshape(-1, 3, 3).squeeze()


def rot_z(theta: SCALAR_OR_ARRAY) -> DOUBLE_ARRAY:
    r"""
    This function performs a right handed rotation about the z axis by angle theta.

    This rotation is defined as:

    .. math::
        \mathbf{R}_z(\theta)=\left[\begin{array}{ccc} \text{cos}(\theta) & -\text{sin}(\theta) & 0 \\
        \text{sin}(\theta) & \text{cos}(\theta) & 0 \\
        0 & 0 & 1 \end{array}\right]

    Theta should be in units of radians and can be a scalar or a vector.  If theta is a vector then each theta value
    will have a corresponding rotation vector down the first axis of the output.  For example::

        >>> from giant.rotations import rot_z
        >>> rot_z([2, 0.5])
        array([[[-0.41614684, -0.90929743,  0.        ],
                [ 0.90929743, -0.41614684,  0.        ],
                [ 0.        ,  0.        ,  1.        ]],
               [[ 0.87758256, -0.47942554,  0.        ],
                [ 0.47942554,  0.87758256,  0.        ],
                [ 0.        ,  0.        ,  1.        ]]])

    :param theta: The angles to form the rotation matrix(ces) for
    :return: The rotation matrix(ces) corresponding to the rotation angle(s)
    """

    # ensure we have an array of theta(s)
    theta = np.atleast_1d(np.asarray(theta)).flatten()

    # form an array of ones the same shape as theta
    ones = np.ones(theta.shape)

    # form an array of zeros the same shape as theta
    zeros = np.zeros(theta.shape)

    # compute the cosine of theta
    ctheta = np.cos(theta)

    # compute the sine of theta
    stheta = np.sin(theta)

    # form and return the matrix(ces)
    return np.vstack([ctheta, -stheta, zeros, stheta, ctheta, zeros, zeros, zeros, ones]).T.reshape(-1, 3, 3).squeeze()


def skew(vector: ARRAY_LIKE) -> DOUBLE_ARRAY:
    r"""
    This function returns a numpy array with the skew symmetric cross product matrix for vector.

    The skew symmetric cross product matrix is defined such that:

    .. math::
        \mathbf{a}\times\mathbf{b}=\left[\mathbf{a}\times\right]\mathbf{b} \\
        \left[\mathbf{a}\times\right] = \left[\begin{array}{ccc} 0 & -a_3 & a_2 \\
        a_3 & 0 & -a_1 \\
        -a_2 & a_1 & 0 \end{array}\right]

    where :math:`\times` indicates the cross product and :math:`\left[\bullet\times\right]` is the skew symmetric cross
    product matrix

    This function is vectorized, therefore you can input multiple vectors as a 3xn array where each column is an
    independent vector.  The resulting skew matrix output will be nx3x3 where the first axis stores each matrix

    :param vector: The vector to compute a skew symmetric matrix for
    :return: The skew symmetric cross product matrix(ces) corresponding to the vector(s)
    """

    vector = _check_vector_array_and_shape(vector)

    if vector.ndim > 1:
        zeros = np.zeros(vector.shape[-1])

    else:
        zeros = 0

    return np.array([zeros, -vector[2], vector[1],
                     vector[2], zeros, -vector[0],
                     -vector[1], vector[0], zeros]).T.reshape(-1, 3, 3).squeeze()


