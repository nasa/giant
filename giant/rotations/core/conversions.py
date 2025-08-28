# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.

"""
Core conversion routines for rotation representations

This module contains core routines for converting between different rotation representations.
All routines are implemented purely on numpy arrays (or array like objects).
"""


from typing import Sequence

import numpy as np

from giant._typing import ARRAY_LIKE, F_SCALAR_OR_ARRAY, DOUBLE_ARRAY, EULER_ORDERS, SCALAR_OR_ARRAY

from giant.rotations.core._helpers import _check_matrix_array_and_shape, _check_quaternion_array_and_shape, _check_vector_array_and_shape
from giant.rotations.core.elementals import rot_x, rot_y, rot_z, skew


__all__ = ['quaternion_to_rotvec', 'quaternion_to_rotmat', 'quaternion_to_euler', 
           'rotvec_to_rotmat', 'rotvec_to_quaternion', 'rotvec_to_euler',
           'rotmat_to_quaternion', 'rotmat_to_euler', 'rotmat_to_rotvec',
           'euler_to_rotmat', 'euler_to_quaternion', 'euler_to_rotvec']


def quaternion_to_rotvec(quaternion: ARRAY_LIKE) -> DOUBLE_ARRAY:
    r"""
    This function converts a rotation quaternion into a rotation vector of the form discussed in
    :ref:`Rotation Representations <rotation-representation-table>`.

    The rotation vector is returned as a numpy array and is formed by:

    .. math::
        \theta = 2*\text{cos}^{-1}(q_s) \\
        \hat{\mathbf{x}} = \frac{\mathbf{q}_v}{\text{sin}(\theta/2)} \\
        \mathbf{v} = \theta\hat{\mathbf{x}}

    This function is also vectorized, meaning that you can specify multiple rotation quaternions to be converted to
    rotation vectors by specifying each quaternion as a column.  Regardless of whether you are converting 1 or many
    quaternions the first axis must have a length of 4.

    This function makes the output have the same number of dimensions as the input.  Therefore, if the input is one
    dimensional, then the output is one dimensional, and if the input is two dimensional then the output will be two
    dimensional.  It also checks for cases when theta is nearly zero (less than 1e-15) and replaces these with the
    identity rotation vector [0, 0, 0].

    :param quaternion: the rotation quaternion(s) to be converted to the rotation vector(s)
    :return: The rotation vector(s) corresponding to the input rotation quaternion(s)
    """

    # ensure we have a numpy array of the quaternion(s)
    quaternion = _check_quaternion_array_and_shape(quaternion)

    # get the rotation angle from the scalar portion of the quaternion
    theta = 2 * np.arccos(quaternion[-1, ...])

    if quaternion.ndim > 1:
        # if we are dealing with 2D quaternion(s)

        # check to see if we have identity quaternion(s)
        small_angle_check = theta < 1e-15

        # get the rotation axis
        e_vec = quaternion[:3] / np.sin(theta / 2)

        # replace the rotation axis with 0 in places where there is an identity quaternion
        e_vec[:, small_angle_check] = np.zeros((3, 1))

        # form the rotation vector
        return theta * e_vec

    else:

        if theta < 1e-15:
            # check to see if the input is an identity quaternion and return 0 if it is

            return np.zeros(3)

        else:

            # get the rotation axis
            e_vec = quaternion[:3] / np.sin(theta / 2)

            # form the rotation vector
            return theta * e_vec


def quaternion_to_rotmat(quaternion: ARRAY_LIKE) -> DOUBLE_ARRAY:
    r"""
    This function converts an attitude quaternion into its equivalent rotation matrix of the form discussed in
    :ref:`Rotation Representations <rotation-representation-table>`.

    Rotation quaternions are converted to rotation matrices by using:

    .. math::
        \mathbf{q}=\left[\begin{array}{c}\mathbf{q}_v \\ q_s\end{array}\right] \\
        \mathbf{T} = (q_s^2-\mathbf{q}_v^T\mathbf{q}_v)\mathbf{I}_{3\times 3}+2\mathbf{q}_v\mathbf{q}_v^T+2q_s
        \left[\mathbf{q}_v\times\right]

    where :math:`\mathbf{q}_v` is the vector portion of the quaternion, :math:`q_s` is the scalar portion of the
    quaternion, :math:`\left[\bullet\times\right]` is the skew symmetric cross product matrix (see :func:`skew`), and
    :math:`\mathbf{I}_{3\times 3}` is a :math:`3\times 3` identity matrix.

    This function is vectorized, meaning that you can specify multiple rotation quaternions to be converted to matrices
    by specifying each quaternion as a column.  Regardless of whether you are converting 1 or many
    quaternions the first axis must have a length of 4.  When converting multiple quaternions, each rotation matrix
    is stacked along the first axis such that the rotation matrix for the rotation quaternion in the first column is
    the 0th index of the first axis, the rotation matrix for the rotation quaternion in the second column is the 1st
    index of the first axis, and so on.  For example::

        >>> from giant.rotations import quaternion_to_rotmat
        >>> from numpy import sqrt
        >>> quaternion_to_rotmat([[0, 0], [1, 1/sqrt(3)], [0, 1/sqrt(3)], [0, 1/sqrt(3)]])
        array([[[-1.        ,  0.        ,  0.        ],
                [ 0.        ,  1.        ,  0.        ],
                [ 0.        ,  0.        , -1.        ]],
               [[-0.33333333, -0.66666667,  0.66666667],
                [ 0.66666667,  0.33333333,  0.66666667],
                [-0.66666667,  0.66666667,  0.33333333]]])

    :param quaternion: The rotation quaternion(s) to be converted to the rotation matrix(ces)
    :return: a numpy array containing the rotation matrix(ces) corresponding to the input quaternion(s)
    """

    # retrieve the numpy array of the input quaternion(s)
    quaternion = _check_quaternion_array_and_shape(quaternion)

    # extract the scalar and vector portion of the quaternion(s)
    qs = quaternion[-1].reshape(-1, 1, 1)
    qv = quaternion[:3].reshape(3, -1)

    # form and return the rotation matrix
    return ((qs ** 2 - (qv * qv).sum(axis=0).reshape(-1, 1, 1)) * np.eye(3) + 2 * np.einsum('ij,jk->jik', qv, qv.T) +
            2 * qs * skew(qv)).squeeze()


def quaternion_to_euler(quaternion: ARRAY_LIKE,
                        order: EULER_ORDERS = 'xyz') -> tuple[F_SCALAR_OR_ARRAY, F_SCALAR_OR_ARRAY, F_SCALAR_OR_ARRAY]:
    """
    This function converts a rotation quaternion to 3 euler angles to be applied to the axes specified in order.

    This function works by first converting the quaternion to a rotation matrix using :func:`quaternion_to_rotmat` and
    then using the function :func:`rotmat_to_euler` to find the euler angles.  See the documentation for those two
    functions for more information.

    This function is vectorized so multiple quaternions can be converted simultaneously by specifying them as columns.

    :param quaternion: The quaternion(s) to be converted to euler angles
    :param order: The order of the rotations
    :return: The euler angles corresponding to the rotation quaternion(s) acording to order
    """

    rmat = quaternion_to_rotmat(quaternion)

    return rotmat_to_euler(rmat, order=order)


def rotvec_to_rotmat(vector: ARRAY_LIKE) -> DOUBLE_ARRAY:
    r"""
    This function converts a rotation vector to a rotation matrix according to the form specified in
    :ref:`Rotation Representations <rotation-representation-table>`.

    The resulting rotation matrix is returned as a numpy array and is computed according to:

    .. math::
        \theta=\left\|\mathbf{v}\right\| \\
        \hat{\mathbf{x}} = \frac{\mathbf{v}}{\theta}\\
        \mathbf{T} = \text{cos}(\theta)\mathbf{I}_{3\times 3}+\text{sin}(\theta)\left[\hat{\mathbf{x}}\times\right]+
        (1-\text{cos}(\theta))\hat{\mathbf{x}}\hat{\mathbf{x}}^T

    where :math:`\mathbf{v}` is the rotation vector, :math:`\theta` is the rotation angle, :math:`\hat{\mathbf{x}}` is
    the rotation axis, :math:`\left[\bullet\times\right]` is the skew symmetric cross product matrix (see :func:`skew`),
    and :math:`\mathbf{I}_{3\times 3}` is a :math:`3\times 3` identity matrix.

    This function is also vectorized, meaning that you can specify multiple rotation vectors to be converted to
    rotation matrices by specifying each vector as a column.  Regardless of whether you are converting 1 or many vectors
    the first axis must have a length of 3.

    :param vector:  The rotation vector(s) to convert to a rotation matrix
    :return: The rotation matrix(ces) corresponding to the rotation vector(s)
    """

    vector = _check_vector_array_and_shape(vector)

    # check to see if we are working with a 2d array
    if vector.ndim > 1:

        # compute the rotation angle(s)
        theta = np.linalg.norm(vector, axis=0, keepdims=True)

        # check for anywhere that the rotation is 0
        zero_test = (theta == 0).flatten()

        # get the rotation axis(es)
        unit = vector / theta

        # compute the sine and cosine values of the angle(s)
        ctheta = np.cos(theta).reshape(-1, 1, 1)
        stheta = np.sin(theta).reshape(-1, 1, 1)

        # for the rotation matrix(ces)
        res = (ctheta * np.eye(3)) + \
              (stheta * skew(unit)) + \
              ((1 - ctheta) * np.einsum('ij,jk->jik', unit, unit.T))

        # replace anywhere that the angle is 0 with an identity matrix and return
        if zero_test.any():
            res[zero_test] = [np.eye(3)] * zero_test.sum()

        return res

    else:

        # get the rotation angle
        theta = np.linalg.norm(vector)

        # check to see if there is a rotation.  If not return the identity matrix
        if theta == 0:
            return np.eye(3)

        # compute the rotation axis
        unit = vector / theta

        # compute the rotation matrix and return
        ctheta = np.cos(theta)

        return ctheta * np.eye(3) + np.sin(theta) * skew(unit) + (1 - ctheta) * np.outer(unit, unit)


def rotvec_to_quaternion(rot_vec: ARRAY_LIKE) -> DOUBLE_ARRAY:
    r"""
    This function converts a rotation vector given as a 3 element Sequence into a rotation quaternion of the form
    discussed in :ref:`Rotation Representations <rotation-representation-table>`.

    The quaternion is returned as a numpy array and is formed by:

    .. math::
        \theta = \left\|\mathbf{v}\right\| \\
        \hat{\mathbf{x}} = \frac{\hat{\mathbf{x}}}{\theta} \\
        \mathbf{q} = \left[\begin{array}{c} \text{sin}(\frac{\theta}{2})\mathbf{x} \\
        \text{cos}(\frac{\theta}{2})\end{array}\right]

    This function is also vectorized, meaning that you can specify multiple rotation vectors to be converted to
    quaternions by specifying each vector as a column.  Regardless of whether you are converting 1 or many vectors
    the first axis must have a length of 3.

    This function makes the output have the same number of dimensions as the input.  Therefore, if the input is one
    dimensional, then the output is one dimensional, and if the input is two dimensional then the output will be two
    dimensional.  It also checks for cases when theta is nearly zero (less than 1e-15) and replaces these with the
    identity quaternion [0, 0, 0, 1].

    :param rot_vec: The rotation vector to convert to a rotation quaternion
    :return: the rotation quaternion(s) corresponding to the input rotation vector(s)
    """

    rot_vec = _check_vector_array_and_shape(rot_vec)

    # get the rotation angle(s)
    theta = np.linalg.norm(rot_vec, axis=0)

    # check the dimensions of the input
    if rot_vec.ndim > 1:

        # check for empty rotation vectors
        small_angle_check = theta < 1e-15

        # form the vector portion of the quaternion
        q_vec = rot_vec / theta * np.sin(theta / 2)

        # form the scalar portion of the quaternion
        q_scal = np.cos(theta / 2)

        # set identity quaternions wherever is required
        q_vec[:, small_angle_check] = np.zeros((3, 1))

        q_scal[small_angle_check] = 1

        # for the total quaternion(s) from the parts
        q = np.vstack([q_vec, q_scal])

    else:
        # check to see if we have an empty rotation vector
        if theta < 1e-15:
            # if we do then return the identity quaternion
            q_vec = np.zeros(3)
            q_scal = 1
        else:
            # otherwise use the formula
            q_vec = np.sin(theta / 2) * rot_vec / theta
            q_scal = np.cos(theta / 2)

        # for the total quaternion from the parts
        q = np.hstack([q_vec, q_scal])

    return q


def rotvec_to_euler(vector: ARRAY_LIKE,
                    order: EULER_ORDERS = 'xyz') -> tuple[F_SCALAR_OR_ARRAY, F_SCALAR_OR_ARRAY, F_SCALAR_OR_ARRAY]:
    """
    This function converts a rotation vector into euler angles in the specified order.
    
    Currently this is done through calls to :func:`.rotvec_to_rotmat` followed by a call 
    to :func:`.rotmat_to_euler`.
    
    :param vector: The rotation vector(s) to convert into Euler angles
    :param order: the desired order of the Euler angles
        
    :returns: The euler angles in the specified order as a tuple of 3 floats or 3 1d numpy arrays
    """
    rmat = rotvec_to_rotmat(vector)
    return rotmat_to_euler(rmat, order=order)


def rotmat_to_quaternion(rotation_matrix: ARRAY_LIKE) -> DOUBLE_ARRAY:
    r"""
    This function converts a rotation matrix into a rotation quaternion of the form
    discussed in :ref:`Rotation Representations <rotation-representation-table>`.

    The quaternion is returned as a numpy array and is formed by:

    .. math::
        q_s = \frac{1}{2}\sqrt{(\text{Tr}(\mathbf{T})+1)}\\
        \mathbf{q}_v = \frac{1}{2}\left[\begin{array}{c}\text{copysign}(\sqrt{1+t_{11}-t_{22}-t_{33}}, t_{32}-t_{23})\\
        \text{copysign}(\sqrt{1-t_{11}+t_{22}-t_{33}}, t_{13}-t_{31})\\
        \text{copysign}(\sqrt{1-t_{11}-t_{22}+t_{33}}, t_{21}-t_{12})\end{array}\right]

    where :math:`\text{Tr}(\bullet)` is the trace operator, :math:`\mathbf{T}` is the rotation matrix to be converted,
    :math:`t_{ij}` is the :math:`i, j` element of :math:`\mathbf{T}`, and :math:`\text{copysign}(a, b)` overwrites the
    sign of :math:`a` with the sign of :math:`b`.

    This function is also vectorized, meaning that you can specify multiple rotation matrices to be converted to
    quaternions by specifying each matrix along the first axis.  Regardless of whether you are converting 1 or many
    matrices the last two axes must have a length of 3.

    This function makes the output have the same number of dimensions as the input.  Therefore, if the input is two
    dimensional, then the output is one dimensional, and if the input is three dimensional then the output will be two
    dimensional.

    :param rotation_matrix: The rotation matrix to convert to a rotation quaternion
    :return: the rotation quaternion(s) corresponding to the input rotation vector(s)
    """

    rotation_matrix = _check_matrix_array_and_shape(rotation_matrix)

    # compute the scalar portion of the quaternion.  The max(..., 0) is to avoid rounding errors.
    q_scalar = 0.5 * np.sqrt(np.maximum(np.trace(rotation_matrix.T) + 1, 0))

    # extract the diagonal elements from the matrix(ces)
    t_diag = np.diagonal(rotation_matrix.T).reshape((-1, 1, 3))

    temp_mat = np.array([[1, -1, -1],
                         [-1, 1, -1],
                         [-1, -1, 1]])

    # form the vector portion of the quaternion.  The max(..., 0) is to avoid rounding errors
    q_vec = np.sqrt(np.maximum((temp_mat * t_diag).sum(axis=-1) + 1, 0)).squeeze() / 2.0  # type:np.ndarray

    # get the skew symmetric values from the rotation matrix
    rotation_skew = rotation_matrix - rotation_matrix.swapaxes(-2, -1)

    # copy the signs from the skew values onto the vector portion(s)
    q_vec = np.copysign(q_vec.T, [-rotation_skew[..., 1, 2],
                                  rotation_skew[..., 0, 2],
                                  -rotation_skew[..., 0, 1]]).squeeze()

    # return as a 1d or 2d array with the quaternions down the first axis
    try:
        return np.hstack([q_vec, q_scalar])

    except ValueError:

        return np.vstack([q_vec, q_scalar])


def rotmat_to_euler(matrix: ARRAY_LIKE,
                    order: EULER_ORDERS = 'xyz') -> tuple[F_SCALAR_OR_ARRAY, F_SCALAR_OR_ARRAY, F_SCALAR_OR_ARRAY]:
    """
    This function converts a rotation matrix to 3 euler angles to be applied to the axes specified in order.

    Order specifies both the axes of the euler angles, and the order they should be applied.  The order is applied left
    to right.  That is, for an order of xyz the rotation will be applied about x, then about y, then about z.

    The returned euler angles will match the order of `order`. That is, if order is xyz then the first angle will
    correspond to the rotation about x, the second angle will correspond to the rotation about y, and the third angle
    will correspond to the rotation about z.  The angles will be returned in radians.

    This function is vectorized, therefore you can input matrix as a nx3x3 stack of rotation matrices down the first
    axis and the results will return the angles for each matrix.  There can only be a single input for order which will
    apply to all cases in this case.

    Math to follow.

    :param matrix: The matrix(ces) to convert to euler angles
    :param order: The order of the rotations
    :return: The euler angles corresponding to the rotation matrix(ces)
    """

    fixed_order = order.upper().lower()

    matrix = _check_matrix_array_and_shape(matrix)

    if fixed_order == 'xyx':

        f1 = matrix[..., 1, 0]
        f2 = matrix[..., 2, 0]
        s1 = matrix[..., 0, 0]
        t1 = matrix[..., 0, 1]
        t2 = -matrix[..., 0, 2]

        return (-np.arctan2(t1, t2), -np.arccos(s1), -np.arctan2(f1, f2))

    elif fixed_order == 'zyx':

        f1 = matrix[..., 1, 2]
        f2 = matrix[..., 2, 2]
        s1 = matrix[..., 0, 2]
        t1 = matrix[..., 0, 1]
        t2 = matrix[..., 0, 0]

        return (-np.arctan2(t1, t2), np.arcsin(s1), -np.arctan2(f1, f2))

    elif fixed_order == 'xzx':

        f1 = matrix[..., 2, 0]
        f2 = -matrix[..., 1, 0]
        s1 = matrix[..., 0, 0]
        t1 = matrix[..., 0, 2]
        t2 = matrix[..., 0, 1]

        return (-np.arctan2(t1, t2), -np.arccos(s1), -np.arctan2(f1, f2))

    elif fixed_order == 'yzx':

        f1 = -matrix[..., 2, 1]
        f2 = matrix[..., 1, 1]
        s1 = matrix[..., 0, 1]
        t1 = -matrix[..., 0, 2]
        t2 = matrix[..., 0, 0]

        return (-np.arctan2(t1, t2), -np.arcsin(s1), -np.arctan2(f1, f2))

    elif fixed_order == 'yxy':

        f1 = matrix[..., 0, 1]
        f2 = -matrix[..., 2, 1]
        s1 = matrix[..., 1, 1]
        t1 = matrix[..., 1, 0]
        t2 = matrix[..., 1, 2]

        return (-np.arctan2(t1, t2), -np.arccos(s1), -np.arctan2(f1, f2))

    elif fixed_order == 'zxy':

        f1 = -matrix[..., 0, 2]
        f2 = matrix[..., 2, 2]
        s1 = matrix[..., 1, 2]
        t1 = -matrix[..., 1, 0]
        t2 = matrix[..., 1, 1]

        return (-np.arctan2(t1, t2), -np.arcsin(s1), -np.arctan2(f1, f2))

    elif fixed_order == 'xzy':

        f1 = matrix[..., 2, 0]
        f2 = matrix[..., 0, 0]
        s1 = matrix[..., 1, 0]
        t1 = matrix[..., 1, 2]
        t2 = matrix[..., 1, 1]

        return (-np.arctan2(t1, t2), np.arcsin(s1), -np.arctan2(f1, f2))

    elif fixed_order == 'yzy':

        f1 = matrix[..., 2, 1]
        f2 = matrix[..., 0, 1]
        s1 = matrix[..., 1, 1]
        t1 = matrix[..., 1, 2]
        t2 = -matrix[..., 1, 0]

        return (-np.arctan2(t1, t2), -np.arccos(s1), -np.arctan2(f1, f2))

    elif fixed_order == 'yxz':

        f1 = matrix[..., 0, 1]
        f2 = matrix[..., 1, 1]
        s1 = matrix[..., 2, 1]
        t1 = matrix[..., 2, 0]
        t2 = matrix[..., 2, 2]

        return (-np.arctan2(t1, t2), np.arcsin(s1), -np.arctan2(f1, f2))

    elif fixed_order == 'zxz':

        f1 = matrix[..., 0, 2]
        f2 = matrix[..., 1, 2]
        s1 = matrix[..., 2, 2]
        t1 = matrix[..., 2, 0]
        t2 = -matrix[..., 2, 1]

        return (-np.arctan2(t1, t2), -np.arccos(s1), -np.arctan2(f1, f2))

    elif fixed_order == 'xyz':

        f1 = -matrix[..., 1, 0]
        f2 = matrix[..., 0, 0]
        s1 = matrix[..., 2, 0]
        t1 = -matrix[..., 2, 1]
        t2 = matrix[..., 2, 2]

        return (-np.arctan2(t1, t2), -np.arcsin(s1), -np.arctan2(f1, f2))

    elif fixed_order == 'zyz':

        f1 = matrix[..., 1, 2]
        f2 = -matrix[..., 0, 2]
        s1 = matrix[..., 2, 2]
        t1 = matrix[..., 2, 1]
        t2 = matrix[..., 2, 0]

        return (-np.arctan2(t1, t2), -np.arccos(s1), -np.arctan2(f1, f2))
    else:
        raise ValueError('Invalid order')
    
def rotmat_to_rotvec(matrix: ARRAY_LIKE) -> DOUBLE_ARRAY:
    """
    Converts a rotation matrix to a rotation vector.
    
    Currently this just calls :func:`.rotmat_to_quaterion` followed by :func:`.quaternion_to_rotvec`
    
    :param matrix: The matrix(ices) to convert
        
    :returns: The rotation vector(s)"""
    
    return quaternion_to_rotvec(rotmat_to_quaternion(matrix))


def euler_to_rotmat(angles: Sequence[SCALAR_OR_ARRAY] | DOUBLE_ARRAY, order: EULER_ORDERS = 'xyz') -> DOUBLE_ARRAY:
    """
    This function converts a sequence of 3 euler angles into a rotation matrix.

    The order of the rotations is specified using the `order` keyword argument which recognizes x, y, and z
    for axes of rotation.  For instance, say you have a rotation sequence of (1) rotate about x by xr, (2) rotate about
    y by yr, and (3) rotate about z by zr then you would specify order as 'xyz', and the angles as [xr, yr, zr] (order
    should correspond to the indices of angles).

    The rotation matrix is formed using the :func:`rot_x`, :func:`rot_y`, and
    :func:`rot_z` functions passing in the corresponding index from the angles input according to the order input.

    :param angles: The euler angles
    :param order: The order to apply the rotations in
    :return: The rotation matrix formed by the euler angles
    :raises ValueError: When the ``order`` string contains a character that is not x, y, or z
    """

    rotation = np.eye(3)

    # loop through the angles and their axes and update the total rotation matrix
    for angle, axis in zip(angles, order):

        if axis.upper().lower() == 'x':
            update = rot_x(angle)
        elif axis.upper().lower() == 'y':
            update = rot_y(angle)
        elif axis.upper().lower() == 'z':
            update = rot_z(angle)
        else:
            raise ValueError('Order must only include x, y, and z.  You entered a {} character'.format(axis))

        rotation = update @ rotation

    return rotation

def euler_to_quaternion(angles: Sequence[SCALAR_OR_ARRAY] | DOUBLE_ARRAY, order: EULER_ORDERS = 'xyz') -> DOUBLE_ARRAY:
    """
    This function converts Euler angles into a a rotation quaternion.
    
    Currently this is just done through a call to :func:`.euler_to_rotmat` followed by a call to :func:`.rotmat_to_quaternion`.
    
    :param angles: The angles to convert
    :param order: the order of the angles
        
    :returns: The rotation quaternion(s)
    """
    return rotmat_to_quaternion(euler_to_rotmat(angles, order))


def euler_to_rotvec(angles: Sequence[SCALAR_OR_ARRAY] | DOUBLE_ARRAY, order: EULER_ORDERS = 'xyz') -> DOUBLE_ARRAY:
    """
    This function converts Euler angles into a a rotation vector.
    
    Currently this is just done through a call to :func:`.euler_to_rotmat` followed by a call to :func:`.rotmat_to_rotvec`.
    
    :param angles: The angles to convert
    :param order: the order of the angles
        
    :returns: The rotation quaternion(s)
    """
    return rotmat_to_quaternion(euler_to_rotmat(angles, order))
