# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


r"""
This module defines a number of useful routines for converting between various attitude and rotation representations
as well as a class which acts as the primary way to express attitude and rotation data in GIANT.

There are a few different rotation representations that are used in this module and their format is described as
follows:

.. _rotation-representation-table:

=================  =====================================================================================================
Representation     Description
=================  =====================================================================================================
quaternion         A 4 element rotation quaternion of the form
                   :math:`\mathbf{q}=\left[\begin{array}{c} q_x \\ q_y \\ q_z \\ q_s\end{array}\right]=
                   \left[\begin{array}{c}\text{sin}(\frac{\theta}{2})\hat{\mathbf{x}}\\
                   \text{cos}(\frac{\theta}{2})\end{array}\right]`
                   where :math:`\hat{\mathbf{x}}` is a 3 element unit vector representing the axis of rotation and
                   :math:`\theta` is the total angle to rotate about that vector.  Note that quaternions are not unique
                   in that the rotation represented by :math:`\mathbf{q}` is the same rotation represented by
                   :math:`-\mathbf{q}`.
rotation vector    A 3 element rotation vector of the form :math:`\mathbf{v}=\theta\hat{\mathbf{x}}` where
                   :math:`\theta` is the total angle to rotate by in radians and :math:`\hat{\mathbf{x}}` is the
                   rotation axis.  Note that rotation vectors are not unique as there is a long and a short vector that
                   both represent the same rotation.
rotation matrix    A :math:`3\times 3` orthonormal matrix representing a rotation such that
                   :math:`\mathbf{T}_B^A\mathbf{y}_A` rotates the 3 element position/direction vector
                   :math:`\mathbf{y}_A` from frame :math:`A` to :math:`B` where :math:`\mathbf{T}_B^A` is the rotation
                   matrix from :math:`A` to :math:`B`.  Rotation matrices uniquely represent a single rotation.
euler angles       A sequence of 3 angles corresponding to a rotation about 3 unit axes.  There are 12 different axis
                   combinations for euler angles.  Mathematically they relate to the rotation matrix as
                   :math:`\mathbf{T}=\mathbf{R}_3(c)\mathbf{R}_2(b)\mathbf{R}_1(a)` where :math:`\mathbf{R}_i(\theta)`
                   represents a rotation about axis :math:`i` (either x, y, or z) by angle :math:`\theta`, :math:`a` is
                   the angle to rotate about the first axis, :math:`b` is angle to rotate about the second axis, and
                   :math:`c` is the angle to rotate about the third axis.
=================  =====================================================================================================

The :class:`Rotation` object is the primary tool that will be used by users.  It offers a convenient constructor which
accepts 3 common rotation representations to initialize the object.  It also offers operator overloading to allow
a sequence of rotations to be performed using the standard multiplication operator ``*``.  Finally, it offers properties
of the three most common rotation representations (quaternion, matrix, rotation vector).

In addition, there are also a number of utilities provided in this module for converting between different
representations of attitudes and rotations, as well as for working with this data.
"""

import copy

import warnings

import sys

from typing import Optional, Union, Sequence, Tuple
from datetime import datetime

import numpy as np

from giant._typing import ARRAY_LIKE, SCALAR_OR_ARRAY, ARRAY_LIKE_2D, Real
from giant import __DepWrapper


__deprecated = {'Attitude': 'Rotation'}


class Rotation:
    """
    A class to represent and manipulate rotations in GIANT.

    The :class:`Rotation` class is the main way that attitude and rotation information is communicated in GIANT.  It
    provides a number of convenient features that make handling attitude data and rotations much easier.  The first of
    these features is is the ability to initialize the class with a rotation vector, rotation matrix, or rotation
    quaternion as described in the :ref:`Rotation Representations <rotation-representation-table>` table. For any case,
    simply provide the constructor with your current representation of your rotation and it will interpret the type of
    input (based on its shape) and store the appropriate data.

    The next feature is property based aliases with caching and setting capabilities.  These properties allow you to
    quickly get any of the 3 primary rotation representations(:attr:`quaternion`, :attr:`matrix`, :attr:`vector`),
    caching the result so the conversion doesn't need to be performed every time you need the representation (but
    smartly knowing when the cache needs updated because the object has been updated).  They also allow you to update
    the entire object in place by directly setting to any of them.

    The final feature is operator overloading so that rotation transformations are easy.  This means that you can do
    things like::

        >>> from giant.rotations import Rotation
        >>> from numpy import pi
        >>> rotation_A2B = Rotation([pi, 0, 0])
        >>> rotation_B2C = Rotation([0, pi/2, 0])
        >>> rotation_A2C = rotation_B2C*rotation_A2B
        >>> rotation_A2C
        Rotation(array([ 7.07106781e-01,  4.32978028e-17, -7.07106781e-01,  4.32978028e-17]))

    In addition to the multiplication operator, the equality operator is also overloaded to check that the
    quaternion representation of two objects is the same.

    In general when a rotation needs to be expressed the :class:`Rotation` object is used in GIANT.
    """

    def __new__(cls, data: Optional[Union[ARRAY_LIKE, 'Rotation']] = None):

        # override the __new__ constructor to return a copy when data is set to a subclass of Rotation
        if isinstance(data, Rotation):
            return data
        else:
            return super().__new__(cls)

    def __init__(self, data: Optional[Union[ARRAY_LIKE, 'Rotation']] = None):
        """
        :param data: The rotation data to initialize the class with
        """

        if isinstance(data, Rotation):
            # do nothing
            return

        # initialize the attributes
        self._quaternion = None
        self._matrix = None
        self._vector = None
        self._mupdate = True
        self._vupdate = True

        # check to see if rotation data was supplied
        if data is None:
            data = [0, 0, 0, 1]

        # interpret the rotation data.
        self.interp_attitude(data)

    @property
    def quaternion(self) -> np.ndarray:
        """
        This property stores the quaternion representation of the attitude as a numpy array.

        It also enables setting the rotation represented for this object. When setting the rotation data for this
        object, the input should have a length of 4 and be convertable to a numpy array.  It should also be of unit
        length (as is required for rotation quaternions).  If the set value is not of unit length then it will be
        automatically normalized before being stored and a warning will be printed.

        In order to make the quaternion representations unique in GIANT, setting a quaternion to this property will
        enforce that the scalar component is positive.
        """

        return self._quaternion

    @quaternion.setter
    def quaternion(self, data):

        if isinstance(data, Rotation):

            data = data.quaternion

        else:

            data = np.asarray(data).flatten().astype(np.float64)

            length = np.linalg.norm(data)

            if not((1-1e-15) <= length <= (1+1e-15)):
                warnings.warn('Non-unit length quaternion ({:e}).  Normalizing'.format(1-length))
                # enforce unit length constraint
                data /= length

            # enforce positive scalar to make quaternions unique
            data *= np.sign(data[-1]) if data[-1] else 1

        if data.size != 4:
            raise ValueError('The quaternion must be length 4')

        self._quaternion = data
        self._mupdate = True
        self._vupdate = True

    @property
    def q(self) -> np.ndarray:
        """
        This is an alias to the :attr:`.quaternion` property.
        """
        return self._quaternion

    @q.setter
    def q(self, data):
        self.quaternion = data

    @property
    def matrix(self) -> np.ndarray:
        """
        This property stores the matrix representation of the attitude as a numpy array.

        It also enables setting the rotation represented for this object. When setting the rotation data for this
        object , the input should be an orthonormal matrix (Sequence of Sequences) of size 3x3.  When setting this
        property, the :attr:`quaternion` property is automatically updated
        """
        if self._mupdate:
            self._matrix = quaternion_to_rotmat(self._quaternion)
            self._mupdate = False

        return self._matrix

    @matrix.setter
    def matrix(self, val):
        self.quaternion = rotmat_to_quaternion(val)

        self._matrix = np.asarray(val)

        self._mupdate = False

    @property
    def vector(self) -> np.ndarray:
        """
        This property stores the vector representation of the attitude as a numpy array.

        It also enables setting the rotation represented for this object. When setting the rotation data for this
        object, the input should be a length 3 rotation vector (Sequence) according to the form specified in the
        :ref:`Rotation Representations <rotation-representation-table>`.  When setting this property, the
        :attr:`quaternion` property is automatically updated.
        """

        if self._vupdate:
            self._vector = quaternion_to_rotvec(self._quaternion)

            self._vupdate = False

        return self._vector

    @vector.setter
    def vector(self, val):

        self.quaternion = rotvec_to_quaternion(val).flatten()

        self._vector = np.asarray(val).flatten()

        self._vupdate = False

    @property
    def q_vector(self) -> np.ndarray:
        """
        This is an alias to the first three elements of the quaternion array (the vector portion of the quaternion)

        This property is read only.
        """

        return self._quaternion[:3]

    @property
    def q_scalar(self) -> float:
        """
        This is an alias to the last element of the quaternion array (the scalar portion of the quaternion)

        This property is read only.
        """

        return self._quaternion[-1]

    def inv(self) -> 'Rotation':
        """
        This method returns the inverse rotation of the current instance as a new ``Rotation`` object.

        The inverse is performed on the quaternion representation since it simply involves negating the vector
        portion of the quaternion.

        See :func:`quaternion_inverse` for more information.

        :return: The inverse rotation
        """

        return quaternion_inverse(self)

    def interp_attitude(self, data: Union[ARRAY_LIKE, 'Rotation']):
        """
        This method interprets attitude data based on its shape and type.

        If the type of the input is an :class:`Rotation` object then the current instance is overwritten with the data
        from the :class:`Rotation` object.  If the type is a sequence then the data is interpreted by size.  If the
        total size of the Sequence is 4 then the data is presumed to be a quaternion.  If the total size of the sequence
        is 3 then the data is presumed to be a rotation vector.  If the total size of the sequence is 9 then the data
        is presumed to be a rotation matrix.

        :raises ValueError: If the size of the input data is not 3, 4, or 9
        :param data: The rotation data to be interpreted
        """

        if isinstance(data, Rotation):

            self.q = data

        else:

            # Ensure the data is in a numpy array
            numpy_data = np.asarray(data)

            if numpy_data.size == 4:
                self.quaternion = numpy_data

            elif numpy_data.size == 3:
                self.vector = numpy_data

            elif numpy_data.size == 9:
                self.matrix = numpy_data

            else:
                raise ValueError('The specified rotation data cannot be interpreted.')

    def __eq__(self, other: Union['Rotation', ARRAY_LIKE]) -> bool:

        # check that other is a rotation object, if not make it into one
        if not isinstance(other, Rotation):
            try:
                other = Rotation(other)
            except ValueError:
                # if we're here then other isn't a representation of rotation that GIANT understands
                return False

        # check that the quaternions are the same
        return (self._quaternion == other.quaternion).all()

    def __mul__(self, other: 'Rotation') -> 'Rotation':

        # use quaternion multiplication
        if isinstance(other, Rotation):

            return Rotation(quaternion_multiplication(self, other))

        else:

            return NotImplemented

    def __repr__(self) -> str:
        return 'Rotation({0!r})'.format(self.q)

    def __str__(self) -> str:
        return str(self.q)

    def rotate(self, other: Union[ARRAY_LIKE, 'Rotation']):
        """
        Performs a left inplace rotation by other.

        Using this method overwrites the data stored in self with self rotated by other.  That is

        .. math::
            \\mathbf{q}_s = \\mathbf{q}_o\\otimes\\mathbf{q}_s

        where :math:`\\mathbf{q}_s` is the attitude quaternion representation of self, :math:`\\mathbf{q}_o` is the
        attitude quaternion representation of other, and :math:`\\otimes` indicates non-hamiltonian multiplication.

        See the :func:`quaternion_multiplication` function for more information.

        :param other: The data to rotate self with
        """

        self.q = Rotation(other) * self

    def copy(self) -> 'Rotation':
        """
        Returns a deep copy of self.

        :return: A deep copy of self breaking all mutability
        """

        return copy.deepcopy(self)


def quaternion_inverse(quaternion: Union[ARRAY_LIKE, Rotation]) -> Union[np.ndarray, Rotation]:
    r"""
    This function provides the inverse of a rotation quaternion of the form discussed in
    :ref:`Rotation Representations <rotation-representation-table>`.

    The inverse of a rotation quaternion is defined such that
    :math:`\mathbf{q}\otimes\mathbf{q}^{-1}=\mathbf{q}_I` where
    :math:`\mathbf{q}_I=\left[\begin{array}{cccc}0&0&0&1\end{array}\right]^T` is the identity quaternion which
    corresponds to the identity matrix (or no rotation) and :math:`\otimes` indicates quaternion multiplication.
    Mathematically this corresponds to negating the vector portion of the quaternion:

    .. math::
        \mathbf{q}=\left[\begin{array}{c}\text{sin}(\frac{\theta}{2})\hat{\mathbf{x}}\\
        \text{cos}(\frac{\theta}{2})\end{array}\right]\\
        \mathbf{q}^{-1}=\left[\begin{array}{c}-\text{sin}(\frac{\theta}{2})\hat{\mathbf{x}}\\
        \text{cos}(\frac{\theta}{2})\end{array}\right]

    This function is also vectorized, meaning that you can specify multiple rotation quaternions to be inversed by
    specifying each quaternion as a column.  Regardless of whether you are converting 1 or many
    quaternions the first axis must have a length of 4.

    This function makes the output have the same number of dimensions as the input.  Therefore, if the input is one
    dimensional, then the output is one dimensional, and if the input is two dimensional then the output will be two
    dimensional. In addition, if you supply this function with an Rotation object then this function will return an
    Rotation object of the inverse rotation.

    :param quaternion: The rotation quaternion(s) to be inverted
    :return: a numpy array or Rotation representing the inverse quaternion corresponding to the input quaternion
    """

    # check to see if the input is an Rotation object
    if isinstance(quaternion, Rotation):
        # retrieve the quaternion array and break the mutability
        quaternion = quaternion.q.copy()

        # negate the vector portion
        quaternion[:3] *= -1

        # return the inverse quaternion as an Rotation object
        return Rotation(quaternion)

    else:
        # ensure the value is an array and break mutability
        quaternion = np.asarray(copy.deepcopy(quaternion))

        # negate the vector portion
        quaternion[:3] *= -1

        # return the inverse quaternion(s) as an array
        return quaternion


def quaternion_multiplication(quaternion_1_in: Union[ARRAY_LIKE, Rotation],
                              quaternion_2_in: Union[ARRAY_LIKE, Rotation]) -> Union[np.ndarray, Rotation]:
    r"""
    This function performs the hamiltonian quaternion multiplication operation.

    The quaternions should be of the form as specified in
    :ref:`Rotation Representations <rotation-representation-table>`.

    The hamiltonian multiplication is defined such that
    `q_from_A_to_C = quaternion_multiplication(q_from_B_to_C, q_from_A_to_B)`

    Mathematically this is given by:

    .. math::
        \mathbf{q}_1\otimes\mathbf{q}_2=\left[\begin{array}{c}q_{s1}\mathbf{q}_{v2} + q_{s2}\mathbf{q}_{v1} +
        \mathbf{q}_{v1}\times\mathbf{q}_{v2}\\
        q_{s1}q_{s2}-\mathbf{q}_{v1}^T\mathbf{q}_{v2}\end{array}\right]

    This function is vectorized, therefore you can input multiple quaternions as a 4xn array where each column is an
    independent quaternion.

    :param quaternion_1_in: The first quaternion to multiply
    :param quaternion_2_in: The second quaternion to multiply
    :return: The non-hamiltonian product of quaternion_1 and quaternion_2
    """

    rquat = False

    if isinstance(quaternion_1_in, Rotation):
        quaternion_1 = quaternion_1_in.q
        rquat = True
    else:
        quaternion_1 = np.asarray(quaternion_1_in)

    if isinstance(quaternion_2_in, Rotation):
        quaternion_2 = quaternion_2_in.q
        rquat = True
    else:
        quaternion_2 = np.asarray(quaternion_2_in)

    if np.shape(quaternion_1)[0] != 4:
        raise ValueError('The length of the first axis must be 4')

    if np.shape(quaternion_2)[0] != 4:
        raise ValueError('The length of the first axis must be 4')

    qs1 = quaternion_1[-1]
    qv1 = quaternion_1[0:3]

    qs2 = quaternion_2[-1]
    qv2 = quaternion_2[0:3]

    qout = np.concatenate([qs1 * qv2 + qs2 * qv1 + np.cross(qv1, qv2, axis=0),
                           [qs1 * qs2 - (qv1 * qv2).sum(axis=0)]], axis=0)

    if rquat:
        return Rotation(qout)

    else:
        return qout


def quaternion_to_rotvec(quaternion: Union[ARRAY_LIKE, Rotation]) -> np.ndarray:
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

    if isinstance(quaternion, Rotation):
        # extract the quaternion portion if the input is an Rotation object
        quaternion = quaternion.q

    else:
        # ensure we have a numpy array of the quaternion(s)
        quaternion = np.asarray(quaternion)

    # check to ensure the first axis has length 4
    if quaternion.shape[0] != 4:
        raise ValueError('The first axis must be length 4')

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


def quaternion_to_rotmat(quaternion: Union[ARRAY_LIKE, Rotation]) -> np.ndarray:
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
    if isinstance(quaternion, Rotation):
        quaternion = quaternion.q

    else:
        quaternion = np.asarray(quaternion)

    # extract the scalar and vector portion of the quaternion(s)
    qs = quaternion[-1].reshape(-1, 1, 1)
    qv = quaternion[:3].reshape(3, -1)

    # form and return the rotation matrix
    return ((qs ** 2 - (qv * qv).sum(axis=0).reshape(-1, 1, 1)) * np.eye(3) + 2 * np.einsum('ij,jk->jik', qv, qv.T) +
            2 * qs * skew(qv)).squeeze()


def quaternion_to_euler(quaternion: Union[ARRAY_LIKE, Rotation],
                        order: str = 'xyz') -> Tuple[SCALAR_OR_ARRAY, SCALAR_OR_ARRAY, SCALAR_OR_ARRAY]:
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


def rotvec_to_rotmat(vector: ARRAY_LIKE) -> np.ndarray:
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

    # ensure we are working with an array
    vector = np.asarray(vector)

    # check to be sure the first axis has length 3
    if vector.shape[0] != 3:
        raise ValueError('First axis must be length 3')

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


def rotvec_to_quaternion(rot_vec: ARRAY_LIKE) -> np.ndarray:
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

    # ensure we have a ndarray
    rot_vec = np.asarray(rot_vec)

    # check to ensure the first axis is length 3
    if rot_vec.shape[0] != 3:
        raise ValueError('The length of the first axis must be 3')

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


def rotmat_to_quaternion(rotation_matrix: ARRAY_LIKE_2D) -> np.ndarray:
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

    # ensure the input is an array and a float
    rotation_matrix = np.asarray(rotation_matrix).astype(np.float64).squeeze()

    # check to ensure the last 2 axes are length 3
    if rotation_matrix.shape[-2:] != (3, 3):
        raise ValueError('Invalid Shape')

    # compute the scalar portion of the quaternion.  The max(..., 0) is to avoid rounding errors.
    q_scalar = 0.5 * np.sqrt(np.maximum(np.trace(rotation_matrix.T) + 1, 0))

    # extract the diagonal elements from the matrix(ces)
    t_diag = np.diagonal(rotation_matrix.T).reshape((-1, 1, 3))

    temp_mat = np.array([[1, -1, -1],
                         [-1, 1, -1],
                         [-1, -1, 1]])

    # form the vector portion of the quaternion.  The max(..., 0) is to avoid rounding errors
    q_vec = np.sqrt(np.maximum((temp_mat * t_diag).sum(axis=-1) + 1, 0)).squeeze() / 2  # type:np.ndarray

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


def rotmat_to_euler(matrix: ARRAY_LIKE_2D,
                    order: str = 'xyz') -> Tuple[SCALAR_OR_ARRAY, SCALAR_OR_ARRAY, SCALAR_OR_ARRAY]:
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

    matrix = np.asarray(matrix)

    if fixed_order == 'xyx':

        f1 = matrix[..., 1, 0]
        f2 = matrix[..., 2, 0]
        s1 = matrix[..., 0, 0]
        t1 = matrix[..., 0, 1]
        t2 = -matrix[..., 0, 2]

        return (-np.arctan2(f1, f2), -np.arccos(s1), -np.arctan2(t1, t2))[::-1]

    elif fixed_order == 'zyx':

        f1 = matrix[..., 1, 2]
        f2 = matrix[..., 2, 2]
        s1 = matrix[..., 0, 2]
        t1 = matrix[..., 0, 1]
        t2 = matrix[..., 0, 0]

        return (-np.arctan2(f1, f2), np.arcsin(s1), -np.arctan2(t1, t2))[::-1]

    elif fixed_order == 'xzx':

        f1 = matrix[..., 2, 0]
        f2 = -matrix[..., 1, 0]
        s1 = matrix[..., 0, 0]
        t1 = matrix[..., 0, 2]
        t2 = matrix[..., 0, 1]

        return (-np.arctan2(f1, f2), -np.arccos(s1), -np.arctan2(t1, t2))[::-1]

    elif fixed_order == 'yzx':

        f1 = -matrix[..., 2, 1]
        f2 = matrix[..., 1, 1]
        s1 = matrix[..., 0, 1]
        t1 = -matrix[..., 0, 2]
        t2 = matrix[..., 0, 0]

        return (-np.arctan2(f1, f2), -np.arcsin(s1), -np.arctan2(t1, t2))[::-1]

    elif fixed_order == 'yxy':

        f1 = matrix[..., 0, 1]
        f2 = -matrix[..., 2, 1]
        s1 = matrix[..., 1, 1]
        t1 = matrix[..., 1, 0]
        t2 = matrix[..., 1, 2]

        return (-np.arctan2(f1, f2), -np.arccos(s1), -np.arctan2(t1, t2))[::-1]

    elif fixed_order == 'zxy':

        f1 = -matrix[..., 0, 2]
        f2 = matrix[..., 2, 2]
        s1 = matrix[..., 1, 2]
        t1 = -matrix[..., 1, 0]
        t2 = matrix[..., 1, 1]

        return (-np.arctan2(f1, f2), -np.arcsin(s1), -np.arctan2(t1, t2))[::-1]

    elif fixed_order == 'xzy':

        f1 = matrix[..., 2, 0]
        f2 = matrix[..., 0, 0]
        s1 = matrix[..., 1, 0]
        t1 = matrix[..., 1, 2]
        t2 = matrix[..., 1, 1]

        return (-np.arctan2(f1, f2), np.arcsin(s1), -np.arctan2(t1, t2))[::-1]

    elif fixed_order == 'yzy':

        f1 = matrix[..., 2, 1]
        f2 = matrix[..., 0, 1]
        s1 = matrix[..., 1, 1]
        t1 = matrix[..., 1, 2]
        t2 = -matrix[..., 1, 0]

        return (-np.arctan2(f1, f2), -np.arccos(s1), -np.arctan2(t1, t2))[::-1]

    elif fixed_order == 'yxz':

        f1 = matrix[..., 0, 1]
        f2 = matrix[..., 1, 1]
        s1 = matrix[..., 2, 1]
        t1 = matrix[..., 2, 0]
        t2 = matrix[..., 2, 2]

        return (-np.arctan2(f1, f2), np.arcsin(s1), -np.arctan2(t1, t2))[::-1]

    elif fixed_order == 'zxz':

        f1 = matrix[..., 0, 2]
        f2 = matrix[..., 1, 2]
        s1 = matrix[..., 2, 2]
        t1 = matrix[..., 2, 0]
        t2 = -matrix[..., 2, 1]

        return (-np.arctan2(f1, f2), -np.arccos(s1), -np.arctan2(t1, t2))[::-1]

    elif fixed_order == 'xyz':

        f1 = -matrix[..., 1, 0]
        f2 = matrix[..., 0, 0]
        s1 = matrix[..., 2, 0]
        t1 = -matrix[..., 2, 1]
        t2 = matrix[..., 2, 2]

        return (-np.arctan2(f1, f2), -np.arcsin(s1), -np.arctan2(t1, t2))[::-1]

    elif fixed_order == 'zyz':

        f1 = matrix[..., 1, 2]
        f2 = -matrix[..., 0, 2]
        s1 = matrix[..., 2, 2]
        t1 = matrix[..., 2, 1]
        t2 = matrix[..., 2, 0]

        return (-np.arctan2(f1, f2), -np.arccos(s1), -np.arctan2(t1, t2))[::-1]


def euler_to_rotmat(angles: Sequence[SCALAR_OR_ARRAY], order: str = 'xyz') -> np.ndarray:
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


def rot_x(theta: SCALAR_OR_ARRAY) -> np.ndarray:
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


def rot_y(theta: SCALAR_OR_ARRAY) -> np.ndarray:
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


def rot_z(theta: SCALAR_OR_ARRAY) -> np.ndarray:
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


def skew(vector: ARRAY_LIKE) -> np.ndarray:
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

    vector = np.asarray(vector)

    if vector.shape[0] != 3:
        raise ValueError('The length of the first axis must be equal to 3')

    if vector.ndim > 1:
        zeros = np.zeros(vector.shape[-1])

    else:
        zeros = 0

    return np.array([zeros, -vector[2], vector[1],
                     vector[2], zeros, -vector[0],
                     -vector[1], vector[0], zeros]).T.reshape(-1, 3, 3).squeeze()


def nlerp(quaternion0: Union[ARRAY_LIKE, Rotation], quaternion1: Union[ARRAY_LIKE, Rotation],
          time: Union[Real, datetime],
          time0: Union[Real, datetime] = 0, time1: Union[Real, datetime] = 1) -> Union[np.ndarray, Rotation]:
    r"""
    This function performs normalized linear interpolation of rotation quaternions.

    NLERP of quaternions involves first performing a linear interpolation between the two vectors, and then normalizing
    the interpolated result to have unit length.  That is:

    .. math::
        \mathbf{q}=\frac{\mathbf{q}_0(1-p)+\mathbf{q}_1p}
        {\left\|\mathbf{q}_0(1-p)+\mathbf{q}_1p\right\|}

    where :math:`\mathbf{q}` is the interpolated quaternion, :math:`\mathbf{q}_0` is the starting quaternion,
    :math:`\mathbf{q}_1` is the ending quaternion, and :math:`p` is the fractional percent of the way between
    :math:`\mathbf{q}_0` and :math:`\mathbf{q}_1` that we want to interpolate at (:math:`p\in[0, 1]`)

    When using this function you can either specify the argument `time` as the fractional percent that you want to
    interpolate at, or specify the keyword arguments `time0` and `time1` to be the times corresponding to the first and
    second quaternion respectively and the function will compute the fractional percent for you.  When using this method
    it is also possible to specify all three of `time`, `time0`, and `time1` as python datetime objects.

    .. warning::
        NLERP is a very fast and efficient interpolation method that is fine for short interpolation intervals; however,
        it does not perform a constant angular velocity interpolation (and instead performs a constant linear velocity
        interpolation), therefore it is not well suited to interpolating over long time intervals. If you need to
        interpolate over larger time intervals it is better to use the :func:`slerp` function which does perform
        constant angular velocity interpolation (but is less efficient).

    :param quaternion0: The starting quaternion(s)
    :param quaternion1: The ending quaternion(s)
    :param time: The time to interpolate the quaternions at, as a fractional percent or as the actual time between
                `time0` and `time1`
    :param time0: the time(s) corresponding to the first quaternion(s). Leave at 0 if you are specifying `time` as a
                  fractional percent
    :param time1: the time(s) corresponding to the second quaternion(s). Leave at 1 if you are specifying `time` as a
                  fractional percent
    :return: The interpolated quaternion(s)
    """

    # compute the fractional percent we are interpolating at
    dt = (time - time0) / (time1 - time0)

    rquat = False

    # extract the quaternion values as arrays and check to see whether we should return and Rotation object
    if isinstance(quaternion0, Rotation):

        q0 = quaternion0.quaternion
        rquat = True

    else:
        q0 = np.asarray(quaternion0)

    if isinstance(quaternion1, Rotation):

        q1 = quaternion1.quaternion
        rquat = True

    else:
        q1 = np.asarray(quaternion1)

    # perform the linear interpolation
    q = q0 * (1 - dt) + q1 * dt

    # perform the normalization
    q /= np.linalg.norm(q, axis=0, keepdims=True)

    # convert to an Rotation object if needed
    if rquat:
        q = Rotation(q)

    # return the interpolated quaternion(s)
    return q


def slerp(quaternion0: Union[ARRAY_LIKE, Rotation], quaternion1: Union[ARRAY_LIKE, Rotation],
          time: Union[Real, datetime],
          time0: Union[Real, datetime] = 0, time1: Union[Real, datetime] = 1) -> Union[np.ndarray, Rotation]:
    r"""
    This function performs spherical linear interpolation of rotation quaternions.

    SLERP of quaternions involves performing a linear interpolation along the great circle arc connecting the two
    quaternions. That is:

    .. math::
        \omega = \text{cos}^{-1}(\mathbf{q}_0^T\mathbf{q}_1)\\
        \mathbf{q}=\mathbf{q}_0\text{cos}(p\omega)+
        \text{sin}(p\omega)\frac{\mathbf{q}_1-\mathbf{q}_0\text{cos}(\omega)}
        {\left\|\mathbf{q}_1-\mathbf{q}_0\text{cos}(\omega)\right\|}\\
        \mathbf{q} = \frac{\mathbf{q}}{\left\|\mathbf{q}\right\|}

    where :math:`\mathbf{q}` is the interpolated quaternion, :math:`\mathbf{q}_0` is the starting quaternion,
    :math:`\mathbf{q}_1` is the ending quaternion, :math:`\omega` is the angle between the first and second quaternion,
    and :math:`p` is the fractional percent of the way between
    :math:`\mathbf{q}_0` and :math:`\mathbf{q}_1` that we want to interpolate at (:math:`p\in[0, 1]`)

    When using this function you can either specify the argument `time` as the fractional percent that you want to
    interpolate at, or specify the keyword arguments `time0` and `time1` to be the times corresponding to the first and
    second quaternion respectively and the function will compute the fractional percent for you.  When using this method
    it is also possible to specify all three of `time`, `time0`, and `time1` as python datetime objects.

    :param quaternion0: The starting quaternion(s)
    :param quaternion1: The ending quaternion(s)
    :param time: The time to interpolate the quaternions at, as a fractional percent or as the actual time between
                `time0` and `time1`
    :param time0: the time(s) corresponding to the first quaternion(s). Leave at 0 if you are specifying `time` as a
                  fractional percent
    :param time1: the time(s) corresponding to the second quaternion(s). Leave at 1 if you are specifying `time` as a
                  fractional percent
    :return: The interpolated quaternion(s)
    """

    # get the fractional percent to interpolate between q0 and q1
    dt = (time - time0) / (time1 - time0)

    rquat = False

    # extact the quaternion as an array and see if we need to return and Rotation object
    if isinstance(quaternion0, Rotation):

        q0 = quaternion0.quaternion.flatten().astype(np.float64)
        rquat = True

    else:
        q0 = np.asarray(quaternion0).flatten().astype(np.float64)

    if isinstance(quaternion1, Rotation):

        q1 = quaternion1.quaternion.flatten().astype(np.float64)
        rquat = True

    else:
        q1 = np.asarray(quaternion1).flatten().astype(np.float64)

    # enforce unit normalization
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)

    # get the cosine of the angle between the quaternions
    cos_angle = np.inner(q0, q1)

    if cos_angle > 0.9995:
        # if the quaternions are really close revert to nlerp
        return nlerp(q0, q1, dt)

    elif cos_angle < 0:
        # if the dot product is negative negate the second quaternion to ensure the shorter path is taken
        q1 *= -1
        cos_angle *= -1

    cos_angle = np.clip(cos_angle, -1, 1)  # ensure the domain for acos (only will leave due to numerical issues)

    angle0 = np.arccos(cos_angle)  # angle between q0 and q1
    angle = angle0 * dt  # angle between q0 and q

    # form an orthonormal basis
    qb = q1 - q0 * cos_angle

    qb /= np.linalg.norm(qb)

    # perform the interpolation
    q = q0 * np.cos(angle) + qb * np.sin(angle)
    q /= np.linalg.norm(q)

    # convert to an Rotation object if necessary
    if rquat:
        q = Rotation(q)

    return q


__dep = __DepWrapper(sys.modules[__name__], __deprecated)

sys.modules[__name__] = __dep
