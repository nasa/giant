from typing import Self, cast

import copy

import numpy as np

from giant.rotations.core.conversions import quaternion_to_euler, rotmat_to_quaternion, quaternion_to_rotmat, quaternion_to_rotvec, rotvec_to_quaternion
from giant.rotations.core.quaternion_math import quaternion_inverse, quaternion_multiplication, quaternion_normalize

from giant._typing import ARRAY_LIKE, DOUBLE_ARRAY, EULER_ORDERS


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

    def __new__(cls, data: ARRAY_LIKE | Self | None = None) -> Self:
        """
        Create  a new Rotation instance or return a copy of an existing one.
        """

        if isinstance(data, Rotation):
            return data
        else:
            return super().__new__(cls)

    def __init__(self, data: ARRAY_LIKE | Self | None = None):
        """
        Initialize the Rotation object.
        
        :param data: The rotation data to initialize the class with
        """

        if isinstance(data, Rotation):
            # do nothing
            return

        # initialize the attributes
        self._quaternion = np.array([0, 0, 0, 1.0])
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
    def quaternion(self) -> DOUBLE_ARRAY:
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
    def quaternion(self, data: ARRAY_LIKE | 'Rotation'):

        if isinstance(data, Rotation):

            self._quaternion = data.quaternion

        else:

            data = np.asanyarray(data, dtype=np.float64).ravel()

            if data.size != 4:
                raise ValueError('The quaternion must be length 4')
            
            self._quaternion = data
            quaternion_normalize(self._quaternion)
            
            
        self._mupdate = True
        self._vupdate = True

    @property
    def matrix(self) -> DOUBLE_ARRAY:
        """
        This property stores the matrix representation of the attitude as a numpy array.

        It also enables setting the rotation represented for this object. When setting the rotation data for this
        object , the input should be an orthonormal matrix (Sequence of Sequences) of size 3x3.  When setting this
        property, the :attr:`quaternion` property is automatically updated
        """
        if self._mupdate:
            self._matrix = quaternion_to_rotmat(self._quaternion)
            self._mupdate = False

        assert self._matrix is not None, "the matrix attribute is somehow None but _mupdate is set to false"
        return self._matrix

    @matrix.setter
    def matrix(self, val: ARRAY_LIKE):
        self.quaternion = rotmat_to_quaternion(val)

        self._matrix = np.asarray(val)

        self._mupdate = False

    @property
    def vector(self) -> DOUBLE_ARRAY:
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

        assert self._vector is not None, "the vector attribute is somehow None but _mupdate is set to false"
        return self._vector

    @vector.setter
    def vector(self, val: ARRAY_LIKE):

        self.quaternion = rotvec_to_quaternion(val).ravel()

        self._vector = np.asarray(val).ravel()

        self._vupdate = False

    @property
    def q_vector(self) -> DOUBLE_ARRAY:
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

        return Rotation(quaternion_inverse(self.quaternion))

    def interp_attitude(self, data: ARRAY_LIKE | 'Rotation'):
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

            self.quaternion = data

        else:

            # Ensure the data is in a numpy array
            numpy_data = np.asanyarray(data, dtype=np.float64)

            if numpy_data.size == 4:
                self.quaternion = numpy_data

            elif numpy_data.size == 3:
                self.vector = numpy_data

            elif numpy_data.size == 9:
                self.matrix = numpy_data

            else:
                raise ValueError('The specified rotation data cannot be interpreted.')

    def __eq__(self, other) -> bool:

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

            return Rotation(quaternion_multiplication(self.quaternion, other.quaternion))

        else:

            return NotImplemented

    def __repr__(self) -> str:
        return 'Rotation({0!r})'.format(self.quaternion)

    def __str__(self) -> str:
        return str(self.quaternion)

    def rotate(self, other: ARRAY_LIKE | 'Rotation'):
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

        self.quaternion = Rotation(other) * self

    def copy(self) -> 'Rotation':
        """
        Returns a deep copy of self.

        :return: A deep copy of self breaking all mutability
        """

        return copy.deepcopy(self)
    
    def as_euler_angles(self, order: EULER_ORDERS = 'xyz') -> tuple[float, float, float]:
        """
        Returns the rotation represent as euler angles in the requested order.
        
        :param order: The order of the angles
        :returns: The angles as a tuple of 3 floats.
        """
        
        return cast(tuple[float, float, float], quaternion_to_euler(self.quaternion, order))
