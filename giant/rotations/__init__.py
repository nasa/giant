import giant.rotations.core
import giant.rotations.frames
import giant.rotations.rotation

from giant.rotations.core import *
from giant.rotations.frames import dynamic_two_vector_frame, two_vector_frame
from giant.rotations.rotation import Rotation

__all__ = ['quaternion_to_rotvec', 'quaternion_to_rotmat', 'quaternion_to_euler', 
           'rotvec_to_rotmat', 'rotvec_to_quaternion', 'rotvec_to_euler',
           'rotmat_to_quaternion', 'rotmat_to_euler', 'rotmat_to_rotvec',
           'euler_to_rotmat', 'euler_to_quaternion', 'euler_to_rotvec',
           'rot_x', 'rot_y', 'rot_z', 'skew',
           'quaternion_normalize', 'quaternion_inverse', 'quaternion_multiplication', 'nlerp', 'slerp',
           'dynamic_two_vector_frame', 'two_vector_frame', 'Rotation']


r"""
This package defines a number of useful routines for converting between various attitude and rotation representations
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

The :class:`.Rotation` object is the primary tool that will be used by users.  It offers a convenient constructor which
accepts 3 common rotation representations to initialize the object.  It also offers operator overloading to allow
a sequence of rotations to be performed using the standard multiplication operator ``*``.  Finally, it offers properties
of the three most common rotation representations (quaternion, matrix, rotation vector).

In addition, there are also a number of utilities provided in this package for converting between different
representations of attitudes and rotations, as well as for working with this data.
"""

