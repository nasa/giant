"""
This module contains fundamental mathematical operations and utilities for rotation
calculations. It has no dependencies on other rotation modules to avoid circular imports.
All functions here are pure mathematical operations that can be used as building blocks
for higher-level rotation representations and conversions.
"""

import giant.rotations.core.conversions
import giant.rotations.core.elementals
import giant.rotations.core.quaternion_math

from giant.rotations.core.conversions import (quaternion_to_rotvec, quaternion_to_rotmat, quaternion_to_euler,
                                              rotvec_to_rotmat, rotvec_to_quaternion, rotvec_to_euler,
                                              rotmat_to_quaternion, rotmat_to_rotvec, rotmat_to_euler,
                                              euler_to_rotmat, euler_to_quaternion, euler_to_rotvec)

from giant.rotations.core.elementals import rot_x, rot_y, rot_z, skew

from giant.rotations.core.quaternion_math import quaternion_normalize, quaternion_inverse, quaternion_multiplication, nlerp, slerp

__all__ = ['quaternion_to_rotvec', 'quaternion_to_rotmat', 'quaternion_to_euler', 
           'rotvec_to_rotmat', 'rotvec_to_quaternion', 'rotvec_to_euler',
           'rotmat_to_quaternion', 'rotmat_to_euler', 'rotmat_to_rotvec',
           'euler_to_rotmat', 'euler_to_quaternion', 'euler_to_rotvec',
           'rot_x', 'rot_y', 'rot_z', 'skew',
           'quaternion_normalize', 'quaternion_inverse', 'quaternion_multiplication', 'nlerp', 'slerp']

