from typing import Literal, Callable

import numpy as np

from giant.rotations.rotation import Rotation

from giant._typing import ARRAY_LIKE, DOUBLE_ARRAY, DatetimeLike


def two_vector_frame(primary_vector: ARRAY_LIKE, secondary_vector: ARRAY_LIKE, 
                     primary_axis: Literal['x', 'y', 'z'], secondary_axis: Literal['x', 'y', 'z'],
                     return_rotation: bool = False) -> DOUBLE_ARRAY | Rotation:
    """
    Compute a 2-vector frame given primary and secondary vectors and their corresponding axes.

    :param primary_vector: The vector defining the primary axis
    :param secondary_vector: The vector providing the constraint for the secondary axis
    :param primary_axis: The axis corresponding to the primary vector (must be x, y, or z)
    :param secondary_axis: The axis corresponding to the secondary vector (must be x, y, or z)
    :param return_reotation: whether to return as a :ref:`.Rotation` object (`True`) or as a numpy array containing the rotation matrix

    :return: rotation defining the transformation to go from the frame the primary and secondary vectors where expressed in to the 2-vector frame 
             either as a 3x3 rotation matrix or as a :ref:`.Rotation` object
    """
    # Normalize the vectors
    primary: np.ndarray = np.asarray(primary_vector) / np.linalg.norm(primary_vector)
    secondary_v: np.ndarray = np.asarray(secondary_vector) / np.linalg.norm(secondary_vector)

    # Determine the third axis based on the right-hand rule
    axes: tuple[str, str, str] = ('x', 'y', 'z')
    primary_axis_str: str = primary_axis.upper().lower()
    secondary_axis_str: str = secondary_axis.upper().lower()
    
    assert primary_axis_str in axes, 'Primary axis must be one of x, y, or z'
    assert secondary_axis_str in axes, 'Secondary axis must be one of x, y, or z'
    third_axis: str = [ax for ax in axes if ax not in [primary_axis_str, secondary_axis_str]][0]

    # Compute the third vector using cross product
    if (axes.index(primary_axis) + 1) % 3 == axes.index(secondary_axis):
        third: np.ndarray = np.cross(primary, secondary_v)
    else:
        third: np.ndarray = np.cross(secondary_v, primary)
    
    third = third / np.linalg.norm(third)

    # Compute the actual secondary vector
    secondary: np.ndarray = np.cross(third, primary)

    # Create the rotation matrix
    frame: np.ndarray = np.array([primary, secondary, third])

    # Rearrange columns to match the specified axes
    col_order: list[int] = [axes.index(primary_axis), axes.index(secondary_axis), axes.index(third_axis)]
    frame = frame[np.argsort(col_order)]

    return frame if not return_rotation else Rotation(frame)


def dynamic_two_vector_frame(primary_vector_func: Callable[[DatetimeLike], np.ndarray], secondary_vector_func: Callable[[DatetimeLike], np.ndarray],
                             primary_axis: Literal['x', 'y', 'z'], secondary_axis: Literal['x', 'y', 'z'],
                             return_rotation: bool = False) -> Callable[[DatetimeLike], np.ndarray | Rotation]:
    """
    Create a dynamic (time dependent) 2-vector frame function.
    
    This is particularly useful in conjunction with the spice interface functionality.
    
    For instance, we can define a time dependent earth nadir frame using the following::
    
    >>> from giant.utilities.spice_interface import SpicePosition
    >>> from giant.rotations import dynamic_two_vector_frame
    >>> from datetime import datetime
    >>> z_dir_fun = SpicePosition('EARTH', 'J2000', 'NONE', 'MY_SPACECRAFT')
    >>> x_const_fun = SpicePosition('SUN', 'J2000', 'NONE', 'MY_SPACECRAFT')
    >>> nadir_frame_fun = dynamic_two_vector_frame(z_dir_fun, x_const_fun, 'z', 'x', return_rotation=True)
    >>> current_rotation_inertial_to_nadir = nadir_frame_fun(datetime.now())

    :param primary_vector_func: Function that returns the primary vector for a given datetime
    :param secondary_vector_func: Function that returns the constraint for the secondary vector for a given datetime
    :param primary_axis: The axis corresponding to the primary vector
    :param secondary_axis: The axis corresponding to the secondary vector
    :param return_reotation: whether to return as a :ref:`.Rotation` object (`True`) or as a numpy array containing the rotation matrix

    :return: callable that returns the rotation matrix for a given datetime
    """
    def frame_at_time(time: DatetimeLike) -> np.ndarray | Rotation:
        primary_vector = primary_vector_func(time)
        secondary_vector = secondary_vector_func(time)
        return two_vector_frame(primary_vector, secondary_vector, primary_axis, secondary_axis, return_rotation=return_rotation)

    return frame_at_time
