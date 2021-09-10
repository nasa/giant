# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


from typing import Optional, Tuple, Union

import numpy as np

from giant.ray_tracer.rays import Rays
from giant.rotations import Rotation

from giant._typing import ARRAY_LIKE


def min_max_to_bounding_box(min_sides: np.ndarray, max_sides: np.ndarray) -> np.ndarray: ...


class AxisAlignedBoundingBox:

    _rotation: Rotation

    def __init__(self, min_sides: np.ndarray, max_sides: np.ndarray, _rotation: Optional[Rotation] = None): ...

    def __reduce__(self) -> Tuple['AxisAlignedBoundingBox', Tuple[np.ndarray, np.ndarray, Optional[Rotation]]]: ...

    def __eq__(self, other: Optional['AxisAlignedBoundingBox']) -> bool: ...

    @property
    def min_sides(self) -> np.ndarray: ...

    @min_sides.setter
    def min_sides(self, val: ARRAY_LIKE): ...

    @property
    def max_sides(self) -> np.ndarray: ...

    @max_sides.setter
    def max_sides(self, val: ARRAY_LIKE): ...

    @property
    def vertices(self) -> np.ndarray: ...

    def trace(self, rays: Rays, return_distances: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: ...

    def compute_intersect(self, ray: Rays, return_distances: bool = False) -> Union[bool, Tuple[bool, np.ndarray]]: ...

    def rotate(self, rotation: Union[Rotation, ARRAY_LIKE]): ...

    def translate(self, translation: ARRAY_LIKE): ...







