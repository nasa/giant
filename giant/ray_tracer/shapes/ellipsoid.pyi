# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


from typing import Optional, Tuple, Union, Callable, Self

import numpy as np

from giant.ray_tracer.shapes.solid import Solid
from giant.ray_tracer.shapes.axis_aligned_bounding_box import AxisAlignedBoundingBox
from giant.ray_tracer.rays import Rays
from giant.rotations import Rotation

from giant._typing import ARRAY_LIKE

def quadratic_equations(a: Union[complex, ARRAY_LIKE],
                        b: Union[complex, ARRAY_LIKE],
                        c: Union[complex, ARRAY_LIKE]) -> Union[Tuple[complex, complex],
                                                               Tuple[np.ndarray, np.ndarray]]: ...

class Ellipsoid(Solid):

    bounding_box: AxisAlignedBoundingBox

    albedo_map: Optional[Callable[[np.ndarray], np.ndarray]]

    id: int

    def __init__(self, center: Optional[ARRAY_LIKE] = None,
                 principal_axes: Optional[ARRAY_LIKE] = None,
                 orientation: Optional[ARRAY_LIKE] = None,
                 ellipsoid_matrix: Optional[ARRAY_LIKE] = None,
                 albedo_map: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 _bounding_box: Optional[AxisAlignedBoundingBox] = None,
                 _id: Optional[int] = None): ...

    @property
    def center(self) -> np.ndarray: ...

    @property
    def principal_axes(self) -> np.ndarray: ...

    @property
    def orientation(self) -> np.ndarray: ...

    @property
    def ellipsoid_matrix(self) -> np.ndarray: ...

    def __reduce__(self) -> Tuple['Ellipsoid', Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                     Optional[Callable[[np.ndarray], np.ndarray]],
                                                     AxisAlignedBoundingBox, int]]: ...

    def compute_bounding_box(self): ...

    def intersect(self, rays: Rays) -> np.ndarray: ...

    def compute_intersect(self, ray: Rays) -> np.ndarray: ...

    def compute_normals(self, locs: np.ndarray) -> np.ndarray: ...

    def compute_albedos(self, body_centered_vecs: np.ndarray) -> np.ndarray: ...

    def rotate(self, rotation: Union[Rotation, ARRAY_LIKE]) -> Self: ...

    def translate(self, translation: np.ndarray) -> Self: ...

    def find_limbs(self, scan_center_dir: np.ndarray,
                   scan_dirs: np.ndarray,
                   observer_position: Optional[np.ndarray] = None) -> np.ndarray: ...

    def compute_limb_jacobian(self, scan_center_dir: np.ndarray, scan_dirs: np.ndarray,
                              limb_points: np.ndarray,
                              observer_position: Optional[np.ndarray] = None) -> np.ndarray: ...
