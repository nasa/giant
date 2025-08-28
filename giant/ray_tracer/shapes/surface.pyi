# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


from typing import Union, Optional, Tuple, Self

import numpy as np

from giant.ray_tracer.shapes.axis_aligned_bounding_box import AxisAlignedBoundingBox
from giant.ray_tracer.shapes.ellipsoid import Ellipsoid
from giant.ray_tracer.shapes.shape import Shape
from giant.ray_tracer.rays import Rays
from giant.rotations import Rotation

from giant._typing import ARRAY_LIKE


def find_limbs_surface(target: ARRAY_LIKE, scan_center_dir: ARRAY_LIKE, scan_dirs: ARRAY_LIKE,
                       observer_position: Optional[ARRAY_LIKE] = None, initial_step: Optional[float]=None,
                       max_iterations: int = 25, rtol: float = 1e-12, atol: float = 1e-12) -> np.ndarray: ...

class Surface(Shape):

    reference_ellipsoid: Ellipsoid

    def find_limbs(self, scan_center_dir: ARRAY_LIKE, scan_dirs: ARRAY_LIKE,
                   observer_position: Optional[ARRAY_LIKE] = None) -> np.ndarray: ...


    def compute_limb_jacobian(self, scan_center_dir: np.ndarray, scan_dirs: np.ndarray,
                              limb_points: np.ndarray,
                              observer_position: Optional[np.ndarray] = None) -> np.ndarray: ...

    def compute_intersect(self, ray: Rays) -> np.ndarray: ...

    def trace(self, rays: Rays, omp: bool = True) -> np.ndarray: ...


class RawSurface(Surface):

    def __init__(self, vertices: ARRAY_LIKE, albedos: Union[ARRAY_LIKE, float], facets: ARRAY_LIKE,
                 normals: Optional[ARRAY_LIKE] = None, compute_bounding_box: bool = True,
                 bounding_box: Optional[AxisAlignedBoundingBox] = None,
                 compute_reference_ellipsoid: bool = True, reference_ellipsoid: Optional[Ellipsoid] = None): ...

    def __reduce__(self) -> Tuple[type[Self], Tuple[np.ndarray, Union[np.ndarray, float], np.ndarray, bool,
                                                      Optional[AxisAlignedBoundingBox], bool, Optional[Ellipsoid]]]: ...

    def __eq__(self, other: object) -> bool: ...

    def merge(self, other: Self,
              compute_bounding_box: bool = True,
              compute_reference_ellipsoid: bool = True) -> Self: ...

    @property
    def facets(self) -> np.ndarray: ...

    @facets.setter
    def facets(self, val: ARRAY_LIKE): ...

    @property
    def albedos(self) -> Union[float, np.ndarray]: ...

    @ albedos.setter
    def albedos(self, val: Union[ARRAY_LIKE, float]): ...

    @property
    def stacked_vertices(self) -> np.ndarray: ...

    @property
    def vertices(self) -> np.ndarray: ...

    @vertices.setter
    def vertices(self, val: ARRAY_LIKE): ...

    @property
    def normals(self) -> np.ndarray: ...

    @normals.setter
    def normals(self, val: ARRAY_LIKE): ...

    @property
    def num_faces(self) -> int: ...

    def rotate(self, rotation: Union[Rotation, ARRAY_LIKE]) -> Self: ...

    def translate(self, translation: ARRAY_LIKE) -> Self: ...

    def compute_bounding_box(self): ...

    def compute_reference_ellipsoid(self): ...


class Surface64(RawSurface): ...


class Surface32(RawSurface): ...
