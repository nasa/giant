


from typing import Union, Optional, Self

import numpy as np

from giant.ray_tracer.rays import Rays
from giant.ray_tracer.shapes.axis_aligned_bounding_box import AxisAlignedBoundingBox

from giant.rotations import Rotation

from giant._typing import ARRAY_LIKE


class Shape:

    bounding_box: AxisAlignedBoundingBox

    def rotate(self, rotation: Union[ARRAY_LIKE, Rotation]) -> Self: ...

    def translate(self, translation: ARRAY_LIKE) -> Self: ...

    def compute_intersect(self, ray: Rays) -> np.ndarray: ...

    def trace(self, rays: Rays) -> np.ndarray: ...

    def find_limbs(self, scan_center_dir: np.ndarray,
                   scan_dirs: np.ndarray,
                   observer_position: Optional[np.ndarray] = None) -> np.ndarray: ...

    def compute_limb_jacobian(self, scan_center_dir: np.ndarray, scan_dirs: np.ndarray,
                              limb_points: np.ndarray,
                              observer_position: Optional[np.ndarray] = None) -> np.ndarray: ...
