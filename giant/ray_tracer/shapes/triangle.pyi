


from typing import Union

import numpy as np

from giant.ray_tracer.shapes.surface import Surface64, Surface32
from giant._typing import ARRAY_LIKE

class Triangle64(Surface64):

    @property
    def sides(self) -> np.ndarray: ...

    def compute_normals(self): ...

    def get_albedo(self, point: ARRAY_LIKE, face_index: Union[ARRAY_LIKE, int]) -> Union[np.ndarray, float]: ...

class Triangle32(Surface32):

    @property
    def sides(self) -> np.ndarray: ...

    def compute_normals(self): ...

    def get_albedo(self, point: ARRAY_LIKE, face_index: Union[ARRAY_LIKE, int]) -> Union[np.ndarray, float]: ...
