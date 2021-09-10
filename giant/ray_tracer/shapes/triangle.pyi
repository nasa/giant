# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


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
