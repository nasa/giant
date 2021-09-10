# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


from typing import Sequence, Union
from numbers import Real
from pathlib import Path

import numpy as np

ARRAY_LIKE = Union[Sequence, np.ndarray]
ARRAY_LIKE_2D = Union[Sequence[Sequence], np.ndarray]
SCALAR_OR_ARRAY = Union[ARRAY_LIKE, Real]

PATH = Union[Path, str]

NONENUM = Union[Real, None]
NONEARRAY = Union[ARRAY_LIKE, None]
