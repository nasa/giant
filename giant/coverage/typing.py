from typing import Sequence, Optional, Union

import numpy as np

from giant._typing import DOUBLE_ARRAY

JACOBIAN_TYPE = Sequence[Optional[DOUBLE_ARRAY]]
DOP_TYPE = Sequence[Union[np.float64, float]]
