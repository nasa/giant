


from typing import Optional

import numpy as np


def sfn_correlator(image: np.ndarray,
                   template: np.ndarray,
                   space_mask: Optional[np.ndarray] = None,
                   intersects: Optional[np.ndarray] = None,
                   search_dist: int = 10,
                   center_predicted: Optional[np.ndarray] = None) -> np.ndarray: ...