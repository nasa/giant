# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


from typing import Optional

import numpy as np


def sfn_correlator(image: np.ndarray,
                   template: np.ndarray,
                   space_mask: Optional[np.ndarray] = None,
                   intersects: Optional[np.ndarray] = None,
                   search_dist: int = 10,
                   center_predicted: Optional[np.ndarray] = None) -> np.ndarray: ...