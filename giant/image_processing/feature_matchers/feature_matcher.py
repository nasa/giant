from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import DTypeLike
from numpy.typing import NDArray
from giant._typing import DOUBLE_ARRAY


class FeatureMatcher(ABC):
    """
    Abstract base class for any image matching process.

    This provides a generalized, high-level interface. The only requirement
    for a concrete implementation is to define a `match_images` method that
    takes two images and returns the coordinates of matching points.
    """

    @abstractmethod
    def match_images(self, image1: NDArray, image2: NDArray) -> DOUBLE_ARRAY:
        """
        Finds and returns correspondences between two images.

        
        :param image1: The first image to match (NumPy array).
        :param image2: The second image to match (NumPy array).

        :returns: A NumPy array of matched keypoint locations with a shape of (N, 2, 2),
                  where N is the number of matches. Each element is structured as
                  [[x1, y1], [x2, y2]], representing the coordinates from
                  image 1 and image 2, respectively.
        """
        pass
    
    