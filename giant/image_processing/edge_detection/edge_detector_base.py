from typing import TypeVar, Generic, cast

from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.typing import NDArray

import scipy.signal as sig

import cv2

from giant.utilities.outlier_identifier import get_outliers
from giant.utilities.mixin_classes.attribute_equality_comparison import AttributeEqualityComparison
from giant.utilities.mixin_classes.attribute_printing import AttributePrinting
from giant.image_processing.otsu import otsu
from giant._typing import DOUBLE_ARRAY


SCHARR_KERNEL: NDArray[np.complex128] = np.array([[ -3-3j, 0-10j,  +3 -3j],
                                                 [-10+0j, 0+ 0j, +10 +0j],
                                                 [ -3+3j, 0+10j,  +3 +3j]]) 
"""
The Scharr kernel for gadient computation in both x and y directions (x is the real, y is the imaginary).
"""


EdgeType = TypeVar('EdgeType')
"""
A typevar for the EdgeDetection meta class specifying the type of the returned edges.
"""


class EdgeDetector(Generic[EdgeType], 
                   AttributePrinting,
                   AttributeEqualityComparison,
                   metaclass=ABCMeta):
    """
    An ABC for edge detectors.
    
    Generally, an edge detector should subclass this class
    """
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.horizontal_mask: NDArray[np.bool] | None = None
        """
        The detected edges in the horizontal direction as a 2d boolean array (`True` where there is an edge) 
        """
        
        self.vertical_mask: NDArray[np.bool] | None = None
        """
        The detected edges in the vertical direction as a 2d boolean array (`True` where there is an edge) 
        """
        
        self.horizontal_gradient: DOUBLE_ARRAY | None = None
        """
        The gradient in the horizontal direction
        """
        
        self.vertical_gradient: DOUBLE_ARRAY | None = None
        """
        The gradient in the vertical direction
        """
        
        self.gradient_magnitude: DOUBLE_ARRAY | None = None
        """
        The magnitude of the gradient vector
        """
    
    def prepare_edge_inputs(self, image: NDArray) -> None:
        """
        This method determines pixel level edges in an image by thresholding the image gradients.

        The image gradients are computed by convolving horizontal and vertical Scharr masks with the image to give the
        horizontal and vertical gradients.  The gradient images are then thresholded using :func:`otsu` to determine
        the strongest gradients in the image.  The strong gradients are then searched for local maxima, which become the
        pixel level edges of the image.

        This function inputs the image and outputs a :class:`.EdgeInputs` structure which contains binary images
        with the horizontal and vertical detected edges, and the horizontal, vertical, and magnitudes of the gradients 
        (for further processing)

        :param image: The image to extract the edges from
        :return: the pixel level edges in the horizontal and vertical directions and the horizontal, vertical, and 
                magnitude gradient arrays.
        """

        # compute the image gradients
        gradients: NDArray[np.complex128] = sig.convolve2d(image, SCHARR_KERNEL, mode='same', boundary='symm')
        horizontal_gradient: DOUBLE_ARRAY = gradients.real # gradient from left to right
        vertical_gradient: DOUBLE_ARRAY = gradients.imag # gradient from top to bottom
        gradient_magnitude: DOUBLE_ARRAY = cast(DOUBLE_ARRAY, abs(gradients))

        # get the absolute of the gradients
        abs_horizontal_gradient = np.abs(horizontal_gradient)
        abs_vertical_gradient = np.abs(vertical_gradient)

        # threshold the edges using Otsu's method
        _, normalized_gradient_binned = otsu(gradient_magnitude, 4)

        # get the number of pixels in each threshold level
        num_pix, _ = np.histogram(normalized_gradient_binned, np.arange(5))
        
        # check for outliers
        outliers = get_outliers(num_pix, sigma_cutoff=3)

        if outliers[0]:
            binned_gradient = normalized_gradient_binned > 1.5
        else:
            _, binned_gradient = otsu(gradient_magnitude, 2)

        # do connected components to throw out individual points
        number_of_labels, labs, stats, _ = cv2.connectedComponentsWithStats(binned_gradient.astype(np.uint8))

        for blob in range(number_of_labels):
            if stats[blob, cv2.CC_STAT_AREA] < 2:
                labs[labs == blob] = 0

        binned_gradient = labs > 0

        # determine the horizontal edges
        horiz_mask = np.zeros(image.shape, dtype=bool)

        # horizontal edges correspond to high vertical gradients
        horiz_mask[5:-5, 2:-2] = (binned_gradient[5:-5, 2:-2] &  # check to see that the overall gradient is large
                                    # check that this is a horizontal edge by checking that the vertical_gradient is
                                    # larger
                                    (abs_vertical_gradient[5:-5, 2:-2] >= abs_horizontal_gradient[5:-5, 2:-2]) &
                                    # check that this is a local maxima horizontally
                                    (abs_vertical_gradient[5:-5, 2:-2] >= abs_vertical_gradient[4:-6, 2:-2]) &
                                    (abs_vertical_gradient[5:-5, 2:-2] > abs_vertical_gradient[6:-4, 2:-2]))

        # determine the vertical edges
        vert_mask = np.zeros(image.shape, dtype=bool)

        # vertical edges correspond to high horizontal gradients
        vert_mask[2:-2, 5:-5] = (binned_gradient[2:-2, 5:-5] &  # check to see that the overall gradient is large
                                    # check that this is a vertical edge by checking that the horizontal_gradient is larger
                                    (abs_horizontal_gradient[2:-2, 5:-5] >= abs_vertical_gradient[2:-2, 5:-5]) &
                                    # check that this is a local maxima vertically
                                    (abs_horizontal_gradient[2:-2, 5:-5] >= abs_horizontal_gradient[2:-2, 4:-6]) &
                                    (abs_horizontal_gradient[2:-2, 5:-5] > abs_horizontal_gradient[2:-2, 6:-4]))

        # perpendicular edges correspond to high rss gradients
        perpendicular_mask = np.zeros(image.shape, dtype=bool)

        perpendicular_mask[5:-5, 5:-5] = (
                binned_gradient[5:-5, 5:-5] &  # check to see if the overall gradient is large
                (gradient_magnitude[5:-5, 5:-5] >= gradient_magnitude[5:-5, 4:-6]) &  # horizontal local maxima
                (gradient_magnitude[5:-5, 5:-5] > gradient_magnitude[5:-5, 6:-4]) &
                (gradient_magnitude[5:-5, 5:-5] >= gradient_magnitude[4:-6, 5:-5]) &  # vertical local maxima
                (gradient_magnitude[5:-5, 5:-5] > gradient_magnitude[6:-4, 5:-5]))

        vert_mask |= perpendicular_mask
        
        self.horizontal_mask = horiz_mask
        self.vertical_mask = vert_mask
        self.horizontal_gradient = horizontal_gradient
        self.vertical_gradient = vertical_gradient
        self.gradient_magnitude = gradient_magnitude
        
    @abstractmethod
    def refine_edges(self, image: NDArray, edges: NDArray[np.int64]) -> EdgeType:
        """
        This method should take prior edge locations and refine them to be more accurate (or just return the input).
        
        This is used in GIANT for things like limb finding, where pixel level edges may be detected externally 
        from the class.
        
        :param image: The image the edges are to be refined in
        :param edges: The rough edge locations that are to be refined
        """
        pass
    
    @abstractmethod 
    def identify_edges(self, image: NDArray) -> EdgeType:
        """
        This method should identify edges in an image and then refine them according to the method.
        
        This differs from :meth:`.refine_edges` in that the rough edge points are not already known.
        
        Generally, this can call refine_edges after determining the rough edge points.
        :param image: the image the edges are to be identified in.
        """
        pass


