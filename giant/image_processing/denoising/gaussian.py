from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import cv2


from giant.image_processing.utilities.image_validation_mixin import ImageValidationMixin
from giant.utilities.mixin_classes import AttributeEqualityComparison, AttributePrinting, UserOptionConfigured
from giant.utilities.options import UserOptions


@dataclass
class GaussianDenoisingOptions(UserOptions):
    
    size: tuple[int, int] = (5, 5)
    """
    Gaussian kernel size width, height.
    
    Must both be positive and odd or 0 which implies they should be computed from sigma.
    """
    
    sigma_x: float = 0.0
    """
    Gaussian kernel standard deviation in x direction.
    
    If both this and sigma_y are 0 then they are computed from the size.
    """
    
    sigma_y: float = 0.0
    """
    Guassian kernel standard deviation in y direction. 
    
    If 0 and sigma_x is not 0 then set to sigma_x.
    If both sigma_x and this are 0 then computed from the size.
    """
    
    border_type: cv2.BorderTypes = cv2.BORDER_DEFAULT
    """
    The pixel extrapolation method.
    
    BORDER_WRAP is not supported.
    """

class GaussianDenoising(UserOptionConfigured[GaussianDenoisingOptions], GaussianDenoisingOptions, 
                        AttributeEqualityComparison, AttributePrinting,
                        ImageValidationMixin):
    """
    Uses gaussian smoothing to reduce noise in an image.
    
    All we do is convolve a 2d gaussian kernel with the image. This has the effect of reducing noise spikes
    but also reduces sharpness in the image.  It can be fairly effective for reducing noise in images only
    of point sources without removing too many dim ones.  It is also generally fast.
    """
    
    allowed_dtypes = [np.float32]
    """
    The allowed datatype.  Technically uint8, int16, uint16, and float64 are also supported but we want the output to be float32 so force everything to that.
    """
    
    def __init__(self, options: GaussianDenoisingOptions | None = None) -> None:
        """
        :param options: the options to configure the class with"""
        super().__init__(GaussianDenoisingOptions, options=options)
    
    def __call__(self, image: NDArray) -> NDArray[np.float32]:
        return cv2.GaussianBlur(self._validate_and_convert_image(image), self.size, self.sigma_x, None, self.sigma_y, self.border_type).astype(np.float32)