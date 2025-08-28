from dataclasses import dataclass

from typing import Literal

import numpy as np
from numpy.typing import NDArray

import cv2


from giant.image_processing.utilities.image_validation_mixin import ImageValidationMixin
from giant.utilities.mixin_classes import AttributeEqualityComparison, AttributePrinting, UserOptionConfigured
from giant.utilities.options import UserOptions


@dataclass
class NlMeansDenoisingOptions(UserOptions):
    
    filter_strength: float = 3.0
    """
    higher values leads to more denoising but more loss of detail.
    """
    
    template_window_size: int = 7
    """
    The size in pixels of the tempalte patch that is used to compute weights.  
    
    Should be odd.
    """
    
    search_window_size: int = 21
    """
    The size in pixels of the window that is used to computed weighted average for a given pixel.
    
    Should be odd.
    
    Higher numbers makes the denoising slower.
    """
    
    norm_type: cv2.NormTypes = cv2.NORM_L2 
    """
    The type of norm used for the weight calculation.
    
    Must be NORM_L1 or NORM_L2
    """

class NlMeansDenoising(UserOptionConfigured[NlMeansDenoisingOptions], NlMeansDenoisingOptions, 
                        AttributeEqualityComparison, AttributePrinting,
                        ImageValidationMixin):
    r"""
    Implements image denoising using Non-local Means Denoising algorithm.
    
    See http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/ for details.
    
    Noise is expected to be gaussian white noise.
    
    This can be fairly effective at removing noise without introducing artifacts, however it can tend to get rid of dim
    point sources as noise.  
    """
    
    allowed_dtypes = [np.uint8, np.uint16]
    """
    The allowed datatype.  
    """
    
    def __init__(self, options: NlMeansDenoisingOptions | None = None) -> None:
        """
        :param options: the options to configure the class with"""
        super().__init__(NlMeansDenoisingOptions, options=options)
    
    def __call__(self, image: NDArray) -> NDArray[np.uint8] | NDArray[np.uint16]:
        return cv2.fastNlMeansDenoising(self._validate_and_convert_image(image), 
                                        dst=None, 
                                        h=[self.filter_strength], 
                                        templateWindowSize=self.template_window_size,
                                        searchWindowSize=self.search_window_size,
                                        normType=self.norm_type) # type: ignore
        

