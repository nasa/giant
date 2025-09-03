from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import cv2


from giant.image_processing.utilities.image_validation_mixin import ImageValidationMixin
from giant.utilities.mixin_classes import AttributeEqualityComparison, AttributePrinting, UserOptionConfigured
from giant.utilities.options import UserOptions


@dataclass
class TVL1DenoisingOptions(UserOptions):
    
    lambda_p: float = 1.0
    """
    Lower values leads to more noise removal but more bluring.  
    
    Must be > 0
    """
    
    n_iter: int = 30
    """
    The number of iterations to run
    """

class TVL1Denoising(UserOptionConfigured[TVL1DenoisingOptions], TVL1DenoisingOptions, 
                        AttributeEqualityComparison, AttributePrinting,
                        ImageValidationMixin):
    r"""
    Implements denoising using the total variation approach with a prime dual algorithm.
   
    Essentially we minimize the following 
    .. math::
        \left\|\left\|\nabla \mathbf{I}\right\|\right\| + \lambda\sum_i\left\|\left\|\mathbf{I}-\mathbf{N}_i\right\|\right\|
        
    where :math:`\mathbf{I}` is the denoised image, :math:`\mathbf{N}_i` are the input images, :math:`\nabla \mathbf{I}` is the
    gradient of the smoothed image, :math:`\lambda` is a tuning parameter balancing between smooth images (smaller)
    and crisp images (higher), and :math:`\left\|\left\|\bullet\right\|\right\|` is the :math:`L_2` norm operator.
    
    Generally, this works ok for denoising but can easily mistake dim point sources for noise and can produce weird ring artifacts around
    point sources, therefore, for star/point source only images it is probably preferable to use something like the GaussianDenoising.
    
    For more details on the algorithm see https://hal.science/hal-00437581v1
    """
    
    allowed_dtypes = [np.uint8]
    """
    The allowed datatype.  
    """
    
    def __init__(self, options: TVL1DenoisingOptions | None = None) -> None:
        """
        :param options: the options to configure the class with"""
        super().__init__(TVL1DenoisingOptions, options=options)
    
    def __call__(self, image: NDArray) -> NDArray[np.uint8]:
        out = np.zeros(image.shape, dtype=np.uint8)
        cv2.denoise_TVL1([self._validate_and_convert_image(image)], out, self.lambda_p, self.n_iter)
        
        return out
