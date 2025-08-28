from dataclasses import dataclass, field, asdict

from typing import Sequence

import numpy as np
from numpy.typing import NDArray, DTypeLike

import cv2

from giant.image_processing.feature_matchers.keypoint_matcher import KeypointMatcher, KeypointMatcherOptions
from giant.image_processing.feature_matchers.flann_configuration import FLANN_INDEX_PARAM_TYPES, FLANNIndexAlgorithmType, FLANNIndexKdTreeParams


@dataclass
class SIFTOptions:
    """
    Options for creating the SIFT instance
    """
    n_features: int = 500
    """
    The number of best features to retain. 
    
    The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
    """
    
    n_octave_layers: int = 3
    """
    The number of layers in each octave. 
    
    3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.
    """
    
    contrast_threshold: float = 0.04
    """
    The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. 
    
    The larger the threshold, the less features are produced by the detector.
    
    .. Note::
        The contrast threshold will be divided by nOctaveLayers when the filtering is applied. When nOctaveLayers is set to default and if you want to use the value used in D. Lowe paper, 0.03, set this argument to 0.09.
        edgeThreshold	The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).
    """
    
    edge_threshold: float = 10	
    """
    The threshold used to filter out edge-like features. 
    
    Note that the its meaning is different from the :attr:`contrast_threshold`, i.e. the larger the `edge_threshold`, the less features are filtered out (more features are retained).
    """
    
    sigma: float = 1.6	
    """
    The sigma of the Gaussian applied to the input image at the octave #0. 
    
    If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
    """
    
    enable_precise_upscale: bool = False
    """
    Whether to enable precise upscaling in the scale pyramid, which maps index to x to 2x.
    
    This prevents localization bias. The option is disabled by default.
    """
    
    
@dataclass
class SIFTKeypointMatcherOptions(KeypointMatcherOptions):
    """
    Options for configuring the ORB keypoint matcher
    """
    
    sift_options: SIFTOptions = field(default_factory=SIFTOptions)
    """
    Options used to configure cv2.ORB.create() which configures how keypoints are detected and described
    """
    
    flann_index_algorithm_type: FLANNIndexAlgorithmType = FLANNIndexAlgorithmType.FLANN_INDEX_KDTREE
    """
    What FLANN algorithm to use. 
    
    For SIFT, FLANN_INDEX_KDTREE is recommended
    """
    
    flann_algorithm_parameters: FLANN_INDEX_PARAM_TYPES = field(default_factory=FLANNIndexKdTreeParams)
    """
    The parameters to configure the index in FLANN.
    
    This should match the flann_index_algorithm_type although OpenCV will not complain if it doesn't.
    """


class SIFTKeypointMatcher(KeypointMatcher, SIFTKeypointMatcherOptions):
    """
    Implementation of KeypointMatcher using SIFT for detection and FLANN for matching.
    """
    
    allowed_dtypes: list[DTypeLike] = [np.uint8]
    
    def __init__(self, options: SIFTKeypointMatcherOptions | None = None):
        """
        :param options: how to configure the class
        """
        super().__init__(SIFTKeypointMatcherOptions, options=options)
        
        self.sift = cv2.SIFT.create(nfeatures=self.sift_options.n_features,
                                    nOctaveLayers=self.sift_options.n_octave_layers,
                                    contrastThreshold=self.sift_options.contrast_threshold,
                                    edgeThreshold=self.sift_options.edge_threshold,
                                    sigma=self.sift_options.sigma,
                                    enable_precise_upscale=self.sift_options.enable_precise_upscale)
        
    def detect_keypoints(self, image):
        """
        Detect keypoint descriptors using SIFT.
        
        :param image: The image to search for keypoints
            
        :returns: Tuple of (keypoints, descriptors)
        """
        
        # Convert image to allowable type
        image = self._validate_and_convert_image(image)
        
        keypoints, descriptors = self.sift.detectAndCompute(image, None) # type: ignore
        
        return keypoints, descriptors
        