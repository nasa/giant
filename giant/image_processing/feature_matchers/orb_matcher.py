from dataclasses import dataclass, field, asdict

from typing import Sequence

import numpy as np
from numpy.typing import NDArray, DTypeLike

import cv2

from giant.image_processing.feature_matchers.keypoint_matcher import KeypointMatcher, KeypointMatcherOptions
from giant.image_processing.feature_matchers.flann_configuration import FLANNIndexAlgorithmType, FLANNIndexLSHParams, \
                                                                        FLANN_INDEX_PARAM_TYPES

from giant.utilities.options import UserOptions
from giant.utilities.mixin_classes.attribute_equality_comparison import AttributeEqualityComparison
from giant.utilities.mixin_classes.attribute_printing import AttributePrinting
from giant.utilities.mixin_classes.user_option_configured import UserOptionConfigured


@dataclass
class ORBOptions:
    """
    Options for creating the ORB instance
    """
    
    n_features: int = 500
    """
    The maximum number of features to retain.
    """
    
    scale_factor: float = 1.2
    """
    Pyramid decimation ratio, greater than 1. 
    
    scaleFactor==2 means the classical pyramid, where each next level has 4x less pixels than the previous, 
    but such a big scale factor will degrade feature matching scores dramatically. On the other hand, too 
    close to 1 scale factor will mean that to cover certain scale range you will need more pyramid levels 
    and so the speed will suffer.
    """
    
    n_levels: int = 8
    """
    The number of pyramid levels. 
    
    The smallest level will have linear size equal to input_image_linear_size/pow(scaleFactor, n_levels - first_level).
    """
    
    edge_threshold: int = 31
    """
    This is size of the border where the features are not detected. 
    
    It should roughly match the patchSize parameter.
    """

    first_level: int = 0
    """
    The level of pyramid to put source image to. 
    
    Previous layers are filled with upscaled source image.
    """
    
    wta_k: int = 2
    """
    The number of points that produce each element of the oriented BRIEF descriptor. 
    
    The default value 2 means the BRIEF where we take a random point pair and compare their brightnesses, so we get 0/1 
    response. Other possible values are 3 and 4. For example, 3 means that we take 3 random points (of course, those 
    point coordinates are random, but they are generated from the pre-defined seed, so each element of BRIEF descriptor 
    is computed deterministically from the pixel rectangle), find point of maximum brightness and output index of the 
    winner (0, 1 or 2). Such output will occupy 2 bits, and therefore it will need a special variant of Hamming distance, 
    denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each bin (that will also 
    occupy 2 bits with possible values 0, 1, 2 or 3).
    """
    
    score_type: int = cv2.ORB_HARRIS_SCORE
    """
    The default ORB_HARRIS_SCORE means that Harris algorithm is used to rank features (the score is written to 
    KeyPoint::score and is used to retain best n_features features); FAST_SCORE is alternative value of the parameter that 
    produces slightly less stable keypoints, but it is a little faster to compute.
    """
    
    patch_size: int = 31
    """
    Size of the patch used by the oriented BRIEF descriptor. 
    
    Of course, on smaller pyramid layers the perceived image area covered by a feature will be larger
    """
    
    fast_threshold: int = 20
    """
    The fast threshold.
    """
    

@dataclass
class ORBKeypointMatcherOptions(KeypointMatcherOptions):
    """
    Options for configuring the ORB keypoint matcher
    """
    
    orb_options: ORBOptions = field(default_factory=ORBOptions)
    """
    Options used to configure cv2.ORB.create() which configures how keypoints are detected and described
    """
    
    flann_index_algorithm_type: FLANNIndexAlgorithmType = FLANNIndexAlgorithmType.FLANN_INDEX_LSH
    """
    What FLANN algorithm to use. 
    
    For ORB, FLANN_INDEX_LSH is recommended
    """
    
    flann_algorithm_parameters: FLANN_INDEX_PARAM_TYPES = field(default_factory=FLANNIndexLSHParams)
    """
    The parameters to configure the index in FLANN.
    
    This should match the flann_index_algorithm_type although OpenCV will not complain if it doesn't.
    """


class OrbKeypointMatcher(KeypointMatcher, ORBKeypointMatcherOptions):
    """
    Implementation of KeypointMatcher using ORB for detection and FLANN for matching.
    """
    
    allowed_dtypes: list[DTypeLike] = [np.uint8]
    
    def __init__(self, options: ORBKeypointMatcherOptions | None = None):
        """
        Initialize the OrbKeypointMatcher.
        
        :param ratio_threshold: Value used to filter keypoint matches (Lowe's ratio test)
        :param detect_kwargs: Dictionary of kwargs for cv2.ORB_create
        :param index_params: Dictionary of kwargs for FlannBasedMatcher
        :param search_params: Dictionary of kwargs for FlannBasedMatcher.knnMatch
        """
        super().__init__(ORBKeypointMatcherOptions, options=options)
        self.orb = cv2.ORB.create(nfeatures=self.orb_options.n_features, scaleFactor=self.orb_options.scale_factor,
                                  nlevels=self.orb_options.n_levels, edgeThreshold=self.orb_options.edge_threshold,
                                  firstLevel=self.orb_options.first_level, WTA_K=self.orb_options.wta_k,
                                  scoreType=self.orb_options.score_type, patchSize=self.orb_options.patch_size,
                                  fastThreshold=self.orb_options.fast_threshold)
        
    def detect_keypoints(self, image: NDArray) -> tuple[Sequence[cv2.KeyPoint], NDArray]:
        """
        Detect keypoint descriptors using ORB.
        
        :param image: The image to search for keypoints
            
        :returns: Tuple of (keypoints, descriptors)
        """
        # make sure the image is the right dtype (though it already should be by this point)
        image = self._validate_and_convert_image(image)
        
        keypoints, descriptors = self.orb.detectAndCompute(image, None) # type: ignore
        
        return keypoints, descriptors
        
