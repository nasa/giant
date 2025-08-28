from abc import abstractmethod, ABC

from dataclasses import dataclass, field, asdict

from typing import Sequence

import numpy as np
from numpy.typing import NDArray, DTypeLike

import cv2

from giant.image_processing.feature_matchers.feature_matcher import FeatureMatcher
from giant.image_processing.feature_matchers.flann_configuration import FLANNIndexAlgorithmType, FLANN_INDEX_PARAM_TYPES, FLANNIndexCompositeParams, FLANNSearchParams
from giant.image_processing.utilities.image_validation_mixin import ImageValidationMixin

from giant.utilities.options import UserOptions
from giant.utilities.mixin_classes.attribute_equality_comparison import AttributeEqualityComparison
from giant.utilities.mixin_classes.attribute_printing import AttributePrinting
from giant.utilities.mixin_classes.user_option_configured import UserOptionConfigured

from giant._typing import DOUBLE_ARRAY


@dataclass
class KeypointMatcherOptions(UserOptions):
    """
    Options for configuring a KeypointMatcher
    """
    
    ratio_threshold: float = 0.9
    """
    The threshold to use in Lowe's ratio test
    """
    
    flann_index_algorithm_type: FLANNIndexAlgorithmType = FLANNIndexAlgorithmType.FLANN_INDEX_COMPOSITE
    """
    What FLANN algorithm to use. 
    """
    
    flann_algorithm_parameters: FLANN_INDEX_PARAM_TYPES = field(default_factory=FLANNIndexCompositeParams)
    """
    The parameters to configure the index in FLANN.
    
    This should match the flann_index_algorithm_type although OpenCV will not complain if it doesn't.
    """
    
    flann_search_parameters: FLANNSearchParams = field(default_factory=FLANNSearchParams)
    """
    The parameters to configure how FLANN performs searches
    """


class KeypointMatcher(UserOptionConfigured[KeypointMatcherOptions], 
                      KeypointMatcherOptions, 
                      AttributeEqualityComparison, 
                      AttributePrinting, 
                      FeatureMatcher, 
                      ImageValidationMixin,
                      ABC):
    """
    Abstract base class defining the interface for keypoint detection and matching.
    """
    
    
    def __init__(self, options_type: type[KeypointMatcherOptions], *args, options: KeypointMatcherOptions | None = None, **kwargs):
        """
        :param options: the option dataclass to configure with
        """
        super().__init__(options_type, *args, options=options, **kwargs)
        
    
    @abstractmethod
    def detect_keypoints(self, image: NDArray) -> tuple[Sequence[cv2.KeyPoint], NDArray]:
        """
        Detect keypoints and compute descriptors in an image.
        
        :param image: The image to search for keypoints
            
        :returns: Tuple of (keypoints, descriptors)
        """
        pass
    
    def match_descriptors(self, descriptors1: NDArray, descriptors2: NDArray) -> tuple[Sequence[cv2.DMatch], Sequence[Sequence[cv2.DMatch]]]:
        """
        Match keypoint descriptors using FLANN and Lowe's ratio test.
        
        :param descriptors1: Descriptors from first image
        :param descriptors2: Descriptors from second image
            
        :returns: Tuple of (good_matches, all_matches)
            
        This can be overridden if desired
        """
        index_params: dict[str, bool | int | float | str]  = {"algorithm": self.flann_index_algorithm_type.value}
        index_params.update(asdict(self.flann_algorithm_parameters))
        flann = cv2.FlannBasedMatcher(index_params, asdict(self.flann_search_parameters))
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
                
        return self.filter_lowes(matches), matches
    
    def match_images(self, image1: NDArray, image2: NDArray) -> DOUBLE_ARRAY:
        """
        Orchestrates the detect-then-match process.

        This concrete method fulfills the `ImageMatcher` interface requirement.
        It uses the abstract `detect_and_compute` and `match_descriptors` methods
        to perform the full matching pipeline.
        
        :param image1: The first image to match keypoints
        :param image2: The second image to match keypoints
            
        :returns: An array of the matched keypoint locations as nx2x2 
        """
        # make sure the images are of the right dtype
        image1 = self._validate_and_convert_image(image1)
        image2 = self._validate_and_convert_image(image2)
        
        kp1, des1 = self.detect_keypoints(image1)
        kp2, des2 = self.detect_keypoints(image2)
        
        cv_matches, _ = self.match_descriptors(des1, des2)
        
        # Convert to numpy array
        matched_keypoints_array = np.array([[kp1[m.queryIdx].pt, kp2[m.trainIdx].pt] for m in cv_matches])
        
        return matched_keypoints_array
    
    def filter_lowes(self, matches: Sequence[Sequence[cv2.DMatch]]) -> Sequence[cv2.DMatch]:
        """
        Filters matches based on the Lowe's ratio test

        :param matches: a sequence of sequences of all the matches computed
        
        :returns: A sequence of the good matches that pass the ratio test.
        """
        
        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                # Filter matches using Lowe's ratio test
                if m.distance < n.distance * self.ratio_threshold:
                    good_matches.append(m)
            elif len(match) == 1:
                good_matches.append(match[0])
        return good_matches
    
