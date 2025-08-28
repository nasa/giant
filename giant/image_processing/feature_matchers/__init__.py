import giant.image_processing.feature_matchers.feature_matcher as feature_matcher
import giant.image_processing.feature_matchers.flann_configuration as flann_configuration
import giant.image_processing.feature_matchers.keypoint_matcher as keypoint_matcher
import giant.image_processing.feature_matchers.orb_matcher as orb_matcher
import giant.image_processing.feature_matchers.sift_matcher as sift_matcher
import giant.image_processing.feature_matchers.roma_matcher as roma_matcher

from giant.image_processing.feature_matchers.feature_matcher import FeatureMatcher
from giant.image_processing.feature_matchers.keypoint_matcher import KeypointMatcher, KeypointMatcherOptions
from giant.image_processing.feature_matchers.flann_configuration import (FLANNCentersInit, 
    FLANNIndexAlgorithmType, FLANNIndexAutotunedParams, FLANNIndexCompositeParams, FLANNIndexHierarchicalParams, 
    FLANNIndexKdTreeParams, FLANNIndexKdTreeSingleParams, FLANNIndexKMeansParams, FLANNIndexLinearParams, 
    FLANNIndexLSHParams, FLANNSearchParams, FLANNDistance)
from giant.image_processing.feature_matchers.orb_matcher import OrbKeypointMatcher, ORBKeypointMatcherOptions
from giant.image_processing.feature_matchers.sift_matcher import SIFTKeypointMatcher, SIFTKeypointMatcherOptions
__all__ = ["OrbKeypointMatcher", "ORBKeypointMatcherOptions", "SIFTKeypointMatcher", "SIFTKeypointMatcherOptions",
           "FLANNCentersInit", "FLANNDistance", "FLANNIndexCompositeParams", "FLANNIndexAutotunedParams", 
           "FLANNIndexHierarchicalParams", "FLANNIndexKdTreeParams", "FLANNIndexKdTreeSingleParams", "FLANNIndexLSHParams", 
           "FLANNSearchParams", "FLANNIndexKMeansParams", "FLANNIndexLinearParams"]

_valid_matchers = ["SIFT", "ORB", "CUSTOM"]
try:
    from giant.image_processing.feature_matchers.roma_matcher import RoMaFeatureMatcher, RoMaFeatureMatcherOptions
    __all__.extend(['RoMaFeatureMatcher', 'RoMaFeatureMatcherOptions'])
    _valid_matchers.append("ROMA")
except ImportError:
    RoMaFeatureMatcherOptions = None
    RoMaFeatureMatcher = None
    
from enum import Enum


FeatureMatcherMethod = Enum("FeatureMatcherMethod", 
                            _valid_matchers,
                            module=__name__)
"""
An enum specifying the valid feature matchers in this version of GIANT

At minimum this will include SIFT, ORB, and CUSTOM (for custom implemented feature matchers using the FeatureMatcher ABC).  
If you have installed romatch (https://github.com/Parskatt/RoMa/tree/main) then ROMA will also be available
"""