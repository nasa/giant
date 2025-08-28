


"""
This package provides a number of image processing techniques for use throughout GIANT.

The functionality provided in this package, is the primary set of tools used for working with image data
throughout GIANT. This package provides routines to identify point sources in an image (:class:`.PointOfInterestFinder`), 
detect edges in an image (:class:`.PixelEdgeDetector`, :class:`.PAESubpixelEdgeDetector`, :class:`.ZernikeRampEdgeDetector`),
perform template matching through cross correlation (:mod:`.correlators`), detect opportunistic features in images 
(:class:`.ORBKeypointMatcher`,).

For many of these methods, there are multiple algorithms that can be used to perform the same task. 

A general user will usually not directly interact with the classes and functions in this class and instead will rely on
the OpNav classes to interact for them.
"""

import cv2
# fix for serializing
cv2.GaussianBlur.__module__ = "cv2"

import giant.image_processing.edge_detection as edge_detection
import giant.image_processing.feature_matchers as feature_matchers
import giant.image_processing.correlators as correlators
import giant.image_processing.image_flattener as image_flattener
import giant.image_processing.image_segmenter as image_segmenter
import giant.image_processing.limb_edge_detection as limb_edge_detection
import giant.image_processing.limb_scanning as limb_scanning
import giant.image_processing.local_maxima as local_maxima
import giant.image_processing.otsu as otsu
import giant.image_processing.peak_finders as peak_finders
import giant.image_processing.point_source_finder as point_source_finder


from giant.image_processing.correlators import cv2_correlator_2d, scipy_correlator_2d, fft_correlator_1d
from giant.image_processing.image_flattener import ImageFlattener, ImageFlattenerOptions, ImageFlattenerOut, ImageFlatteningNoiseApprox
from giant.image_processing.image_segmenter import ImageSegmenter, ImageSegmenterOptions, ImageSegmenterOut
from giant.image_processing.limb_edge_detection import LimbEdgeDetection, LimbEdgeDetectionOptions
from giant.image_processing.limb_scanning import LimbScanner, LimbScannerOptions
from giant.image_processing.local_maxima import local_maxima
from giant.image_processing.otsu import otsu
from giant.image_processing.peak_finders import parabolic_peak_finder_1d, pixel_level_peak_finder_1d, pixel_level_peak_finder_2d, quadric_peak_finder_2d
from giant.image_processing.point_source_finder import PointOfInterestFinder, POIFinderOut, PointOfInterestFinderOptions
from giant.image_processing.edge_detection import (PixelEdgeDetector, PAESubpixelEdgeDetector, PAESubpixelEdgeDetectorOptions, 
                                                   ZernikeRampEdgeDetector, ZernikeRampEdgeDetectorOptions, EdgeDetectionMethods)
from giant.image_processing.feature_matchers import (FeatureMatcher, KeypointMatcher, KeypointMatcherOptions, FLANNCentersInit, FLANNIndexAlgorithmType, 
                                                     FLANNIndexAutotunedParams, FLANNIndexCompositeParams, FLANNIndexHierarchicalParams, FLANNIndexKdTreeParams,
                                                     FLANNIndexKdTreeSingleParams, FLANNIndexKMeansParams, FLANNIndexLSHParams, FLANNIndexLinearParams,
                                                     FLANNSearchParams, FLANNDistance, OrbKeypointMatcher, ORBKeypointMatcherOptions, SIFTKeypointMatcher,
                                                     SIFTKeypointMatcherOptions, RoMaFeatureMatcher, RoMaFeatureMatcherOptions)

__all__ = ["cv2_correlator_2d", "scipy_correlator_2d", "fft_correlator_1d",
           "ImageFlattener", "ImageFlattenerOptions", "ImageFlattenerOptions", "ImageFlatteningNoiseApprox",
           "ImageSegmenter", "ImageSegmenterOptions", "ImageSegmenterOut",
           "LimbEdgeDetection", "LimbEdgeDetectionOptions", 
           "LimbScanner", "LimbScannerOptions",
           "local_maxima",
           "otsu",
           "parabolic_peak_finder_1d", "pixel_level_peak_finder_1d", "pixel_level_peak_finder_2d", "quadric_peak_finder_2d",
           "PointOfInterestFinder", "PointOfInterestFinderOptions", "POIFinderOut",
           "PixelEdgeDetector", "PAESubpixelEdgeDetector", "PAESubpixelEdgeDetectorOptions", "ZernikeRampEdgeDetector", "ZernikeRampEdgeDetectorOptions", "EdgeDetectionMethods",
           "FLANNCentersInit", "FLANNIndexAlgorithmType", "FLANNIndexAutotunedParams", "FLANNIndexCompositeParams", "FLANNIndexHierarchicalParams", "FLANNIndexKdTreeParams",
           "FLANNIndexKdTreeSingleParams", "FLANNIndexKMeansParams", "FLANNIndexLSHParams", "FLANNIndexLinearParams",
           "FLANNSearchParams", "FLANNDistance", "OrbKeypointMatcher", "ORBKeypointMatcherOptions", "SIFTKeypointMatcher", "SIFTKeypointMatcherOptions",
           "RoMaFeatureMatcher", "RoMaFeatureMatcherOptions"]

