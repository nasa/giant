from enum import Enum, auto

import giant.image_processing.edge_detection.edge_detector_base as edge_detector_base
import giant.image_processing.edge_detection.pixel_edge_detector as pixel_edge_detector
import giant.image_processing.edge_detection.pae_subpixel_edge_detector as pae_subpixel_edge_detector
import giant.image_processing.edge_detection.zernike_ramp_edge_detector as zernike_ramp_edge_detector   

from giant.image_processing.edge_detection.edge_detector_base import EdgeDetector
from giant.image_processing.edge_detection.pixel_edge_detector import PixelEdgeDetector
from giant.image_processing.edge_detection.pae_subpixel_edge_detector import PAESubpixelEdgeDetector, PAESubpixelEdgeDetectorOptions
from giant.image_processing.edge_detection.zernike_ramp_edge_detector import ZernikeRampEdgeDetector, ZernikeRampEdgeDetectorOptions


__all__ = ["PixelEdgeDetector", 
           "PAESubpixelEdgeDetector", "PAESubpixelEdgeDetectorOptions", 
           "ZernikeRampEdgeDetector", "ZernikeRampEdgeDetectorOptions"]


class EdgeDetectionMethods(Enum):
    """
    An enum specifying the available edge detection techniques
    """
    
    PIXEL_EDGE_DETECTOR = auto()
    """
    Detect edges to pixel level accuracy
    """
    
    PAE_SUBPIXEL_EDGE_DETECTOR = auto()
    """
    Detect edges to subpixel level accuracy using the Partial Area Effect method.
    
    See https://www.researchgate.net/publication/233397974_Accurate_Subpixel_Edge_Location_based_on_Partial_Area_Effect for more details
    """
    
    ZERNIKE_RAMP_EDGE_DETECTOR = auto()
    """
    Detect edges to subpixel level accuracy using the Zernike Ramp method.
    
    See https://arc.aiaa.org/doi/full/10.2514/1.A33692?mobileUi=0 for more details
    """
    
    CUSTOM_DETECTOR = auto()
    """
    Detect edges with a custom, user implemented detector.
    
    See the :class:`.EdgeDetector` abstract base class for the required interface.
    """
    
    
 