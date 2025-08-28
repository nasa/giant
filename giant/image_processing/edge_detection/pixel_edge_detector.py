import numpy as np
from numpy.typing import NDArray

from giant.image_processing.edge_detection.edge_detector_base import EdgeDetector


class PixelEdgeDetector(EdgeDetector[NDArray[np.int64]]):
    """
    This class implements a basic pixel level edge detector using the Scharr methodology.
    
    Essentially, a Scharr filter is convolved with the image and then Otsu's method is used 
    to automatically threshold the image gradients to detect the strongest signals, which 
    presumably correspond to edges.
    """
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def refine_edges(self, image: NDArray, edges: NDArray[np.int64]) -> NDArray[np.int64]:
        """
        Refines the pixel level edge locations to pixel level edge locations.
        
        This does nothing except return the input...
        
        :param image: The image to use to refine the edges
        :param edges: the edges to refine
        :return: the refined edges.
        """
        return edges
    
    def identify_edges(self, image: NDArray) -> NDArray[np.int64]:
        """
        This method identifies pixel level edges in the image. 
        
        :param image: The image to detect the edges in.
        :return: The edges as a 2xn array with x (column) components in the first axis and y (row) components in the second axis
        """
        if self.horizontal_mask is None or self.vertical_mask is None:
            self.prepare_edge_inputs(image)
            
        return np.vstack(np.where(self.horizontal_mask | self.vertical_mask))[::-1] # type: ignore
    