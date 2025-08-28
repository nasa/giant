from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from giant.utilities.options import UserOptions
from giant.utilities.mixin_classes.user_option_configured import UserOptionConfigured

from giant.image_processing.edge_detection.edge_detector_base import EdgeDetector
from giant.image_processing.edge_detection.pixel_edge_detector import PixelEdgeDetector

from giant._typing import DOUBLE_ARRAY

    
FIRST_ORDER_REAL_MOMENTS = np.array([[-.0147, -.0469, 0, .0469, .0147],
                                     [- .0933, -.0640, 0, .0640, .0933],
                                     [-.1253, -.0640, 0, .0640, .1253],
                                     [-.0933, -.0640, 0, .0640, .0933],
                                     [-.0147, -.0469, 0, .0469, .0147]])
"""
First order real component of Zernike Moments

This is used in the zernike moment sub-pixel edge detection routines
"""

FIRST_ORDER_IMAGINARY_MOMENTS = FIRST_ORDER_REAL_MOMENTS.T
"""
First order imaginary component of Zernike Moments

This is used in the zernike moment sub-pixel edge detection routines
"""

SECOND_ORDER_MOMENTS = np.array([[.0177, .0595, .0507, .0595, .0177],
                                 [.0595, -.0492, -.1004, -.0492, .0595],
                                 [.0507, -.1004, -.1516, -.1004, .0507],
                                 [.0595, -.0492, -.1004, -.0492, .0595],
                                 [.0177, .0595, .0507, .0595, .0177]])
"""
Second order Zernike Moments

This is used in the zernike moment sub-pixel edge detection routines
"""


@dataclass
class ZernikeRampEdgeDetectorOptions(UserOptions):
    edge_width: float = 0.5
    """
    A tuning parameter for the Zernike Ramp method specifying half the total edge width in pixels.
    
    Typically this is set to 1.66*sigma where sigma is the point spread function full width half maximum for the 
    camera.
    """
    
    
class ZernikeRampEdgeDetector(UserOptionConfigured[ZernikeRampEdgeDetectorOptions], 
                              EdgeDetector[DOUBLE_ARRAY], 
                              ZernikeRampEdgeDetectorOptions):
    """
    This class implements a subpixel edge detector using the Zernike Ramp method.

    The Zernike Ramp method is described in detail in https://arc.aiaa.org/doi/full/10.2514/1.A33692?mobileUi=0.
    It uses precomputed Zernike moments and their inner products with the image data around pixel-level edges
    to compute subpixel corrections.

    There is one tuning parameter for this method, :attr:`.edge_width`, which specifies half the total edge width in pixels.
    This is typically set to 1.66*sigma, where sigma is the point spread function full width half maximum for the camera.

    This class first identifies pixel-level edges using a :class:`.PixelEdgeDetector`, then refines them to subpixel
    accuracy using the Zernike Ramp method.
    """

    
    def __init__(self, options: ZernikeRampEdgeDetectorOptions | None = None):
        """
        :param options: The options configuring the Zernike ramp subpixel edge detector.
        """
        super().__init__(ZernikeRampEdgeDetectorOptions, options=options)
        
        # create a local pixel level edge instance to do the initial detection
        self._pixel_edge_detector = PixelEdgeDetector()
        
    def refine_edges(self, image: NDArray, edges: DOUBLE_ARRAY) -> DOUBLE_ARRAY:
        """
        This method refines edge locations using the Zernike Ramp method described in
        https://arc.aiaa.org/doi/full/10.2514/1.A33692?mobileUi=0.

        The subpixel edge locations are found by computing the inner product between precomputed Zernike moments
        and the image data around the pixel level edges, and then computing a correction to the pixel level
        edge (see the paper for details).

        There is one tuning parameter for this method and that is the half edge width which is specified in the
        :attr:`.zernike_edge_width` attribute.  This should be set to roughly half the total edge length in pixels,
        which is typically approximately 1.66*sigma where sigma is the point spread function full width half maximum
        for the camera.

        This method returns a 2xn array of subpixel edge points, leaving the pixel level edge points for areas where it
        failed.

        :param image: The image which the edge points index into
        :param edges: the pixel level edge points to be refined. 
        :return: A 2xn array of subpixel edge points (col [x], row[y])
        """
        
        if (self.horizontal_mask is None or 
            self.vertical_mask is None or 
            self.horizontal_gradient is None or 
            self.vertical_gradient is None or 
            self.gradient_magnitude is None):
            self.prepare_edge_inputs(image)
            
        assert (self.horizontal_gradient is not None and 
                self.horizontal_mask is not None and 
                self.vertical_gradient is not None and 
                self.vertical_mask is not None and 
                self.gradient_magnitude is not None), "This should never happen"
        
        starts = np.maximum(edges-2, 0)
        stops = np.minimum(edges+3, [[image.shape[1]], [image.shape[0]]])

        subpixel_edges = []

        edge_width_squared = self.edge_width ** 2
        # loop through each edge
        for edge, start, stop in zip(edges.T, starts.T, stops.T):

            if ((stop - start) < 5).any():
                # we are too close to the edge so just keep the pixel level point
                subpixel_edges.append(edge)
                continue

            sub_img = image[start[1]:stop[1], start[0]:stop[0]]

            # compute the correlation between the moment and the image data
            first_order_imaginary_correlation = (FIRST_ORDER_IMAGINARY_MOMENTS*sub_img).sum()
            first_order_real_correlation = (FIRST_ORDER_REAL_MOMENTS*sub_img).sum()
            second_order_correlation = (SECOND_ORDER_MOMENTS*sub_img).sum()

            # determine the edge normal
            angle = np.arctan2(first_order_imaginary_correlation, first_order_real_correlation)
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)

            # determine the ratio of the correlations
            ratio = second_order_correlation / (first_order_real_correlation*cos_angle +
                                                first_order_imaginary_correlation*sin_angle)

            # solve for the distance along hte normal we need to perturb
            if self.edge_width > 0.01:
                location = (1 - edge_width_squared -
                            np.sqrt((edge_width_squared-1)**2 - 2*edge_width_squared*ratio))/edge_width_squared
            else:
                location = ratio

            if np.abs(location) < 0.9:
                subpixel_edges.append(edge+2.5*location*np.array([cos_angle, sin_angle]))
            else:
                # if we're here then we didn't get a good fit
                subpixel_edges.append(edge)

        return np.vstack(subpixel_edges).T
        
    
    def identify_edges(self, image: NDArray) -> DOUBLE_ARRAY:
        """
        This method identifies pixel level edges in the image and then refines them to the subpixel locations using the PAE method.
        
        The pixel edges are detected using the :class:`.PixelEdgeDetector`.  They are then refined using :meth:`.refine_edges`.
        
        :param image: The image to detect the edges in.
        :return: The subpixel edges as a 2xn array with x (column) components in the first axis and y (row) components in the second axis
        """
        
        if self.horizontal_mask is None or self.vertical_mask is None or self.horizontal_gradient is None or self.vertical_gradient is None:
            self.prepare_edge_inputs(image)
            
        # copy over the infomration to the pixel edge detector 
        self._pixel_edge_detector.horizontal_gradient = self.horizontal_gradient
        self._pixel_edge_detector.vertical_gradient= self.vertical_gradient
        self._pixel_edge_detector.horizontal_mask = self.horizontal_mask
        self._pixel_edge_detector.vertical_mask = self.vertical_mask
        self._pixel_edge_detector.gradient_magnitude = self.gradient_magnitude
        
        # get the pixel level edges and refine them
        return self.refine_edges(image, self._pixel_edge_detector.identify_edges(image).astype(np.float64))
    

