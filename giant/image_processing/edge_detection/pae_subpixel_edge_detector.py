from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from giant.utilities.options import UserOptions
from giant.utilities.mixin_classes.user_option_configured import UserOptionConfigured

from giant.image_processing.edge_detection.edge_detector_base import EdgeDetector
from giant.image_processing.edge_detection.pixel_edge_detector import PixelEdgeDetector

from giant._typing import DOUBLE_ARRAY


@dataclass
class PAESubpixelEdgeDetectorOptions(UserOptions):
    a01: float = 0.125
    """
    The 0, 1 coefficient (upper middle) of the Gaussian kernel representing the blurring experienced in the images being 
    processed for the PAE sub-pixel edge method.

    By default this is set to 0.125 assuming a 2D gaussian kernel with a sigma of 1 pixel in each axis.  If you know a 
    better approximation of the gaussian kernel that represents the point spread function in the image (combined with any 
    gaussian blurring applied to the image to smooth out noise) then you may get better results from the PAE method by 
    updating this value.

    https://www.researchgate.net/publication/233397974_Accurate_Subpixel_Edge_Location_based_on_Partial_Area_Effect
    """
    
    a11: float = 0.0625
    """
    The 1, 1 coefficient (upper right) of the Gaussian kernel representing the blurring experienced in the images being 
    processed for the PAE sub-pixel edge method.

    By default this is set to 0.0625 assuming a 2D gaussian kernel with a sigma of 1 pixel in each axis.  If you know a 
    better approximation of the gaussian kernel that represents the point spread function in the image (combined with any 
    gaussian blurring applied to the image to smooth out noise) then you may get better results from the PAE method by 
    updating this value.

    https://www.researchgate.net/publication/233397974_Accurate_Subpixel_Edge_Location_based_on_Partial_Area_Effect
    """
    
    order: int = 2
    """
    This specifies whether to fit a linear (1) or quadratic (2) to the limb in the PAE method.  
    
    Typically quadratic produces the best results.
    """

class PAESubpixelEdgeDetector(UserOptionConfigured[PAESubpixelEdgeDetectorOptions], 
                              EdgeDetector[DOUBLE_ARRAY], 
                              PAESubpixelEdgeDetectorOptions):
    """
    This class can be used to identify and refine to the subpixel location of edges in an image using the partial area effect method.

    Edges are defined as places in the image where the illumination values abruptly transition from light to dark
    or dark to light.  The algorithms in this method are based off of the Partial Area Effect as discussed in
    http://www.sciencedirect.com/science/article/pii/S0262885612001850

    First edges are detected at the pixel level by using a gradient based edge detection method.  The edges are then
    refined to subpixel accuracy using the PAE.  Tests have shown that the PAE achieves accuracy better than 0.1
    pixels in most cases.

    There are three tuning parameters for the PAE technique.  The first tuning parameter is the :attr:`.order`, which specifies 
    whether a linear or quadratic fit is used to refine the edge location.  It should have a value of 1 or 2.  
    The second and third are the :attr:`.a01` and :att:`.a11` attributes.  These specify the upper middle and upper right 
    coeficients of the expected 3x3 Gaussian PSF that described how edges are blurred in the image. 
    """
    
    def __init__(self, options: PAESubpixelEdgeDetectorOptions | None) -> None:
        super().__init__(PAESubpixelEdgeDetectorOptions, options=options)
        
        # create a local pixel level edge instance to do the initial detection
        self._pixel_edge_detector = PixelEdgeDetector()
    
    @staticmethod 
    def _split_pos_neg_edges(horizontal_gradient: DOUBLE_ARRAY, vertical_gradient: DOUBLE_ARRAY,
                            edges: DOUBLE_ARRAY) -> tuple[DOUBLE_ARRAY, DOUBLE_ARRAY]:
        """
        This function splits diagonal edges into positive/negative bins

        :param horizontal_gradient: The horizontal gradient array
        :param vertical_gradient: The vertical gradient array
        :param edges: The edge array containing the pixel location of the edges as [x, y]
        :return: The edges split into positive and negative groupings
        """

        # check with edges are positive edges
        positive_check = horizontal_gradient[edges[1], edges[0]] * vertical_gradient[edges[1], edges[0]] > 0

        # split and return the binned edges
        return edges[:, positive_check], edges[:, ~positive_check]
    
    def _compute_pae_delta(self, sum_a: NDArray, sum_b: NDArray, sum_c: NDArray,
                           int_a: NDArray, int_b: NDArray) -> DOUBLE_ARRAY:
        """
        This method computes the subpixel location of an edge using the pae method within a pixel.

        This method is vectorized so multiple edges can be refined at the same time.

        Essentially this method either fits a line or a parabola to the edge based off of the intensity data surrounding
        the edge.  if :attr:`pae_order` is set to 1, then a linear fit is made.  If it is set to 2 then a parabola fit
        is made.

        :param sum_a: The sum of the first row or first column (depending on whether this is a horizontal or vertical
                     edge)
        :param sum_b: The sum of the middle row or column (depending on whether this is a horizontal or vertical edge)
        :param sum_c: The sum of the final row or column (depending on whether this is a horizontal or vertical edge)
        :param int_a: The average intensity to the positive side of the edge
        :param int_b: The average intensity to the negative side of the edge
        :return: The offset in the local pixel for the subpixel edge locations.
        """

        a_coef = (self.order - 1) * (sum_a + sum_c - 2 * sum_b) / (2 * (int_b - int_a))
        c_coef = ((2 * sum_b - 7 * (int_b + int_a)) /
                  (2 * (int_b - int_a)) -
                  a_coef * (1 + 24 * self.a01 + 48 * self.a11) / 12)

        c_coef[(np.abs(c_coef) > 1) | ~np.isfinite(c_coef)] = 0

        return c_coef

    def refine_edges(self, image: NDArray, edges: NDArray[np.int64]) -> DOUBLE_ARRAY:
        """
        This method refines pixel level edges to subpixel level using the PAE method.

        The PAE method is explained at https://www.sciencedirect.com/science/article/pii/S0262885612001850 and is not
        discussed in detail here.  In brief, a linear or parabolic function is fit to the edge data based off of the
        intensity data in the pixels surrounding the edge locations.

        To use this method, you must input the image, as well as a 2xn array of [[x], [y]] edges to be refined.
        
        The edges are refined and returned as a 2D array with the x locations in the first row and the y locations 
        in the second row.

        :param image:  The image the edges are being extracted from
        :param edges: The pixel level edges from the image as a 2D array with x in the first row and y in the
                            second row
        :return: The subpixel edge locations as a 2d array with the x values in the first row and the y values in the
                 second row (col [x], row [y])
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
            
        image = image.astype(np.float64)
        
        edges = edges.T
        
        # split into horizontal and vertical edges 
        horizontal_edges = edges[self.horizontal_mask[edges[:, 1], edges[:, 0]]].T # type: ignore
        vertical_edges = edges[self.vertical_mask[edges[:, 1], edges[:, 0]]].T # type: ignore
        
        horiz_pos_edges, horiz_neg_edges = self._split_pos_neg_edges(self.horizontal_gradient, self.vertical_gradient,
                                                                     horizontal_edges)
        vert_pos_edges, vert_neg_edges = self._split_pos_neg_edges(self.horizontal_gradient, self.vertical_gradient,
                                                                   vertical_edges)

        # process the horizontal edges

        # precompute the indices
        prm4 = horiz_pos_edges[1] - 4
        prm3 = horiz_pos_edges[1] - 3
        prm2 = horiz_pos_edges[1] - 2
        prm1 = horiz_pos_edges[1] - 1
        pr = horiz_pos_edges[1]
        prp1 = horiz_pos_edges[1] + 1
        prp2 = horiz_pos_edges[1] + 2
        prp3 = horiz_pos_edges[1] + 3
        prp4 = horiz_pos_edges[1] + 4
        pcm1 = horiz_pos_edges[0] - 1
        pc = horiz_pos_edges[0]
        pcp1 = horiz_pos_edges[0] + 1

        nrm4 = horiz_neg_edges[1] - 4
        nrm3 = horiz_neg_edges[1] - 3
        nrm2 = horiz_neg_edges[1] - 2
        nrm1 = horiz_neg_edges[1] - 1
        nr = horiz_neg_edges[1]
        nrp1 = horiz_neg_edges[1] + 1
        nrp2 = horiz_neg_edges[1] + 2
        nrp3 = horiz_neg_edges[1] + 3
        nrp4 = horiz_neg_edges[1] + 4
        ncm1 = horiz_neg_edges[0] - 1
        nc = horiz_neg_edges[0]
        ncp1 = horiz_neg_edges[0] + 1

        # calculate the average intensity on either side of the edge
        # above the edge for positive sloped edges
        int_top_pos = image[[prm3, prm4, prm4], [pcm1, pcm1, pc]].sum(axis=0) / 3

        # below the edge for positive sloped edges
        int_bot_pos = image[[prp3, prp4, prp4], [pcp1, pcp1, pc]].sum(axis=0) / 3

        # above the edge for negative sloped edges
        int_top_neg = image[[nrm3, nrm4, nrm4], [ncp1, ncp1, nc]].sum(axis=0) / 3

        # below the edge for negative sloped edges
        int_bot_neg = image[[nrp3, nrp4, nrp4], [ncm1, ncm1, nc]].sum(axis=0) / 3

        # sum the columns of intensity for the positive slop edges
        sum_left_pos_slope = image[[prm2, prm1, pr, prp1, prp2, prp3, prp4],
                                    [pcm1, pcm1, pcm1, pcm1, pcm1, pcm1, pcm1]].sum(axis=0)
        sum_mid_pos_slope = image[[prm3, prm2, prm1, pr, prp1, prp2, prp3],
                                    [pc, pc, pc, pc, pc, pc, pc]].sum(axis=0)
        sum_right_pos_slope = image[[prm4, prm3, prm2, prm1, pr, prp1, prp2],
                                    [pcp1, pcp1, pcp1, pcp1, pcp1, pcp1, pcp1]].sum(axis=0)

        # sum the columns of intensity for the negative slop edges
        sum_left_neg_slope = image[[nrm4, nrm3, nrm2, nrm1, nr, nrp1, nrp2],
                                    [ncm1, ncm1, ncm1, ncm1, ncm1, ncm1, ncm1]].sum(axis=0)
        sum_mid_neg_slope = image[[nrm3, nrm2, nrm1, nr, nrp1, nrp2, nrp3],
                                    [nc, nc, nc, nc, nc, nc, nc]].sum(axis=0)
        sum_right_neg_slope = image[[nrm2, nrm1, nr, nrp1, nrp2, nrp3, nrp4],
                                    [ncp1, ncp1, ncp1, ncp1, ncp1, ncp1, ncp1]].sum(axis=0)

        # calculate the coefficient for the partial area for the positive slopes
        dy_pos_slope = self._compute_pae_delta(sum_left_pos_slope, sum_mid_pos_slope, sum_right_pos_slope,
                                                int_top_pos, int_bot_pos)

        # calculate the subpixel edge locations for the positive slope edges
        sp_horiz_edges_pos = horiz_pos_edges.astype(np.float64)
        sp_horiz_edges_pos[1] -= dy_pos_slope

        # calculate the coefficient for the partial area for the positive slopes
        dy_neg_slope = self._compute_pae_delta(sum_left_neg_slope, sum_mid_neg_slope, sum_right_neg_slope,
                                                int_top_neg, int_bot_neg)

        # calculate the subpixel edge locations for the negative slope edges
        sp_horiz_edges_neg = horiz_neg_edges.astype(np.float64)
        sp_horiz_edges_neg[1] -= dy_neg_slope

        # process the vertical edges

        # precompute the indices
        pcm4 = vert_pos_edges[0] - 4
        pcm3 = vert_pos_edges[0] - 3
        pcm2 = vert_pos_edges[0] - 2
        pcm1 = vert_pos_edges[0] - 1
        pc = vert_pos_edges[0]
        pcp1 = vert_pos_edges[0] + 1
        pcp2 = vert_pos_edges[0] + 2
        pcp3 = vert_pos_edges[0] + 3
        pcp4 = vert_pos_edges[0] + 4
        prm1 = vert_pos_edges[1] - 1
        pr = vert_pos_edges[1]
        prp1 = vert_pos_edges[1] + 1

        ncm4 = vert_neg_edges[0] - 4
        ncm3 = vert_neg_edges[0] - 3
        ncm2 = vert_neg_edges[0] - 2
        ncm1 = vert_neg_edges[0] - 1
        nc = vert_neg_edges[0]
        ncp1 = vert_neg_edges[0] + 1
        ncp2 = vert_neg_edges[0] + 2
        ncp3 = vert_neg_edges[0] + 3
        ncp4 = vert_neg_edges[0] + 4
        nrm1 = vert_neg_edges[1] - 1
        nr = vert_neg_edges[1]
        nrp1 = vert_neg_edges[1] + 1

        # calculate the average intensity on either side of the edge
        # left of the edge for positive sloped edges
        int_left_pos = image[[prm1, prm1, pr], [pcm3, pcm4, pcm4]].sum(axis=0) / 3

        # right of the edge for positive sloped edges
        int_right_pos = image[[prp1, prp1, pr], [pcp3, pcp4, pcp4]].sum(axis=0) / 3

        # left of the edge for negative sloped edges
        int_left_neg = image[[nrp1, nrp1, nr], [ncm3, ncm4, ncm4]].sum(axis=0) / 3

        # right of the edge for negative sloped edges
        int_right_neg = image[[nrm1, nrm1, nr], [ncp3, ncp4, ncp4]].sum(axis=0) / 3

        # sum the rows of intensity for the positive slop edges
        sum_top_pos_slope = image[[prm1, prm1, prm1, prm1, prm1, prm1, prm1],
                                    [pcm2, pcm1, pc, pcp1, pcp2, pcp3, pcp4]].sum(axis=0)
        sum_mid_pos_slope = image[[pr, pr, pr, pr, pr, pr, pr],
                                    [pcm3, pcm2, pcm1, pc, pcp1, pcp2, pcp3]].sum(axis=0)
        sum_bottom_pos_slope = image[[prp1, prp1, prp1, prp1, prp1, prp1, prp1],
                                        [pcm4, pcm3, pcm2, pcm1, pc, pcp1, pcp2]].sum(axis=0)

        # sum the rows of intensity for the negative slop edges
        sum_top_neg_slope = image[[nrm1, nrm1, nrm1, nrm1, nrm1, nrm1, nrm1],
                                    [ncm4, ncm3, ncm2, ncm1, nc, ncp1, ncp2]].sum(axis=0)
        sum_mid_neg_slope = image[[nr, nr, nr, nr, nr, nr, nr],
                                    [ncm3, ncm2, ncm1, nc, ncp1, ncp2, ncp3]].sum(axis=0)
        sum_bottom_neg_slope = image[[nrp1, nrp1, nrp1, nrp1, nrp1, nrp1, nrp1],
                                        [ncm2, ncm1, nc, ncp1, ncp2, ncp3, ncp4]].sum(axis=0)

        # calculate the coefficient for the partial area for the positive slopes
        dx_pos_slope = self._compute_pae_delta(sum_top_pos_slope, sum_mid_pos_slope, sum_bottom_pos_slope,
                                                int_left_pos, int_right_pos)

        # calculate the subpixel edge locations for the positive slope edges
        sp_vert_edges_pos = vert_pos_edges.astype(np.float64)
        sp_vert_edges_pos[0] -= dx_pos_slope

        # calculate the coefficient for the partial area for the positive slopes
        dx_neg_slope = self._compute_pae_delta(sum_top_neg_slope, sum_mid_neg_slope, sum_bottom_neg_slope,
                                                int_left_neg, int_right_neg)

        # calculate the subpixel edge locations for the negative slope edges
        sp_vert_edges_neg = vert_neg_edges.astype(np.float64)
        sp_vert_edges_neg[0] -= dx_neg_slope

        # return the subpixel edges
        return np.hstack([sp_horiz_edges_pos, sp_horiz_edges_neg, sp_vert_edges_pos, sp_vert_edges_neg])
   
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
        return self.refine_edges(image, self._pixel_edge_detector.identify_edges(image))

