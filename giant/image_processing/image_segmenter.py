from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

import cv2

from giant.image_processing.otsu import otsu

from giant.utilities.options import UserOptions
from giant.utilities.mixin_classes.user_option_configured import UserOptionConfigured
from giant.utilities.mixin_classes.attribute_equality_comparison import AttributeEqualityComparison
from giant.utilities.mixin_classes.attribute_printing import AttributePrinting


class ImageSegmenterOut(NamedTuple):
    labeled_image: NDArray[np.int32]
    """
    An image with labeled foreground objects (>=0) of the same shape as the input and dtype np.int32
    """
    
    foreground_image: NDArray[np.uint8]
    """
    A boolean array the same shape as the input with 0 in the background pixels and 1 in the foreground pixels
    """
    
    stats: NDArray[np.int32]
    """
    The stats vector returned from opencv's connectectedComponentsWithStats as a nx5 array.
    
    The columns are [left x coordinate, top y coordinate, width, height, area] in pixels.
    
    It is best to access the appropriate column using `cv2.CC_STAT_LEFT`, `cv2.CC_STAT_TOP`, `cv2.CC_STAT_WIDTH`, 
    `cv2.CC_STAT_HEIGHT`, or `cv2.CC_STAT_AREA` in case opencv ever changes the order.
    
    Each row corresponds to the object number in the labeled image
    """
    
    centroids: NDArray[np.float64]
    """
    The unweighted centroids of each blob in the image as a nx2 array (x, y) in pixels.
    """


@dataclass
class ImageSegmenterOptions(UserOptions):
    otsu_levels: int = 2
    """
    This sets the number of levels to attempt to segment the histogram into for Otsu based multi level thresholding.
    
    See the :func:`.otsu` function for more details.
    """

    minimum_segment_area: int = 10
    """
    This sets the minimum area for a segment to be considered not noise.
    
    Segments with areas less than this are discarded as noise spikes
    """

    minimum_segment_dn: float = 200
    """
    The minimum that the average DN for a segment must be for it to not be discarded as the background.

    Segments with average DNs less than this are discarded as the background
    """
    

class ImageSegmenter(UserOptionConfigured[ImageSegmenterOptions], 
                     ImageSegmenterOptions, 
                     AttributePrinting, 
                     AttributeEqualityComparison):
    """
    This class segments images into foreground and background objects using a multi-level Otsu thresholding approach.

    It is configured using the :class:`ImageSegmenterOptions` dataclass, which specifies parameters such as the number
    of Otsu levels, minimum segment area, and minimum segment DN.

    The main functionality is provided by the :meth:`__call__` method, which takes an input image and returns
    a :class:`ImageSegmenterOut` named tuple containing the segmentation results.
    
    Example usage::
        segmenter = ImageSegmenter()
        segment_results = segmenter(image)
    """
    
    def __init__(self, options: ImageSegmenterOptions | None = None) -> None:
        """
        :param options: the options configuring the image segmenter. If `None`, defaults will be used
        """
        
        super().__init__(ImageSegmenterOptions, options=options)
        
    def __call__(self, image: NDArray) -> ImageSegmenterOut:
        """
        This method attempts to segment images into foreground/background objects.

        The objects are segmented by
        #. Performing a multi-level Otsu threshold on the image
        #. Choosing all but the bottom level from Otsu as likely foreground.
        #. Performing connected components on all the likely foreground objects
        #. Rejecting connected objects where the DN is less than the :attr:`minimum_segment_dn`
        #. Rejecting connected objects where the area is less than the :attr:`minimum_segment_area`

        The resulting objects are returned as a label matrix, where values >=1 indicate a pixel containing a foreground
        object (values of 0 are the background object). In addition, the statistics about the foreground objects are
        returned.

        :param image: The image to attempt to segment
        :return: A named tuple containing the label array, a boolean array specifying foreground objects, 
                 a stats array about the labels in order, and the centroids of the segments
        """
        
        # threshold the image
        levels, thresholded = otsu(image, self.otsu_levels)

        if float(levels[0]) > self.minimum_segment_dn:
            print(f'warning, the minimum Otsu level is greater than the minimum segment DN. This could indicate that '
                  f'there is an issue with your settings.\n\tminimum_segment_dn = {self.minimum_segment_dn}\n\t'
                  f'otsu_level = {levels[0]}')

        foreground_image = (thresholded >= 1).astype(np.uint8)

        _, labeled, stats, centroids = cv2.connectedComponentsWithStats(foreground_image)

        out_labeled = -np.ones(labeled.shape, dtype=np.int32)

        out_stats = []
        out_centroids = []

        stored_ind = 0

        sorted_labs = np.argsort(-stats[:, cv2.CC_STAT_AREA])  # sort the labels by size

        for ind in sorted_labs:

            stat = stats[ind]
            centroid = centroids[ind]

            if stat[cv2.CC_STAT_AREA] < self.minimum_segment_area:
                break  # since we are going in reverse size order if we get here we're done

            boolean_label = labeled == ind
            if np.median(image[boolean_label]) < self.minimum_segment_dn:
                continue

            out_labeled[boolean_label] = stored_ind
            out_stats.append(stat)
            out_centroids.append(centroid)
            stored_ind += 1

        return ImageSegmenterOut(out_labeled, foreground_image, np.array(out_stats), np.array(out_centroids))

    