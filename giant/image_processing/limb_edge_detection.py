from dataclasses import dataclass 

from typing import cast

import numpy as np
from numpy.typing import NDArray

import cv2

from giant.image_processing.edge_detection import PAESubpixelEdgeDetector, PAESubpixelEdgeDetectorOptions, \
                                                  ZernikeRampEdgeDetector, ZernikeRampEdgeDetectorOptions, \
                                                  PixelEdgeDetector, EdgeDetectionMethods, EdgeDetector
from giant.image_processing.otsu import otsu
from giant.utilities.outlier_identifier import get_outliers
from giant.utilities.options import UserOptions
from giant.utilities.mixin_classes.user_option_configured import UserOptionConfigured
from giant.utilities.mixin_classes.attribute_equality_comparison import AttributeEqualityComparison
from giant.utilities.mixin_classes.attribute_printing import AttributePrinting

from giant._typing import ARRAY_LIKE, DOUBLE_ARRAY


@dataclass
class LimbEdgeDetectionOptions(UserOptions):
    """
    This dataclass specifies the options which control how limbs are identified using edge detection.
    """
    
    edge_detection_type: EdgeDetectionMethods = EdgeDetectionMethods.ZERNIKE_RAMP_EDGE_DETECTOR
    """
    The edge detection type to use.
    
    If this is set to :attr:`.EdgeDetectionMethods.CUSTOM_DETECTOR` then the :attr:`custom_edge_detection_class` must be specified.
    
    If this is set to :attr:`.EdgeDetectionMethods.CUSTOM_DETECTOR`, :attr:`.EdgeDetectionMethods.PAE_SUBPIXEL_EDGE_DETECTOR` or :attr:`.EdgeDetectionMethods.ZERNIKE_RAMPE_EDGE_DETECTOR`
    then you can configure the detector using the :attr:`edge_detection_options` attribute.
    
    From this class, we only use the :meth:`.EdgeDetector.refine_edges` method to refine the pixel level limb points we detected.
    """
    
    custom_edge_detection_class: type[EdgeDetector] | None = None
    """
    A custom edge detector class implementing the :class:`.EdgeDetector` API.
    
    This will be initialized by providing the "attr:`edge_detection_options` attribute as the only argument.
    """
    
    edge_detection_options: object | ZernikeRampEdgeDetectorOptions | PAESubpixelEdgeDetectorOptions | None = None
    """
    The options dataclass to initialize the edge detector with.
    
    This is provided as the only argument to initialization if the :attr:`edge_detection_type` is set to anything by :attr:`.EdgeDetectionMethods.PIXEL_EDGE_DETECTOR`
    """
    
    step: int = 1
    """
    The step size to sample limb points at in units of pixels 
    """
    
    surround_check_size: int | None = None
    """
    How many pixels to use before/after the limb in the sun direction to check that the intensity changes from darker to brighter (indicating an illuminated limb).
    
    That is, we will check that the median of image intensity of the previous `surround_check_size` pixels along the scan line is lower than the median of the 
    image intensity of the next `surround_check_size` pixels along the scan line.
    
    If this is set to `None` then we use approximately 1/8 of the length of the scan line.
    """
    
    minimum_surround_check_size: int = 4
    """
    The minimum number of pixels before/after the limb to include in checking that the image goes from dark to light at the limb point.
    
    If there are not this many pixels before or after a potential limb to check (due to being close to the boundary of the image) then 
    the potential limb is discarded.  Therefore, you want this to be a decent size so you can get a reasonable sense of the intensity 
    before/after the potential limb point, but if set too large too many true limbs could be unnecessarily discarded.
    """
    
    
class LimbEdgeDetection(UserOptionConfigured[LimbEdgeDetectionOptions], 
                        LimbEdgeDetectionOptions,
                        AttributePrinting,
                        AttributeEqualityComparison):
    
    edge_detector: EdgeDetector
    
    def __init__(self, options: LimbEdgeDetectionOptions | None = None) -> None:
        super().__init__(LimbEdgeDetectionOptions, options=options)
        
        # set up the edge detector
        match self.edge_detection_type:
            case EdgeDetectionMethods.PIXEL_EDGE_DETECTOR:
                self.edge_detector = PixelEdgeDetector()
            case EdgeDetectionMethods.PAE_SUBPIXEL_EDGE_DETECTOR:
                if self.edge_detection_options is not None and not isinstance(self.edge_detection_options, PAESubpixelEdgeDetectorOptions):
                    raise ValueError('edge_detection_options must be None or an instance of PAESubpixelEdgeDetectorOptions to use the PAE_SUBPIXEL_EDGE_DETECTOR')
                self.edge_detector = PAESubpixelEdgeDetector(options=self.edge_detection_options)
            case EdgeDetectionMethods.ZERNIKE_RAMP_EDGE_DETECTOR:
                if self.edge_detection_options is not None and not isinstance(self.edge_detection_options, ZernikeRampEdgeDetectorOptions):
                    raise ValueError('edge_detection_options must be None or an instance of ZernikeRampEdgeDetectorOptions to use the ZERNIKE_RAMP_EDGE_DETECTOR')
                self.edge_detector = ZernikeRampEdgeDetector(options=self.edge_detection_options)
            case EdgeDetectionMethods.CUSTOM_DETECTOR:
                if self.custom_edge_detection_class is None or not issubclass(self.custom_edge_detection_class, EdgeDetector):
                    raise ValueError('custom_edge_detection_class must be specified as a subclass of EdgeDetecto if CUSTOM_DETECTOR is chosen')
                self.edge_detector = self.custom_edge_detection_class(options=self.edge_detection_options)

    def identify_subpixel_limbs(self, image: NDArray, illum_dir: ARRAY_LIKE, num_objs: int = 1) -> list[DOUBLE_ARRAY]:
        r"""
        This method identifies illuminated limbs in an image to sub-pixel accuracy.

        The input to this method is the image to have the limbs extracted from, the illumination direction in the image,
        and the number of objects that limbs are to be extracted from in the image.  The output is a list of arrays
        or subpixel limb points with each element of the list being a 2d array of the limb points for the
        i\ :sup:`th` object. The limb arrays are 2xn where n is the number of limb points and the first row
        corresponds to the x locations of the limb points in the image and the second row corresponds to the y
        locations of the limb points in the image.

        This method works by first thresholding the image to extract the foreground objects from the background using
        the :func:`otsu` function, and then identifying complete objects using connected components.  For each connected
        object up to `num_objs` objects, the limb points are extracted by scanning along the `illum_dir` vector to the
        first edge pixel encountered.  Then the edge level pixels are refined to subpixel accuracy  using one of the
        subpixel edge detection routines.

        :param image: The image to have the limbs extracted from
        :param illum_dir:  The direction of the incoming sunlight in the image
        :param num_objs: The number of objects to extract limbs from
        :return: A list of 2D arrays containing the xy subpixel limb points for each object in the image
        """
        # convert the image to uint8 if it isn't already
        if image.dtype != np.uint8:
            # noinspection PyArgumentList
            image = image.astype(np.float64) - image.min()
            image *= 255 / image.max()
            image = np.round(image).astype(np.uint8)

        # first, try to split the image into 4 bins with Otsu thresholding
        _, labels = otsu(image, 4)

        # get the number of pixels in each threshold level
        num_pix, _ = np.histogram(labels, np.arange(5))
        
        # check for outliers
        outliers = get_outliers(num_pix, sigma_cutoff=3)

        # handle the outliers
        if outliers.any():
            # check if levels 2 and 3 are also noise
            if (np.sqrt(2)*num_pix[1:].sum()) > num_pix[0]:
                outliers[:3] = True

        else:
            if (np.sqrt(2)*num_pix[1:].sum()) > num_pix[0]:
                outliers[:3] = True
            else:
                outliers[0] = True

        # also try with just the number of objects expected and take whichever gives more
        _, labels2 = otsu(image, num_objs+1)

        n_from2 = labels2.sum()

        # create a binary image where only the non-outlier pixels are turned on
        if n_from2 > num_pix[~outliers].sum():
            # already have 1 on the bright poritions
            connected_mat = labels2
        else:
            connected_mat = (labels == np.arange(4)[~outliers].reshape(-1, 1, 1)).any(axis=0)

        # do connected components
        _, __, stats, centroids = cv2.connectedComponentsWithStats(connected_mat.astype(np.uint8))

        # sort based on area size
        sorted_labs = np.argsort(-stats[:, cv2.CC_STAT_AREA])

        limbs = []
        for ind, blob in enumerate(sorted_labs[1:]):
            # if we have considered the maximum number of objects already
            if ind == num_objs:
                break

            # throw out blobs which are smaller than 10 pixels
            if stats[blob, cv2.CC_STAT_AREA] < 10:
                continue

            # extract the area around the blob from the image
            extra_bounds = 10
            top_left = stats[blob, [cv2.CC_STAT_TOP, cv2.CC_STAT_LEFT]] - extra_bounds
            bottom_right = top_left + stats[blob, [cv2.CC_STAT_HEIGHT, cv2.CC_STAT_WIDTH]] + 2 * extra_bounds + 1

            top_left[top_left < 0] = 0
            bottom_right[bottom_right < 0] = 0

            sub_image = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

            # determine the centroid of the current blob
            centroid = cast(DOUBLE_ARRAY, centroids[blob] - top_left[::-1])

            # check to be sure we have an actual object
            if sub_image.size == 0:
                continue

            # identify the subpixel limbs and store them
            limbs.append(self._locate_limbs(sub_image, centroid, illum_dir) + top_left[::-1].reshape(2, 1))

        return limbs

    def _locate_limbs(self, region: NDArray, centroid: DOUBLE_ARRAY, illum_dir: ARRAY_LIKE) -> np.ndarray:
        """
        This method identifies limb points in a region.

        This method combines the :meth:`identify_pixel_edges`, :meth:`_pixel_limbs`, and a subpixel method based off
        of the :attr:`.subpixel_method` attribute to determine the pixel level limb points in the region.  It inputs the
        region being considered, the centroid of the object in the region, and the illumination direction.  It outputs
        the subpixel limbs from the region.

        :param region: The imaging region being considered as a 2D array of illumination data
        :param centroid: The centroid of the blob in the region (typically provided by the opencv connected components
                         with stats function).
        :param illum_dir: The illumination direction in the region begin considered
        :return: the limb locations in the image
        """
        
        # detect edges in the image
        self.edge_detector.prepare_edge_inputs(region)

        # pick out the pixels corresponding to the limbs
        pixel_limbs = self._pixel_limbs(centroid, illum_dir, region)
            
        if not pixel_limbs.size:
            return pixel_limbs
        
        # refine to subpixel and return
        return self.edge_detector.refine_edges(region, pixel_limbs).astype(np.float64)

    def _pixel_limbs(self, centroid: DOUBLE_ARRAY, illum_dir: ARRAY_LIKE, image: NDArray) -> NDArray[np.int64]:
        """
        This method identifies pixel level limb points from a binary image of edge points.

        A limb is defined as the first edge point encountered by a scan vector in the direction of the illumination
        direction.  The limb points are extracted by (1) selecting starting locations for the scan vectors along a line
        perpendicular to the illumination direction spaced :attr:`step` pixels apart and then (2) scanning from these starting
        points in the illumination direction to identify the first edge point that is along the line.

        This method inputs a binary image with true values in the pixels which contain edges, the centroid of the object
        being considered in the binary image, the illumination direction, and the :attr:`step` size. It outputs the pixel level
        edges as a 2D array with the x values in the first row and the y values in the second row.

        :param centroid: The centroid of the object being considered
        :param illum_dir: the illumination direction in the `edge_mask` image
        :param image: the image being processed
        :return: The pixel level limb locations as a 2D integer array with the x values in the first row and the y values in the
                 second row
        """
        
        illum_dir_np = np.array(illum_dir, dtype=np.float64)
        # get the illumination gradient in the sun direction
        assert self.edge_detector.horizontal_gradient is not None and self.edge_detector.vertical_gradient is not None, "the gradients can't be none at this point"
        illum_grad: DOUBLE_ARRAY = self.edge_detector.horizontal_gradient*illum_dir_np[0] + self.edge_detector.vertical_gradient*illum_dir_np[1]

        # set the minimum gradient for a limb to be 3/4 of an SD away from the mean
        limb_grad_min = np.median(illum_grad) + 0.75*np.std(illum_grad)

        # determine how far we need to travel from the centroid to start our scan lines
        line_length = float(np.sqrt(np.sum(np.power(illum_grad.shape, 2))))

        # determine the direction to offset our scan stars
        perpendicular_direction = illum_dir_np[::-1].copy()
        perpendicular_direction[0] *= -1

        # get the middle of the start positions of our scan lines
        # middle start position of scan
        scan_start_middle = centroid - line_length * illum_dir_np

        # choose scan starting locations
        scan_starts = scan_start_middle.reshape(2, 1) + \
            np.arange(-line_length, line_length + 1, self.step).reshape(1, -1) * perpendicular_direction.reshape(2, -1)

        # where we'll store the limbs
        limbs = []

        # for each scan line
        for scan_line_number in range(scan_starts.shape[-1]):
            # get the points along the scan line
            sl = np.round(
                scan_starts[:, scan_line_number, np.newaxis] + np.arange(0, 2*line_length+1, 1) * illum_dir_np.reshape(2, -1)
            ).astype(int)
            _, sl_index = np.unique(sl, axis=1, return_index=True)
            # make sure we preserve order
            scan_line = sl[:, np.sort(sl_index)]

            # filter out invalid scan points
            scan_line = scan_line[:, (scan_line >= 0).all(axis=0) & 
                                  (scan_line < [[illum_grad.shape[1]], [illum_grad.shape[0]]]).all(axis=0)]

            # if none of the line is valid continue
            if not scan_line.size:
                continue
            # get the gradient values along the scan line
            scan_gradient = illum_grad[scan_line[1], scan_line[0]].ravel()

            # look for large gradients
            outliers = get_outliers(scan_gradient, 2)

            # get the step along the line the outliers fall out
            outlier_numbers = np.argwhere(outliers)

            for outlier_number in outlier_numbers.ravel():
                outlier_number = int(outlier_number)

                # make sure that the outlier is not at the end or beginning
                if (outlier_number == 0) or (outlier_number == (scan_gradient.size - 1)):
                    continue

                # make sure this is greater than the min gradient
                if scan_gradient[outlier_number] < limb_grad_min:
                    continue

                # check if we are at a local maxima
                if scan_gradient[outlier_number] < scan_gradient[outlier_number-1]:
                    continue
                if scan_gradient[outlier_number] < scan_gradient[outlier_number+1]:
                    continue

                # determine how many surrounding pixels we need to check
                # distance between the current step and the end points
                back_distance: int = outlier_number - 1
                forward_distance: int = scan_line.shape[1] - outlier_number - 1
                
                if self.surround_check_size is None:
                    # if the check size wasn't specified use 1/8 of the scan line length
                    check_size = scan_line.shape[1] // 8
                else:
                    check_size = self.surround_check_size

                check_size = min(check_size, back_distance, forward_distance)
                # if we're at the edge discard this
                if check_size < self.minimum_surround_check_size:
                    continue

                # check if the backwards is less than the forwards
                backwards_points = scan_line[:, outlier_number-check_size:outlier_number]
                backwards_median = np.median(image[backwards_points[1], backwards_points[0]])
                forwards_points = scan_line[:, outlier_number+1:outlier_number+1+check_size]
                forwards_median = np.median(image[forwards_points[1], forwards_points[0]])
                if backwards_median >= forwards_median:
                    continue

                # if we made it here we have a limb
                limbs.append(scan_line[:, outlier_number])
                break

        if limbs:
            limbs = np.vstack(limbs).T
        else:
            limbs = np.array(limbs)
        
        return limbs

