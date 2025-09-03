"""
This module provides functionality for identifying the locations of points of interest 
(unresolved bright spots) in an image and fitting point spread functions to them.
"""

from dataclasses import dataclass, field

from typing import cast, NamedTuple

import numpy as np
from numpy.typing import NDArray

import cv2


from giant.point_spread_functions.psf_meta import PointSpreadFunction
from giant.point_spread_functions.gaussians import IterativeGeneralizedGaussian

from giant.image_processing.image_flattener import ImageFlattener, ImageFlattenerOptions, ImageFlatteningNoiseApprox

from giant.utilities.options import UserOptions
from giant.utilities.mixin_classes.user_option_configured import UserOptionConfigured
from giant.utilities.mixin_classes.attribute_equality_comparison import AttributeEqualityComparison
from giant.utilities.mixin_classes.attribute_printing import AttributePrinting

from giant._typing import ARRAY_LIKE


class FindPPOIInROIOut(NamedTuple):
    """
    This named tuple represents the output of the :meth:`.find_poit_in_roi` method.
    """
    
    peak_location: list[NDArray[np.int64]]
    """
    The location of the peak pixel in each detected point of interest
    """
    
    stats: list[NDArray[np.int32]]
    """
    The stats of each detected point of interest blob.
    
    Each element corresponds to the same element in :ref:`.poi_peak_locations`
    
    See `OpenCV connectedComponentsWithStats <https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f>`_ 
    for details
    """
    
    peak_snr: list[float]
    """
    The peak signal to noise ratio of the blob containing the detected point of interest.
    
    Each element corresponds to the same element in :ref:`.poi_peak_locations`
    """


class POIFinderOut(NamedTuple):
    """
    This named tuple represents the output of the point of interest finding process.

    It contains the centroids of the identified points of interest, their intensities,
    the fitted point spread functions, blob statistics, and peak signal-to-noise ratios.
    """
    
    centroids: NDArray[np.float64]
    """
    The identied centroids of the points of interest
    """
    
    centroid_intensity: NDArray
    """
    The intensity of the image at the location of the centroid of the point of interest.
    
    Each element corresponds to the same element in ref:`.poi_centroids`
    """
    
    point_spread_functions: list[PointSpreadFunction]
    """
    The fit point spread function for each point of interest.
    
    Each element corresponds to the same element in ref:`.poi_centroids`
    """
    
    stats: NDArray[np.int32]
    """
    The stats of each detected point of interest blob.
    
    See `OpenCV connectedComponentsWithStats <https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f>`_ 
    for details
    
    Each element corresponds to the same element in ref:`.poi_centroids`
    """
    
    peak_snr: NDArray[np.float64]
    """
    The peak signal to noise ratio of the blob containing the detected point of interest.
    
    Each element corresponds to the same element in ref:`.poi_centroids`
    """
    

@dataclass
class PointOfInterestFinderOptions(UserOptions):
    """
    This class defines the options used to configure the PointOfInterestFinder.

    It includes parameters for the point spread function to be fit, image flattening,
    centroid size, minimum and maximum blob size, threshold for point of interest
    detection, and saturation rejection.

    These options allow for fine-tuning the point of interest detection process
    to suit different types of images and detection requirements.
    """
    
    point_spread_function: type[PointSpreadFunction] = IterativeGeneralizedGaussian
    """
    The PSF object that estimates the center of a region of interest.
    
    This should be of the form::
    
        res = centroiding(x, y, illums)
        x0, y0 = res.centroid
        
    where x0, y0 is the subpixel center of the blob, [...] are optional outputs containing information about the 
    fit, x, y are arrays of the column and row locations corresponding to illums, and illums are the illumination 
    values at x, y.  
    
    There are a few built in options for centroiding in the :mod:`.point_spread_functions` package or you can build
    your own.
    """
    
    image_flattener_options: ImageFlattenerOptions = field(default_factory=ImageFlattenerOptions)
    """
    The options to use when flattening the image
    """
    
    centroid_size: int = 1
    """
    This specifies how many pixels to include when identifying a centroid in a region of interest.
    
    This sets the +/- number from the peak brightness pixel in both axes (so that a value of 1 means
    a 3x3 grid will be considered, a value of 2 will result in a 5x5 grid, etc).  
    """
    
    min_size: int = 2
    """
    This specifies the minimum number of pixels that must be connected for a blob to be considered a point of 
    interest.

    Individual pixels are clumped using a connected components algorithm, and then the size of each blob is compared
    against this value.  See :meth:`.locate_subpixel_poi_in_roi` for more details.
    """

    max_size: int = 50
    """
    This specifies the maximum number of pixels that must be connected for a blob to be considered a point of 
    interest.

    Individual pixels are clumped using a connected components algorithm, and then the size of each blob is compared
    against this value.  see :meth:`.locate_subpixel_poi_in_roi` for more details.
    """

    threshold: float = 8
    """
    This specifies the sigma multiplier to use when identifying a pixel as a point of interest.
    
    The sigma multiplier is applied to a rough noise estimate of the image (see 
    :meth:`.flatten_image_and_get_noise_level`) and then any pixels above this DN value are labeled as interesting 
    pixels that require further processing (see :meth:`.locate_subpixel_poi_in_roi`).
    """

    reject_saturation: bool = True
    """
    This boolean flag specifies whether to ignore clumps of pixels that contain saturated DN values when identifying 
    points of interest in an image.
    
    Set to True to reject any clumps containing saturated pixels.
    """


class PointOfInterestFinder(UserOptionConfigured[PointOfInterestFinderOptions],
                            PointOfInterestFinderOptions,
                            AttributeEqualityComparison,
                            AttributePrinting):
    """
    This class is responsible for finding and refining points of interest in an image.

    It uses a combination of image flattening, thresholding, and point spread function fitting
    to identify and locate subpixel centers of points of interest.

    The class is configured using the PointOfInterestFinderOptions, which allows for
    customization of various parameters such as the point spread function, image flattening
    options, centroid size, minimum and maximum blob size, threshold, and saturation rejection.

    Key methods:
    - find_poi_in_roi: Identifies pixel-level centers of points of interest in a region of interest.
    - refine_locations: Estimates subpixel centers of blobs given pixel-level locations.
    - __call__: Combines find_poi_in_roi and refine_locations to identify subpixel locations of points of interest.
    """
    
    def __init__(self, options: PointOfInterestFinderOptions | None = None) -> None:
        """
        :param options: The options configuring this class
        """
        super().__init__(PointOfInterestFinderOptions, options=options)
        
        self.image_flattener = ImageFlattener(self.image_flattener_options)
        
    @staticmethod
    def corners_to_roi(row_corners: ARRAY_LIKE, column_corners: ARRAY_LIKE) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        This method provides a convenient way to convert a set of corners to a region of interest that can be passed to
        :meth:`find_poi_in_roi` and :meth:`locate_subpixel_poi_in_roi`.

        This method finds the minimum and maximum row and column from row_corners and column_corners, respectively, and
        then makes a call to meshgrid using these bounds, reversing the output so it is row, col instead of col, row.

        The results from this function can be used to directly index into an image

        >>> import numpy
        >>> import giant.image_processing as gimp
        >>> im = numpy.random.randn(500, 600)
        >>> local_row_corners = [5.5, 3, 6.5, 8.9]
        >>> local_column_corners = [4.3, 2.7, 3.3, 7.8]
        >>> roi = im[gimp.ImageProcessing.corners_to_roi(local_row_corners, local_column_corners)]
        >>> (roi == im[3:10, 2:9]).all()
        True

        :param row_corners: a list of corner row locations
        :param column_corners: a list of corner column locations
        :return: row, column subscripts into an image as a tuple of ndarrays of type int
        """

        # get the bounds
        min_row, min_col = int(np.floor(np.min(row_corners))), int(np.floor(np.min(column_corners)))

        max_row, max_col = int(np.ceil(np.max(row_corners))), int(np.ceil(np.max(column_corners)))

        # return the subscripts
        return cast(tuple[NDArray[np.int64], NDArray[np.int64]], 
                    tuple(np.meshgrid(np.arange(min_row, max_row + 1), np.arange(min_col, max_col + 1), indexing='ij')))

    def find_poi_in_roi(self, image: np.ndarray, region: tuple[np.ndarray, np.ndarray] | None = None) \
            -> FindPPOIInROIOut:
        """
        This method identifies pixel level centers for all points of interest inside of some region of interest.

        A point of interest is defined as any grouping of *n* pixels that are above :attr:`.threshold` *
        **standard_deviation** where :attr:`min_size` <= *n* <= :attr:`.max_size`.  The **standard_deviation**
        is computed using the :meth:`.flatten_image_and_get_noise_level` method.
        Pixels are defined to be grouped if they are neighboring:

        .. code-block:: none

            nnnnn
            nyyyn
            nyoyn
            nyyyn
            nnnnn

        therefore any pixels labeled ``y`` are grouped with ``o`` whereas any pixels labeled ``n`` are not.

        This method will ignore any blobs that contain saturated pixels if :attr:`.reject_saturation` is set to True
        and the ``image`` object has an attribute :attr:`~.OpNavImage.saturation` containing the saturation level for
        the image.

        This method will also return the connected components stats (see
        `OpenCV connectedComponentsWithStats <https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html
        #ga107a78bf7cd25dec05fb4dfc5c9e765f>`_
        for details) and the peak signal to noise ratio for each detection.

        :param image: The image being considered
        :param region: The region of the image to consider
        :return: the pixel level locations of the points of interest in the region of interest (row, col), 
                 the connected component stats and the peak signal to noise ratio for each detection if
                 :attr:`.return_stats` is set to True.
        """


        # select the sub image we are considering
        if region is not None:
            roi_start = [np.min(region[1]), np.min(region[0])]

            big_roi = image[tuple(region)]

        else:
            roi_start = [0, 0]

            big_roi = image
            
        
        roi, noise_estimate, slices = self.image_flattener(big_roi)
        # get the flat image and approximate noise level(s) in the image
        if self.image_flattener_options.image_flattening_noise_approximation == ImageFlatteningNoiseApprox.GLOBAL:
            # detect pixels of interest by thresholding the flattened image at some multiple of the noise level
            snr = roi / noise_estimate
        else:
            # if we're doing local flattening and noise estimation
            # detect pixels of interest by thresholding the flattened image at some multiple of the noise level
            snr = np.zeros(big_roi.shape, dtype=np.float64)
            
            assert isinstance(noise_estimate, list) and slices is not None

            # for each region, compute the "snr" for each detection
            for noise, slices in zip(noise_estimate, slices):
                if noise < 1e-6:
                    continue
                flat_sliced = roi[slices[0], slices[1]]
                snr[slices[0], slices[1]] = flat_sliced / noise

        interesting_pix = snr > self.threshold

        # use connected components to blob the pixels together into single objects
        _, __, stats, ___ = cv2.connectedComponentsWithStats(interesting_pix.astype(np.uint8))

        poi_subs = []
        out_stats = []
        out_snrs = []

        # loop through each grouping of pixels
        for blob in stats:
            blob = cast(NDArray[np.int32], blob) # type hint for pylance
            left: int = blob[cv2.CC_STAT_LEFT]
            top: int = blob[cv2.CC_STAT_TOP]
            width: int = blob[cv2.CC_STAT_WIDTH]
            height: int = blob[cv2.CC_STAT_HEIGHT]
            area: int = blob[cv2.CC_STAT_AREA]

            if self.max_size >= area >= self.min_size:
                
                # get the image of the blob
                poi_roi = roi[top:top + height, left:left + width]

                # if we want to reject blobs that are affected by saturation
                if self.reject_saturation:

                    # ignore blobs where a portion of the blob is saturated
                    if (poi_roi >= getattr(image, 'saturation', np.finfo(np.float64).max)).any():
                        continue

                # get the subscript to the maximum illumination value within the current component and append it to the
                # return list

                # get the x/y location by unraveling the index (and reversing the order
                local_subs = np.unravel_index(np.nanargmax(poi_roi), poi_roi.shape)[::-1] 
                # store the results translated back to the full image and the statistics
                poi_subs.append(local_subs + blob[[0, 1]] + roi_start)
                out_stats.append(blob)
                out_snrs.append(snr[blob[1]:blob[1] + blob[3],
                                    blob[0]:blob[0] + blob[2]].max())

        return FindPPOIInROIOut(poi_subs, out_stats, out_snrs)

    def refine_locations(self, image: NDArray, points_of_interest: FindPPOIInROIOut) \
            -> POIFinderOut:
        """
        This method is used to estimate the subpixel centers of blobs in an image given the pixel level location of the
        blobs.

        The method operates by performing a user specified centroiding algorithm on the image area surrounding the
        specified pixel level centers of the points of interest.  The centroiding algorithm should typically be a
        subclass of :class:`.PointSpreadFunction`, however it can be any object with a ``fit`` method that inputs  3
        array like parameters with the first two being pixel locations and the last being DN values and returns a
        object with a ``centroid`` attribute which provides the (x, y) location of the centroid.  The centroiding
        algorithm is specified using the :attr:`.centroiding` attribute. The size of the area considered in the
        centroiding algorithm can be specified in the :attr:`.centroid_size` attribute.

        This method returns both the subpixel centers of the points of interest as well as the illumination values of
        the pixels containing the subpixel centers of the points of interest.  Optionally, stats about the blobs that
        the centroid was fit to and then full centroid fit can be returned if ``stats`` and ``snrs`` are not
        None and :attr:`.save_psf` is set to True, respectively.

        Note that if a centroid fit is unsuccessful then no information is returned for that point.  Therefore the
        output arrays lengths will be less than or equal to the length of the input array.

        This method is designed to be used in conjunction with the :meth:`find_poi_in_roi` method; however, it can be
        used with any rough identification method so long as the input format is correct.

        :param image: The image to be processed
        :param points_of_interest: The named tuple containing the pixel level information about the points of interest
        :return: The subpixel centers of the points of interest (col, row), the intensity of the center of the pois,
                 the fit :class:`.PointSpreadFunction` to the pois, the stats from the blobs containing the pois,
                 and the peak SNR of the blobs containing the pois.
        """

        # initialize lists for output
        points = []
        illums = []
        psfs = []
        out_stats = []
        out_snrs = []

        # loop through the pixel level points of interest
        for ind, center in enumerate(points_of_interest.peak_location):

            column_array = np.arange(center[0] - self.centroid_size,
                                     center[0] + self.centroid_size + 1)
            row_array = np.arange(center[1] - self.centroid_size,
                                  center[1] + self.centroid_size + 1)
            col_check = (column_array >= 0) & (column_array <= image.shape[1] - 1)
            row_check = (row_array >= 0) & (row_array <= image.shape[0] - 1)
            # valid_check = col_check & row_check
            cols, rows = np.meshgrid(column_array[col_check],
                                     row_array[row_check])

            # if col_check and row_check:
            if cols.size >= 0.5*(2*self.centroid_size + 1)**2:

                sampled_image = image[rows, cols].astype(np.float64)

                psf = self.point_spread_function.fit(cols - cols.mean(), rows - rows.mean(), sampled_image)

                psf.shift_centroid([cols.mean(), rows.mean()])

                x0, y0 = psf.centroid  
                
                # if we're outside the image or the fit failed skip this one
                if (x0 < 0) or (y0 < 0) or (np.isnan((x0, y0)).any()):
                    continue

                # check to be sure we haven't deviated too far from the original peak of interest (avoid poorly
                # conditioned systems)
                if (np.abs(np.asarray(center) - np.asarray([x0, y0]).flatten()) <= 3).all():
                    points.append([x0, y0])
                    illums.append(image[tuple(center[::-1])])
                    psfs.append(psf)
                    out_stats.append(points_of_interest.stats[ind])
                    out_snrs.append(points_of_interest.peak_snr[ind])

        return POIFinderOut(np.array(points, dtype=np.float64),
                            np.array(illums, dtype=image.dtype),
                            psfs,
                            np.array(out_stats, dtype=np.int32),
                            np.array(out_snrs, dtype=np.float64))

    def __call__(self, image: np.ndarray, region: tuple[np.ndarray, np.ndarray] | None = None) \
            -> POIFinderOut:
        """
        This method identifies the subpixel locations of points of interest in an image.

        This method is simply a convenient way of combining :meth:`find_poi_in_roi` and :meth:`refine_locations` and
        calls these two methods directly, feeding the results of the first into the second.

        :param image: The image to be processed
        :param region: The region of interest to consider as 2 numpy arrays of indices into the images or None
        :return: The subpixel centers of the points of interest (col, row), the intensity of the center of the pois,
                 the fit :class:`.PointSpreadFunction` to the pois, the stats from the blobs containing the pois,
                 and the peak SNR of the blobs containing the pois as a named tuple.
        """
 
        return self.refine_locations(image, self.find_poi_in_roi(image, region))
