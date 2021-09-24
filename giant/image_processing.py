# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This module provides a number of image processing techniques for use throughout GIANT.

The class provided by this module, :class:`ImageProcessing`, is the primary tool used for working with image data
throughout GIANT. This class provides routines to identify point sources in an image (:meth:`.find_poi_in_roi`,
:meth:`.refine_locations`, :meth:`.locate_subpixel_poi_in_roi`), detect subpixel edges in an image (:meth:`.pae_edges`),
perform template matching through cross correlation (:meth:`.correlate`), and denoise/flatten an image and get its noise
level (:meth:`.flatten_image_and_get_noise_level`, :meth:`.denoise_image`).

For many of these methods, there are multiple algorithms that can be used to perform the same task.  The
:class:`ImageProcessing` class makes it easy to change what algorithm is being used by simply switching out one function
object for another.  There are a few selections of different algorithms that can be used already provided by this
module, and users can easily write their own algorithms and swap them in by following the instructions in the
:class:`ImageProcessing` class.

A general user will usually not directly interact with the classes and functions in this class and instead will rely on
the OpNav classes to interact for them.
"""

from typing import Callable, Iterable, Tuple, Union, List, Dict, Optional
from enum import Enum

import cv2
import numpy as np
import scipy.signal as sig
from scipy.optimize import fmin
from scipy.fftpack.helper import next_fast_len


from giant._typing import ARRAY_LIKE, ARRAY_LIKE_2D, Real
from giant.utilities.outlier_identifier import get_outliers
from giant.point_spread_functions import PointSpreadFunction, Gaussian


# fix for Serializing
cv2.GaussianBlur.__module__ = 'cv2'


# compute the image sobel masks
HORIZONTAL_KERNEL = np.array([[1, 0, -1],
                              [2, 0, -2],
                              [1, 0, -1]])
"""
The horizontal Sobel kernel for convolving with an image when computing the horizontal image gradients.

https://www.researchgate.net/publication/239398674_An_Isotropic_3x3_Image_Gradient_Operator
"""

VERTICAL_KERNEL = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])
"""
The vertical Sobel kernel for convolving with an image when computing the vertical image gradients.

https://www.researchgate.net/publication/239398674_An_Isotropic_3x3_Image_Gradient_Operator
"""

PAE_A01 = 0.125
"""
The 0, 1 coefficient (upper middle) of the Gaussian kernel representing the blurring experienced in the images being 
processed for the PAE sub-pixel edge method.

By default this is set to 0.125 assuming a 2D gaussian kernel with a sigma of 1 pixel in each axis.  If you know a 
better approximation of the gaussian kernel that represents the point spread function in the image (combined with any 
gaussian blurring applied to the image to smooth out noise) then you may get better results from the PAE method by 
updating this value.

https://www.researchgate.net/publication/233397974_Accurate_Subpixel_Edge_Location_based_on_Partial_Area_Effect
"""

PAE_A11 = 0.0625
"""
The 1, 1 coefficient (upper left) of the Gaussian kernel representing the blurring experienced in the images being 
processed for the PAE sub-pixel edge method.

By default this is set to 0.0625 assuming a 2D gaussian kernel with a sigma of 1 pixel in each axis.  If you know a 
better approximation of the gaussian kernel that represents the point spread function in the image (combined with any 
gaussian blurring applied to the image to smooth out noise) then you may get better results from the PAE method by 
updating this value.

https://www.researchgate.net/publication/233397974_Accurate_Subpixel_Edge_Location_based_on_Partial_Area_Effect
"""

# Store the Zernike Moments
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


class SubpixelEdgeMethods(Enum):
    """
    This enumeration provides the valid options for subpixel edge detection methods.

    You should be sure to use one of these values when setting to the :attr:`.subpixel_method` attribute of the
    :class:`.ImageProcessing` class.
    """

    PIXEL = "PIXEL"
    """
    Pixel level edges, no refining
    """

    PAE = "PAE"
    """
    Use Partial Area Effect to compute subpixel edge locations.
    
    See :meth:`.refine_edges_pae` for details.
    """

    ZERNIKE_RAMP = "ZERNIKE_RAMP"
    """
    Use Zernike Ramp to compute subpixel edge locations
    
    See :meth:`.refine_edges_zernike_ramp` for details.
    """


class ImageFlatteningNoiseApprox(Enum):
    """
    This enumeration provides the valid options for flattening an image and determining the noise levels when
    identifying points of interest in :meth:`.ImageProcessing.find_poi_in_roi`

    You should be sure to use one of these values when setting to the :attr:`.image_flattening_noise_approximation`
    attribute of the :class:`.ImageProcessing` class.
    """

    GLOBAL = "GLOBAL"
    """
    Globally flatten the image and estimate the noise level from it.
    
    In this the image in flattened by subtracting a median filtered version of the image from it and a single noise 
    level is approximated for the entire image either through sampling or through the :attr:`.dark_pixels` of the image.
    
    For most OpNav cases this is sufficient and fast.
    """

    LOCAL = "LOCAL"
    """
    Locally flatten the image and estimate the noise levels for each local region

    In this the image in flattened by splitting it into regions, estimating a linear background gradient in each region,
    and the subtracting the estimated background gradient from the region to get the flattened region.  An individual 
    noise level is estimated for each of these regions through sampling.

    This technique allows much dimmer points of interest to be extracted without overwhelming with noise, but it is 
    generally much slower and is unnecessary for all but detailed analyses.
    """


def local_maxima(data_grid: ARRAY_LIKE_2D) -> np.ndarray:
    """
    This function returns a boolean mask selecting all local maxima from a 2d array.

    A local maxima is defined as any value that is greater than or equal to all of the values surrounding it.  That is,
    given:

    .. code::

        +---+---+---+
        | 1 | 2 | 3 |
        +---+---+---+
        | 4 | 5 | 6 |
        +---+---+---+
        | 7 | 8 | 9 |
        +---+---+---+

    value 5 is a local maxima if and only if it is greater than or equal to values 1, 2, 3, 4, 6, 7, 8, 9.

    For edge cases, only the valid cells are checked (ie value 1 would be checked against values 2, 4, 5 only).

    >>> from giant.image_processing import local_maxima
    >>> im = [[0, 1, 2, 20, 1],
    ...       [5, 2, 1, 3, 1],
    ...       [0, 1, 2, 10, 1],
    ...       [1, 2, -1, -2, -5]]
    >>> local_maxima(im)
    array([[False, False, False,  True, False],
           [ True, False, False, False, False],
           [False, False, False,  True, False],
           [False,  True, False, False, False]], dtype=bool)

    :param data_grid: The grid of values to search for local maximas
    :return: A 2d boolean array with `True` where the data_grid values are local maximas
    """

    # make sure the array is numpy
    array2d = np.atleast_2d(data_grid)

    # check the interior points
    test = ((array2d >= np.roll(array2d, 1, 0)) &
            (array2d >= np.roll(array2d, -1, 0)) &
            (array2d >= np.roll(array2d, 1, 1)) &
            (array2d >= np.roll(array2d, -1, 1)) &
            (array2d >= np.roll(np.roll(array2d, 1, 0), 1, 1)) &
            (array2d >= np.roll(np.roll(array2d, -1, 0), 1, 1)) &
            (array2d >= np.roll(np.roll(array2d, 1, 0), -1, 1)) &
            (array2d >= np.roll(np.roll(array2d, -1, 0), -1, 1))
            )

    # test the edges
    # test the top
    test[0] = array2d[0] >= array2d[1]
    test[0, :-1] &= (array2d[0, :-1] >= array2d[0, 1:]) & (array2d[0, :-1] >= array2d[1, 1:])
    test[0, 1:] &= (array2d[0, 1:] >= array2d[0, :-1]) & (array2d[0, 1:] >= array2d[1, :-1])

    # test the left
    test[:, 0] = array2d[:, 0] >= array2d[:, 1]
    test[:-1, 0] &= (array2d[:-1, 0] >= array2d[1:, 0]) & (array2d[:-1, 0] >= array2d[1:, 1])
    test[1:, 0] &= (array2d[1:, 0] >= array2d[:-1, 0]) & (array2d[1:, 0] >= array2d[:-1, 1])

    # test the right
    test[:, -1] = array2d[:, -1] >= array2d[:, -2]
    test[:-1, -1] &= (array2d[:-1, -1] >= array2d[1:, -1]) & (array2d[:-1, -1] >= array2d[1:, -2])
    test[1:, -1] &= (array2d[1:, -1] >= array2d[:-1, -1]) & (array2d[1:, -1] >= array2d[:-1, -2])

    # test the bottom
    test[-1] = array2d[-1] >= array2d[-2]
    test[-1, :-1] &= (array2d[-1, :-1] >= array2d[-1, 1:]) & (array2d[-1, :-1] >= array2d[-2, 1:])
    test[-1, 1:] &= (array2d[-1, 1:] >= array2d[-1, :-1]) & (array2d[-1, 1:] >= array2d[-2, :-1])

    # send out the results
    return test


def cv2_correlator_2d(image: np.ndarray, template: np.ndarray, flag: int = cv2.TM_CCOEFF_NORMED) -> np.ndarray:
    """
    This function performs a 2D cross correlation between ``image`` and ``template`` and returns the correlation surface
    using the `OpenCV matchTemplate function <http://docs.opencv.org/3.1.0/d4/dc6/tutorial_py_template_matching.html>`_.

    The input ``image`` and ``template`` are first converted to single precision (as is required by matchTemplate) and
    then given to the matchTemplate function.

    The flag indicates the correlation coefficients to calculate (in general you will want ``cv2.TM_CCOEFF_NORMED`` for
    normalized cross correlation).  For more information about this function see the OpenCV documentation at
    https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html

    Each pixel of the correlation surface returned by this function represents the correlation value when the center of
    the template is placed at this location.  Thus, the location of any point in the template can be found by

    >>> import numpy
    >>> from giant.image_processing import cv2_correlator_2d
    >>> example_image = numpy.random.randn(200, 200)
    >>> example_template = example_image[30:60, 45:60]
    >>> surf = cv2_correlator_2d(example_image, example_template)
    >>> temp_middle = numpy.floor(numpy.array(example_template.shape)/2)
    >>> template_point = numpy.array([0, 0])  # upper left corner
    >>> template_point - temp_middle + numpy.unravel_index(surf.argmax(), surf.shape)
    array([30., 45.])

    :param image: The image that the template is to be matched against
    :param template: the template that is to be matched against the image
    :param flag: A flag indicating the correlation coefficient to be calculated
    :return: A surface of the correlation coefficients for each overlap between the template and the image.
    """

    # calculate what the size of the correlation surface should be and pad the image with 0s
    size_diff = np.array(template.shape) / 2
    upper = np.ceil(size_diff).astype(int)
    lower = np.floor(size_diff).astype(int)

    original_shape = image.shape
    image = np.pad(image.astype(np.float32), [(lower[0], upper[0]), (lower[1], upper[1])], 'constant')

    # perform the correlation
    cor_surf = cv2.matchTemplate(image, template.astype(np.float32), flag)

    # return the correlation surface of the appropriate size
    return cor_surf[:original_shape[0], :original_shape[1]]


def _normalize_xcorr_2d(image: np.ndarray, zero_mean_temp: np.ndarray, corr_surf: np.ndarray) -> np.ndarray:
    """
    This function calculates normalized correlation coefficients between the template and the image based off of the
    non-normalized correlation between temp and image.

    This method works by computing the local standard deviation and mean of the image for each overlay of the template,
    then dividing the correlation surface by the difference of these values (roughly at least).

    This function is used inside of both :func:`spatial_correlator_2d` and :func:`fft_correlator_2d` to normalize the
    correlation surfaces.  Typically it is not used explicitly by the user.

    :param image: the image that was correlated against
    :param zero_mean_temp: the zero mean version of the template that was correlated
    :param corr_surf: the non-normalized correlation surface to be normalized
    :return: the normalized correlation surface
    """

    # the following code is based off of MATLAB's normxcorr2 which is based off of
    # Lewis, J. P. "Fast normalized cross-correlation." Vision interface. Vol. 10. No. 1. 1995.
    def local_sum(in_mat: np.ndarray, shape: tuple):
        """
        Compute the integral of in_mat over the given search areas.

        :param in_mat: the matrix to be integrated
        :param shape: the size of the search areas
        :return: a matrix containing the integral of in_mat for a search area overlaid starting at each pixel of in_mat
        """

        # first, pad in_mat so that the template can be overlaid on the borders as well
        in_mat = np.pad(in_mat, [(shape[0], shape[0]), (shape[1], shape[1])], 'constant')

        # calculate the cumulative summation along the first axis (down each row)
        sum1 = in_mat.cumsum(0)

        # calculate the running sums for the rows
        temp1 = sum1[shape[0]:-1] - sum1[:(-shape[0] - 1)]

        # calculate the cumulative summation along the second axis (down each column)
        sum2 = temp1.cumsum(1)

        # calculate the running sums for the cols
        return sum2[:, shape[1]:-1] - sum2[:, :(-shape[1] - 1)]

    # get the integral of the images under the template for each overlay
    local_means = local_sum(image, zero_mean_temp.shape)  # this is the template.size*mean of the image within the
    # template window for every overlay of the template
    local_sum_squares = local_sum(image * image,
                                  zero_mean_temp.shape)  # this is the sum of the squares of the image within
    # the template window for every overlay of the template

    # calculate the variance of the image under the template for the area overlaid under each image and ensure the
    # variance is positive or zero (it will only be negative due to numerical precision issues)
    local_variance = local_sum_squares - local_means ** 2 / zero_mean_temp.size
    local_variance[local_variance < 0] = 0

    # calculate the variance of the template itself
    temp_variance = (zero_mean_temp ** 2).sum()

    # calculate the product of the local standard deviations of the image and the standard deviation of the template
    # (this is the same as the square root of the product of the variances)
    std_image_std_template = np.sqrt(local_variance * temp_variance)

    # calculate the normalized correlation coefficients
    res = corr_surf / std_image_std_template

    # check to make sure that machine precision and divide by zero errors haven't given us any invalid answers
    res[np.abs(res) > 1 + np.sqrt(np.finfo(np.float64).eps)] = 0
    # this step shouldn't be necessary due to the previous step but its basically instantaneous so keep it in to be safe
    res[std_image_std_template == 0] = 0

    return res


def fft_correlator_2d(image: ARRAY_LIKE_2D, template: ARRAY_LIKE_2D) -> np.ndarray:
    """
    This function performs normalized cross correlation between a template and an image in the frequency domain.

    The correlation is performed over the full image, aligning the center of the template with every pixel in the image.
    (Note that this means that if the center of the template should be outside of the image this function will not
    work.)

    The correlation in this method is roughly performed by

    #. take the 2D fourier transform of the image and the fliplr/flipud template
    #. multiply each term of the frequency image and template together
    #. take the inverse fourier transform of the product from step 2.
    #. normalize the correlation coefficients

    Each pixel of the correlation surface returned by this function represents the correlation value when the center of
    the template is placed at this location.  Thus, the location of any point in the template can be found by

    >>> import numpy as numpy
    >>> from giant.image_processing import fft_correlator_2d
    >>> example_image = numpy.random.randn(200, 200)
    >>> example_template = example_image[30:60, 45:60]
    >>> surf = fft_correlator_2d(example_image, example_template)
    >>> temp_middle = numpy.floor(numpy.array(example_template.shape)/2)
    >>> template_point = numpy.array([0, 0])  # upper left corner
    >>> template_point - temp_middle + numpy.unravel_index(surf.argmax(), surf.shape)
    array([30., 45.])

    :param image: The image that the template is to be matched against
    :param template: the template that is to be matched against the image
    :return: A surface of the correlation coefficients for each overlap between the template and the image.
    """

    # perform the correlation in the frequency domain.  Note that template needs to be fliplr/flipud due to the
    # definition of correlation

    # use the zero mean template to simplify some steps later
    zero_mean_temp = template - template.mean()
    corr_surf = sig.fftconvolve(image, zero_mean_temp[::-1, ::-1], 'full')
    # the preceding is mostly equivalent to the following but it does a better job of handling the shapes to make things
    # faster
    # fft_shape = np.array(image.shape)+np.array(template.shape)-1
    # image_fft = np.fft.rfft2(image, s=fft_shape)
    # template_fft = np.fft.rfft2(template[::-1, ::-1], s=fft_shape)
    # corr_surf = np.fft.irfft2(image_fft*template_fft, s=fft_shape)

    # this forms the un-normalized correlation surface.  Now we need to normalize:
    res = _normalize_xcorr_2d(image, zero_mean_temp, corr_surf)

    # get the output size for output type of "same"
    diff = (np.array(res.shape) - np.array(image.shape)) / 2

    lower = np.floor(diff).astype(int)
    upper = np.ceil(diff).astype(int)

    # return the correlation surface for type "same"
    return res[lower[0]:-upper[0], lower[1]:-upper[1]]


def spatial_correlator_2d(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    This function performs normalized cross correlation directly (spatial correlation).

    The correlation is performed over the full image, aligning the center of the template with every pixel in the image.
    (Note that this means that if the center of the template should be outside of the image this function will not
    work.)

    Each pixel of the correlation surface returned by this function represents the correlation value when the center of
    the template is placed at this location.  Thus, the location of any point in the template can be found by

    >>> import numpy
    >>> from giant.image_processing import spatial_correlator_2d
    >>> example_image = numpy.random.randn(200, 200)
    >>> example_template = example_image[30:60, 45:60]
    >>> surf = spatial_correlator_2d(example_image, example_template)
    >>> temp_middle = numpy.floor(numpy.array(example_template.shape)/2)
    >>> template_point = numpy.array([0, 0])  # upper left corner -- replace 0, 0 with whichever template location you
    >>> # want (starting with the upper left as 0, 0).
    >>> template_point - temp_middle + numpy.unravel_index(surf.argmax(), surf.shape)
    array([30., 45.])

    :param image: The image that the template is to be matched against
    :param template: the template that is to be matched against the image
    :return: A surface of the correlation coefficients for each overlap between the template and the image.
    """

    image = image.copy()
    template = template.copy()

    zero_mean_temp = template - template.mean()

    corr_surf = sig.convolve2d(image, zero_mean_temp[::-1, ::-1], 'full')

    # this forms the un-normalized correlation surface.  Now we need to normalize:
    res = _normalize_xcorr_2d(image, zero_mean_temp, corr_surf)

    # get the output size for output type of "same"
    diff = (np.array(res.shape) - np.array(image.shape)) / 2

    lower = np.floor(diff).astype(int)
    upper = np.ceil(diff).astype(int)

    # return the correlation surface for type "same"
    return res[lower[0]:-upper[0], lower[1]:-upper[1]]


def _normalize_xcorr_1d(extracted: np.ndarray, zero_mean_predicted: np.ndarray, corr_lines: np.ndarray) -> np.ndarray:
    """
    This function normalizes correlation coefficients between 1d lines based off of the non-normalized correlation
    between the 1d lines. This method works by computing the local standard deviation and mean of the extracted for
    each overlay of the temps, then dividing the correlation surface by the difference of these values (roughly
    at least). This function is used inside of :func:`n1d_correlate` to normalize the
    correlation surfaces.

    :param extracted: the extracted scan lines that were correlated against (each image should be contained in the last
                      axis
    :param zero_mean_predicted: the zero mean versions of the predicted_lines that were correlated
    :param corr_lines: the non-normalized correlation lines to be normalized
    :return: the normalized correlation lines
    """

    # The following code is based off of MATLAB's normxcorr2 which is based off of
    # Lewis, J. P. "Fast normalized cross-correlation." Vision interface. Vol. 10. No. 1. 1995.

    def local_sum(in_mat, shape) -> np.ndarray:
        """
        Compute the integral of in_mat over the given search areas.

        :param in_mat: the matrix to be integrated
        :param shape: the size of the search areas
        :return: a matrix containing the integral of in_mat for a search area overlaid starting at each pixel of in_mat
        """

        # First, pad in_mat so that the template can be overlaid on the borders as well
        in_mat = np.pad(in_mat, [(0, 0), (shape[-1], shape[-1])], 'constant')

        # Calculate the cumulative summation along the second axis (down each column)
        sum2 = in_mat.cumsum(1)

        # Calculate the running sums for the cols
        return sum2[:, shape[1]:-1] - sum2[:, :(-shape[1] - 1)]

    # Get the integral of the extracted lines under the template for each overlay
    local_means = local_sum(extracted, zero_mean_predicted.shape)  # this is the predicted.size*mean of the extracted

    # Within the predicted window for every overlay of the predicted
    local_sum_square = local_sum(extracted * extracted,
                                 zero_mean_predicted.shape)  # This is the sum of the squares of the image within
    # The template window for every overlay of the template

    # Calculate the variance of the extracted lines under the predicted_lines for the area overlaid under each image and
    # Ensure the variance is positive or zero (it will only be negative due to numerical precision issues)
    local_variance = local_sum_square - local_means ** 2 / zero_mean_predicted.shape[-1]
    local_variance[local_variance < 0] = 0

    # Calculate the variance of the template itself
    temp_variance = (zero_mean_predicted ** 2).sum(axis=-1, keepdims=True)

    # Calculate the product of the local standard deviations of the image and the standard deviation of the template
    # (This is the same as the square root of the product of the variances)
    std_image_std_template = np.sqrt(local_variance * temp_variance)

    # Calculate the normalized correlation coefficients
    res = corr_lines / std_image_std_template

    # Check to make sure that machine precision and divide by zero errors haven't given us any invalid answers
    res[np.abs(res) > 1 + np.sqrt(np.finfo(np.float64).eps)] = 0

    # This step shouldn't be necessary due to the previous step but its basically instantaneous so keep it in to be safe
    res[std_image_std_template == 0] = 0
    return res


def _fft_correlate_1d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    This function performs FFT based correlation on nd arrays of 1d scan lines.

    :param a: array of 1d scan lines
    :param b: array of 1d scan lines
    :return: array of spatial correlation values
    """
    # Determine the size of the correlation surface for type "full"
    n = a.shape[-1] + b.shape[-1] - 1

    # Get the next fast fft length
    fft_size = next_fast_len(n)

    # Transform the input values into the frequency domain
    a_fft = np.fft.rfft(a, n=fft_size)
    b_fft = np.fft.rfft(b, n=fft_size)

    # Perform the correlation and transform back to the spatial domain
    cc = np.fft.irfft(a_fft * b_fft.conj(), n=fft_size)

    return np.hstack([cc[..., -b.shape[-1] + 1:], cc[..., :a.shape[-1]]])


def fft_correlator_1d(extracted_lines: np.ndarray, predicted_lines: np.ndarray) -> np.ndarray:
    """
    This function performs 1d correlation based on extracted lines and predicted lines.

    Each line of the input matrices should be a pair of scan lines to be correlated.  The result of this function
    will be a numpy array of correlation coefficients for the cross correlation of the lines.

    The correlation is computed using discrete fourier transforms to transform the scan lines into the frequency domain.
    The correlation is then performed in the frequency domain and then transformed back into the spatial domain.
    Finally, the spatial correlation lines are normalized to have values between -1 and 1 in the usual sense.

    :param extracted_lines: array of extracted lines to be correlated
    :param predicted_lines: array of predicted lines to be correlated
    :return: array of correlation coefficients for each scan line pair.
    """

    # Subtract the mean from each template to reduce the complexity later
    zero_mean_pred_lines = predicted_lines - predicted_lines.mean(axis=-1, keepdims=True)

    # Get the un-normalized correlation lines (for a "full" correlation)
    un_normalized_corr_lines = _fft_correlate_1d(extracted_lines, zero_mean_pred_lines)

    # Normalize the correlation coefficients to be between -1 and 1
    res = _normalize_xcorr_1d(extracted_lines, zero_mean_pred_lines, un_normalized_corr_lines)[..., 1:]

    # Select only correlation coefficients for type "same"
    diff = (res.shape[-1] - extracted_lines.shape[-1]) / 2

    # Determine the regions of the correlation lines that are valid
    lower = int(np.floor(diff))
    upper = int(np.ceil(diff))

    # Only return the valid regions
    out = res[..., lower:-upper]

    return out


def otsu(image: np.ndarray, n: int) -> Tuple[List[Real], np.ndarray]:
    """
    This function performs multilevel Otsu thresholding on a 2D array.

    Otsu thresholding is a technique by which the optimal threshold is chosen so as to split a 2D array based on the
    peaks in its histogram.  In multilevel thresholding, we choose multiple optimal thresholds so that multiple peaks
    are separated.  This process is described in
    "Otsu N, A Threshold Selection Method from Gray-Level Histograms, IEEE Trans. Syst. Man Cybern. 1979;9:62-66."

    To use this function, simply input the image and the number of times you want to split the histogram.  The function
    will then return the optimal threshold values used to bin the image (n-1 thresholds), and a labeled image where each
    bin has its own number (n labels).  Note that the function will convert the image to a uint8 image if it is not
    already, and the thresholds will correspond to the uint8 image.

    This function uses the opencv threhold function to perform the thresholding when n=2 and is based off of the
    MATLAB function otsu
    (https://www.mathworks.com/matlabcentral/fileexchange/26532-image-segmentation-using-otsu-thresholding?s_tid=prof_contriblnk)
    for when n>=3.

    >>> import numpy
    >>> from giant.image_processing import otsu
    >>> from giant.point_spread_functions import Gaussian
    >>> im = numpy.zeros((100, 100), dtype=numpy.float64)
    >>> x, y = numpy.meshgrid(numpy.arange(10), numpy.arange(10))
    >>> psf = Gaussian(sigma_x=1.5, sigma_y=0.7, amplitude=100, centroid_x=5, centroid_y=5)
    >>> im[50:60, 50:60] = psf.evaluate(x, y)
    >>> thresh, labeled_im = otsu(im, 3)
    >>> print(thresh)
    [0.24723526470339388, 2.235294117647059]

    :param image: The grayscale image to be thresholded as a numpy array
    :param n: The number of times to bin the image (for a binary threshold n=2)
    :return: The n-1 threshold values and the labeled image with the background
             being labeled 0 and each subsequent bin being labeled with the next integer (ie 1, 2, 3, ...)
    """

    # convert the image to uint 8 (Assume it is already grayscale)
    if image.dtype != np.uint8:
        # noinspection PyArgumentList
        delta_conv = image.min()
        iu8 = image.astype(np.float64) - delta_conv
        multi_conv = 255 / iu8.max()
        iu8 = np.round(iu8 * multi_conv).astype(np.uint8)
    else:
        iu8 = image
        delta_conv = 0
        multi_conv = 1

    if n == 2:
        threshold, labeled_image = cv2.threshold(iu8, 0, 1, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

        threshold = float(threshold)
        threshold /= multi_conv
        threshold += delta_conv

        threshold = image.dtype.type(threshold)

        return [threshold], labeled_image

    else:
        # get the unique dn values at uint8 level
        unique_iu8 = np.unique(iu8.ravel())

        range_unique_image = np.arange(1, unique_iu8.size + 1)

        # generate a histogram of the values
        hist, _ = np.histogram(iu8, np.hstack([unique_iu8, [256]]))

        # estimate the pdf by scaling back so the integral is equal to 1
        pdf = hist / hist.sum()

        range_unique_image_pdf = range_unique_image * pdf

        if n == 3:

            # determine the zeroth and first-order cumulative moments
            w = pdf.cumsum()
            mu = range_unique_image_pdf.cumsum()

            w0 = w
            w2 = pdf[::-1].cumsum()[::-1]

            w0, w2 = np.meshgrid(w0, w2, indexing='ij')

            mu0 = mu / w

            mu2 = (range_unique_image_pdf[::-1].cumsum() / pdf[::-1].cumsum())[::-1]

            mu0, mu2 = np.meshgrid(mu0, mu2, indexing='ij')

            w1 = 1 - w0 - w2

            w1[w1 < 0] = np.nan

            mu0mue = mu0 - mu[-1]
            mu2mue = mu2 - mu[-1]
            w0mu0mue = w0 * mu0mue
            w2mu2mue = w2 * mu2mue

            sigma2b = w0mu0mue * mu0mue + w2mu2mue * mu2mue + (w0mu0mue + w2mu2mue) ** 2 / w1

            sigma2b[~np.isfinite(sigma2b)] = 0

            k = sigma2b.ravel().argmax()

            k1, k2 = np.unravel_index(k, sigma2b.shape)

            labeled_image = np.zeros(image.shape, dtype=np.float64)

            labeled_image[(iu8 > unique_iu8[k1]) & (iu8 <= unique_iu8[k2])] = 1

            labeled_image[iu8 > unique_iu8[k2]] = 2

            thresholds = np.array([unique_iu8[k1], unique_iu8[k2]], dtype=np.float64)

            thresholds /= multi_conv
            thresholds += delta_conv

            # noinspection PyTypeChecker
            out_thresh = thresholds.astype(image.dtype).tolist()  # type: list

            for ind, t in enumerate(out_thresh):
                out_thresh[ind] = min(max(t, image[labeled_image == ind].max()), image[labeled_image == ind+1].min())

            return out_thresh, labeled_image

        else:

            mut = range_unique_image_pdf.sum()
            sig2t = ((range_unique_image - mut) ** 2 * pdf).sum()

            def sig_fun(ik: np.ndarray) -> float:
                """
                A temporary function for passing to the optimizer

                :param ik:
                :return:
                """

                ik = np.round(ik * (unique_iu8.size - 1) + 1.000000000001)
                ik = np.sort(ik)

                if ((ik < 1) | (ik > unique_iu8.size)).any():
                    return 1

                ik = np.hstack([0, ik, unique_iu8.size]).astype(int)

                sigma2bi = 0

                for ii in range(n):
                    wj = pdf[ik[ii]:ik[ii + 1]].sum()

                    if wj == 0:
                        return 1

                    muj = (np.arange(ik[ii] + 1, ik[ii + 1] + 1) * pdf[ik[ii]:ik[ii + 1]]).sum() / wj
                    sigma2bi += wj * (muj - mut) ** 2

                return 1 - sigma2bi / sig2t

            k0 = np.linspace(0, 1, n + 1)[1:-1]

            kk = fmin(sig_fun, k0, xtol=1, disp=False)

            kk = np.round(kk * (unique_iu8.size - 1)).astype(int)

            labeled_image = np.zeros(image.shape, dtype=np.float64)

            labeled_image[iu8 > unique_iu8[kk[n - 2]]] = n - 1
            for i in range(n - 2):
                labeled_image[(iu8 > unique_iu8[kk[i]]) & (iu8 <= unique_iu8[kk[i + 1]])] = i + 1

            # put back into the original image values
            thresholds = unique_iu8[kk[:n - 2]].astype(np.float64)
            thresholds /= multi_conv
            thresholds += delta_conv

            # noinspection PyTypeChecker
            out_thresh = thresholds.astype(image.dtype).tolist()  # type: list

            for ind, t in enumerate(out_thresh):
                out_thresh[ind] = min(max(t, image[labeled_image == ind].max()), image[labeled_image == ind+1].min())

            return out_thresh, labeled_image


def pixel_level_peak_finder_2d(surface: ARRAY_LIKE_2D, blur: bool = True) -> np.ndarray:
    """
    This function returns a numpy array containing the (x, y) location of the maximum surface value
    to pixel level accuracy.

    Optionally, a blur can be applied to the surface before locating the peak to attempt to remove high frequency noise.

    :param surface: A surface, or image, to use
    :param blur: A flag to indicate whether to apply Gaussian blur to image
    :return: The (x, y) location of the maximum surface values to pixel level accuracy.
    """
    surface = np.array(surface)

    if blur:
        # Do this to try to avoid spikes due to noise aligning
        surface = cv2.GaussianBlur(surface, (5, 5), 1)

    return np.flipud(np.unravel_index(np.argmax(surface), np.shape(surface)))


def quadric_peak_finder_2d(surface: ARRAY_LIKE_2D, fit_size: int = 1, blur: bool = True,
                           shift_limit: int = 3) -> np.ndarray:
    r"""
    This function returns a numpy array containing the (x, y) location of the maximum surface value
    which corresponds to the peak of the fitted quadric surface to subpixel accuracy.

    First, this function calls :func:`pixel_level_peak_finder_2d` to identify the pixel location of the peak of the
    correlation surface.  It then fits a 2D quadric to the pixels around the peak and solves for the center of the
    quadric to be the peak value.  The quadric equation that is fit is

    .. math::
        z = Ax^2+By^2+Cxy+Dx+Ey+F

    where :math:`z` is the height of the correlation surface at location :math:`(x,y)`, and :math:`A--F` are the
    coefficients to be fit.  The fit is performed in an algebraic least squares sense.
    The location of the peak of the surface is then given by:

    .. math::
        \left[\begin{array}{c}x_p \\ y_p\end{array}\right] = \frac{1}{4AB-C^2}\left[\begin{array}{c} CE-2BD\\
        CD-2AE\end{array}\right]

    where :math:`(x_p,y_p)` is the location of the peak.

    If the peak is invalid because it is too close to the edge, the fit failed, or the parabolic fit moved
    the peak too far from the pixel level peak then the result is returned as NaNs.

    :param surface: A surface, or image, to use
    :param fit_size: Number of pixels around the peak that are used in fitting the paraboloid
    :param blur: A flag to indicate whether to apply Gaussian blur to the correlation surface to filter out high
                 frequency noise
    :param shift_limit: maximum difference from the pixel level peak to the fitted peak for the fitted peak to be
                        accepted
    :return: The (x, y) location corresponding to the peak of fitted quadric surface to subpixel accuracy
    """

    # make sure we have an array
    surface = np.asarray(surface)

    # find the pixel level peak
    max_col, max_row = pixel_level_peak_finder_2d(surface, blur=blur)

    # if we're too close to the edge return NaNs
    if ((max_row - fit_size) < 0) or ((max_row + fit_size) >= surface.shape[0]) or \
            ((max_col - fit_size) < 0) or ((max_col + fit_size) >= surface.shape[1]):
        return np.array([np.nan, np.nan])

    # set up the columns/rows we will fit the peak to
    deltas = np.arange(-fit_size, fit_size + 1)
    cols, rows = np.meshgrid(max_col + deltas, max_row + deltas)

    cols = cols.flatten()
    rows = rows.flatten()

    # form the jacobian matrix for the leas squares
    jac_matrix = np.array([cols * cols, rows * rows, cols * rows, cols, rows, np.ones(rows.shape)]).T

    # perform the least squares fit
    coefs = np.linalg.lstsq(jac_matrix, surface[rows, cols].flatten(), rcond=None)[0]

    # extract the peak column and row
    peak_col = (coefs[2] * coefs[4] - 2 * coefs[1] * coefs[3]) / (4 * coefs[0] * coefs[1] - coefs[2] ** 2)
    peak_row = (coefs[2] * coefs[3] - 2 * coefs[0] * coefs[4]) / (4 * coefs[0] * coefs[1] - coefs[2] ** 2)

    # Check if peak of fitted parabolic surface is outside the correlation surface:
    if peak_col > (surface.shape[1] - 1) or peak_row > (surface.shape[0] - 1):
        return np.array([np.nan, np.nan])

    # Check if peak pixel and peak of fitted parabolic surface are reasonably close:
    if (abs(max_col - peak_col) > shift_limit) or (abs(max_row - peak_row) > shift_limit):
        return np.array([np.nan, np.nan])

    # Fit is valid, return the fit:
    return np.array([peak_col, peak_row])


def pixel_level_peak_finder_1d(correlation_lines: np.ndarray) -> np.ndarray:
    """
    This function returns a numpy array containing the location of the maximum surface value
    to pixel level accuracy for each row of the input matrix.

    :return: The location of the maximum surface values to pixel level accuracy.
    """
    # noinspection PyTypeChecker
    out = np.argmax(correlation_lines, axis=-1)[..., np.newaxis]  # type: np.ndarray
    return out


def parabolic_peak_finder_1d(correlation_lines: np.ndarray, fit_size=1):
    r"""
    Find the subpixel maximum location along each row.

    First, this function calls :func:`pixel_level_peak_finder_1d` to identify the location of the peak of each row.
    It then fits a parabola to the values around the peak and solves for the center of the
    parabola to be the peak value.  The parabola equation that is fit is

    .. math::
        y = Ax^2+Bx+C

    where :math:`y` is the value of the correlation line at location :math:`x`, and :math:`A-C` are the
    coefficients to be fit.  The fit is performed in an algebraic least squares sense.
    The location of the peak of the surface is then given by:

    .. math::
        x_p = \frac{-B}{2A}

    where :math:`x_p` is the location of the peak.

    :param correlation_lines: array of correlation lines
    :param fit_size: number of values on each side to include in the parabola fit
    :return: array of subpixel centers for each row
    """
    # Get the pixel level correlation surface
    max_cols = pixel_level_peak_finder_1d(correlation_lines)

    # Determine which values to include in the parabola fit
    deltas = np.arange(-fit_size, fit_size + 1)

    # Get the original shape of the correlation lines
    base_shape = correlation_lines.shape[:-1]

    # Reshape the correlation lines to be only 2d
    correlation_lines = correlation_lines.reshape(np.prod(base_shape), -1)

    # Get the column indices for the fit
    cols = (max_cols.reshape((np.prod(base_shape), 1)) + deltas.reshape(1, -1))

    # Build the jacobian matrix
    jac_matrix = np.rollaxis(np.array([cols * cols, cols, np.ones(cols.shape)]), 0, -1)

    # Build the rhs
    rhs = correlation_lines[np.ogrid[:correlation_lines.shape[0], :0][:1] + [None, cols]]

    # Fit the paraboloid using LLS
    solus = np.linalg.solve(jac_matrix @ jac_matrix.swapaxes(-1, -2), jac_matrix @ rhs)

    # Return the subpixel center
    return (-solus[..., 1, :] / (2 * solus[..., 0, :])).reshape(base_shape + (-1,))


class ImageProcessing:
    """
    This class is a collection of various image processing techniques used throughout GIANT.

    All image processing techniques for the GIANT algorithms are contained in this class.  This includes
    centroiding algorithms for stars and unresolved bodies, algorithms for extracting bright spots from an image
    (particularly useful in the detection of stars and unresolved bodies), denoising algorithms,
    and edge detection algorithms.  The class essentially works as a container for the various options required for
    each technique.  It also makes it easier to pass data between different functions that may be required for
    individual algorithms.

    In general, users will not directly interact with this class, as it is used internally by many other GIANT
    routines.
    """

    def __init__(self, centroiding: PointSpreadFunction = Gaussian,
                 image_denoising: Callable = cv2.GaussianBlur,
                 denoising_args: Optional[Tuple] = ((3, 3), 0), denoising_kwargs: Optional[Dict] = None,
                 denoise_flag: bool = False,
                 pae_threshold: Union[float, int] = 40, pae_order: int = 2, centroid_size: int = 1,
                 correlator: Callable = cv2_correlator_2d, correlator_kwargs: Optional[Dict] = None,
                 poi_min_size: int = 2, poi_max_size: int = 50, poi_threshold: Union[float, int] = 8,
                 reject_saturation: bool = True, subpixel_method: SubpixelEdgeMethods = SubpixelEdgeMethods.PAE,
                 save_psf: bool = False, return_stats: bool = False, zernike_edge_width: float = 0.5,
                 otsu_levels: int = 2, minimum_segment_area: int = 10, minimum_segment_dn: Real = 200,
                 image_flattening_noise_approximation: ImageFlatteningNoiseApprox = ImageFlatteningNoiseApprox.GLOBAL,
                 flattening_kernel_size: int = 7):
        """
        :param centroiding: A callable object which takes 3 positional arguments and estimates the centers of a ROI
        :param image_denoising: A callable object with takes an image as the first positional argument and returns the
                                denoised image
        :param denoising_args: The rest of the positional arguments for the image_denoising callable
        :param denoising_kwargs: the keyword arguments for the image_denoising callable as a dictionary
        :param denoise_flag: A flag to indicate whether to denoise the image before applying the other techniques
        :param pae_threshold: The threshold for identifying pixel level edges in the PAE method
        :param pae_order:  The order of fit for the PAE refinement (must be 1 or 2)
        :param centroid_size:  half of the area passed to the centroiding function for refining the poi positions
        :param correlator: The cross correlation function to use
        :param correlator_kwargs: Key word arguments to pass to the correlator function
        :param poi_min_size: The minimum size for blobs to be considered points of interest
        :param poi_max_size: The maximum size for blobs to be considered points of interest
        :param poi_threshold: The threshold for coarsely identifying potential points of interest
        :param reject_saturation: A flag indicating whether to reject blobs that contain saturated pixels when
                                  performing poi identification.  Note that the saturation dn value must be stored in
                                  a `saturation` attribute for each image object being considered
        :param subpixel_method: An enumeration specifying which method to use for identifying subpixel edges
        :param save_psf: A flag specifying whether to save the fit psf in the centroiding methods
        :param return_stats: A flag specifying whether to return stats about each point of interest in the locate_poi
                             methods
        :param zernike_edge_width: The expected width of the edges for the zernike ramp edge method.
        :param otsu_levels: The number of levels to attempt to split the histogram by for Otsu thresholding.
        :param minimum_segment_dn: The minimum average DN for a segment to be considered foreground instead of
                                   background
        :param minimum_segment_area: The minimum area for a segment to be considered foreground instead of
                                     noise in pixels squared.
        :param image_flattening_noise_approximation: A
        """

        self.centroiding = centroiding  # type: PointSpreadFunction
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

        self.save_psf = save_psf  # type: bool
        """
        A boolean flag specifying whether to save the point spread function fit.
        
        If this parameter is set to ``true`` then resulting PSF object from the :attr:`centroiding` attribute is saved
        in addition to just the centroid.  To ensure that the fit statistics are also saved for each PSF
        ensure the :attr:`~.PointSpreadFunction.save_residuals` class attribute on the PSF object is set to ``True`` as
        well.
        """

        self.image_denoising = image_denoising  # type: Callable
        """
        A callable that is used to decrease the effects of noise in an image.
        
        This should take the form of::
        
            denoised_image = image_denoising(original_image, *denoising_args, *denoising_kwargs)
            
        where ``original_image`` is the original 2D grayscale image as a numpy array, ``denoising_args`` are additional
        positional arguments to the image_denoising callable in a list, denoising_kwargs are a dictionary of key word
        arguments to pass to the image_denoising method, and denoised_image is a grayscale 2D image containing
        the noise suppressed version of the input image.
        
        By default this applies a 2D Gaussian blurring kernel of size 3, 3 to the image to suppress the noise effects.
        """

        if isinstance(subpixel_method, str):
            subpixel_method = subpixel_method.upper()

        self.subpixel_method = SubpixelEdgeMethods(subpixel_method)  # type: SubpixelEdgeMethods
        """
        An enumeration (string) specifying what subpixel edge refinement method to use.
        
        This can specified as an attribute of the :class:`SubpixelEdgeMethods` enumeration directly or as a string
        that corresponds to that enumeration.  
        """

        self.zernike_edge_width = zernike_edge_width  # type: float
        """
        A tuning parameter for the Zernike Ramp method specifying half the total edge width in pixels.
        
        Typically this is set to 1.66*sigma where sigma is the point spread function full width half maximum for the 
        camera.
        """

        self.denoising_args = []  # type: list
        """
        A list of additional arguments to pass to the :attr:`.image_denoising` callable after the image.
        
        This list is expanded using the typical python expansion.
        """

        if denoising_args is not None:
            self.denoising_args = denoising_args

        self.denoising_kwargs = {}  # type: dict
        """
        A dictionary of keyword arguments to pass to the :attr:`.image_denoising` callable after the image.

        This dictionary is expanded using the typical python expansion.
        """
        if denoising_kwargs is not None:
            self.denoising_kwargs = denoising_kwargs

        self.denoise_flag = denoise_flag  # type: bool
        """
        A boolean specifying whether to apply the :attr:`.image_denoising` callable before applying other image 
        processing routines to an image.
        
        Set this attribute to True to apply the denoising routine and False to not apply the denoising routine.
        """

        self.correlator = correlator  # type: Callable
        """
        A callable that is used to perform cross correlation between an image and a template
        
        This should take the image as the first argument, the template as the second argument, and
        correlator_kwargs as the key word arguments.  That is, it should be of the form::

            cor_surf = correlator(image, template, **correlator_kwargs)

        where cor_surf is the correlation surface.  By default this is set to :func:`.cv2_correlator_2d`.
        """

        self.correlator_kwargs = {}  # type: dict
        """
        A dictionary of keyword arguments to pass to the :attr:`.correlator` callable after the image and the template.

        This dictionary is expanded using the typical python expansion.
        """
        if correlator_kwargs is not None:
            self.correlator_kwargs = correlator_kwargs

        self.pae_threshold = pae_threshold  # type: float
        """
        This tuning parameter specifies the minimum absolute image gradient for a location in an image to be considered 
        an edge for the Partial Area Effect Method.
        """

        self.pae_order = pae_order  # type: int
        """
        This specifies whether to fit a linear (1) or quadratic (2) to the limb in the PAE method.  
        
        Typically quadratic produces the best results.
        """

        self.centroid_size = centroid_size  # type: int
        """
        This specifies how many pixels to include when identifying a centroid in a region of interest.
        
        This sets the +/- number from the peak brightness pixel in both axes (so that a value of 1 means
        a 3x3 grid will be considered, a value of 2 will result in a 5x5 grid, etc).  
        """

        self.poi_threshold = poi_threshold  # type: float
        """
        This specifies the sigma multiplier to use when identifying a pixel as a point of interest.
        
        The sigma multiplier is applied to a rough noise estimate of the image (see 
        :meth:`.flatten_image_and_get_noise_level`) and then any pixels above this DN value are labeled as interesting 
        pixels that require further processing (see :meth:`.locate_subpixel_poi_in_roi`).
        """

        self.poi_min_size = poi_min_size  # type: int
        """
        This specifies the minimum number of pixels that must be connected for a blob to be considered a point of 
        interest.

        Individual pixels are clumped using a connected components algorithm, and then the size of each blob is compared
        against this value.  See :meth:`.locate_subpixel_poi_in_roi` for more details.
        """

        self.poi_max_size = poi_max_size  # type: int
        """
        This specifies the maximum number of pixels that must be connected for a blob to be considered a point of 
        interest.

        Individual pixels are clumped using a connected components algorithm, and then the size of each blob is compared
        against this value.  see :meth:`.locate_subpixel_poi_in_roi` for more details.
        """

        self.reject_saturation = reject_saturation  # type: bool
        """
        This boolean flag specifies whether to ignore clumps of pixels that contain saturated DN values when identifying 
        points of interest in an image.
        
        Set to True to reject any clumps containing saturated pixels.
        """

        self.return_stats = return_stats  # type: bool
        """
        This boolean flag specifies whether to return statistics about each blob when identifying points of interest in 
        the image.
        """

        self.otsu_levels = otsu_levels  # type int
        """
        This sets the number of levels to attempt to segment the histogram into for Otsu based multi level thresholding.
        
        See the :func:`.otsu` function for more details.
        
        This is used in method :meth:`segment_image`
        """

        self.minimum_segment_area = minimum_segment_area  # type int
        """
        This sets the minimum area for a segment to be considered not noise.
        
        Segments with areas less than this are discarded as noise spikes
        
        This is used in method :meth:`segment_image`
        """

        self.minimum_segment_dn = float(minimum_segment_dn)  # type float
        """
        The minimum that the average DN for a segment must be for it to not be discarded as the background.

        Segments with average DNs less than this are discarded as the background
        
        This is used in method :meth:`segment_image`
        """

        if isinstance(image_flattening_noise_approximation, str):
            image_flattening_noise_approximation = image_flattening_noise_approximation.upper()

        self.image_flattening_noise_approximation: ImageFlatteningNoiseApprox = ImageFlatteningNoiseApprox(
            image_flattening_noise_approximation
        )
        """
        This specifies whether to globally flatten the image and compute a single noise level or to locally do so.
        
        Generally global is sufficient for star identification purposes.  If you are trying to extract very dim stars 
        (or particles) then you may need to use the ``'LOCAL'`` option, which is much better for low SNR targets but 
        much slower.
        
        This is used in :meth:`find_poi_in_roi` and :meth:`flatten_image_and_get_noise_level`
        """

        self.flattening_kernel_size: int = flattening_kernel_size
        """
        This specifies the half size of the kernel to use when locally flattening an image.  
        
        If you are using global flattening of an image this is ignored.
        
        The size of the kernel/region used in flattening the image will be ``2*flattening_kernel_size+1``.
        
        This is used in :meth:`flatten_image_and_get_noise_level`.
        """


    def __repr__(self) -> str:

        ip_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                ip_dict[key] = value

        return (self.__module__ + "." + self.__class__.__name__ +
                "(" + ', '.join(['{}={!r}'.format(k, v) for k, v in ip_dict.items()]) + ")")

    def __str__(self) -> str:
        ip_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Callable):
                value = value.__module__ + "." + value.__name__

            if not key.startswith("_"):
                ip_dict[key] = value

        return (self.__module__ + "." + self.__class__.__name__ +
                "(" + ', '.join(['{}={!s}'.format(k, v) for k, v in ip_dict.items()]) + ")")

    def segment_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        :return: The label array, stats array about the labels in order, and the centroids of the segments
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
                continue

            boolean_label = labeled == ind
            if np.median(image[boolean_label]) < self.minimum_segment_dn:
                break  # since we are going in reverse size order if we get here we're done

            out_labeled[boolean_label] = stored_ind
            out_stats.append(stat)
            out_centroids.append(centroid)
            stored_ind += 1

        return out_labeled, foreground_image, np.array(out_stats), np.array(out_centroids)

    @staticmethod
    def _global_flat_image_and_noise(image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        This method is used to sample the noise level of an image, as well as return a flattened version of the image.

        The image is flattened by subtracting off a median filtered copy of the image from the image itself

        The standard deviation of the noise level in the image is estimated by either calculating the standard deviation
        of flattened user defined dark pixels for the image (contained in the :attr:`.OpNavImage.dark_pixels` attribute)
        or by calculating the standard deviation of 2,000 randomly sampled differences between pixel pairs of the
        flattened image spaced 5 rows and 5 columns apart.

        This method is used by :meth:`locate_subpixel_poi_in_roi` in order to make the point of interest identification
        easier.

        :param image: The image to be flattened and have the noise level estimated for
        :return: The flattened image and the noise level as a tuple
        """

        # flatten the image by subtracting a median blurred copy of the image.  Using a blurring kernel of 5x5.
        flat_image = (image.astype(np.float32) - cv2.medianBlur(image.copy().astype(np.float32), 5))

        dark_pixels = getattr(image, 'dark_pixels', None)
        if dark_pixels is not None:  # if there are identified dark pixels
            # flatten the dark pixels using a median filter
            flat_dark = dark_pixels.astype(np.float64) - \
                        sig.medfilt(dark_pixels.astype(np.float64))

            # compute the standard deviation of the flattened dark pixels
            standard_deviation = np.nanstd(flat_dark) / 2

        else:  # otherwise, sample the image to determine the randomness
            # determine the randomness of the image by sampling at 10000 randomly selected points compared with point +5
            # rows and +5 cols from those points

            im_shape = flat_image.shape

            dist = np.minimum(np.min(im_shape) - 1, 5)

            if dist <= 0:
                raise ValueError('the input image is too small...')

            # get the total possible number of starting locations
            num_pix = float(np.prod(np.array(im_shape) - dist))  # type: float

            # sample at most 1 quarter of the available starting locations
            num_choice = int(np.minimum(num_pix // 4, 2000))

            # choose a random sample of starting locations
            start_rows, start_cols = np.unravel_index(np.random.choice(np.arange(int(num_pix)), num_choice,
                                                                       replace=False),
                                                      np.array(im_shape) - dist)

            # get the other half of the sample
            next_rows = start_rows + dist
            next_cols = start_cols + dist

            # compute the standard deviation of the difference between the star points and hte next points.  This
            # measures the noise in the image and sets the threshold for identifiable stars.
            data = (flat_image[next_rows, next_cols] - flat_image[start_rows, start_cols]).ravel()

            # reject outliers from the data using MAD
            outliers = get_outliers(data)

            # compute the standard deviation
            standard_deviation = np.nanstd(data[~outliers]) / 2

        return flat_image, standard_deviation

    # TODO: This would probably be better as a cython function where we can do parallel processing
    def _local_flat_image_and_noise(self, image) -> Tuple[np.ndarray, List[float], List[Tuple[slice, slice]]]:
        """
        This method flattens the image and approximates the noise over regions of the image.

        This is not intended by the user, instead use :meth:`flatten_image_and_get_noise_level`.

        :param image: The image which is to be flattened and have noise levels estimated for
        :return: The flattened image, a list of noise values for regions of the image, and a list of tuples of slices
                 describing the regions of the image
        """

        # get the shape of the image
        img_shape = image.shape

        # make sure that the image is double, also copy it to ensure that we don't mess up the original
        flat_image = image.astype(np.float32).copy()

        # start the region center at the kernel size
        current_row = self.flattening_kernel_size
        current_col = self.flattening_kernel_size

        # initialize the lists for return
        noises, slices = [], []

        # loop rows through until we've processed the whole image
        while current_row < img_shape[0]:
            # get the row bounds and slice
            lower_row = current_row - self.flattening_kernel_size
            upper_row = min(current_row + self.flattening_kernel_size + 1, img_shape[0])
            row_slice = slice(lower_row, upper_row)

            # loop through columns until we've processed the whole image
            while current_col < img_shape[1]:
                # get the column bounds and slice
                lower_column = current_col - self.flattening_kernel_size
                upper_column = min(current_col + self.flattening_kernel_size + 1, img_shape[1])
                column_slice = slice(lower_column, upper_column)

                # get the row/column labels that we are working with
                rows, cols = np.mgrid[row_slice, column_slice]

                # get the region from the original image we are editing
                region = image[row_slice, column_slice].astype(np.float32)

                # compute the background of the region using least squares [1, x, y] @ [A, B, C] = bg
                h_matrix = np.vstack([np.ones(rows.size), cols.ravel(), rows.ravel()]).T.astype(np.float32)
                background = np.linalg.lstsq(h_matrix, region.ravel(), rcond=None)[0].ravel()

                # flatten the region by subtracting the linear background approximation
                flat_image[row_slice, column_slice] -= (h_matrix@background.reshape(3, 1)).reshape(region.shape)

                # store the slices
                slices.append((row_slice, column_slice))

                # update the current column we're centered on
                current_col += 2 * self.flattening_kernel_size + 1

            # update the current row/column we're centered on
            current_row += 2 * self.flattening_kernel_size + 1
            current_col = self.flattening_kernel_size

        # make sure we're extra flat by flattening the flat image with a median blur.
        flat_image: np.ndarray = (flat_image - cv2.medianBlur(flat_image.copy(), 5))

        for local_slice in slices:
            region = flat_image[local_slice[0], local_slice[1]].ravel()
            selections = np.random.choice(np.arange(int(region.size)), int(region.size//2), replace=False)

            selected_region: np.ndarray = region[selections]

            outliers = get_outliers(selected_region)

            if outliers.sum() > selections.size//2:

                local_std: float = selected_region.std()
            else:
                local_std: float = selected_region[~outliers].std()

            noises.append(local_std)

        return flat_image, noises, slices

    def flatten_image_and_get_noise_level(self, image: np.ndarray) -> Union[Tuple[np.ndarray, float],
                                                                            Tuple[np.ndarray, List[float],
                                                                                  List[Tuple[slice, slice]]]]:
        """
        This method is used to sample the noise level of an image, as well as return a flattened version of the image.

        There are 2 techniques for flattening the image.

        In the first, ``GLOBAL`` technique: the image is flattened by subtracting off a median filtered copy of the
        image from the image itself

        The standard deviation of the noise level in the image is then estimated by either calculating the standard
        deviation of flattened user defined dark pixels for the image (contained in the :attr:`.OpNavImage.dark_pixels`
        attribute) or by calculating the standard deviation of 2,000 randomly sampled differences between pixel pairs of
        the flattened image spaced 5 rows and 5 columns apart.

        In the second, ``LOCAL`` technique: the image is split into regions based on :attr:`flattening_kernel_size`.
        For each region, a linear background gradient is estimated and subtracted from the region.  The global flattened
        image is then flattened further by subtracting off a median filtered copy of the flattened image.

        The standard deviation of the noise level is then computed for each region by sampling about half of the points
        in the flattened region and computing the standard deviation of the flattened intensity values.  In this case
        3 values are returned, the flattened image, the list of noise values for each region, and a list of slices
        defining the regions that were processed.

        This method is used by :meth:`locate_subpixel_poi_in_roi` in order to make the point of interest identification
        easier.

        :param image: The image to be flattened and have the noise level estimated for
        :return: The flattened image and the noise level as a tuple, or the flattened image, the noise levels as a list,
                 and a list of slices of tuples specifying the regions of the image the noise levels apply to.
        """

        if self.image_flattening_noise_approximation == ImageFlatteningNoiseApprox.GLOBAL:
            return self._global_flat_image_and_noise(image)
        else:
            return self._local_flat_image_and_noise(image)

    @staticmethod
    def corners_to_roi(row_corners: Iterable, column_corners: Iterable) -> Tuple[np.ndarray, np.ndarray]:
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
        return tuple(np.meshgrid(np.arange(min_row, max_row + 1), np.arange(min_col, max_col + 1), indexing='ij'))

    # noinspection SpellCheckingInspection
    def find_poi_in_roi(self, image: np.ndarray,
                        region: Optional[Tuple[np.ndarray, np.ndarray]] = None) \
            -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]]:
        """
        This method identifies pixel level centers for all points of interest inside of some region of interest.

        A point of interest is defined as any grouping of *n* pixels that are above :attr:`.poi_threshold` *
        **standard_deviation** where :attr:`poi_min_size` <= *n* <= :attr:`.poi_max_size`.  The **standard_deviation**
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

        If the :attr:`.return_stats` attribute is set to True, then this method will also return the connected
        components stats (see
        `OpenCV connectedComponentsWithStats <https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html
        #ga107a78bf7cd25dec05fb4dfc5c9e765f>`_
        for details) and the peak signal to noise ratio for each detection.

        :param image: The image being considered
        :param region: The region of the image to consider
        :return: the pixel level locations of the points of interest in the region of interest (row, col).  Optionally
                 returns the connected component stats and the peak signal to noise ratio for each detection if
                 :attr:`.return_stats` is set to True.
        """


        # select the sub image we are considering
        if region is not None:
            roi_start = [np.min(region[1]), np.min(region[0])]

            big_roi = image[tuple(region)]

        else:
            roi_start = [0, 0]

            big_roi = image

        # get the flat image and approximate noise level(s) in the image
        if self.image_flattening_noise_approximation == ImageFlatteningNoiseApprox.GLOBAL:
            roi, standard_deviation = self.flatten_image_and_get_noise_level(big_roi)

            # detect pixels of interest by thresholding the flattened image at some multiple of the noise level
            snr = roi / standard_deviation
        else:
            # if we're doing local flattening and noise estimation
            roi, noise_estimates, slices = self.flatten_image_and_get_noise_level(big_roi)

            # detect pixels of interest by thresholding the flattened image at some multiple of the noise level
            snr = np.zeros(big_roi.shape, dtype=np.float64)

            # for each region, compute the "snr" for each detection
            for noise, slices in zip(noise_estimates, slices):
                flat_sliced = roi[slices[0], slices[1]]
                snr[slices[0], slices[1]] = flat_sliced / noise

        interesting_pix = snr > self.poi_threshold

        # use connected components to blob the pixels together into single objects
        _, __, stats, ___ = cv2.connectedComponentsWithStats(interesting_pix.astype(np.uint8))

        poi_subs = []
        out_stats = []
        out_snrs = []

        # loop through each grouping of pixels
        for blob in stats:

            if self.poi_max_size >= blob[-1] >= self.poi_min_size:

                # if we want to reject blobs that are affected by saturation
                if self.reject_saturation and hasattr(image, 'saturation'):

                    # ignore blobs where a portion of the blob is saturated
                    if (big_roi[blob[1]:blob[1] + blob[3], blob[0]:blob[0] + blob[2]] >= image.saturation).any():
                        continue

                # get the subscript to the maximum illumination value within the current component and append it to the
                # return list
                poi_roi = roi[blob[1]:blob[1] + blob[3],
                              blob[0]:blob[0] + blob[2]]

                # get the x/y location by unraveling the index (and reversing the order
                local_subs = np.unravel_index(np.nanargmax(poi_roi), poi_roi.shape)[::-1]  # type: np.ndarray
                # store the results translated back to the full image and the statistics
                poi_subs.append(local_subs + blob[[0, 1]] + roi_start)
                out_stats.append(blob)
                out_snrs.append(snr[blob[1]:blob[1] + blob[3],
                                    blob[0]:blob[0] + blob[2]].max())

        if self.return_stats:
            return poi_subs, out_stats, out_snrs
        else:
            return poi_subs

    def refine_locations(self, image: np.ndarray, image_subs: Iterable[np.ndarray],
                         stats: Optional[List[np.ndarray]] = None,
                         snrs: Optional[List[np.ndarray]] = None) \
            -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray],
                     Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]],
                     Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]]:
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
        :param image_subs: The pixel level locations of interest to be refined
        :param stats: An optional input of stats about the blobs.  This is not used in this function but is passed
                      through, removing any blobs where a centroid was not found.
        :param snrs: An optional input of signal to noise ratios from the blobs.  This is not used in this function but
                     is passed through, removing any blobs where a centroid was not found.
        :return: The subpixel centers of the points of interest as well as the illumination values (col, row)
        """

        # initialize lists for output
        star_points = []
        star_illums = []
        star_psfs = []
        out_stats = []
        out_snrs = []

        # loop through the pixel level points of interest
        for ind, center in enumerate(image_subs):

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

                # perform the fit
                # if self.save_psf:
                #     [x0, y0], psf, rss = self.centroiding(cols,
                #                                           rows,
                #                                           sampled_image,
                #                                           save_psf=self.save_psf)
                # else:
                #     x0, y0 = self.centroiding(cols,
                #                               rows,
                #                               sampled_image,
                #                               save_psf=self.save_psf)

                psf = self.centroiding.fit(cols, rows, sampled_image)

                x0, y0 = psf.centroid

                # if we're outside the image or the fit failed skip this one
                if (x0 < 0) or (y0 < 0) or (np.isnan((x0, y0)).any()):
                    continue

                # check to be sure we haven't deviated too far from the original peak of interest (avoid poorly
                # conditioned systems)
                if (np.abs(np.asarray(center) - np.asarray([x0, y0]).flatten()) <= 3).all():
                    star_points.append([x0, y0])
                    star_illums.append(image[tuple(center[::-1])])
                    star_psfs.append(psf)
                    if stats is not None:
                        out_stats.append(stats[ind])
                        out_snrs.append(snrs[ind])

        # determine which form the output should take
        if self.save_psf:
            if stats is not None:
                return np.asarray(star_points).T, np.asarray(star_illums), np.array(star_psfs), out_stats, out_snrs
            else:
                return np.asarray(star_points).T, np.asarray(star_illums), np.array(star_psfs)
        else:
            if stats is not None:
                return np.asarray(star_points).T, np.asarray(star_illums), out_stats, out_snrs
            else:
                return np.asarray(star_points).T, np.asarray(star_illums)

    def locate_subpixel_poi_in_roi(self, image: np.ndarray,
                                   region: Optional[Tuple[np.ndarray, np.ndarray]] = None) \
            -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray],
                     Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]],
                     Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]]:

        """
        This method identifies the subpixel locations of points of interest in an image.

        This method is simply a convenient way of combining :meth:`find_poi_in_roi` and :meth:`refine_locations` and
        calls these two methods directly, feeding the results of the first into the second.

        Note that if the :attr:`.denoise_flag` is set to true then this method will first pass the image through the
        :meth:`.denoise_image` method.

        :param image: The image to be processed
        :param region: The region of interest to consider as 2 numpy arrays of indices into the images or None
        :return: The subpixel centers of the points of interest as well as the illumination values, plus
                 optionally details about the point spread function fit if
                 :attr:`.save_psf` is set to True and the blob statistics and SNR values for each blob if
                 :attr:`.return_stats` is set to true
        """

        # denoise the image if requested
        if self.denoise_flag:
            image = self.denoise_image(image)
            flip_denoise_flag = True
            self.denoise_flag = False
        else:
            flip_denoise_flag = False

        # first get the rough locations of points of interest
        if self.return_stats:
            image_inds, stats, snrs = self.find_poi_in_roi(image, region=region)

            # refine the rough locations and return the results
            res = self.refine_locations(image, image_inds, stats, snrs)
            if flip_denoise_flag:
                self.denoise_flag = True
            return res
        else:
            image_inds = self.find_poi_in_roi(image, region=region)

            # refine the rough locations and return the results
            res = self.refine_locations(image, image_inds)
            if flip_denoise_flag:
                self.denoise_flag = True
            return res

    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        This method is used to optionally denoise the image before a number of the other techniques contained in this
        class.

        The method applies the denoising technique specified in the :attr:`.image_denoising` attribute.  The returned
        2D numpy array is the image after applying the denoising technique.

        :param image: The image to be denoised
        :return: The denoised image
        """

        return self.image_denoising(image, *self.denoising_args, **self.denoising_kwargs)

    def correlate(self, image: np.ndarray, template: np.ndarray) -> np.ndarray:
        """
        This method generates a cross correlation surface between template and image.

        The method applies the correlation function specified in the :attr:`.correlator` attribute.  The returned
        2D array in general will be the same size as the image (though this is controlled by the
        :attr:`.correlator` attribute) where each element will represent the correlation score between the template and
        the image when the center of the template is aligned with the corresponding element in the image.  Therefore,
        to get the location of a template in an image one would do

            >>> from giant.image_processing import ImageProcessing
            >>> import numpy
            >>> ip = ImageProcessing()
            >>> local_image = numpy.random.randn(200, 200)
            >>> local_template = local_image[30:60, 45:60]
            >>> surf = ip.correlate(local_image, local_template)
            >>> temp_middle = numpy.floor(numpy.array(local_template.shape)/2)
            >>> template_point = numpy.array([0, 0])  # upper left corner
            >>> template_point - temp_middle + numpy.unravel_index(surf.argmax(), surf.shape)
            array([30., 45.])

        :param image: The image to be matched against
        :param template: The template to find in the image
        :return: The normalized correlation surface
        """

        return self.correlator(image, template, **self.correlator_kwargs)

    # TODO: unit tests for all of the following
    def identify_subpixel_limbs(self, image: np.ndarray, illum_dir: ARRAY_LIKE, num_objs: int = 1) -> List[np.ndarray]:
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

        # denoise the image if requested
        if self.denoise_flag:
            image = self.denoise_image(image)
            self.denoise_flag = False
            flip_denoise_flag = True
        else:
            flip_denoise_flag = False

        # convert the image to uint8 if it isn't already
        if image.dtype != np.uint8:
            # noinspection PyArgumentList
            image = image.astype(np.float64) - image.min()
            image *= 255 / image.max()
            image = image.astype(np.uint8)

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

        # create a binary image where only the non-outlier pixels are turned on
        connected_mat = (labels == np.arange(4)[~outliers].reshape(-1, 1, 1)).any(axis=0)

        # do connected components
        _, labs2, stats, centroids = cv2.connectedComponentsWithStats(connected_mat.astype(np.uint8))

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
            centroid = centroids[blob] - top_left[::-1]

            # check to be sure we have an actual object
            if sub_image.size == 0:
                continue

            # identify the subpixel limbs and store them
            limbs.append(self._locate_limbs(sub_image, centroid, illum_dir) + top_left[::-1].reshape(2, 1))

        if flip_denoise_flag:
            self.denoise_flag = True
        return limbs

    def identify_pixel_edges(self, image: np.ndarray, split_horizontal_vertical: bool = False,
                             return_gradient: bool = False) -> Union[np.ndarray,
                                                                     Tuple[np.ndarray, np.ndarray],
                                                                     Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                           np.ndarray],
                                                                     Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                           np.ndarray, np.ndarray]]:
        """
        This method determines pixel level edges in an image by thresholding the image gradients.

        The image gradients are computed by convolving horizontal and vertical Sobel masks with the image to give the
        horizontal and vertical gradients.  The gradient images are then thresholded using :func:`otsu` to determine
        the strongest gradients in the image.  The strong gradients are then searched for local maxima, which become the
        pixel level edges of the image.

        This function inputs the image and outputs a binary image with true values corresponding to the edge locations
        in the image.  Optionally, if the ``split_horizontal_vertical`` argument is set to True, the 2 binary images are
        returned, the first with true values in locations containing horizontal edges, and the second with true values
        in locations containing vertical edges.  Finally, if the `return_gradient` argument is set to true, then the
        horizontal, vertical, and magnitude gradient arrays are returned as well.

        :param image: The image to extract the edges from
        :param split_horizontal_vertical: A flag specifying whether to return the vertical and horizontal edges
                                          separately or combined
        :param return_gradient: A flag specifying whether to return the gradient arrays or not
        :return: the pixel level edges (either as a single boolean array or a split boolean array) and optionally the
                 horizontal, vertical, and magnitude gradient arrays.
        """

        # blur the image
        if self.denoise_flag:
            image = self.denoise_image(image)
            self.denoise_flag = False
            flip_denoise_flag = True
        else:
            flip_denoise_flag = False

        # compute the image gradients
        horizontal_gradient = sig.fftconvolve(image, HORIZONTAL_KERNEL, 'same')  # gradient from left to right
        vertical_gradient = sig.fftconvolve(image, VERTICAL_KERNEL, 'same')  # gradient from top to bottom
        normalized_gradient = np.sqrt(horizontal_gradient ** 2 + vertical_gradient ** 2)

        # get the absolute of the gradients
        abs_horizontal_gradient = np.abs(horizontal_gradient)
        abs_vertical_gradient = np.abs(vertical_gradient)

        # fix the edges since they can be wonky
        normalized_gradient[:, 0] = 0
        normalized_gradient[:, -1] = 0
        normalized_gradient[0, :] = 0
        normalized_gradient[-1, :] = 0

        # threshold the edges using Otsu's method
        _, normalized_gradient_binned = otsu(normalized_gradient, 4)

        # get the number of pixels in each threshold level
        num_pix, _ = np.histogram(normalized_gradient_binned, np.arange(5))
        
        # check for outliers
        outliers = get_outliers(num_pix, sigma_cutoff=3)

        if outliers[0]:
            binned_gradient = normalized_gradient_binned > 1.5
        else:
            _, binned_gradient = otsu(normalized_gradient, 2)

        # do connected components to throw out individual points
        number_of_labels, labs, stats, _ = cv2.connectedComponentsWithStats(binned_gradient.astype(np.uint8))

        for blob in range(number_of_labels):
            if stats[blob, cv2.CC_STAT_AREA] < 2:
                labs[labs == blob] = 0

        binned_gradient = labs > 0

        # determine the horizontal edges
        horiz_mask = np.zeros(image.shape, dtype=bool)

        # horizontal edges correspond to high vertical gradients
        horiz_mask[5:-5, 2:-2] = (binned_gradient[5:-5, 2:-2] &  # check to see that the overall gradient is large
                                  # check that this is a horizontal edge by checking that the vertical_gradient is
                                  # larger
                                  (abs_vertical_gradient[5:-5, 2:-2] >= abs_horizontal_gradient[5:-5, 2:-2]) &
                                  # check that this is a local maxima horizontally
                                  (abs_vertical_gradient[5:-5, 2:-2] >= abs_vertical_gradient[4:-6, 2:-2]) &
                                  (abs_vertical_gradient[5:-5, 2:-2] > abs_vertical_gradient[6:-4, 2:-2]))

        # determine the vertical edges
        vert_mask = np.zeros(image.shape, dtype=bool)

        # vertical edges correspond to high horizontal gradients
        vert_mask[2:-2, 5:-5] = (binned_gradient[2:-2, 5:-5] &  # check to see that the overall gradient is large
                                 # check that this is a vertical edge by checking that the horizontal_gradient is larger
                                 (abs_horizontal_gradient[2:-2, 5:-5] >= abs_vertical_gradient[2:-2, 5:-5]) &
                                 # check that this is a local maxima vertically
                                 (abs_horizontal_gradient[2:-2, 5:-5] >= abs_horizontal_gradient[2:-2, 4:-6]) &
                                 (abs_horizontal_gradient[2:-2, 5:-5] > abs_horizontal_gradient[2:-2, 6:-4]))

        # perpendicular edges correspond to high rss gradients
        perpendicular_mask = np.zeros(image.shape, dtype=bool)

        perpendicular_mask[5:-5, 5:-5] = (
                binned_gradient[5:-5, 5:-5] &  # check to see if the overall gradient is large
                (normalized_gradient[5:-5, 5:-5] >= normalized_gradient[5:-5, 4:-6]) &  # horizontal local maxima
                (normalized_gradient[5:-5, 5:-5] > normalized_gradient[5:-5, 6:-4]) &
                (normalized_gradient[5:-5, 5:-5] >= normalized_gradient[4:-6, 5:-5]) &  # vertical local maxima
                (normalized_gradient[5:-5, 5:-5] > normalized_gradient[6:-4, 5:-5]))

        vert_mask |= perpendicular_mask

        # determine what to return

        if flip_denoise_flag:
            self.denoise_flag = True
        if split_horizontal_vertical:
            if return_gradient:
                return horiz_mask, vert_mask, horizontal_gradient, vertical_gradient, normalized_gradient
            else:
                return horiz_mask, vert_mask
        else:
            if return_gradient:
                return horiz_mask | vert_mask, horizontal_gradient, vertical_gradient, normalized_gradient
            else:
                return horiz_mask | vert_mask

    @staticmethod
    def _split_pos_neg_edges(horizontal_gradient: np.ndarray, vertical_gradient: np.ndarray,
                             edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method splits diagonal edges into positive/negative bins

        :param horizontal_gradient: The horizontal gradient array
        :param vertical_gradient: The vertical gradient array
        :param edges: The edge array containing the pixel location of the edges as [x, y]
        :return: The edges split into positive and negative groupings
        """

        # check with edges are positive edges
        positive_check = horizontal_gradient[edges[1], edges[0]] * vertical_gradient[edges[1], edges[0]] > 0

        # split and return the binned edges
        return edges[:, positive_check], edges[:, ~positive_check]

    def _compute_pae_delta(self, sum_a: np.ndarray, sum_b: np.ndarray, sum_c: np.ndarray,
                           int_a: np.ndarray, int_b: np.ndarray) -> np.ndarray:
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

        a_coef = (self.pae_order - 1) * (sum_a + sum_c - 2 * sum_b) / (2 * (int_b - int_a))
        c_coef = ((2 * sum_b - 7 * (int_b + int_a)) /
                  (2 * (int_b - int_a)) -
                  a_coef * (1 + 24 * PAE_A01 + 48 * PAE_A11) / 12)

        c_coef[np.abs(c_coef) > 1] = 0

        return c_coef

    def pae_edges(self, image: np.ndarray) -> np.ndarray:
        """
        This method locates edges in an image with subpixel accuracy.

        Edges are defined as places in the image where the illumination values abruptly transition from light to dark
        or dark to light.  The algorithms in this method are based off of the Partial Area Effect as discussed in
        http://www.sciencedirect.com/science/article/pii/S0262885612001850

        First edges are detected at the pixel level by using a gradient based edge detection method.  The edges are then
        refined to subpixel accuracy using the PAE.  Tests have shown that the PAE achieves accuracy better than 0.1
        pixels in most cases.

        There are two tuning parameters for the PAE method.  One is the :attr:`.pae_threshold`.  This is the threshold
        for detecting pixel level edges (ie the absolute value of the gradient of the image must be above this threshold
        for an edge to be identified).  The second tuning
        parameter is the :attr:`.pae_order`.  The :attr:`.pae_order` specifies whether a linear or quadratic fit is used
        to refine the edge location.  It should have a value of 1 or 2.

        Note that this method returns all edges in an image.  If you are looking for just limbs, check out the
        :meth:`identify_subpixel_limbs` method instead

        :param image: The image to be processed
        :return: a 2xn numpy array of the subpixel edge locations (cols [x], rows [y])
        """

        # optionally denoise the image before estimating the subpixel centers (recommended)
        if self.denoise_flag:
            image_smoothed = self.denoise_image(image)
            self.denoise_flag = False
            flip_denoise_flag = True
        else:
            flip_denoise_flag = False
            image_smoothed = image

        # everything is implemented in refine_edges_pae so just do that...
        res = self.refine_edges_pae(image_smoothed)
        if flip_denoise_flag:
            self.denoise_flag = True
        return res

    def refine_edges_pae(self, image: np.ndarray,
                         pixel_edges: Optional[np.ndarray] = None,
                         horizontal_mask: Optional[np.ndarray] = None,
                         vertical_mask: Optional[np.ndarray] = None,
                         horizontal_gradient: Optional[np.ndarray] = None,
                         vertical_gradient: Optional[np.ndarray] = None) -> np.ndarray:
        """
        This method refines pixel level edges to subpixel level using the PAE method.

        The PAE method is explained at https://www.sciencedirect.com/science/article/pii/S0262885612001850 and is not
        discussed in detail here.  In brief, a linear or parabolic function is fit to the edge data based off of the
        intensity data in the pixels surrounding the edge locations.

        To use this function, you can either input just an image, in which case the pixel level edges will be
        detected using the :meth:`identify_pixel_edges` method, or you can also specify the pixel level edges, the
        mask specifying which edges are horizontal, the mask specifying which edges are vertical, and the horizontal and
        vertical gradient arrays for the image.  The edges are refined and returned as a 2D array with the x
        locations in the first row and the y locations in the second row.

        :param image:  The image the edges are being extracted from
        :param pixel_edges: The pixel level edges from the image as a 2D array with x in the first row and y in the
                            second row
        :param horizontal_mask: A binary mask which selects the horizontal edges from the `pixel_edges` parameter
        :param vertical_mask: A binary mask which selects the vertical edges from the `pixel_edges` parameter
        :param horizontal_gradient: The horizontal image gradient
        :param vertical_gradient: The vertical image gradient
        :return: The subpixel edge locations as a 2d array with the x values in the first row and the y values in the
                 second row (col [x], row [y])
        """

        # if the pixel level edges have not been supplied then calculate them
        if pixel_edges is None:

            (horizontal_mask, vertical_mask,
             horizontal_gradient, vertical_gradient, _) = self.identify_pixel_edges(image,
                                                                                    split_horizontal_vertical=True,
                                                                                    return_gradient=True)

            horizontal_edges = np.vstack(np.where(horizontal_mask)[::-1])
            vertical_edges = np.vstack(np.where(vertical_mask)[::-1])

        else:
            horizontal_edges = pixel_edges[:, horizontal_mask[pixel_edges[1], pixel_edges[0]]]
            vertical_edges = pixel_edges[:, vertical_mask[pixel_edges[1], pixel_edges[0]]]

        if self.denoise_flag:
            image = self.denoise_image(image)
            self.denoise_flag = False
            flip_denoise_flag = True
        else:
            flip_denoise_flag = False

        image = image.astype(np.float64)

        # group the pixel level edges into edges with positive and negative slopes
        horiz_pos_edges, horiz_neg_edges = self._split_pos_neg_edges(horizontal_gradient, vertical_gradient,
                                                                     horizontal_edges)
        vert_pos_edges, vert_neg_edges = self._split_pos_neg_edges(horizontal_gradient, vertical_gradient,
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
        if flip_denoise_flag:
            self.denoise_flag = True
        return np.hstack([sp_horiz_edges_pos, sp_horiz_edges_neg, sp_vert_edges_pos, sp_vert_edges_neg])

    def _locate_limbs(self, region: np.ndarray, centroid: np.ndarray, illum_dir: np.ndarray) -> np.ndarray:
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

        # get the pixel level edges
        (horiz_edges, vert_edges,
         horizontal_gradient, vertical_gradient,
         normalized_gradient) = self.identify_pixel_edges(region, split_horizontal_vertical=True, return_gradient=True)

        # determine the limb edges
        limbs = self._pixel_limbs(horiz_edges | vert_edges, centroid, illum_dir)

        if self.subpixel_method.name == 'PAE':
            limbs = self.refine_edges_pae(region, pixel_edges=limbs,
                                          horizontal_mask=horiz_edges, vertical_mask=vert_edges,
                                          horizontal_gradient=horizontal_gradient, vertical_gradient=vertical_gradient)
        elif self.subpixel_method.name == "ZERNIKE_RAMP":
            limbs = self.refine_edges_zernike_ramp(region, pixel_edges=limbs)

        else:
            # do nothing and just return the pixel limbs
            limbs = np.array(limbs)

        return limbs

    def refine_edges_zernike_ramp(self, image: np.ndarray, pixel_edges: Optional[np.ndarray] = None) -> np.ndarray:
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
        :param pixel_edges: the pixel level edge points to be refined. If none then they will be computed for the whole
                            image
        :return: A 2xn array of subpixel edge points (col [x], row[y])
        """

        if pixel_edges is None:

            edge_mask = self.identify_pixel_edges(image, split_horizontal_vertical=False, return_gradient=False)

            pixel_edges = np.vstack(np.where(edge_mask)[::-1])

        starts = np.maximum(pixel_edges-2, 0)
        stops = np.minimum(pixel_edges+3, [[image.shape[1]], [image.shape[0]]])

        subpixel_edges = []

        edge_width_squared = self.zernike_edge_width ** 2
        # loop through each edge
        for edge, start, stop in zip(pixel_edges.T, starts.T, stops.T):

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
            if self.zernike_edge_width > 0.01:
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

    @staticmethod
    def _pixel_limbs(edge_mask: np.ndarray, centroid: np.ndarray, illum_dir: np.ndarray, step: int = 1) -> np.ndarray:
        """
        This method identifies pixel level limb points from a binary image of edge points.

        A limb is defined as the first edge point encountered by a scan vector in the direction of the illumination
        direction.  The limb points are extracted by (1) selecting starting locations for the scan vectors along a line
        perpendicular to the illumination direction spaced `step` pixels apart and then (2) scanning from these starting
        points in the illumination direction to identify the first edge point that is along the line.

        This method inputs a binary image with true values in the pixels which contain edges, the centroid of the object
        being considered in the binary image, the illumination direction, and the step size. It outputs the pixel level
        edges as a 2D array with the x values in the first row and the y values in the second row.

        :param edge_mask: A binary image with true values in the pixels containing edges.
        :param centroid: The centroid of the object being considered
        :param illum_dir: the illumination direction in the `edge_mask` image
        :param step: The step size to sample for limb points at
        :return: The pixel level limb locations as a 2D array with the x values in the first row and the y values in the
                 second row
        """

        # identify the pixel level edges
        edge_y, edge_x = np.where(edge_mask)
        edge_points = np.vstack([edge_x, edge_y]).astype(np.float64)
        if edge_points.shape[-1] > 100000:
            return np.array([])
        
        # determine how far we need to travel from the centroid to start our scan lines
        line_length = np.sqrt(np.sum(np.power(edge_mask.shape, 2)))

        # determine the maximum distance an edge can be from a scan line for it to belong to that scan line
        max_distance = np.minimum(10, np.ceil(np.prod(edge_mask.shape)/edge_y.size/2))
        # max_distance = 1.1 * np.sqrt(2) * step / 2

        # determine the direction to offset our scan stars
        perpendicular_direction = illum_dir[::-1].copy()
        perpendicular_direction[0] *= -1

        # get the middle of the start positions of our scan lines
        # middle start position of scan
        scan_start_middle = centroid - line_length * illum_dir

        # choose scan starting locations
        scan_starts = scan_start_middle.reshape(2, 1) + \
            np.arange(-line_length, line_length + 1, step).reshape(1, -1) * perpendicular_direction.reshape(2, -1)

        # compute the vector from the scan starts to the potential limb points
        scan_start2edge_points = edge_points - scan_starts.T.reshape(-1, 2, 1)

        # compute the distance from the edge points to the scan lines by taking the projection of the edge points
        # onto the scan line
        edge_distances = np.abs(perpendicular_direction.reshape(1, 1, 2) @ scan_start2edge_points).squeeze()

        # compute the distance from the scan start to each potential limb point
        scan_start2edge_points_dist = np.linalg.norm(scan_start2edge_points, axis=1)

        # locate which points are within the maximum distance from the scan line
        limb_points_check = edge_distances < max_distance

        # choose the closest edge point from the scan starts
        limbs = []
        for scan_line in range(scan_starts.shape[-1]):
            lpc = limb_points_check[scan_line]
            if lpc.any():
                potential_limbs = np.where(limb_points_check[scan_line])[0]
                real_limb = potential_limbs[np.argmin(scan_start2edge_points_dist[scan_line, lpc])]

                edges = edge_points[:, real_limb].astype(int)
                limbs.append(edges)

        if limbs:
            limbs = np.vstack(limbs).T

        return limbs
