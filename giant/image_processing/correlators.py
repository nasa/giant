from typing import cast, Callable

import numpy as np

from numpy.typing import NDArray

import cv2

from scipy.fftpack import next_fast_len
import scipy.signal as sig

from giant._typing import DOUBLE_ARRAY


CORRLATOR_SIGNATURE = Callable[[NDArray, NDArray], DOUBLE_ARRAY]
"""
The call signature for a correlator function.

The inputs are image, template as NDArrays, and then should return a float64 double precision array with the correlation surface.
The correlation surface should be returnd as the same size as the provided image such that the peak of the surface corresponds 
to aligning the middle of the temlate with that pixel.
"""


def cv2_correlator_2d(image: NDArray, template: NDArray, flag: int = cv2.TM_CCOEFF_NORMED) -> DOUBLE_ARRAY:
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
    valid_dtypes = (np.uint8, np.int8, np.float32)
    if not any(np.issubdtype(image.dtype, vdtype) for vdtype in valid_dtypes):
        image = image.astype(np.float32)
    if not any(np.issubdtype(template.dtype, vdtype) for vdtype in valid_dtypes):
        template = template.astype(np.float32)
    

    # calculate what the size of the correlation surface should be and pad the image with 0s
    size_diff = np.array(template.shape) / 2
    upper = np.ceil(size_diff).astype(int)
    lower = np.floor(size_diff).astype(int)

    original_shape = image.shape
    image = np.pad(image, [(lower[0], upper[0]), (lower[1], upper[1])], 'constant')

    # perform the correlation
    cor_surf = cast(DOUBLE_ARRAY, cv2.matchTemplate(image, template, flag))

    # return the correlation surface of the appropriate size
    return cor_surf[:original_shape[0], :original_shape[1]]


def _normalize_xcorr_2d(image: NDArray, zero_mean_temp: NDArray, corr_surf: DOUBLE_ARRAY) -> DOUBLE_ARRAY:
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
    def local_sum(in_mat: NDArray, shape: tuple[int, ...]):
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
    local_sum_squares = local_sum(image * image, zero_mean_temp.shape)  # this is the sum of the squares of the image within
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


def scipy_correlator_2d(image: NDArray, template: NDArray) -> DOUBLE_ARRAY:
    """
    This function performs normalized cross correlation between a template and an image using scipy.signal.convolve2d

    The correlation is performed over the full image, aligning the center of the template with every pixel in the image.
    (Note that this means that if the center of the template should be outside of the image this function will not
    work.)
    
    convolve2d will automatically switch between using frequency domain convolution and spatial domain convolution depending
    on the size of the template.

    Each pixel of the correlation surface returned by this function represents the correlation value when the center of
    the template is placed at this location.  Thus, the location of any point in the template can be found by

    >>> import numpy as numpy
    >>> from giant.image_processing import fft_correlator_2d
    >>> example_image = numpy.random.randn(200, 200)
    >>> example_template = example_image[30:60, 45:60]
    >>> surf = scipy_correlator_2d(example_image, example_template)
    >>> temp_middle = numpy.floor(numpy.array(example_template.shape)/2)
    >>> template_point = numpy.array([0, 0])  # upper left corner
    >>> template_point - temp_middle + numpy.unravel_index(surf.argmax(), surf.shape)
    array([30., 45.])

    :param image: The image that the template is to be matched against
    :param template: the template that is to be matched against the image
    :return: A surface of the correlation coefficients for each overlap between the template and the image.
    """

    # perform the convolution.  Note that template needs to be fliplr/flipud due to the
    # definition of correlation

    # use the zero mean template to simplify some steps later
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


def _normalize_xcorr_1d(extracted: NDArray, zero_mean_predicted: DOUBLE_ARRAY, corr_lines: DOUBLE_ARRAY) -> DOUBLE_ARRAY:
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

    def local_sum(in_mat: NDArray, shape: tuple[int, ...]) -> NDArray:
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


def _fft_correlate_1d(a: NDArray, b: NDArray) -> DOUBLE_ARRAY:
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


def fft_correlator_1d(extracted_lines: NDArray, predicted_lines: NDArray) -> DOUBLE_ARRAY:
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
