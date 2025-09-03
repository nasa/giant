"""
This module provides functionality for finding the (subpixel) peak of a 2d surface or of 1d lines.
"""

import numpy as np

from numpy.typing import NDArray

import cv2

from giant._typing import ARRAY_LIKE


def pixel_level_peak_finder_2d(surface: ARRAY_LIKE, blur: bool = True) -> NDArray[np.integer]:
    """
    This function returns a numpy array containing the (x, y) location of the maximum surface value
    to pixel level accuracy.

    Optionally, a blur can be applied to the surface before locating the peak to attempt to remove high frequency noise.

    :param surface: A surface, or image, to use
    :param blur: A flag to indicate whether to apply Gaussian blur to image
    :return: The (x, y) location of the maximum surface values to pixel level accuracy.
    :raises ValueError: If the provided surface is not 2 dimensional
    """
    array_surface: NDArray = np.asanyarray(surface)
    
    if array_surface.ndim != 2:
        raise ValueError('The surface must be 2d to use this function')

    if blur:
        valid_dtypes = (np.uint8, np.uint16, np.int16, np.float32, np.float64)
        
        if not any(np.issubdtype(array_surface.dtype, vdtype) for vdtype in valid_dtypes):
            array_surface = array_surface.astype(np.float64)
        # Do this to try to avoid spikes due to noise aligning
        array_surface = cv2.GaussianBlur(array_surface, (5, 5), 1, None, 1, cv2.BORDER_CONSTANT)

    return np.flipud(np.unravel_index(np.argmax(array_surface), array_surface.shape))


def quadric_peak_finder_2d(surface: ARRAY_LIKE, fit_size: int = 1, blur: bool = True,
                           shift_limit: int = 3) -> NDArray[np.float64]:
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
    :raises ValueError: If the provided surface is not 2 dimensional
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

    cols = cols.ravel()
    rows = rows.ravel()

    # form the jacobian matrix for the least squares
    jac_matrix = np.array([cols * cols, rows * rows, cols * rows, cols, rows, np.ones(rows.shape)]).T

    # perform the least squares fit
    coefs = np.linalg.lstsq(jac_matrix, surface[rows, cols].ravel(), rcond=None)[0]

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


def pixel_level_peak_finder_1d(correlation_lines: np.ndarray) -> NDArray[np.integer]:
    """
    This function returns a numpy array containing the location of the maximum surface value
    to pixel level accuracy for each row of the input matrix.

    :return: The location of the maximum surface values to pixel level accuracy.
    """
    out = np.argmax(correlation_lines, axis=-1)[..., np.newaxis] 
    return out


def parabolic_peak_finder_1d(correlation_lines: np.ndarray, fit_size: int = 1) -> NDArray[np.float64]:
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
    jac_matrix = np.rollaxis(np.array([cols * cols, cols, np.ones(cols.shape)], dtype=np.float64), 0, -1)

    # Build the rhs
    rhs: NDArray[np.float64] = correlation_lines[np.ogrid[:correlation_lines.shape[0], :0][:1] + (None, cols)]

    # Fit the paraboloid using LLS
    solus = np.linalg.solve(jac_matrix @ jac_matrix.swapaxes(-1, -2), jac_matrix @ rhs)

    # Return the subpixel center
    return (-solus[..., 1, :] / (2 * solus[..., 0, :])).reshape(base_shape + (-1,))
