"""
This module provides a function for performing automatic optimal thresholding based on histogram peaks referred to as the Otsu method.
"""

import numpy as np
from numpy.typing import NDArray

from scipy.optimize import fmin

import cv2


def otsu(image: NDArray, n: int) -> tuple[list[float], NDArray[np.uint8]]:
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

        return [threshold], labeled_image.astype(np.uint8)

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

            labeled_image = np.zeros(image.shape, dtype=np.uint8)

            labeled_image[(iu8 > unique_iu8[k1]) & (iu8 <= unique_iu8[k2])] = 1

            labeled_image[iu8 > unique_iu8[k2]] = 2

            thresholds = np.array([unique_iu8[k1], unique_iu8[k2]], dtype=np.float64)

            thresholds /= multi_conv
            thresholds += delta_conv

            # noinspection PyTypeChecker
            out_thresh: list = thresholds.astype(image.dtype).tolist() 

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

            labeled_image = np.zeros(image.shape, dtype=np.uint8)

            labeled_image[iu8 > unique_iu8[kk[n - 2]]] = n - 1
            for i in range(n - 2):
                labeled_image[(iu8 > unique_iu8[kk[i]]) & (iu8 <= unique_iu8[kk[i + 1]])] = i + 1

            # put back into the original image values
            thresholds = unique_iu8[kk[:n - 2]].astype(np.float64)
            thresholds /= multi_conv
            thresholds += delta_conv

            # noinspection PyTypeChecker
            out_thresh: list = thresholds.astype(image.dtype).tolist() 

            for ind, t in enumerate(out_thresh):
                try:
                    out_thresh[ind] = min(max(t, image[labeled_image == ind].max()), image[labeled_image == ind+1].min())
                except:
                    out_thresh[ind] = np.nan
            return out_thresh, labeled_image

