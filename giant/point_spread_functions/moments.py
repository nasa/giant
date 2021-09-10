# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
Defines a PSF object for estimating centroids using a moment (center-of-illumination) algorithm.
"""
import numpy as np

from .psf_meta import PointSpreadFunction

from .._typing import NONENUM, ARRAY_LIKE, Real, NONEARRAY


class Moment(PointSpreadFunction):
    """
    This class implements a moment based (center of illumination) algorithm for locating the centroid of a PSF.

    This class implements a fully functional PSF object for GIANT, however, because it does not actually model how light
    is spread out, if applied to an image or scan lines it just returns the input unaltered.  Also, since this isn't
    actually an estimation, the covariance and residuals are undefined so these are always set to NaN.

    :Note: This object can be biased toward the center of an image if it is applied naively.  You must be careful in
           selecting which points to pass to this function.
    """

    def __init__(self, centroid_x: NONENUM = None, centroid_y: NONENUM = None, **kwargs):
        """
        :param centroid_x: The x component of the centroid in pixels
        :param centroid_y: The y component of the centroid in pixels
        """

        self.centroid_x = 0.0  # type: float
        """
        The x location of the centroid
        """

        self.centroid_y = 0.0  # type: float
        """
        The y location of the centroid
        """

        if centroid_x is not None:
            self.centroid_x = float(centroid_x)

        if centroid_y is not None:
            self.centroid_y = float(centroid_y)

        super().__init__(**kwargs)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Just returns the input image as an array

        :param image: The image to apply the psf to
        :return: the unaltered image as an array
        """

        return image

    def apply_1d(self, image_1d: np.ndarray, direction: NONEARRAY = None,
                 step: Real = 1) -> np.ndarray:
        """
        Just returns the input scan lines as is.

        :param image_1d: the scan lines to apply the PSF to
        :param direction: the direction of the scan lines
        :param step: the step size of the scan lines
        :return: the unaltered scan lines
        """

        return image_1d

    def generate_kernel(self) -> np.ndarray:
        """
        Returns a 3x3 array of zeros except the center which is one because this does nothing.

        :return: The nothing kernel
        """

        return np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    def evaluate(self, x: ARRAY_LIKE, y: ARRAY_LIKE) -> np.ndarray:
        """
        Returns an array of zeros the same shape of ``x``/``y``.

        :param x: The x values to evaluate at
        :param y: The y values to evaluate at
        :return: An array of zeros
        """

        return np.zeros(x.shape, dtype=np.float64)

    @classmethod
    def fit(cls, x: ARRAY_LIKE, y: ARRAY_LIKE, z: ARRAY_LIKE) -> __qualname__:
        r"""
        This function identifies the centroid of the PSF for the input data using a moment algorithm (center of
        illumination).

        .. math::
            x_0 = \frac{\sum{\mathbf{x}\mathbf{I}}}{\sum{\mathbf{I}}} \qquad
            y_0 = \frac{\sum{\mathbf{y}\mathbf{I}}}{\sum{\mathbf{I}}}

        :param x: The x values underlying the surface the PSF is to be fit to
        :param y: The y values underlying the surface the PSF is to be fit to
        :param z: The z or "height" values of the surface the PSF is to be fit to
        :return: An instance of the PSF that best fits the provided data
        """
        # make sure the inputs are flattened numpy arrays
        x = np.array(x).ravel()
        y = np.array(y).ravel()
        z = np.array(z).ravel()

        # perform the moment algorithm
        x0 = np.sum(x * z) / np.sum(z)
        y0 = np.sum(y * z) / np.sum(z)

        out = cls(centroid_x=x0, centroid_y=y0)

        return out

    @property
    def centroid(self) -> np.ndarray:
        """
        The location of the center of the PSF as an (x, y) length 2 numpy array.

        This property is used to enable the PSF class to be used in identifying the center of
        illumination in image processing (see :attr:`.ImageProcessing.centroiding`).

        :return: The (x, y) location of the peak of the PSF as a 1D numpy array
        """

        return np.array([self.centroid_x, self.centroid_y])

    @property
    def residual_rss(self) -> NONENUM:
        """
        The rss of the residuals (undefined).

        :return: NaN or ``None`` since this is undefined.
        """

        if self.save_residuals:
            return np.nan
        else:
            return None

    @property
    def residual_mean(self) -> NONENUM:
        """
        The mean of the residuals (undefined).

        :return: NaN or ``None`` since this is undefined.
        """

        if self.save_residuals:
            return np.nan
        else:
            return None

    @property
    def residual_std(self) -> NONENUM:
        """
        The standard deviation of the residuals (undefined).

        :return: NaN or ``None`` since this is undefined.
        """

        if self.save_residuals:
            return np.nan
        else:
            return None

    @property
    def covariance(self) -> NONEARRAY:
        """
        The covariance of the fit (undefined).

        :return: A 2x2 array of NaN or ``None`` since this is undefined.
        """

        if self.save_residuals:
            return np.nan*np.zeros((2, 2), dtype=np.float64)
        else:
            return None

    def volume(self) -> float:
        """
        The volume is undefined for a moment PSF so just return 0

        :return: 0
        """
        return 0.0
