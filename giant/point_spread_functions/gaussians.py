# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
Defines PSF subclasses for representing/fitting various forms of 2D Gaussian functions.

Note that in this module we assume that the resulting illumination profile for a point source as captured by a detector
is Gaussian, which is not strictly true.  In actuality, if the PSF is well modelled by a Gaussian function then the
captured profile will be integrals of the portions of the Gaussian contained within each pixel.  These integrals are not
easily defined which makes fitting much more cumbersome and cost intensive (requiring numeric approximations for the
Jacobians).  In general, however, particularly for a well sampled PSF (where the FWHM is larger than the
pixel pitch of the detector), this distinction is negligible, particularly when it comes to estimating the centroid of
the PSF, which is typically our primary goal.  Providing integrated Gaussian PSFs for extra precise estimation is
something that is currently under development.
"""

from abc import ABCMeta

from typing import Optional

import numpy as np
import cv2

from .psf_meta import (KernelBasedApply1DPSF, KernelBasedCallPSF,
                       InitialGuessIterativeNonlinearLSTSQPSF, SizedPSF,
                       InitialGuessIterativeNonlinearLSTSQPSFwBackground)

from .._typing import ARRAY_LIKE, Real, NONEARRAY


class _GaussianSkeleton(KernelBasedApply1DPSF, SizedPSF, metaclass=ABCMeta):
    """
    This internal class defines common attributes, properties, and methods for Gaussian based PSFs.

    This should not be used by the user as it is not functional as is.
    """

    def __init__(self, size: Optional[int] = None, amplitude: Optional[Real] = None,
                 centroid_x: Optional[Real] = 0, centroid_y: Optional[Real] = 0,
                 sigma_x: Optional[Real] = 1, sigma_y: Optional[Real] = 0, **kwargs):

        self.sigma_x = 1.0  # type: float
        """
        The Gaussian RMS width in the x direction in pixels.
        """

        if not ((sigma_x is None) or (sigma_x == 0)):
            self.sigma_x = float(sigma_x)

        self.sigma_y = sigma_x
        """
        The Gaussian RMS width in the y direction in pixels.
        """

        if not ((sigma_y is None) or (sigma_y == 0)):
            self.sigma_y = float(sigma_y)

        self.amplitude = 0.0  # type: float
        """
        The amplitude of the Gaussian.

        This specifies how much energy the Gaussian function increases or decreases the signal by.  

        Typically this is set so that the kernel does not increase or decrease the signal, which can be achieved by 
        using the :meth:`normalize_amplitude` method.
        """

        if (amplitude is None) or amplitude == 0:
            self.normalize_amplitude()
        else:
            self.amplitude = float(amplitude)

        self.centroid_x = 0.0  # type: float
        """
        The x location of the peak of the Gaussian kernel

        When applying the gaussian kernel, the centroid doesn't matter as it will always be applied as if it was 
        centered on a pixel.  In general this is just used when estimating a kernel to locate the peak of the PSF in 
        the image.
        """

        if centroid_x is not None:
            self.centroid_x = float(centroid_x)

        self.centroid_y = 0.0  # type: float
        """
        The y location of the peak of the Gaussian kernel

        When applying the gaussian kernel, the centroid doesn't matter as it will always be applied as if it was 
        centered on a pixel.  In general this is just used when estimating a kernel to locate the peak of the PSF in 
        the image.
        """

        if centroid_y is not None:
            self.centroid_y = float(centroid_y)

        self._residuals = None  # type: NONEARRAY

        self._covariance = None  # type: NONEARRAY

        super().__init__(size=size, **kwargs)

    def __repr__(self) -> str:
        """
        Print out the PSF parameters.
        """

        important_vars = []
        important_values = []

        for attr in self.__dict__.keys():

            if ((not attr.startswith('_')) and ('residual' not in attr) and ('covariance' not in attr)
                    and ('centroid' != attr) and ('size' != attr)):
                important_vars.append(attr)
                important_values.append(getattr(self, attr))

        format_string = self.__class__.__name__ + '(' + '={}, '.join(important_vars) + ' = {})'

        return format_string.format(*important_values)


    @property
    def residuals(self) -> np.ndarray:
        """
        A 1D array containing residuals of the fit of this Gaussian Model to data.

        These are only populated when initialized by :meth:`fit` and when the class attribute :attr:`save_residuals`
        is set to true.
        """

        return self._residuals

    @property
    def centroid(self) -> np.ndarray:
        """
        The location of the peak of the Gaussian PSF as a length 2 numpy array (x, y).

        This is equivalent to ``np.array([psf.centroid_x, psf.centroid_y])`` where ``psf`` is an initialized instance of
        this class
        """

        return np.array([self.centroid_x, self.centroid_y])

    @property
    def residual_mean(self) -> Optional[float]:
        """
        The mean of the post-fit residuals after fitting this PSF model to data.

        If this instance is not the result of a fit (:meth:`fit`) or if :attr:`save_residuals` is ``False`` then this
        will return None
        """

        if self.residuals is not None:
            return float(np.mean(self.residuals))
        else:
            return None

    @property
    def residual_std(self) -> Optional[float]:
        """
        The standard deviation of the post-fit residuals after fitting this PSF model to data.

        If this instance is not the result of a fit (:meth:`fit`) or if :attr:`save_residuals` is ``False`` then this
        will return None
        """

        if self.residuals is not None:
            return float(np.std(self.residuals))
        else:
            return None

    @property
    def residual_rss(self) -> Optional[float]:
        """
        The sum of squares of the post-fit residuals after fitting this PSF model to data.

        If this instance is not the result of a fit (:meth:`fit`) or if :attr:`save_residuals` is ``False`` then this
        will return None
        """

        if self.residuals is not None:
            return float(np.square(self.residuals).sum())
        else:
            return None

    @property
    def covariance(self) -> Optional[np.ndarray]:
        r"""
        The formal covariance of the PSF parameters after fitting this PSF model to data.

        If this instance is not the result of a fit (:meth:`fit`) of if :attr:`save_residuals` is ``False`` then this
        will return None.
        """

        return self._covariance

    def apply_1d(self, image_1d: np.ndarray, direction: Optional[np.ndarray] = None,
                 step: Real = 1) -> np.ndarray:
        """
        Applies the defined PSF using the stored parameters to the 1D image scans provided.

        ``image_1d`` can be a 2D array but in that case each row will be treated as an independent 1D scan.

        For non-symmetric PSFs, a ``direction`` argument can be supplied which should be the direction in the image of
        each scan line.  This can be used to determine the appropriate cross-section of the PSF to use for applying to
        the 1D scans (if applicable).  If no direction is provided then the x direction [1, 0] is assumed.

        :param image_1d: The scan line(s) to be blurred using the PSF
        :param direction: The direction for the 1D cross section of the PSF.  This should be either None, a length 2
                          array, or a shape nx2 array where n is the number of scan lines
        :param step: The step size of the lines being blurred.
        :return: an array containing the input after blurring with the defined PSF
        """
        size = max(int(4 * max(self.sigma_x, self.sigma_y) + 0.5), 3)

        # resize so that half the size is evenly divisible by step
        size = step * (((size / 2) // step) + 1) * 2

        return self.apply_1d_sized(image_1d, size, direction, step)

    def normalize_amplitude(self) -> None:
        r"""
        Calculate and store the amplitude that makes the volume under the gaussian surface equal to 1

        This is defined as

        .. math::
            A = \frac{1}{2\pi\sigma_x\sigma_y}
        """

        self.amplitude = 1 / (2 * np.pi * self.sigma_x * self.sigma_y)

    def volume(self) -> float:
        r"""
        Calculate the volume under the gaussian function

        This is defined as

        .. math::
            V = 2\pi A\sigma_x\sigma_y}
        """

        return self.amplitude*2*np.pi*self.sigma_x*self.sigma_y

    def determine_size(self) -> None:
        r"""
        Sets the size for the kernel based on the width of the PSF.

        This is defined as

        .. math::
            s=\text{floor}(4*\text{max}(\sigma_x, \sigma_y)+0.5))

        which corresponds roughly to the size required to capture at least 2 sigma of the PSF along its widest axis, or
        95.45%  of the curve.
        """

        if not np.isnan([self.sigma_x, self.sigma_y]).any():
            self.size = max(int(4 * max(self.sigma_x, self.sigma_y) + 0.5), 3)

            # make sure its odd
            if (self.size % 2) == 0:
                self.size += 1
        else:
            self.size = 3


class Gaussian(_GaussianSkeleton):
    r"""
    A class for representing and fitting a standard (non-rotated) 2D gaussian point spread function.

    This class represents a 2D Gaussian function of the form

    .. math::
        f(x, y) = A e^{\left(-\left[\frac{(x-x_0)^2}{2\sigma_x^2}+\frac{(y-y_0)^2}{2\sigma_y^2}\right]\right)}

    where :math:`A` is the :attr:`amplitude` of the PSF, :math:`\sigma_x` is the Gaussian RMS width in the x direction,
    :math:`\sigma_y` is the Gaussian RMS width in the y direction, and :math:`(x_0,y_0)` is the centroid of the Gaussian
    (location of the peak response).

    This class can be used for both estimating a Gaussian fit to an observed PSF (using the :meth:`fit` class method to
    create an instance) as well as for applying the represented PSF to 1D scan lines (using :meth:`apply_1d`) and 2D
    images (using the *call* capabilities of an instance of this class).  In addition, if generated from a fit to data,
    this class will store the residuals and statistics about the residuals of the fit if the class attribute
    :attr:`save_residuals` is set to True before calling :meth:`fit`.

    This class can be used anywhere GIANT expects a point spread function.
    """


    def __init__(self, sigma_x: Optional[Real] = 1, sigma_y: Optional[Real] = 0, size: Optional[int] = None,
                 amplitude: Optional[Real] = None, centroid_x: Optional[Real] = 0, centroid_y: Optional[Real] = 0,
                 **kwargs):
        """
        :param sigma_x: The Gaussian RMS width in the x direction in pixels
        :param sigma_y: The Gaussian RMS Width in the y direction in pixels.  If set to 0 or ``None`` this is set to be
                        the same as ``sigma_x``
        :param size: The size of the kernel to use when applying this PSF in pixels.  If set to 0 or ``None`` will be
                     computed based on the Gaussian RSM widths.
        :param amplitude: The amplitude of the gaussian kernel to use when applying this PSF.  If set to 0 or ``None``
                          this will be computed so that the kernel does not increase/decrease the total signal.
        :param centroid_x: The x location of the peak of the Gaussian PSF in pixels.  This is not used when applying the
                           PSF, but it is used when fitting the PSF.  Typically this is not specified by the user.
        :param centroid_y: The y location of the peak of the Gaussian PSF in pixels.  This is not used when applying the
                           PSF, but it is used when fitting the PSF.  Typically this is not specified by the user.
        """

        # call the super class
        super().__init__(size=size, amplitude=amplitude, centroid_x=centroid_x, centroid_y=centroid_y,
                         sigma_x=sigma_x, sigma_y=sigma_y, **kwargs)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the represented Gaussian PSF to a 2D array.
        :param image: The array to apply the PSF to
        :return: The resulting array after applying the PSF as a numpy array.
        """
        return cv2.GaussianBlur(image, (self.size, self.size), self.sigma_x, sigmaY=self.sigma_y)

    @classmethod
    def fit(cls, x: ARRAY_LIKE, y: ARRAY_LIKE, z: ARRAY_LIKE) -> 'Gaussian':
        r"""
        This fits a 2d gaussian function to a surface using least squares estimation.

        The fit assumes that z = f(x, y) where f is the gaussian function (and thus z is the "height" of the gaussian
        function).

        The fit performed is for a 2D gaussian function of the form

        .. math::
            z = f(x, y) = A e^{\left(-\left[\frac{(x-x_0)^2}{2\sigma_x^2}+\frac{(y-y_0)^2}{2\sigma_y^2}\right]\right)}

        The estimation in this function is performed by transforming the gaussian function into the logspace, which
        allows us to perform a true linear least squares fit without iteration but overweights the tails of the
        function.  Therefore, it is best to constrain the data you are fitting to be near the peak of the PSF to ensure
        that too much noise is not being given extra weight in the fit.

        If the fit is unsuccessful due to a rank deficient matrix or a fit of a hyperbolic surface the resulting data
        will be set to np.nan.

        :param x: The x values underlying the surface the PSF is to be fit to
        :param y: The y values underlying the surface the PSF is to be fit to
        :param z: The z or "height" values of the surface the PSF is to be fit to
        :return: The initialized PSF with values according to the fit
        """

        # make sure the arrays are flat and numpy arrays
        x = np.array(x).ravel()
        y = np.array(y).ravel()
        z = np.array(z).ravel()

        # form the Jacobian matrix
        # we are fitting to a model of the form
        coefficients = np.vstack([np.power(x, 2), x, np.power(y, 2), y, np.ones(x.shape)]).T

        try:  # try to generate the least squares solution unless the system is rank deficient

            # get the solution to the least squares problem
            solution = np.linalg.lstsq(coefficients, np.log(z), rcond=None)[0]

            sigma_x = np.sqrt(-1 / (2 * solution[0]))
            sigma_y = np.sqrt(-1 / (2 * solution[2]))

            x0 = solution[1] * sigma_x ** 2
            y0 = solution[3] * sigma_y ** 2

            amplitude = np.exp(solution[4] + x0 ** 2 / (2 * sigma_x ** 2) + y0 ** 2 / (2 * sigma_y ** 2))

            out = cls(sigma_x=sigma_x, sigma_y=sigma_y, amplitude=amplitude, centroid_x=x0, centroid_y=y0)

            # check for something bad
            if (sigma_x < 0) or (sigma_y < 0):
                out.update_state(None)
                return out

            if cls.save_residuals:
                computed = out.evaluate(x, y)
                out._residuals = z - computed

                jacobian = out.compute_jacobian(x, y, computed)

                # assume a single uncertainty for all measurements
                out._covariance = np.linalg.pinv(jacobian.T @ jacobian) * out.residual_std ** 2

        except np.linalg.linalg.LinAlgError:
            out = cls(np.nan, sigma_y=np.nan, amplitude=np.nan, centroid_x=np.nan, centroid_y=np.nan)

        return out

    def compute_jacobian(self, x: np.ndarray, y: np.ndarray, computed: np.ndarray):
        r"""
        This method computes the Jacobian of the PSF with respect to a change in the state.

        It returns a mx5 matrix defined as

        .. math::
            \mathbf{J} = \left[\begin{array}{ccccc} \frac{\partial f}{\partial x_0} &
            \frac{\partial f}{\partial y_0} &
            \frac{\partial f}{\partial \sigma_x} &
            \frac{\partial f}{\partial \sigma_y} &
            \frac{\partial f}{\partial A}\end{array}\right]=\left[\begin{array}{ccccc}
            \frac{x-x_0}{\sigma_x^2}f(x, y) &
            \frac{y-y_0}{\sigma_y^2}f(x, y) &
            \frac{(x-x_0)^2}{\sigma_x^3}f(x, y) &
            \frac{(y-y_0)^2}{\sigma_y^3}f(x, y) &
            \frac{f(x, y)}{A}\end{array}\right]

        :param x: The x values to evaluate the Jacobian at as a length m array
        :param y: The y values to evaluate the Jacobian at as a length m array
        :param computed: The PSF evaluated at x and y as a length m array
        :return: The Jacobian matrix as a mx5 numpy array
        """

        jacobian = np.vstack([computed * (x - self.centroid_x) / (self.sigma_x ** 2),
                              computed * (y - self.centroid_y) / (self.sigma_y ** 2),
                              computed * (x - self.centroid_x) ** 2 / (self.sigma_x ** 3),
                              computed * (y - self.centroid_y) ** 2 / (self.sigma_y ** 3),
                              computed / self.amplitude]).T

        return jacobian

    def update_state(self, update: NONEARRAY) -> None:
        r"""
        Updates the current values based on the provided update vector.

        The provided update vector is in the order of :math:`[x_0, y_0, \sigma_x, \sigma_y, A]`.

        If the update vector is set to ``None`` then sets everything to NaN to indicate a bad fit.

        :param update: The vector of additive updates to apply
        """

        if update is None:
            self.size = np.nan
            self.sigma_x = np.nan
            self.sigma_y = np.nan
            self.amplitude = np.nan
            self.centroid_x = np.nan
            self.centroid_y = np.nan
        else:
            self.centroid_x += update[0]
            self.centroid_y += update[1]
            self.sigma_x += update[2]
            self.sigma_y += update[3]
            self.amplitude += update[4]

    @property
    def covariance(self) -> Optional[np.ndarray]:
        r"""
        The formal covariance of the PSF parameters after fitting this PSF model to data.

        If this instance is not the result of a fit (:meth:`fit`) of if :attr:`save_residuals` is ``False`` then this
        will return None.

        The order of the state vector (and thus the covariance matrix) is :math:`[x_0, y_0, \sigma_x, \sigma_y, A]`.
        """

        return self._covariance

    def evaluate(self, x: ARRAY_LIKE, y: ARRAY_LIKE) -> np.ndarray:
        r"""
        This method evaluates the PSF at the given x and y.

        This method is not intended to be used to apply the PSF for an image (use the callable capability of the class
        instead for this).  Instead it simply computes the height of the PSF above the xy-plane at the requested
        locations.

        Specifically, this method computes

        .. math::
            z = f(x, y) = A e^{\left(-\left[\frac{(x-x_0)^2}{2\sigma_x^2}+\frac{(y-y_0)^2}{2\sigma_y^2}\right]\right)}

        :param x: The x locations the height of the PSF is to be calculated at.
        :param y: The y locations the height of the PSF is to be calculated at.
        :return: A numpy array containing the height of the PSF at the requested locations the same shape as x and y.
        """

        # ensure things are arrays
        x = np.array(x)
        y = np.array(y)

        extras = super().evaluate(x, y)

        if extras is None:
            extras = 0

        return self.amplitude * np.exp(-(x - self.centroid_x) ** 2 / (2 * self.sigma_x ** 2) -
                                       (y - self.centroid_y) ** 2 / (2 * self.sigma_y ** 2)) + extras


class IterativeGaussian(Gaussian, InitialGuessIterativeNonlinearLSTSQPSF):
    """
    A class for representing and fitting a standard (non-rotated) 2D Gaussian point spread function using iterative
    non-linear least squares.

    This class only differs from the :class:`Gaussian` class in the way the :meth:`fit` class method works.  In this
    version, the fit is performed using iterative non-linear least squares, which is typically more accurate than using
    the logarithmic transformation to do linear least squares at the expense of more computation time.

    For more details about the model this works with, see the :class:`Gaussian` documentation.

    This class can be used anywhere GIANT expects a point spread function
    """


    @classmethod
    def fit(cls, x: ARRAY_LIKE, y: ARRAY_LIKE, z: ARRAY_LIKE) -> 'IterativeGaussian':
        r"""
        This fits a 2d gaussian function to a surface using iterative non-linear least squares estimation.

        The fit assumes that z = f(x, y) where f is the gaussian function (and thus z is the "height" of the gaussian
        function).

        The fit performed is for a 2D gaussian function of the form

        .. math::
            z = f(x, y) = A e^{\left(-\left[\frac{(x-x_0)^2}{2\sigma_x^2}+\frac{(y-y_0)^2}{2\sigma_y^2}\right]\right)}

        The estimation in this function is performed iteratively.  First, a transformed fit is performed using the
        :meth:`.Gaussian.fit` method.  This initial fit is then refined using iterative non-linear least squares to
        remove biases that can be introduced in the transformed fit.

        If the fit is unsuccessful due to a rank deficient matrix or a fit of a hyperbolic surface the resulting data
        will be set to np.nan.

        :param x: The x values underlying the surface the PSF is to be fit to
        :param y: The y values underlying the surface the PSF is to be fit to
        :param z: The z or "height" values of the surface the PSF is to be fit to
        :return: The initialized PSF with values according to the fit
        """

        out = cls.fit_lstsq(x, y, z)

        # check if something isn't right
        if (out.sigma_x < 0) or (out.sigma_y < 0):
            out.update_state(None)
            return out

        return out


class IterativeGaussianWBackground(Gaussian, InitialGuessIterativeNonlinearLSTSQPSFwBackground):
    r"""
    A class for representing and fitting the superposition of a standard (non-rotated) 2D Gaussian point spread function
    and a linear background gradiant using iterative non-linear least squares.

    This class differs from the :class:`Gaussian` class in the way the :meth:`fit` class method works and in the fact
    that it adds a background gradient to the model.  In this version, the fit is performed using iterative non-linear
    least squares, which is typically more accurate than using the logarithmic transformation to do linear least squares
    at the expense of more computation time.

    The specific model that is fit is given by

    .. math::
        f(x, y) = A e^{\left(-\left[\frac{(x-x_0)^2}{2\sigma_x^2}+\frac{(y-y_0)^2}{2\sigma_y^2}\right]\right)} + Bx+Cy+D

    This class can be used anywhere GIANT expects a point spread function
    """

    def __init__(self, sigma_x: Optional[Real] = 1, sigma_y: Optional[Real] = 0, size: Optional[int] = None,
                 amplitude: Optional[Real] = None, centroid_x: Optional[Real] = 0, centroid_y: Optional[Real] = 0,
                 bg_b_coef: Optional[Real] = None, bg_c_coef: Optional[Real] = None, bg_d_coef: Optional[Real] = None,
                 **kwargs):
        """
        :param sigma_x: The Gaussian RMS width in the x direction in pixels
        :param sigma_y: The Gaussian RMS Width in the y direction in pixels.  If set to 0 or ``None`` this is set to be
                        the same as ``sigma_x``
        :param size: The size of the kernel to use when applying this PSF in pixels.  If set to 0 or ``None`` will be
                     computed based on the Gaussian RSM widths.
        :param amplitude: The amplitude of the gaussian kernel to use when applying this PSF.  If set to 0 or ``None``
                          this will be computed so that the kernel does not increase/decrease the total signal.
        :param centroid_x: The x location of the peak of the Gaussian PSF in pixels.  This is not used when applying the
                           PSF, but it is used when fitting the PSF.  Typically this is not specified by the user.
        :param centroid_y: The y location of the peak of the Gaussian PSF in pixels.  This is not used when applying the
                           PSF, but it is used when fitting the PSF.  Typically this is not specified by the user.
        :param bg_b_coef: The x slope of the background gradient
        :param bg_c_coef: They y slope of the background gradient
        :param bg_d_coef: The constant offset of the background gradient
        """

        super().__init__(sigma_x=sigma_x, sigma_y=sigma_y, size=size, amplitude=amplitude, centroid_x=centroid_x,
                         centroid_y=centroid_y, bg_b_coef=bg_b_coef, bg_c_coef=bg_c_coef, bg_d_coef=bg_d_coef,
                         **kwargs)

    @classmethod
    def fit(cls, x: ARRAY_LIKE, y: ARRAY_LIKE, z: ARRAY_LIKE) -> 'IterativeGaussianWBackground':
        r"""
        This fits a 2d gaussian function to a surface using iterative non-linear least squares estimation.

        The fit assumes that z = f(x, y) where f is the gaussian function (and thus z is the "height" of the gaussian
        function).

        The fit performed is for a 2D gaussian function of the form

        .. math::
            z = f(x, y) = A e^{\left(-\left[\frac{(x-x_0)^2}{2\sigma_x^2}+\frac{(y-y_0)^2}{2\sigma_y^2}\right]\right)}
            +Bx+Cy+D

        The estimation in this function is performed iteratively.  First, the rough background is estimated and removed.
        Then, a transformed fit is performed using the super class's fit method on the data with the rough background
        removed.  This initial fit is then refined using iterative non-linear least squares on the original data to
        remove biases that might have been introduced in the non-iterative fit.

        If the fit is unsuccessful due to a rank deficient matrix or a fit of a hyperbolic surface the resulting data
        will be set to np.nan.

        :param x: The x values underlying the surface the PSF is to be fit to
        :param y: The y values underlying the surface the PSF is to be fit to
        :param z: The z or "height" values of the surface the PSF is to be fit to
        :return: The initialized PSF with values according to the fit
        """

        out = cls.fit_lstsq(x, y, z)

        # check if something isn't right
        if (out.sigma_x < 0) or (out.sigma_y < 0):
            out.update_state(None)
            return out

        return out

    def update_state(self, update: NONEARRAY) -> None:
        r"""
        Updates the current values based on the provided update vector.

        The provided update vector is in the order of :math:`[x_0, y_0, \sigma_x, \sigma_y, A, B, C, D]`.

        If the update vector is set to ``None`` then sets everything to NaN to indicate a bad fit.

        :param update: The vector of additive updates to apply
        """

        super().update_state(update)
        if update is not None:
            self.apply_update_bg(update[-3:])
        else:
            self.apply_update_bg(update)

    def compute_jacobian(self, x: np.ndarray, y: np.ndarray, computed: np.ndarray) -> np.ndarray:
        r"""
        This method computes the Jacobian of the PSF with respect to a change in the state.

        It returns a mx8 matrix defined as

        .. math::
            \mathbf{J} = \left[\begin{array}{cccccccc} \frac{\partial f}{\partial x_0} &
            \frac{\partial f}{\partial y_0} &
            \frac{\partial f}{\partial \sigma_x} &
            \frac{\partial f}{\partial \sigma_y} &
            \frac{\partial f}{\partial A} &
            \frac{\partial f}{\partial B} &
            \frac{\partial f}{\partial C} &
            \frac{\partial f}{\partial D}\end{array}\right]=\left[\begin{array}{cccccccc}
            \frac{x-x_0}{\sigma_x^2}f(x, y) &
            \frac{y-y_0}{\sigma_y^2}f(x, y) &
            \frac{(x-x_0)^2}{\sigma_x^3}f(x, y) &
            \frac{(y-y_0)^2}{\sigma_y^3}f(x, y) &
            \frac{f(x, y)}{A} &
            x & y & 1\end{array}\right]

        :param x: The x values to evaluate the Jacobian at as a length m array
        :param y: The y values to evaluate the Jacobian at as a length m array
        :param computed: The PSF evaluated at x and y as a length m array
        :return: The Jacobian matrix as a mx8 numpy array
        """

        return self.compute_jacobian_all(x, y, computed)


class GeneralizedGaussian(KernelBasedCallPSF, _GaussianSkeleton):
    r"""
    A class for representing and fitting a generalized (rotated) 2D gaussian point spread function.

    This class represents a 2D Gaussian function of the form

    .. math::
        f(x, y) = A e^{\left(-\left[a(x-x_0)^2 + 2b (x-x_0)(y-y_0) + c (y-y_0)^2\right]\right)}

    where :math:`A` is the :attr:`amplitude` of the PSF, :math:`a` is the coefficient for :math:`(x-x_0)^2`,
    :math:`c` is the coefficient for :math:`(y-y_0)^2`,  :math:`b` is the coefficient for :math:`(x-x_0)(y-y_0)`, and
    :math:`(x_0,y_0)` is the centroid of the Gaussian (location of the peak response).

    This is equivalent to a function of the form

    .. math::
        f(x, y) = A e^{-\left[\begin{array}{cc} (x-x_0) & (y-y_0)\end{array}\right]\mathbf{B}\mathbf{S}\mathbf{B}^T
        \left[\begin{array}{c}(x-x_0) \\ (y-y_0)\end{array}\right]}

    where

    .. math::
        \mathbf{B} = \left[\begin{array}{cc} \text{cos}(\theta) & -\text{sin}(\theta) \\
        \text{sin}(\theta) & \text{cos}(\theta)\end{array}\right] \\
        \mathbf{S} = \left[\begin{array}{cc} \frac{1}{\sigma_x^2} & 0 \\ 0 & \frac{1}{\sigma_y^2}\end{array}\right]

    :math:`\theta` is the angle between the x-axis and the principal axis of the Gaussian, :math:`sigma_x` is the
    Gaussian RMS width in the semi-major axis direction, and :math:`\sigma_x` is the  RMS width in the semi-minor axis
    direction.

    When creating an instance of this class you can specify either ``a, b, c`` or ``sigma_x, sigma_y, theta`` and the
    class will convert and store appropriately.  This class also allows you to retrieve either ``a, b, c`` or
    ``sigma_x, sigma_y, theta``.

    This class can be used for both estimating a Gaussian fit to an observed PSF (using the :meth:`fit` class method to
    create an instance) as well as for applying the represented PSF to 1D scan lines (using :meth:`apply_1d) and 2D
    images (using the *call* capabilities of an instance of this class).  In addition, if generated from a fit to data,
    this class will store the residuals and statistics about the residuals of the fit if the class attribute
    :attr:`save_residuals` is set to True before calling :meth:`fit`.

    This class can be used anywhere GIANT expects a point spread function.
    """

    def __init__(self, a_coef: Optional[Real] = None, b_coef: Optional[Real] = None, c_coef: Optional[Real] = None,
                 sigma_x: Optional[Real] = None, sigma_y: Optional[Real] = None, theta: Optional[Real] = None,
                 amplitude: Optional[Real] = None, centroid_x: Optional[Real] = 0, centroid_y: Optional[Real] = 0,
                 size: Optional[int] = None, **kwargs):
        """
        :param a_coef: The a coefficient of the Gaussian polynomial
        :param b_coef: The b coefficient of the Gaussian polynomial
        :param c_coef: The c coefficient of the Gaussian polynomial
        :param sigma_x: The Gaussian RMS width in the x direction in pixels
        :param sigma_y: The Gaussian RMS Width in the y direction in pixels.  If set to 0 or ``None`` this is set to be
                        the same as ``sigma_x``
        :param sigma_x: The angle between the x-axis and the principal axis of the Gaussian in radians.
        :param size: The size of the kernel to use when applying this PSF in pixels.  If set to 0 or ``None`` will be
                     computed based on the Gaussian RSM widths.
        :param amplitude: The amplitude of the gaussian kernel to use when applying this PSF.  If set to 0 or ``None``
                          this will be computed so that the kernel does not increase/decrease the total signal.
        :param centroid_x: The x location of the peak of the Gaussian PSF in pixels.  This is not used when applying the
                           PSF, but it is used when fitting the PSF.  Typically this is not specified by the user.
        :param centroid_y: The y location of the peak of the Gaussian PSF in pixels.  This is not used when applying the
                           PSF, but it is used when fitting the PSF.  Typically this is not specified by the user.
        """

        general_check = a_coef is not None and b_coef is not None and c_coef is not None
        rotated_check = sigma_x is not None and sigma_y is not None and theta is not None
        if not general_check and not rotated_check:
            # raise ValueError('One of (a_coef, b_coef, c_coef) or (sigma_x, sigma_y, theta) must be not None')
            # we're probably just trying to blank initialize
            pass
        else:
            general_any_check = a_coef is not None or b_coef is not None or c_coef is not None
            rotated_any_check = sigma_x is not None or sigma_y is not None or theta is not None

            if general_any_check and rotated_any_check:
                raise ValueError('One of (a_coef, b_coef, c_coef) or (sigma_x, sigma_y, theta) must be not None')

        self.a_coef = 1.0  # type: float
        """
        The :math:`(x-x_0)^2` coefficient from the exponential component of the generalized 2D Gaussian.
        """

        self.b_coef = 0.0  # type: float
        """
        The :math:`(x-x_0)(y-y_0)` coefficient from the exponential component of the generalized 2D Gaussian.
        """

        self.c_coef = 1.0  # type: float
        """
        The :math:`(y-y_0)^2` coefficient from the exponential component of the generalized 2D Gaussian.
        """

        self.sigma_x = 1.0  # type: float
        """
        The RMS width in the semi-major axis direction.  
        """

        self.sigma_y = 1.0  # type: float
        """
        The RMS width in the semi-minor axis direction.  
        """

        self.theta = 0.0  # type: float
        """
        The angle between the semi-major axis and the x axis
        """

        if general_check:
            self.a_coef = a_coef
            self.b_coef = b_coef
            self.c_coef = c_coef

            self._convert_to_pa()
        elif rotated_check:
            self.sigma_x = float(sigma_x)
            self.sigma_y = float(sigma_y)
            self.theta = float(theta)
            self._convert_to_general()

        super().__init__(sigma_x=self.sigma_x, sigma_y=self.sigma_y, size=size, centroid_x=centroid_x,
                         centroid_y=centroid_y, amplitude=amplitude, **kwargs)

    def _convert_to_general(self) -> None:
        r"""
        Calculate the general coefficients from the principal axis definition.

        .. math::
            a = \frac{\text{cos}^2(\theta)}{2\sigma_x^2} + \frac{\text{sin}^2(\theta)}{2\sigma_y^2} \\
            b = \text{cos}(\theta)\text{sin}(\theta)}\left(\frac{1}{2\sigma_y^2} - \frac{1}{2\sigma_x^2}\right) \\
            c = \frac{\text{sin}^2(\theta)}{2\sigma_x^2} + \frac{\text{cos}^2(\theta)}{2\sigma_y^2}
        """
        # TODO: not certain why we need to divide by 2 here, but its what the internet says...
        self.a_coef = (np.cos(self.theta) ** 2 / (self.sigma_x ** 2) + np.sin(self.theta) ** 2 / (
                self.sigma_y ** 2)) / 2
        self.b_coef = (np.sin(self.theta) * np.cos(self.theta) * (1 / self.sigma_y ** 2 - 1 / self.sigma_x ** 2)) / 2
        self.c_coef = (np.sin(self.theta) ** 2 / (self.sigma_x ** 2) + np.cos(self.theta) ** 2 / (
                self.sigma_y ** 2)) / 2

    def _convert_to_pa(self) -> None:
        r"""
        Calculate the principal axis coefficients from the general definition.

        .. math::
            \sigma_x = \sqrt{a+c+\sqrt{(a+c)^2-4(ac-b^2)}} \\
            \sigma_y = \sqrt{a+c-\sqrt{(a+c)^2-4(ac-b^2)}} \\
            \theta = \text{arctan2}(2*b, \sigma_x^2-2*c)
        """

        sig, r = np.linalg.eigh([[2*self.a_coef, 2*self.b_coef], [2*self.b_coef, 2*self.c_coef]])

        self.sigma_x = 1/np.sqrt(sig[1])
        self.sigma_y = 1/np.sqrt(sig[0])

        # self.theta = np.arccos(r[0, 0])
        self.theta = np.arctan2(r[1, 1], -r[0, 1])

        # make sure theta is between 0 and 180
        if self.theta < 0:
            self.theta += np.pi

    @classmethod
    def fit(cls, x: ARRAY_LIKE, y: ARRAY_LIKE, z: ARRAY_LIKE) -> 'GeneralizedGaussian':
        r"""
        This fits a generalized (rotated) 2d gaussian function to a surface using least squares estimation.

        The fit assumes that z = f(x, y) where f is the gaussian function (and thus z is the "height" of the gaussian
        function).

        The fit performed is for a generalized 2D gaussian function of the form

        .. math::
            z = f(x, y) = A e^{\left(-\left[a(x-x_0)^2 + 2b (x-x_0)(y-y_0) + c (y-y_0)^2\right]\right)}

        The estimation in this function is performed by transforming the gaussian function into the logspace, which
        allows us to perform a true linear least squares fit without iteration but overweights the tails of the
        function.  Therefore, it is best to constrain the data you are fitting to be near the peak of the PSF to ensure
        that too much noise is not being given extra weight in the fit.

        If the fit is unsuccessful due to a rank deficient matrix or a fit of a hyperbolic surface the resulting data
        will be set to np.nan.

        :param x: The x values underlying the data the gaussian surface is to be fit to
        :param y: The y values underlying the data the gaussian surface is to be fit to
        :param z: The z or "height" values for the gaussian surface
        :return: The initialized PSF with values according to the fit
        """

        # make sure the arrays are flat and numpy arrays
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        z = np.asarray(z).ravel()

        # form the Jacobian matrix
        # we are fitting to a model of the form
        coefficients = np.vstack([x ** 2, x, x * y, y, y ** 2, np.ones(x.shape)]).T

        try:  # try to generate the least squares solution unless the system is rank deficient

            # get the solution to the least squares problem
            solution = np.linalg.lstsq(coefficients, np.log(z), rcond=None)[0]

            # extract the meaningful parts from the solution vector
            a = -solution.item(0)
            b = -solution.item(2) / 2.
            c = -solution.item(4)
            # y0 = (solution.item(3) * a - 2 * solution.item(1) * b) / (a * c - 4 * b ** 2)
            y0 = (b*solution.item(1)-a*solution.item(3))/(2*(b**2-a*c))
            # x0 = (solution.item(1) - 2 * b * y0) / a
            x0 = (solution.item(3)-2*c*y0)/(2*b)
            # amplitude = np.exp(solution.item(5) + a * x0 ** 2 - 2 * b * x0 * y0 + c * y0 ** 2)
            amplitude = np.exp(solution.item(5)+a*x0**2+2*b*x0*y0+c*y0**2)

            out = cls(a_coef=a, b_coef=b, c_coef=c, amplitude=amplitude, centroid_x=x0, centroid_y=y0)

            if cls.save_residuals:
                computed = out.evaluate(x, y)
                out._residuals = z - computed

                # compute the non-transformed jacobian to the get the formal covariance
                jacobian = out.compute_jacobian(x, y, computed)

                # assume a single uncertainty for all measurements
                out._covariance = np.linalg.pinv(jacobian.T @ jacobian) * out.residual_std ** 2

        except (np.linalg.linalg.LinAlgError, ZeroDivisionError):  # if the LHS was singular
            out = cls(a_coef=np.nan, b_coef=np.nan, c_coef=np.nan, amplitude=np.nan,
                      centroid_x=np.nan, centroid_y=np.nan)

            if cls.save_residuals:
                out._residuals = np.nan*z
                out._covariance = np.nan*np.ones((6, 6), dtype=np.float64)

        # if we fit a hyperbolic surface then return nans
        if (out.a_coef < 0) or (out.c_coef < 0):
            out.update_state(None)

        return out

    def compute_jacobian(self, x: np.ndarray, y: np.ndarray, computed: np.ndarray):
        r"""
        This method computes the Jacobian of the PSF with respect to a change in the state.

        This is used internally for computing the covariance.  It returns a 5xn matrix defined as

        .. math::
            \mathbf{J} = \left[\begin{array}{cccccc} \frac{\partial f}{\partial x_0} &
            \frac{\partial f}{\partial y_0} &
            \frac{\partial f}{\partial a} &
            \frac{\partial f}{\partial b} &
            \frac{\partial f}{\partial c} &
            \frac{\partial f}{\partial A} \end{array}\right]=\left[\begin{array}{c}
            \left(2a(x-x_0)+2b(y-y_0)\right)f(x, y) \\
            \left(2c(y-y_0)+2b(x-x_0)\right)f(x, y) \\
            -(x-x_0)^2f(x, y) \\
            -2(x-x_0)(y-y_0)f(x, y) \\
            -(y-y_0)^2f(x, y) \\
            \frac{f(x, y)}{A} \end{array}\right]^T

        :param x: The x values to evaluate the Jacobian at as a length m array
        :param y: The y values to evaluate the Jacobian at as a length m array
        :param computed: The PSF evaluated at x and y as a length m array
        :return: The Jacobian matrix as a 5xm numpy array
        """

        delta_x = x - self.centroid_x
        delta_y = y - self.centroid_y
        jacobian = np.vstack([computed * (2 * self.a_coef * delta_x + 2 * self.b_coef * delta_y),
                              computed * (2 * self.c_coef * delta_y + 2 * self.b_coef * delta_x),
                              computed * (-delta_x ** 2),
                              computed * (-2 * delta_x * delta_y),
                              computed * (-delta_y ** 2),
                              computed / self.amplitude]).T

        return jacobian

    def update_state(self, update: NONEARRAY) -> None:
        r"""
        Updates the current values based on the provided update vector.

        The provided update vector is in the order of :math:`[x_0, y_0, a, b, c, A]`.

        If the update vector is set to ``None`` then sets everything to NaN to indicate a bad fit.

        :param update: The vector of additive updates to apply
        """

        if update is None:
            self.size = np.nan
            self.sigma_x = np.nan
            self.sigma_y = np.nan
            self.theta = np.nan
            self.a_coef = np.nan
            self.b_coef = np.nan
            self.c_coef = np.nan
            self.amplitude = np.nan
            self.centroid_x = np.nan
            self.centroid_y = np.nan
        else:
            self.centroid_x += update[0]
            self.centroid_y += update[1]
            self.a_coef += update[2]
            self.b_coef += update[3]
            self.c_coef += update[4]
            self.amplitude += update[5]

    @property
    def covariance(self) -> Optional[np.ndarray]:
        r"""
        The formal covariance of the PSF parameters after fitting this PSF model to data.

        If this instance is not the result of a fit (:meth:`fit`) of if :attr:`save_residuals` is ``False`` then this
        will return None.

        The order of the state vector (and thus the covariance matrix) is :math:`[x_0, y_0, a, b, c, A]`.
        """

        return self._covariance

    def evaluate(self, x: ARRAY_LIKE, y: ARRAY_LIKE) -> np.ndarray:
        r"""
        This method evaluates the PSF at the given x and y.

        This method is not intended to be used to apply the PSF for an image (use the callable capability of the class
        instead for this).  Instead it simply computes the height of the PSF above the xy-plane at the requested
        locations.

        Specifically, this method computes

        .. math::
            z = f(x, y) = A e^{\left(-\left[a(x-x_0)^2 + 2b (x-x_0)(y-y_0) + c (y-y_0)^2\right]\right)}



        :param x: The x locations the height of the PSF is to be calculated at.
        :param y: The y locations the height of the PSF is to be calculated at.
        :return: A numpy array containing the height of the PSF at the requested locations the same shape as x and y.
        """

        # ensure things are arrays
        x = np.array(x)
        y = np.array(y)
        delta_x = x - self.centroid_x
        delta_y = y - self.centroid_y
        extras = super().evaluate(x, y)

        if extras is None:
            extras = 0

        return self.amplitude * np.exp(
            -(self.a_coef * delta_x ** 2 + 2 * self.b_coef * delta_x * delta_y + self.c_coef * delta_y ** 2)) + extras


class IterativeGeneralizedGaussian(GeneralizedGaussian, InitialGuessIterativeNonlinearLSTSQPSF):
    """
    A class for representing and fitting a generalized (rotated) 2D Gaussian point spread function using iterative
    non-linear least squares.

    This class only differs from the :class:`GeneralizedGaussian` class in the way the :meth:`fit` class method works.
    In this version, the fit is performed using iterative non-linear least squares, which is typically more accurate
    than using the logarithmic transformation to do linear least squares at the expense of more computation time.

    For more details about the model this works with, see the :class:`GeneralizedGaussian` documentation.

    This class can be used anywhere GIANT expects a point spread function
    """

    @classmethod
    def fit(cls, x: ARRAY_LIKE, y: ARRAY_LIKE, z: ARRAY_LIKE) -> __qualname__:
        r"""
        This fits a 2d gaussian function to a surface using iterative non-linear least squares estimation.

        The fit assumes that z = f(x, y) where f is the gaussian function (and thus z is the "height" of the gaussian
        function).

        The fit performed is for a 2D gaussian function of the form

        .. math::
            z = f(x, y) = A e^{\left(-\left[a(x-x_0)^2 + 2b (x-x_0)(y-y_0) + c (y-y_0)^2\right]\right)}

        The estimation in this function is performed iteratively.  First, a transformed fit is performed using the
        :meth:`.Gaussian.fit` method.  This initial fit is then refined using iterative non-linear least squares to
        remove biases that can be introduced in the transformed fit.

        If the fit is unsuccessful due to a rank deficient matrix or a fit of a hyperbolic surface the resulting data
        will be set to np.nan.

        :param x: The x values underlying the data the gaussian surface is to be fit to
        :param y: The y values underlying the data the gaussian surface is to be fit to
        :param z: The z or "height" values for the gaussian surface
        :return: The initialized PSF with values according to the fit
        """

        # do the fit using the default setup from InitialGuessIterativeNonlinearLSTSQPSF
        out = cls.fit_lstsq(x, y, z)  # type: IterativeGeneralizedGaussian

        # check if something isn't right
        if (out.a_coef < 0) or (out.b_coef < 0):
            out.update_state(None)
            return out

        # finalize things
        out._convert_to_pa()
        out.determine_size()

        return out


class IterativeGeneralizedGaussianWBackground(GeneralizedGaussian, InitialGuessIterativeNonlinearLSTSQPSFwBackground):
    r"""
    A class for representing and fitting the superposition of a standard (non-rotated) 2D Gaussian point spread function
    and a linear background gradiant using iterative non-linear least squares.

    This class differs from the :class:`GeneralizedGaussian` class in the way the :meth:`fit` class method works and in
    the fact that it adds a background gradient to the model.  In this version, the fit is performed using iterative
    non-linear least squares, which is typically more accurate than using the logarithmic transformation to do linear
    least squares at the expense of more computation time.

    The specific model that is fit is given by

    .. math::
        f(x, y) = A e^{\left(-\left[a(x-x_0)^2+b(x-x_0)(y-y_0)+c(y-y_0)^2\right]\right)} + Bx+Cy+D

    This class can be used anywhere GIANT expects a point spread function
    """

    def __init__(self, a_coef: Optional[Real] = None, b_coef: Optional[Real] = None, c_coef: Optional[Real] = None,
                 sigma_x: Optional[Real] = None, sigma_y: Optional[Real] = None, theta: Optional[Real] = None,
                 amplitude: Optional[Real] = None, centroid_x: Optional[Real] = 0, centroid_y: Optional[Real] = 0,
                 size: Optional[int] = None, bg_b_coef: Optional[float] = None, bg_c_coef: Optional[float] = None,
                 bg_d_coef: Optional[float] = None):
        """
        :param a_coef: The a coefficient of the Gaussian polynomial
        :param b_coef: The b coefficient of the Gaussian polynomial
        :param c_coef: The c coefficient of the Gaussian polynomial
        :param sigma_x: The Gaussian RMS width in the x direction in pixels
        :param sigma_y: The Gaussian RMS Width in the y direction in pixels.  If set to 0 or ``None`` this is set to be
                        the same as ``sigma_x``
        :param sigma_x: The angle between the x-axis and the principal axis of the Gaussian in radians.
        :param size: The size of the kernel to use when applying this PSF in pixels.  If set to 0 or ``None`` will be
                     computed based on the Gaussian RSM widths.
        :param amplitude: The amplitude of the gaussian kernel to use when applying this PSF.  If set to 0 or ``None``
               this will be computed so that the kernel does not increase/decrease the total signal.
        :param centroid_x: The x location of the peak of the Gaussian PSF in pixels.  This is not used when applying the
                           PSF, but it is used when fitting the PSF.  Typically this is not specified by the user.
        :param centroid_y: The y location of the peak of the Gaussian PSF in pixels.  This is not used when applying the
                           PSF, but it is used when fitting the PSF.  Typically this is not specified by the user.
        :param bg_b_coef: The x slope of the background gradient
        :param bg_c_coef: They y slope of the background gradient
        :param bg_d_coef: The constant offset of the background gradient
        """

        super().__init__(a_coef=a_coef, b_coef=b_coef, c_coef=c_coef, sigma_x=sigma_x, sigma_y=sigma_y, theta=theta,
                         size=size, amplitude=amplitude, centroid_x=centroid_x, centroid_y=centroid_y,
                         bg_b_coef=bg_b_coef, bg_c_coef=bg_c_coef, bg_d_coef=bg_d_coef)

    @classmethod
    def fit(cls, x: ARRAY_LIKE, y: ARRAY_LIKE, z: ARRAY_LIKE) -> 'IterativeGaussianWBackground':
        r"""
        This fits a 2d gaussian function to a surface using iterative non-linear least squares estimation.

        The fit assumes that z = f(x, y) where f is the gaussian function (and thus z is the "height" of the gaussian
        function).

        The fit performed is for a 2D gaussian function of the form

        .. math::
            z = f(x, y) = A e^{\left(-\left[a(x-x_0)^2 + 2b (x-x_0)(y-y_0) + c (y-y_0)^2\right]\right)}
            +Bx+Cy+D

        The estimation in this function is performed iteratively.  First, the rough background is estimated and removed.
        Then, a transformed fit is performed using the super class's fit method on the data with the rough background
        removed.  This initial fit is then refined using iterative non-linear least squares on the original data to
        remove biases that might have been introduced in the non-iterative fit.

        If the fit is unsuccessful due to a rank deficient matrix or a fit of a hyperbolic surface the resulting data
        will be set to np.nan.

        :param x: The x values underlying the surface the PSF is to be fit to
        :param y: The y values underlying the surface the PSF is to be fit to
        :param z: The z or "height" values of the surface the PSF is to be fit to
        :return: The initialized PSF with values according to the fit
        """

        out = cls.fit_lstsq(x, y, z)

        # check if something isn't right
        if (out.sigma_x < 0) or (out.sigma_y < 0):
            out.update_state(None)
            return out

        return out

    def update_state(self, update: NONEARRAY) -> None:
        r"""
        Updates the current values based on the provided update vector.

        The provided update vector is in the order of :math:`[x_0, y_0, \sigma_x, \sigma_y, A, B, C, D]`.

        If the update vector is set to ``None`` then sets everything to NaN to indicate a bad fit.

        :param update: The vector of additive updates to apply
        """

        super().update_state(update)
        if update is not None:
            self.apply_update_bg(update[-3:])
        else:
            self.apply_update_bg(update)

    def compute_jacobian(self, x: np.ndarray, y: np.ndarray, computed: np.ndarray) -> np.ndarray:
        r"""
        This method computes the Jacobian of the PSF with respect to a change in the state.

        It returns a mx8 matrix defined as

        .. math::
            \mathbf{J} = \left[\begin{array}{cccccccc} \frac{\partial f}{\partial x_0} &
            \frac{\partial f}{\partial y_0} &
            \frac{\partial f}{\partial \sigma_x} &
            \frac{\partial f}{\partial \sigma_y} &
            \frac{\partial f}{\partial A} &
            \frac{\partial f}{\partial B} &
            \frac{\partial f}{\partial C} &
            \frac{\partial f}{\partial D}\end{array}\right]=\left[\begin{array}{cccccccc}
            \frac{x-x_0}{\sigma_x^2}f(x, y) &
            \frac{y-y_0}{\sigma_y^2}f(x, y) &
            \frac{(x-x_0)^2}{\sigma_x^3}f(x, y) &
            \frac{(y-y_0)^2}{\sigma_y^3}f(x, y) &
            \frac{f(x, y)}{A} &
            x & y & 1\end{array}\right]

        :param x: The x values to evaluate the Jacobian at as a length m array
        :param y: The y values to evaluate the Jacobian at as a length m array
        :param computed: The PSF evaluated at x and y as a length m array
        :return: The Jacobian matrix as a mx8 numpy array
        """

        return self.compute_jacobian_all(x, y, computed)
