# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


r"""
Provides abstract base classes for the construction of Point Spread Function classes for GIANT.

In this module there are a number of abstract base classes (ABCs) that define the interface for and provide some common
functionality for point spread functions (PSFs) in GIANT.  In general, most users will not interact with these directly,
and will instead use a predefined implementation of a PSF, however, if you wish to define a new PSF for GIANT to use
then you will likely benefit from what is available in this module.

To define a new PSF you will most likely want to subclass at least one of the ABCs defined in this module, which will
help you to be sure you've defined all the interfaces GIANT expects and possibly add some shared functionality so that
you don't need to reinvent the wheel.  However, this is not strictly necessary.  While type checkers will complain if
you don't at least inherit from :class:`PointSpreadFunction`, GIANT will not actually error so long as you have defined
all the appropriate interfaces (so called duck typing).

Use
---

To implement a fully function custom PSF for GIANT you must at minimum implement the following methods and attributes

================================================= ======================================================================
Method/Attribute                                  Use
================================================= ======================================================================
:attr:`~PointSpreadFunction.save_residuals`       A class attribute which determines whether to save information about
                                                  the residuals from attempting to fit the PSF to data.  If ``True``
                                                  then you should save the residual statistics and formal fit
                                                  covariance.
:meth:`~PointSpreadFunction.__call__`             The built in call method for the class.  This method should apply the
                                                  defined PSF in the current instance to a 2D image, returning the image
                                                  after the PSF has been applied.
:meth:`~PointSpreadFunction.apply_1d`             An analogous method to :meth:`~PointSpreadFunction.__call__` but for
                                                  applying the PSF to (a) 1D scan line(s) instead of a 2D image.
:meth:`~PointSpreadFunction.generate_kernel`      A method that generates a square unit kernel (sums to 1) of the
                                                  current instance of the PSF.
:meth:`~PointSpreadFunction.evaluate`             A method that evaluates the current instance of the PSF at provided
                                                  x and y locations
:meth:`~PointSpreadFunction.fit`                  A class method which fits an instance of the PSF to supplied data and
                                                  returns and initialized version of the PSF with the fit parameters.
:attr:`~PointSpreadFunction.centroid`             A property which returns the location of the peak of the PSF as a
                                                  as a length 2 numpy array.
:attr:`~PointSpreadFunction.residual_rss`         A property which returns the root sum of squares of the rss of the fit
                                                  of the PSF iff the current instance was made by a call to
                                                  :meth:`~PointSpreadFunction.fit` and
                                                  :attr:`~PointSpreadFunction.save_residuals` is ``True``.  If either of
                                                  these are not True then returns ``None``.
:attr:`~PointSpreadFunction.residual_std`         A property which returns the standard deviation of the rss of the fit
                                                  of the PSF iff the current instance was made by a call to
                                                  :meth:`~PointSpreadFunction.fit` and
                                                  :attr:`~PointSpreadFunction.save_residuals` is ``True``.  If either of
                                                  these are not True then returns ``None``.
:attr:`~PointSpreadFunction.covariance`           A property which returns the formal covariance of the fit
                                                  of the PSF iff the current instance was made by a call to
                                                  :meth:`~PointSpreadFunction.fit` and
                                                  :attr:`~PointSpreadFunction.save_residuals` is ``True``.  If either of
                                                  these are not True then returns ``None``.
:attr:`~PointSpreadFunction.volume`               A method which computes the total volume under the PSF (integral from
                                                  :math:`-\inf` to :math:`\inf`)
================================================= ======================================================================

Implementing these, plus whatever else is needed internally for the functionality of the PSF, will result in a PSF class
that can be used throughout GIANT.

For examples of how this is done, refer to the pre-defined PSFs in :mod:`.gaussians`.
"""

from abc import ABCMeta, abstractmethod

from typing import Optional, Tuple

import numpy as np
from scipy.fftpack.helper import next_fast_len

import cv2

from .._typing import ARRAY_LIKE, Real, NONEARRAY


def _fft_convolve_1d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    This function performs FFT based convolution on nd arrays of 1d scan lines.

    :param a: array of 1d scan lines
    :param b: array of 1d scan lines
    :return: array of spatial correlation values
    """
    # Determine the size of the correlation surface for type "full"
    n = a.shape[-1] + b.shape[-1] - 1

    # Get the next fast fft length
    fftn = next_fast_len(n)

    # Transform the input values into the frequency domain
    a_fft = np.fft.rfft(a, n=fftn)
    b_fft = np.fft.rfft(b, n=fftn)

    # Perform the correlation and transform back to the spatial domain
    cc = np.fft.irfft(a_fft * b_fft, n=fftn)

    # extract the proper data.
    dif = (n - a.shape[-1]) // 2
    return cc[..., dif:dif + a.shape[-1]]


class PointSpreadFunction(metaclass=ABCMeta):
    """
    This abstract base class serves as the template for implementing a point spread function in GIANT.

    A point spread function models how a camera spreads out a point source of light across multiple pixels in an image.
    GIANT uses PSFs both for making rendered templates more realistic for correlation in :mod:`.relative_opnav` and for
    centroiding of stars and unresolved bodies for center finding (:mod:`.unresolved`), attitude estimation
    (:mod:`.stellar_opnav`), and camera calibration (:mod:`.calibration`).

    In general, a PSF class is assigned to the :attr:`.ImageProcessing.psf` attribute and an initialized version of the
    class is assigned to the :attr:`.Camera.psf` attribute.  GIANT will then use the specified PSF wherever it is
    needed. For more details refer to the :mod:`~giant.point_spread_functions` package documentation.

    This class serves as a prototype for implementing a PSF in GIANT.  It defines all the interfaces that GIANT expects
    for duck typing as abstract methods and properties to help you know you've implemented everything you need.

    .. note:: Because this is an ABC, you cannot create an instance of this class (it will raise a ``TypeError``)
    """

    save_residuals = False  # type: bool
    """
    This class attribute specifies whether to save the residuals when fitting the specified PSF to data.

    Saving the residuals can be important for in depth analysis but can use a lot of space when many fits are being 
    performed and stored so this defaults to off.  To store the residuals simply set this to ``True`` before 
    initialization.
    """

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Point spread functions are callable on images and should apply the stored PSF to the image and return the
        result.

        :param image: The image the PSF is to be applied to
        :return: The image after applying the PSF
        """

    @abstractmethod
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

    @abstractmethod
    def generate_kernel(self) -> np.ndarray:
        """
        Generates a square kernel centered at the centroid of the PSF normalized to have a volume (sum) of 1.

        Essentially this evaluates :math:`z = f(x, y)` for x in :math:`[x_0-size//2, x_0+size//2]` and y in
        :math:`[y_0-size//2, y_0+size//2]` where x0 is the x location of the centroid of the PSF and y0 is the y
        location of the centroid of the PSF.

        The resulting values are then normalized to sum to 1 so that the result can be applied using convolution
        without changing the overall signal level.

        :return: A normalized kernel of the PSF centered at the centroid
        """

    @abstractmethod
    def evaluate(self, x: ARRAY_LIKE, y: ARRAY_LIKE) -> np.ndarray:
        """
        This method evaluates the PSF at the given x and y.

        This method is not intended to be used to apply the PSF for an image (use the callable capability of the class
        instead for this).  Instead it simply computes the height of the PSF above the xy-plane at the requested
        locations.

        :param x: The x locations the height of the PSF is to be calculated at.
        :param y: The y locations the height of the PSF is to be calculated at.
        :return: A numpy array containing the height of the PSF at the requested locations the same shape as x and y.
        """

    @classmethod
    @abstractmethod
    def fit(cls, x: ARRAY_LIKE, y: ARRAY_LIKE, z: ARRAY_LIKE) -> __qualname__:
        """
        This function fits the defined PSF to the input data and returns an initialize version of the class based on the
        fit.

        The fit assumes that z = f(x, y) where f is the PSF (and thus z is the "height" of the PSF).

        If the fit is unsuccessful then this should set the attributes of the PSF to NaN to indicate to the rest of
        GIANT that the fit failed.

        :param x: The x values underlying the surface the PSF is to be fit to
        :param y: The y values underlying the surface the PSF is to be fit to
        :param z: The z or "height" values of the surface the PSF is to be fit to
        :return: An instance of the PSF that best fits the provided data
        """

    @property
    @abstractmethod
    def centroid(self) -> np.ndarray:
        """
        This should return the centroid or peak of the initialized PSF as a x, y length 2 numpy array.

        This property is used to enable the PSF class to be used in identifying the center of
        illumination in image processing (see :attr:`.ImageProcessing.centroiding`).
        """

    @property
    @abstractmethod
    def residual_rss(self) -> Optional[float]:
        """
        This should return residual sum of squares (RSS) of the post-fit residuals from fitting this PSF to the data.

        If the PSF is not the result of a fit or the :attr:`save_residuals` is ``False`` this will return ``None``.
        """

    @property
    @abstractmethod
    def residual_mean(self) -> Optional[float]:
        """
        This should return the mean of the post-fit residuals from fitting this PSF to the data.

        If the PSF is not the result of a fit or the :attr:`save_residuals` is ``False`` this will return ``None``.
        """

    @property
    @abstractmethod
    def residual_std(self) -> Optional[float]:
        """
        This should return the standard deviation of the post-fit residuals from fitting this PSF to the data.

        If the PSF is not the result of a fit or the :attr:`save_residuals` is ``False`` this will return ``None``.
        """

    @property
    @abstractmethod
    def covariance(self) -> Optional[np.ndarray]:
        """
        This should return the formal covariance of the PSF parameters (if the PSF was fit and not initialized).

        If the PSF is not the result of a fit or the :attr:`save_residuals` is ``False`` this will return ``None``.
        """

    @abstractmethod
    def volume(self) -> float:
        """
        This should compute the total volume contained under the PSF.

        :return: The total volume contained under the PSF
        """

    def compare(self, other: __qualname__) -> float:
        """
        For real PSFs, this method generates how well the PSF matches another between 0 and 1, with 1 being a perfect
        match and 0 being a horrible match.

        Typically this is evaluated as the clipped pearson product moment coefficient between the kernels of the 2 psfs.
        """

        return float(np.clip(np.corrcoef(self.generate_kernel().ravel(), other.generate_kernel().ravel())[0, 1], 0, 1))


class KernelBasedApply1DPSF(PointSpreadFunction, metaclass=ABCMeta):
    """
    This ABC adds concrete common functionality for applying the initialized PSF to 1D scan lines to
    :class:`.PointSpreadFunction`.

    The implementation that is shared by most PSFs for 1D scan lines is stored in :meth:`apply_1d_sized`.  This method,
    which isn't part of the actual interface GIANT expects, is used for applying the specified PSF to 1D scan lines
    if the size of the required kernel is known.  Therefore, when implementing method:`apply_1d`, all you need to do is
    calculate the required size of the 1D kernel and then dispatch to :meth:`apply_1d_sized`.  An example of this can
    be seen in :class:`.Gaussian`.
    """

    def apply_1d_sized(self, image_1d: np.ndarray, size: int,
                       direction: Optional[np.ndarray] = None, step: Real = 1) -> np.ndarray:
        """
        Applies the defined PSF using the stored parameters to the 1D image scans provided with a given kernel size.

        ``image_1d`` can be a 2D array but in that case each row will be treated as an independent 1D scan.

        For non-symmetric PSFs, a ``direction`` argument can be supplied which should be the direction in the image of
        each scan line.  This can be used to determine the appropriate cross-section of the PSF to use for applying to
        the 1D scans (if applicable).  If no direction is provided then the x direction [1, 0] is assumed.

        This method works by sampling the PSF in the (optionally) specified direction(s) centered around the centroid of
        the PSF according to the input ``size``. These kernels are then applied to the input scan lines using a Fourier
        transform, and the resulting scan lines are returned.

        :param image_1d: The scan line(s) to be blurred using the PSF
        :param size: The size of the kernel to use when convolving the PSF with the scan line
        :param direction: The direction for the 1D cross section of the PSF.  This should be either None, a length 2
                          array, or a shape nx2 array where n is the number of scan lines
        :param step: The step size of the lines being blurred.
        :return: an array containing the input after blurring with the defined PSF
        """

        # make sure direction is in the right format/shape
        if direction is None:
            direction = np.array([[1, 0]])
        else:
            direction = np.array(direction)

            if (direction.shape[1] != 2) and (direction.shape[0] == 2):
                direction = direction.T

        # get everything the right shapes
        image_1d = np.atleast_2d(image_1d)
        number_rows = max(image_1d.shape[0], direction.shape[0])
        lines = np.broadcast_to(image_1d, (number_rows, image_1d.shape[1]))
        directions = np.broadcast_to(direction, (number_rows, direction.shape[1]))

        # determine the number of steps we need to take
        steps = np.arange(-size / 2, size / 2 + step, step)

        # get the lines we are querying along (centered at the centroid)
        queries = directions.reshape((-1, 1, 2)) * steps.reshape((1, -1, 1)) + self.centroid.reshape((1, 1, 2))

        kx = queries[..., 0]
        ky = queries[..., 1]

        # compute and normalize the kernels
        kernels = self.evaluate(kx, ky)
        kernels /= kernels.sum(axis=1, keepdims=True)

        # apply the kernels
        return _fft_convolve_1d(lines, kernels)


class KernelBasedCallPSF(PointSpreadFunction, metaclass=ABCMeta):
    """
    This ABC adds concrete common functionality for applying the initialized PSF to 2D images to
    :class:`.PointSpreadFunction`.

    The implementation that is shared by most PSFs for 2D images is stored in :meth:`__call__`.  This method,
    works by generating a square kernel of the PSF by a call to :meth:`generate_kernel` and then convolving the kernel
    with the image.  For most PSFs, this form will be used, although a few like :class:`Gaussian` may have a further
    optimized call sequence.
    """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        This method generates a kernel and then convolves it with the input image.

        The kernel is generated by a call the :meth:`generate_kernel`.  The kernel is then applied with border
        replication to the image either using fourier transforms or a spatial algorithm, whichever is faster.

        :param image: The image the PSF is to be applied to
        :return: The image after applying the PSF
        """

        # generate the kernel
        kernel = self.generate_kernel()

        # apply it through filtering
        return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)


class SizedPSF(PointSpreadFunction, metaclass=ABCMeta):
    """
    This ABC adds common functionality for a PSF where the required size can be determine algorithmically.

    Specifically, this adds an instance attribute :attr:`size` which stores the size of the PSF,
    a new abstract method :meth:`determine_size` which should be implemented to algorithmically determine the size of
    the kernel required for the PSF, and concrete method :meth:`generate_kernel`, which generates a square unit kernel
    based on the :attr:`size`.
    """

    def __init__(self, size: Optional[int] = None, **kwargs):
        """
        :param size: The size of the kernel to generate.
        """

        self.size = 1  # type: int
        """
        The size of the kernel to return on a call to :meth:`generate_kernel`.
        
        Typically this should be an odd number to ensure that the kernel is square and centered.
        """

        if (size is None) or size == 0:
            self.determine_size()
        else:
            self.size = int(size)

        super().__init__(**kwargs)

    @abstractmethod
    def determine_size(self) -> None:
        """
        Sets the size required for the kernel algorithmically.

        Typically this is based on the width of the PSF.

        The determined size should be stored in the instance attribute :attr:`size`
        """

        pass

    def generate_kernel(self, size: Optional[int] = None):
        r"""
        Generates a square kernel centered at the centroid of the PSF normalized to have a volume (sum) of 1 for the
        size input or specified in the :attr:`size` attribute.

        Essentially this evaluates :math:`z = f(x, y)` for x in :math:`[x_0-size//2, x_0+size//2]` and y in
        :math:`[y_0-size//2, y_0+size//2]` where x0 is the x location of the centroid of the PSF and y0 is the y
        location of the centroid of the PSF.

        The resulting values are then normalized to sum to 1 so that the result can be applied using convolution
        without changing the overall signal level.

        :param size: The size of the kernel to generate (ie return a (size, size) shaped array).  Overrides the
                     :attr:`size` attribute.
        :return: A normalized kernel of the PSF centered at the centroid
        """

        if size is None:
            size = self.size

        grid_x, grid_y = np.meshgrid(np.arange(self.centroid[0] - size // 2, self.centroid[0] + size // 2 + 1, 1),
                                     np.arange(self.centroid[1] - size // 2, self.centroid[1] + size // 2 + 1, 1))

        # just ensure we didn't end up with too many
        grid_x = grid_x[:size, :size]
        grid_y = grid_y[:size, :size]

        kernel = self.evaluate(grid_x, grid_y)
        kernel /= kernel.sum()

        return kernel


    def compare(self, other: __qualname__) -> float:
        """
        For real PSFs, this method generates how well the PSF matches another between 0 and 1, with 1 being a perfect
        match and 0 being a horrible match.

        Typically this is evaluated as the clipped pearson product moment coefficient between the kernels of the 2 psfs.
        """

        size = min(max(self.size, other.size), 10)

        return float(np.clip(np.corrcoef(self.generate_kernel(size=size).ravel(),
                                         other.generate_kernel(size=size).ravel())[0, 1], 0, 1))


class IterativeNonlinearLSTSQPSF(PointSpreadFunction, metaclass=ABCMeta):
    """
    This ABC defines common attributes, properties, and methods for Iterative Non-linear least squares estimation of
    a Point Spread function.

    This class is typically not used by the user except when implementing a new PSF class that uses iterative nonlinear
    least squares to fit the PSF to data.

    To use this class when implementing a new PSF, simply subclass it and then override the abstract methods
    :meth:`compute_jacobian` and :meth:`update_state` (in addition to the required abstract methods from the typical
    :class:`PointSpreadFunction` ABC) according to the PSF you are implementing.  You may also want to override the
    default class attributes :attr:`max_iter`, :attr:`atol`, and :attr:`rtol`, which control when to break out of the
    iterations.

    Once you have overridden the abstract methods (and possibly the class attributes), you simply need to call the
    :meth:`converge` method from somewhere within the :meth:`~PointSpreadFunction.fit` method after initializing the
    class with the initial guess of the PSF parameters.  The :meth:`converge` method will then perform iterative
    non-linear least squares until convergence or the maximum number of iterations have been performed according the the
    :attr:`max_iter`, :attr:`atol`, and :attr:`rtol` class attributes.  The converged solution will be stored as the
    updated class parameters
    """

    max_iter = 20  # type: int
    """
    An integer defining the maximum number of iterations to attempt in the iterative least squares solution.
    """

    atol = 1e-10  # type: float
    """
    The absolute tolerance cut-off for the iterative least squares. (The iteration will cease when the new estimate is 
    within this tolerance for every element from the previous estimate)
    """

    rtol = 1e-10  # type: float
    """
    The relative tolerance cut-off for the iterative least squares. (The iteration will cease when the maximum percent 
    change in the state vector from one iteration to the next is less than this value)
    """

    @abstractmethod
    def compute_jacobian(self, x: np.ndarray, y: np.ndarray, computed: np.ndarray) -> np.ndarray:
        r"""
        This method computes the Jacobian of the PSF with respect to a change in the state.

        Mathematically, it should return the nxm matrix

        .. math::
            \mathbf{J} = \frac{\partial f(x, y)}{\partial \mathbf{t}}

        where :math:`f(x,y)` is the function being fit, :math:`\mathbf{t}` is a length m vector of the state parameters,
        and :math:`\mathbf{J}` is the Jacobian matrix

        :param x: The x values to evaluate the Jacobian at as a length n array
        :param y: The y values to evaluate the Jacobian at as a length n array
        :param computed: :math:`f(x,y)` evaluated at x and y as a length n array.
                         This is provided for efficiency and convenience as the evaluated function is frequently needed
                         in the computation of the Jacobian and it is definitely needed in the non-linear least squares.
                         If not needed for computing the Jacobian this can safely be ignored.
        :return: The Jacobian matrix as a nxm numpy array, with n being the number of measurements and m being the
                 number of state parameters being estimated
        """

        pass

    @abstractmethod
    def update_state(self, update: NONEARRAY) -> None:
        """
        Updates the current values based on the provided update vector.

        The provided update vector is in the order according to order of the columns returned from
        :meth:`compute_jacobian`.

        If the input is ``None`` then this method should set the state parameters to NaN to indicate to the rest of
        GIANT that the estimation failed.

        :param update: The vector of additive updates to apply or None to indicate that the fit failed
        """

        pass

    def converge(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[NONEARRAY, NONEARRAY]:
        """
        Performs iterative non-linear least squares on a PSF model until convergence has been reached for a function of
        the form :math:`z=f(x, y)`

        Iterative non-linear least squares is performed by linearizing at each step about the current best estimate of
        the state.  This means that for each iteration, the Jacobian matrix is computed based off the current best
        estimate of the state, and then used to form a linear approximation of the model using a Taylor expansion.
        The resulting estimate is a delta from the current state, so that it is typically applied by adding the
        resulting update state vector to the existing states (although in some instances such as for rotations, a more
        complicated update application may be needed.

        Iterative non-linear least squares typically needs an adequate initial guess to ensure convergence, therefore,
        it is recommended that the state of the class be appropriately initialized before calling this method (what is
        appropriate is dependent on the PSF itself.

        The iteration performed in this method can be controlled using the :attr:`max_iter`, :attr:`atol`, and
        :attr:`rtol` class attributes which control the maximum number of iterations to attempt for convergence, the
        absolute tolerance criteria for convergence, and the relative tolerance criteria for convergence respectively.

        This method use :meth:`compute_jacobian` to return the Jacobian matrix for the current estimate of the state
        vector and :meth:`update_state` to apply the estimated update at each iteration step, therefore, these methods
        should expect the same order of state elements.

        If the iteration diverges then this method will call :meth:`update_state` with ``None`` as the argument, which
        should typically indicate that the state parameters should be set to NaN so that other GIANT algorithms are
        aware the PSF fit failed.

        If :attr:`~PointSpreadFunction.save_residuals` is set to True, then this function will return a vector of the
        residuals and the covariance matrix from the fit as numpy arrays.  Otherwise it returns None, None.

        :param x: The x locations of the expected values as a 1D array
        :param y: The y locations of the expected values as a 1D array
        :param z: The expected values to fit to as a 1D array
        :return: Either (residuals, covariance) as (n,) and (m,m) arrays if :attr:`save_residuals` is ``True`` or
                 ``(None, None)``.
        """
        computed = self.evaluate(x, y)
        residuals = z - computed

        residual_norm_old = np.inf
        jacobian = self.compute_jacobian(x, y, computed)

        for ind in range(self.max_iter):
            # break self if the jacobian is bad
            if not np.isfinite(jacobian).all():
                self.update_state(None)
                return None, None

            # compute the update
            update = np.linalg.lstsq(jacobian, residuals, rcond=None)[0]

            # update the state
            self.update_state(update)

            # compute the residuals and jacobian after the update
            computed = self.evaluate(x, y)
            residuals = z - computed

            residual_norm_new = np.linalg.norm(residuals)
            jacobian = self.compute_jacobian(x, y, computed)

            # check for convergence/divergence
            if (np.abs(update) <= self.atol).all():
                break

            elif abs(residual_norm_new - residual_norm_old)/residual_norm_new < self.rtol:
                break

            elif residual_norm_old < residual_norm_new:
                self.update_state(None)
                if self.save_residuals:

                    return residuals, np.nan*np.ones((jacobian.shape[1], jacobian.shape[1]), dtype=np.float64)

                else:
                    return None, None

            else:
                residual_norm_old = residual_norm_new

        if self.save_residuals:

            try:
                return residuals, np.linalg.pinv(jacobian.T @ jacobian) * residuals.var()
            except np.linalg.linalg.LinAlgError:
                return residuals, np.zeros((jacobian.shape[1], jacobian.shape[1]), dtype=float)*np.nan

        else:
            return None, None


class IterativeNonlinearLSTSQwBackground(IterativeNonlinearLSTSQPSF, metaclass=ABCMeta):
    r"""
    This class provides support for estimating the superposition of the PSF and a linear background gradient.

    This class is typically not used by the user except when implementing a new PSF class that uses iterative nonlinear
    least squares to fit the PSF to data.

    Beyond the typical implementation in :class:`IterativeNonLinearLSTSQ`, this class provides a concrete
    implementation of methods :meth:`compute_jacobian_bg`, :meth:`evaluate_bg`, and :meth:`apply_update_bg` which
    handle the linear background gradient of the form

    .. math::
        f_{bg}(x, y) = f(x,y)+Bx+Cy+D

    where :math:`f_{bg}(x,y)` is the PSF with the background, :math:`f(x,y)` is the PSF without the background,
    :math:`B` is the slope of the gradient in the x direction, :math:`C` is the slope of the gradient in the y direction
    and :math:`D` is the constant background level.

    The way this class should be used it to subclass it, then in the regular :meth:`compute_jacobian` method call
    :meth:`compute_jacobian_bg` and include with the rest of your Jacobian matrix (typically at the end),
    then in :meth:`evaluate` call :meth:`evaluate_bg` and add the results to the PSF, and finally in
    :meth:`update_states` call :meth:`update_states_bg` inputting the portion of the state vector that contains the
    background update according to where you added it to your existing Jacobian.

    The background terms are stored in instance attributes :attr:`bg_b_coef`, :attr:`bg_c_coef`, and :attr:`bg_d_coef`.
    """

    def __init__(self, bg_b_coef: Optional[Real] = None, bg_c_coef: Optional[Real] = None,
                 bg_d_coef: Optional[Real] = None, **kwargs):
        """
        :param bg_b_coef: The x slope of the background gradient
        :param bg_c_coef: They y slope of the background gradient
        :param bg_d_coef: The constant offset of the background gradient
        """
        self.bg_b_coef = 0.0   # type: float
        """
        The x slope of the background gradient
        """

        if bg_b_coef is not None:
            self.bg_b_coef = float(bg_b_coef)

        self.bg_c_coef = 0.0  # type: float
        """
        The y slope of the background gradient
        """

        if bg_c_coef is not None:
            self.bg_c_coef = float(bg_c_coef)

        self.bg_d_coef = 0.0  # type: float
        """
        The constant offset of the background gradient
        """

        if bg_d_coef is not None:
            self.bg_d_coef = float(bg_d_coef)

        super().__init__(**kwargs)

    @staticmethod
    def compute_jacobian_bg(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""
        This computes the Jacobian matrix for the background terms.

        Mathematically this is

        .. math::
            \mathbf{J}_{bg} = \left[\begin{array}{ccc}\frac{\partial f_{bg}(x,y)}{\partial B} &
            \frac{\partial f_{bg}(x,y)}{\partial C} & \frac{\partial f_{bg}(x,y)}{\partial D}\end{array}\right]=
            \left[\begin{array}{ccc} x & y & 1\end{array}\right]

        The results from this function should be appended to the rest of the Jacobian matrix using ``hstack``.

        :param x: The x values underlying the data the surface is to be fit to
        :param y: The y values underlying the data the surface is to be fit to
        :return: The Jacobian for the background as a nx3 numpy array
        """

        return np.hstack([x.reshape(-1, 1), y.reshape(-1, 1), np.ones((x.size, 1), dtype=np.float64)])

    def apply_update_bg(self, bg_update: NONEARRAY) -> None:
        """
        This applies the background update to the background state

        This typically should be called from the regular :meth:`~IterativeNonlinearLSTSQPSF.update_state` and only fed
        the components of the update vector that correspond to the background Jacobian matrix.

        :param bg_update: The update to apply to the background terms as a length 3 array
        """

        if bg_update is not None:
            self.bg_b_coef += bg_update[0]
            self.bg_c_coef += bg_update[1]
            self.bg_d_coef += bg_update[2]
        else:
            self.bg_b_coef = np.nan
            self.bg_c_coef = np.nan
            self.bg_d_coef = np.nan

    def evaluate_bg(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        This computes the background component at locations ``x`` and ``y``.

        The background component is defined as

        .. math::
            Bx+Cy+D

        :param x: The x values where the background is to be computed at
        :param y: The y values where the background is to be computed at
        :return: The background according to the model
        """

        return self.bg_b_coef*x+self.bg_c_coef*y+self.bg_d_coef

    @classmethod
    def fit_bg(cls, x: ARRAY_LIKE, y: ARRAY_LIKE, z: ARRAY_LIKE) -> __qualname__:
        """
        This method tries to fit the background using linear least squares without worrying about any PSF included.

        This is useful if you need to subtract off a rough estimate of the background before attempting to fit the PSF
        for an initial guess.  The results of the fit are stored in the :attr:`bg_b_coef`, :attr:`bg_c_coef`, and
        :attr:`bg_d_coef`.

        :param x: The x values underlying the data the background is to be fit to
        :param y: The y values underlying the data the background is to be fit to
        :param z: The z or "height" values for the background
        :return: The initialized BG PSF with values according to the fit for the background only
        """

        # make numpy
        x = np.array(x).ravel()
        y = np.array(y).ravel()
        z = np.array(z).ravel()

        # initialize the class
        out = cls()

        # get the Jacobian matrix
        jac = cls.compute_jacobian_bg(x, y)

        # do the least squares estimation
        update = np.linalg.lstsq(jac, z, rcond=None)[0]

        # store the results
        out.apply_update_bg(update)

        return out


class InitialGuessIterativeNonlinearLSTSQPSF(IterativeNonlinearLSTSQPSF, metaclass=ABCMeta):
    """
    This class provides a fit class method which generates the initial guess from a subclass and then converges to a
    better solution using iterative Nonlinear LSTSQ.

    This class is designed to work where you have a non-iterative but biased class for estimating the defined PSF (as
    is done with Gaussian PSFs by using a logarithmic transformation).  If that is the case, and the unbiased estimator
    class uses the same attributes and the biased estimator class, then you can use this as is to add the ability to get
    the biased estimate and then correct it.  Otherwise you will need to do things yourself and shouldn't bother with
    this class.

    To use this class, override the :meth:`~PointSpreadFunction.fit` method, and then call
    ``super().fit_lstsq(x, y, z)``

    This also adds 2 instance attributes :attr:`_residuals` and :attr:`_covariance` which store the covariance and
    residuals of the fit if requested.
    """

    def __init__(self):

        super().__init__()

        self._covariance = None  # type: NONEARRAY
        """
        The covariance of the fit as a nxn array (for n state elements) or None, depending on if
        :attr:`~PointSpreadFunction.save_residuals` is ``True``.
        """

        self._residuals = None  # type: NONEARRAY
        """
        The residuals of the fit as a length m array (for m observations) or None, depending on if
        :attr:`~PointSpreadFunction.save_residuals` is ``True``.
        """

    @classmethod
    def fit_lstsq(cls, x: ARRAY_LIKE, y: ARRAY_LIKE, z: ARRAY_LIKE) -> __qualname__:
        """
        This fits a PSF to a surface using iterative non-linear least squares estimation.

        The estimation in this function is performed iteratively.  First, a non-iterative fit is performed using the
        super class's fit method.  This initial fit is then refined using iterative non-linear least squares to
        remove biases that might have been introduced in the non-iterative fit..

        If the fit is unsuccessful due to a rank deficient matrix then :meth:`~IterativeNonlinearLSTSQPSF.update_states`
        will be called which will likely result in the state parameters being set to NaN.

        :param x: The x values underlying the surface the PSF is to be fit to
        :param y: The y values underlying the surface the PSF is to be fit to
        :param z: The z or "height" values of the surface the PSF is to be fit to
        :return: The initialized PSF with values according to the fit
        """

        # make sure the arrays are flat and numpy arrays
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        if (z.ndim == 2) & (max(z.shape) > 5):
            # do the initial fit only using the 9 pixels closest to the center
            pix_center = np.array(z.shape) // 2

            r_slice = slice(pix_center[0] - 1, pix_center[0] + 2)
            c_slice = slice(pix_center[1] - 1, pix_center[1] + 2)

            out = super(cls, cls).fit(x[r_slice, c_slice], y[r_slice, c_slice], z[r_slice, c_slice])
        else:
            out = super(cls, cls).fit(x, y, z)

        # make things flat
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()

        # converge to the better solution and capture the residuals, maybe
        out._residuals, out._covariance = out.converge(x, y, z)

        return out


class InitialGuessIterativeNonlinearLSTSQPSFwBackground(IterativeNonlinearLSTSQwBackground, metaclass=ABCMeta):
    """
    This class provides a fit class method which generates the initial guess from a subclass and then converges to a
    better solution using iterative Nonlinear LSTSQ including a background gradient.

    This class is designed to work where you have a non-iterative but biased class for estimating the defined PSF (as
    is done with Gaussian PSFs by using a logarithmic transformation).  If that is the case, and the unbiased estimator
    class uses the same attributes and the biased estimator class, then you can use this as is to add the ability to get
    the biased estimate and then correct it along with the background gradient.  Otherwise you will need to do things
    yourself and shouldn't bother with this class.

    To use this class, override the :meth:`~PointSpreadFunction.fit` method, and then call
    ``super().fit_lstsq(x, y, z)``

    This also adds 2 instance attributes :attr:`_residuals` and :attr:`_covariance` which store the covariance and
    residuals of the fit if requested.
    """

    def __init__(self, bg_b_coef: Optional[Real] = None, bg_c_coef: Optional[Real] = None,
                 bg_d_coef: Optional[Real] = None, **kwargs):
        """
        :param bg_b_coef: The x slope of the background gradient
        :param bg_c_coef: They y slope of the background gradient
        :param bg_d_coef: The constant offset of the background gradient
        """

        super().__init__(bg_b_coef, bg_c_coef, bg_d_coef, **kwargs)

        self._covariance = None  # type: NONEARRAY
        """
        The covariance of the fit as a nxn array (for n state elements) or None, depending on if
        :attr:`~PointSpreadFunction.save_residuals` is ``True``.
        """

        self._residuals = None  # type: NONEARRAY
        """
        The residuals of the fit as a length m array (for m observations) or None, depending on if
        :attr:`~PointSpreadFunction.save_residuals` is ``True``.
        """

    @classmethod
    def fit_lstsq(cls, x: ARRAY_LIKE, y: ARRAY_LIKE, z: ARRAY_LIKE) -> __qualname__:
        """
        This fits a PSF to a surface using iterative non-linear least squares estimation.

        The estimation in this function is performed iteratively.  First, the rough background is estimated and removed.
        Then, a non-iterative fit is performed using the super class's fit method on the data with the rough background
        removed.  This initial fit is then refined using iterative non-linear least squares to
        remove biases that might have been introduced in the non-iterative fit.

        If the fit is unsuccessful due to a rank deficient matrix then
        :meth:`~.IterativeNonlinearLSTSQwBackground.update_states` will be called which will likely result in the state
        parameters being set to NaN.

        :param x: The x values underlying the surface the PSF is to be fit to
        :param y: The y values underlying the surface the PSF is to be fit to
        :param z: The z or "height" values of the surface the PSF is to be fit to
        :return: The initialized PSF with values according to the fit
        """

        # make sure the arrays are flat and numpy arrays
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        # fit just the background
        bg = cls.fit_bg(x, y, z)

        # subtract off the rough background
        z_no_bg = z - bg.evaluate_bg(x, y).reshape(z.shape)

        if (z.ndim == 2) & (max(z.shape) > 5):
            # do the initial fit only using the 9 pixels closest to the center
            pix_center = np.array(z.shape) // 2

            r_slice = slice(pix_center[0] - 1, pix_center[0] + 2)
            c_slice = slice(pix_center[1] - 1, pix_center[1] + 2)

            use_z = z_no_bg[r_slice, c_slice]

            out = super(cls, cls).fit(x[r_slice, c_slice], y[r_slice, c_slice], use_z-use_z.min()+1)
        else:
            out = super(cls, cls).fit(x, y, z_no_bg)

        # noinspection PyArgumentList
        # at this point we assume that cls is fully functional and takes the same arguments as the attributes of
        # its super class
        out.bg_b_coef = bg.bg_b_coef
        out.bg_c_coef = bg.bg_c_coef
        out.bg_d_coef = bg.bg_d_coef

        # make things flat
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()

        # converge to the better solution and capture the residuals, maybe
        out._residuals, out._covariance = out.converge(x, y, z)

        return out

    def compute_jacobian_all(self, x: np.ndarray, y: np.ndarray, computed: np.ndarray) -> np.ndarray:
        r"""
        This method computes the Jacobian of the PSF with respect to a change in the state.

        Mathematically, it should return the nxm matrix

        .. math::
            \mathbf{J} = \frac{\partial f(x, y)}{\partial \mathbf{t}}

        where :math:`f(x,y)` is the function being fit, :math:`\mathbf{t}` is a length m vector of the state parameters,
        and :math:`\mathbf{J}` is the Jacobian matrix.  This specific implementation appends the background Jacobian to
        the normal PSF Jacobian for estimating background terms.

        :param x: The x values to evaluate the Jacobian at as a length n array
        :param y: The y values to evaluate the Jacobian at as a length n array
        :param computed: :math:`f(x,y)` evaluated at x and y as a length n array.
                         This is provided for efficiency and convenience as the evaluated function is frequently needed
                         in the computation of the Jacobian and it is definitely needed in the non-linear least squares.
                         If not needed for computing the Jacobian this can safely be ignored.
        :return: The Jacobian matrix as a nxm numpy array, with n being the number of measurements and m being the
                 number of state parameters being estimated
        """

        computed_no_bg = computed - self.evaluate_bg(x, y)

        return np.hstack([super(self.__class__, self).compute_jacobian(x, y, computed_no_bg),
                          self.compute_jacobian_bg(x, y)])

    def evaluate(self, x: ARRAY_LIKE, y: ARRAY_LIKE) -> np.ndarray:
        """
        This method evaluates the PSF at the given x and y.

        This method is not intended to be used to apply the PSF for an image (use the callable capability of the class
        instead for this).  Instead it simply computes the height of the PSF above the xy-plane at the requested
        locations.

        :param x: The x locations the height of the PSF is to be calculated at.
        :param y: The y locations the height of the PSF is to be calculated at.
        :return: A numpy array containing the height of the PSF at the requested locations the same shape as x and y.
        """

        extras = super().evaluate(x, y)

        if extras is None:
            extras = 0

        return self.evaluate_bg(x, y) + extras
