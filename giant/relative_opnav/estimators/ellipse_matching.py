# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


r"""
This module provides the capability to locate the relative position of a regular target body (well modelled by a
triaxial ellipsoid) by matching the observed ellipse of the limb in an image with the ellipsoid model of the target.

Description of the Technique
----------------------------

Ellipse matching is a form of OpNav which produces a full 3DOF relative position measurement between the target and the
camera.  Conceptually, it does this by comparing the observed size of a target in an image to the known size of the
target in 3D space to determine the range, and fits an ellipse to the observed target to locate the center in the image.
As such, this can be a very powerful measurement because it is insensitive to errors in the a priori knowledge of your
range to the target, unlike cross correlation, provides more information than just the bearing to the target
for processing in a filter, and is more computationally efficient.  That being said, the line-of-sight/bearing component
of the estimate is generally slightly less accurate than cross correlation (when there is good a priori knowledge of the
shape and the range to the target). This is because ellipse matching only makes use of the visible limb, while cross
correlation makes use of all of the visible target.

While conceptually the ellipse matching algorithm computes both a bearing and a range measurement, in actuality, a
single 3DOF position estimate is computed in a least squares sense, not 2 separate measurements.  The steps to extract
this measurement are:

#. Identify the observed illuminated limb of the target in the image being processed using
   :meth:`.ImageProcessing.identify_subpixel_limbs`
#. Solve the least squares problem

   .. math::
       \left[\begin{array}{c}\bar{\mathbf{s}}'^T_1 \\ \vdots \\ \bar{\mathbf{s}}'^T_m\end{array}\right]
       \mathbf{n}=\mathbf{1}_{m\times 1}

   where :math:`\bar{\mathbf{s}}'_i=\mathbf{B}\mathbf{s}_i`,  :math:`\mathbf{s}_i`, is a unit vector in the camera frame
   through an observed limb point in an image (computed using :meth:`~.CameraModel.pixels_to_unit`),
   :math:`\mathbf{B}=\mathbf{Q}\mathbf{T}^C_P`, :math:`\mathbf{Q}=\text{diag}(1/a, 1/b, 1/c)`, :math:`a-c` are the size
   of the principal axes of the tri-axial ellipsoid representing the target, and :math:`\mathbf{T}^C_P` is the rotation
   matrix from the principal frame of the target shape to the camera frame.

#. Compute the position of the target in the camera frame using

    .. math::
        \mathbf{r}=-(\mathbf{n}^T\mathbf{n}-1)^{-0.5}\mathbf{T}_C^P\mathbf{Q}^{-1}\mathbf{n}

    where :math:`\mathbf{r}` is the position of the target in camera frame, :math:`\mathbf{T}_C^P` is the rotation from
    the principal frame of the target ellipsoid to the camera frame, and all else is as defined previously.

Further details on the algorithm can be found `here <https://arc.aiaa.org/doi/full/10.2514/1.G000708>`_.

.. note::

    This implements limb based OpNav for regular bodies.  For irregular bodies, like asteroids and comets, see
    :mod:`.limb_matching`.

Typically this technique is used once the body is fully resolved in the image (around at least 50 pixels in apparent
diameter) and then can be used as long as the limb is visible in the image.

Tuning
------

There are a few parameters to tune for this method.  The main thing that may make a difference is the choice and tuning
for the limb extraction routines.  There are 2 categories of routines you can choose from.  The first is image
processing, where the limbs are extracted using only the image and the sun direction.  To tune the image processing limb
extraction routines you can adjust the following :class:`.ImageProcessing` settings:

========================================= ==============================================================================
Parameter                                 Description
========================================= ==============================================================================
:attr:`.ImageProcessing.denoise_flag`     A flag specifying to apply :meth:`~.ImageProcessing.denoise_image` to the
                                          image before attempting to locate the limbs.
:attr:`.ImageProcessing.image_denoising`  The routine to use to attempt to denoise the image
:attr:`.ImageProcessing.subpixel_method`  The subpixel method to use to refine the limb points.
========================================= ==============================================================================

Other tunings are specific to the subpixel method chosen and are discussed in :mod:`.image_processing`.

The other option for limb extraction is limb scanning.  In limb scanning predicted illumination values based on the
shape model and a prior state are correlated with extracted scan lines to locate the limbs in the image.  This technique
can be quite accurate (if the shape model is accurate) but is typically much slower and the extraction must be repeated
each iteration.  The general tunings to use for limb scanning are from the :class:`.LimbScanner` class:

============================================ ===========================================================================
Parameter                                    Description
============================================ ===========================================================================
:attr:`.LimbScanner.number_of_scan_lines`    The number of limb points to extract from the image
:attr:`.LimbScanner.scan_range`              The extent of the limb to use centered on the sun line in radians (should
                                             be <= np.pi/2)
:attr:`.LimbScanner.number_of_sample_points` The number of samples to take along each scan line
============================================ ===========================================================================

There are a few other things that can be tuned but they generally have limited effect.  See the :class:`.LimbScanner`
class for more details.

In addition, there is one knob that can be tweaked on the class itself.

========================================== =============================================================================
Parameter                                  Description
========================================== =============================================================================
:attr:`.LimbMatching.extraction_method`    Chooses the limb extraction method to be image processing or limb scanning.
========================================== =============================================================================

Beyond this, you only need to ensure that you have a fairly accurate ellipsoid model of the target, the knowledge of the
sun direction in the image frame is good, and the knowledge of the rotation between the principal frame and the camera
frame is good.

Use
---

The class provided in this module is usually not used by the user directly, instead it is usually interfaced with
through the :class:`.RelativeOpNav` class using the identifier :attr:`~.RelativeOpNav.ellipse_matching`.  For more
details on using the :class:`.RelativeOpNav` interface, please refer to the :mod:`.relnav_class` documentation.  For
more details on using the technique class directly, as well as a description of the ``details`` dictionaries produced
by this technique, refer to the following class documentation.
"""

import warnings

from enum import Enum

from typing import List, Optional, Union, Tuple, Callable

import numpy as np

from scipy.interpolate import RegularGridInterpolator, interp1d

from giant.relative_opnav.estimators.estimator_interface_abc import RelNavEstimator, RelNavObservablesType
from giant.relative_opnav.estimators.moment_algorithm import MomentAlgorithm

from giant.ray_tracer.shapes import Ellipsoid
from giant.ray_tracer.rays import Rays
from giant.ray_tracer.illumination import IlluminationModel, McEwenIllumination

from giant.image_processing import ImageProcessing, parabolic_peak_finder_1d, fft_correlator_1d
from giant.camera import Camera
from giant.image import OpNavImage
from giant.ray_tracer.scene import Scene, SceneObject
from giant.point_spread_functions import PointSpreadFunction
from giant._typing import NONEARRAY, Real


class LimbExtractionMethods(Enum):
    """
    This enumeration provides the valid options for the limb extraction methods that can be used on the image.
    """

    LIMB_SCANNING = "LIMB_SCANNING"
    """
    Extract limbs from the image through 1D cross correlation of predicted and observed intensity profiles along scan 
    vectors.

    This method relies on the a priori knowledge of the state vector therefore the limbs are re-extracted after each 
    iteration.
    """

    EDGE_DETECTION = "EDGE_DETECTION"
    """
    Extract limbs from the image using edge detection image processing techniques.  

    Because this does not rely on the a priori knowledge of the state vector and only considers the image and the sun 
    direction in the image, this is only performed once.  The specific edge detection technique and other parameters can
    be set in the :class:`.ImageProcessing` class.  

    The edges are extracted using the :meth:`.ImageProcessing.identify_subpixel_limbs` method.
    """


class LimbScanner:
    """
    This class is used to extract limbs from an image and pair them to surface points on the target.

    This is done by first determining the surface points on the limb based on the shape model, the scan center vector,
    and the sun direction vector.  Once these surface points have been identified (using :meth:.Shape.find_limbs`) they
    are projected onto the image to generate the predicted limb locations in the image.  Then the image is sampled
    along the scan line through each predicted limb location and the scan center location in the image using the
    ``image_interpolator`` input to get the observed intensity line.  In addition, the scan line is rendered using
    ray tracing to generate the predicted intensity line.  The predicted intensity lines and the extracted intensity
    lines are then compared using cross correlation to find the shift that best aligns them.  This shift is then applied
    to the predicted limb locations in the image along the scan line to get the extracted limb location in the image.
    This is all handled by the :meth:`extract_limbs` method.

    There are a few tuning options for this class.  The first collection affects the scan lines that are used to extract
    the limb locations from the image.  The :attr:`number_of_scan_lines` sets the number of generated scan lines and
    directly corresponds to the number of limb points that will be extracted from the image.  In addition,
    the :attr:`scan_range` attribute sets the angular extent about the sun direction vector that these scan lines will
    be evenly distributed. Finally, the :attr:`number_of_sample_points` specifies how many samples to take along the
    scan lines for both the extracted and predicted intensity lines and corresponds somewhat to how accurate the
    resulting limb location will be. (Generally a higher number will lead to a higher accuracy though this is also
    limited by the resolution of the image and the shape model itself.  A higher number also will make things take
    longer.)

    In addition to the control over the scan lines, you can adjust the :attr:`brdf` which is used to generate the
    predicted intensity lines (although this will generally not make much difference) and you can change what peak
    finder is used to find the subpixel peaks of the correlation lines.

    This technique requires decent a priori knowledge of the relative state between the target and the camera for it to
    work.  At minimum it requires that the scan center be located through both the observed target location in the image
    and the target shape model placed at the current relative position in the scene.  If this isn't guaranteed by your
    knowledge then you can use something like the :mod:`.moment_algorithm` to correct the gross errors in your a priori
    knowledge as is done by :class:`.LimbMatching`.

    Generally you will not use this class directly as it is used by the :class:`.LimbMatching` class.  If you want to
    use it for some other purpose however, simply provide the required initialization parameters, then use
    :meth:`extract_limbs` to extract the limbs from the image.
    """

    def __init__(self, scene: Scene, camera: Camera, psf: PointSpreadFunction, number_of_scan_lines: int = 51,
                 scan_range: Real = 3 * np.pi / 4, number_of_sample_points: int = 501,
                 brdf: Optional[IlluminationModel] = None, peak_finder: Callable = parabolic_peak_finder_1d):
        r"""
        :param scene: The scene containing the target(s) and the light source
        :param camera: The camera containing the camera model
        :param psf: The point spread function to apply to the predicted intensity lines.  This should have a
                    :meth:`~.PointSpreadFunction.apply_1d`
        :param number_of_scan_lines: The number of scan lines to generate (number of extracted limb points)
        :param scan_range: The angular extent about the sun direction vector to distribute the scan lines through.  This
                           Should be in units of radians and should generally be less than :math:`\pi`.
        :param number_of_sample_points: The number of points to sample along each scan line
        :param brdf: The illumination model to use to render the predicted scan intensity lines
        :param peak_finder: The peak finder to find the peak of each correlation line.  This should assume that each
                            row of the input array is a correlation line that the peak needs to be found for.
        """
        self.scene: Scene = scene
        """
        The scene containing the target(s) and the light source
        """

        self.camera: Camera = camera
        """
        The camera containing the camera model
        """

        self.psf: PointSpreadFunction = psf
        """
        The point spread function to apply to the predicted intensity lines.

        This should provide a :meth:`~.PointSpreadFunction.apply_1d` that accepts in a numpy array where each 
        row is an intensity line and returns the blurred intensity lines as a numpy array.
        """

        self.number_of_scan_lines: int = number_of_scan_lines
        """
        The number of scan lines to generate/limb points to extract
        """

        self.scan_range: float = float(scan_range)
        r"""
        The extent about the illumination direction in radians in which to distribute the scan lines.

        The scan lines are distributed +/- scan_range/2 about the illumination direction.  This therefore should 
        generally be less than :math:`\frac{\pi}{2}` unless you are 100% certain that the phase angle is perfectly 0
        """

        self.number_of_sample_points: int = number_of_sample_points
        """
        The number of points to sample each scan line along for the extracted/predicted intensity lines
        """

        self.brdf: IlluminationModel = brdf
        """
        The illumination function to use to render the predicted scan lines.
        """

        # set the default if it wasn't specified
        if self.brdf is None:
            self.brdf = McEwenIllumination()

        self.peak_finder: Callable = peak_finder
        """
        the callable to use to return the peak of the correlation lines.
        """

        self.predicted_illums: NONEARRAY = None
        """
        The predicted intensity lines from rendering the scan lines.

        This will be a ``number_of_scan_lines`` by ``number_of_sample_points`` 2d array where each row is a scan line.

        This will be ``None`` until :meth:`extract_limbs` is called
        """

        self.extracted_illums: NONEARRAY = None
        """
        The extracted intensity lines from sampling the image.

        This will be a ``number_of_scan_lines`` by ``number_of_sample_points`` 2d array where each row is a scan line.

        This will be ``None`` until :meth:`extract_limbs` is called
        """

        self.correlation_lines: NONEARRAY = None
        """
        The correlation lines resulting from doing 1D cross correlation between the predicted and extracted scan lines.

        This will be a ``number_of_scan_lines`` by ``number_of_sample_points`` 2d array where each row is a correlation 
        line.

        This will be ``None`` until :meth:`extract_limbs` is called
        """

        self.correlation_peaks: NONEARRAY = None
        """
        The peaks of the correlation lines.

        This will be a ``number_of_scan_lines`` length 1d array where each element is the peak of the corresponding 
        correlation line.

        This will be ``None`` until :meth:`extract_limbs` is called
        """

    def predict_limbs(self, scan_center: np.ndarray, line_of_sight_sun: np.ndarray, target: SceneObject,
                      camera_temperature: Real) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the limb locations for a given target in the camera frame.

        This is done by

        #. get the angle between the illumination vector and the x axis of the image
        #. Generate :attr:`number_of_scan_lines` scan angles evenly distributed between the sun angle -
           :attr:`scan_range` /2 the and sun angle + :attr:`scan_range` /2
        #. convert the image scan line directions into directions in the camera frame
        #. use :meth:`.Shape.find_limbs` to find the limbs of the target given the scan center and the scan directions
           in the camera frame

        The limbs will be returned as a 3xn array in the camera frame.

        This method is automatically called by :meth:`extract_limbs` and will almost never be used directly, however,
        it is exposed for the adventurous types.

        :param scan_center: the beginning of the scan in the image (pixels)
        :param line_of_sight_sun: the line of sight to the sun in the image (pixels)
        :param target: The target the limbs are to be predicted for
        :param camera_temperature: The temperature of the camera
        :return: The predicted limb locations in the camera frame
        """

        # Get the angle of the illumination direction from the x axis in the image
        angle_sun = np.arctan2(line_of_sight_sun[1], line_of_sight_sun[0])

        # Set the scan angles +/- scan range around the sun direction
        scan_angles = np.linspace(angle_sun - self.scan_range / 2, angle_sun + self.scan_range / 2,
                                  self.number_of_scan_lines)

        # get the scan directions in the image
        scan_dirs_pixels = np.vstack([np.cos(scan_angles), np.sin(scan_angles)])

        # get the line of sight to the target in the camera frame
        scan_center_camera = self.camera.model.pixels_to_unit(scan_center, temperature=camera_temperature)
        scan_center_camera /= scan_center_camera[-1]

        # get the scan directions in the camera frame
        scan_dirs_camera = self.camera.model.pixels_to_unit(scan_center.reshape(2, 1) + scan_dirs_pixels,
                                                            temperature=camera_temperature)
        scan_dirs_camera /= scan_dirs_camera[-1]
        scan_dirs_camera -= scan_center_camera
        scan_dirs_camera /= np.linalg.norm(scan_dirs_camera, axis=0, keepdims=True)

        # get the limbs body centered
        limbs = target.shape.find_limbs(scan_center_camera, scan_dirs_camera, target.position.ravel())

        # return the limbs in the camera frame
        return limbs + target.position.reshape(3, 1), scan_dirs_pixels, scan_dirs_camera

    def extract_limbs(self, image_interpolator: Callable, camera_temperature: Real, target: SceneObject,
                      scan_center: np.ndarray, line_of_sight_sun: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This method extracts limb points in an image and pairs them to surface points that likely generated them.

        This is completed through the used of 1D cross correlation.

        #. The predicted limb locations in the image and the scan lines are determined using :meth:`predict_limbs`
        #. Scan lines are generated along the scan directions and used to create extracted intensity lines by sampling
           the image and predicted intensity lines by rendering the results of a ray trace along the scan line.
        #. The predicted and extracted intensity lines are cross correlated in 1 dimension :func:`.fft_correlator_1d`
        #. The peak of each correlation line is found using :attr:`peak_finder`.
        #. the peak of the correlation surface is translated into a shift between the predicted and extracted limb
           location in the image and used to compute the extracted limb location.

        The resulting predicted surface points, predicted image points, observed image points, and scan directions
        in the camera frame are then all returned as numpy arrays.

        :param image_interpolator: A callable which returns the interpolated image values for provides [y,x] locations
                                   in the image
        :param camera_temperature: The temperature of the camera in degrees at the time the image was captured
        :param target: The target we are looking for limb points for
        :param scan_center: The center where all of our scan lines will start
        :param line_of_sight_sun:  The line of sight of the sun in the image
        :return: The predicted surface points in the camera frame as a 3xn array, the predicted limbs in the image as a
                 2xn array, the observed limbs in the image as a 2xn array, and the scan directions in the camera frame
                 as a 3xn array of unit vectors where n is the :attr:`number_of_scan_lines`
        """

        # predict the limb locations
        predicted_limbs_camera, scan_dirs, scan_dirs_camera = self.predict_limbs(scan_center, line_of_sight_sun,
                                                                                 target, camera_temperature)
        predicted_limbs_pixels = self.camera.model.project_onto_image(predicted_limbs_camera,
                                                                      temperature=camera_temperature)

        # set the distance to search along each scan line 2 times the apparent radius of the target in the image
        # noinspection PyArgumentList
        apparent_radius_pixels = np.linalg.norm(predicted_limbs_pixels - scan_center.reshape(2, 1), axis=0).max()

        search_dist = 2 * apparent_radius_pixels

        # Create an array of where we want to interpolate the image at/shoot rays through
        search_distance_array = np.linspace(-search_dist, search_dist, self.number_of_sample_points)

        # Create an interpolator to figure out the distance from the scan center for the subpixel locations of
        # the correlation
        distance_interpolator = interp1d(np.arange(self.number_of_sample_points), search_distance_array)

        # Get the center of each scan line
        center = (self.number_of_sample_points - 1) // 2

        # Only take the middle of the predicted scan lines since we know the limb will lie in that region
        template_selection = self.number_of_sample_points // 4

        # Determine the deltas to apply to the limb locations
        search_deltas = scan_dirs[:2].T.reshape((-1, 2, 1), order='F') * search_distance_array

        # Get the pixels that we are sampling in the image along each scan line
        search_points_image = search_deltas + predicted_limbs_pixels.reshape((1, 2, -1), order='F')

        # Flatten everything to just 2d matrices instead of nd matrices
        sp_flat = np.hstack(search_points_image)

        # Select the template portion
        sp_flat_template = np.hstack(search_points_image[...,
                                     center - template_selection:center + template_selection + 1])

        # Compute the direction vector through each pixel we are sampling in the template
        direction_vectors = self.camera.model.pixels_to_unit(sp_flat_template, temperature=camera_temperature)

        # Build the rays we are going to trace to determine our predicted scan lines
        render_rays = Rays(np.zeros(3), direction_vectors)

        # Get the predicted scan line illumination inputs
        illum_inputs = self.scene.get_illumination_inputs(render_rays)

        # Compute the scan line illuminations
        illums = self.brdf(illum_inputs).reshape(search_points_image.shape[0], 2 * template_selection + 1)

        # Apply the psf to the predicted illuminations and store the scan lines
        self.predicted_illums = self.psf(illums)

        # Extract the scan line DN values from the image
        self.extracted_illums = image_interpolator(sp_flat[::-1].T).reshape(search_points_image.shape[0],
                                                                            search_points_image.shape[-1])

        # Do the 1d correlations between the extracted and predicted scan lines
        self.correlation_lines = fft_correlator_1d(self.extracted_illums, self.predicted_illums)

        # Find the peak of each correlation line
        self.correlation_peaks = self.peak_finder(self.correlation_lines)

        distances = distance_interpolator(self.correlation_peaks.ravel()).reshape(self.correlation_peaks.shape)

        observed_limbs_pixels = distances.reshape(1, -1) * scan_dirs + predicted_limbs_pixels

        return predicted_limbs_camera, predicted_limbs_pixels, observed_limbs_pixels, scan_dirs_camera


class EllipseMatching(RelNavEstimator):
    """
    This class implements GIANT's version of limb based OpNav for regular bodies.

    The class provides an interface to perform limb based OpNav for each target body that is predicted to be in an
    image.  It does this by looping through each target object contained in the :attr:`.Scene.target_objs` attribute
    that is requested.  For each of the targets, the algorithm:

    #. If using limb scanning to extract the limbs, and requested with :attr:`recenter`, identifies the center of
       brightness for each target using the :mod:`.moment_algorithm` and moves the a priori target to be along that line
       of sight
    #. Extracts the observed limbs from the image and pairs them to the target
    #. Estimates the relative position between the target and the image using the observed limbs and the steps discussed
       in the :mod:.ellipse_matching` documentation
    #. Uses the estimated position to get the predicted limb surface location and predicted limb locations in the image

    When all of the required data has been successfully loaded into an instance of this class, the :meth:`estimate`
    method is used to perform the estimation for the requested image.  The results are stored into the
    :attr:`observed_bearings` attribute for the observed limb locations and the :attr:`observed_positions` attribute for
    the estimated relative position between the target and the camera. In addition, the predicted location for the limbs
    for each target are stored in the :attr:`computed_bearings` attribute and the a priori relative position between the
    target and the camera is stored in the :attr:`computed_positions` attribute. Finally, the details about the fit are
    stored as a dictionary in the appropriate element in the :attr:`details` attribute.  Specifically, these
    dictionaries will contain the following keys.

    =========================== ========================================================================================
    Key                         Description
    =========================== ========================================================================================
    ``'Covariance'``            The 3x3 covariance matrix for the estimated relative position in the camera frame based
                                on the residuals.  This is only available if successful
    ``'Surface Limb Points'``   The surface points that correspond to the limb points in the target fixed target
                                centered frame.
    ``'Failed'``                A message indicating why the fit failed.  This will only be present if the fit failed
                                (so you could do something like ``'Failed' in limb_matching.details[target_ind]`` to
                                check if something failed.  The message should be a human readable description of what
                                called the failure.
    =========================== ========================================================================================

    .. warning::
        Before calling the :meth:`estimate` method be sure that the scene has been updated to correspond to the correct
        image time.  This class does not update the scene automatically.
    """

    technique: str = 'ellipse_matching'
    """
    The name of the technique identifier in the :class:`.RelativeOpNav` class.
    """

    observable_type: List[RelNavObservablesType] = [RelNavObservablesType.LIMB, RelNavObservablesType.RELATIVE_POSITION]
    """
    The type of observables this technique generates.
    """

    def __init__(self, scene: Scene, camera: Camera, image_processing: ImageProcessing,
                 limb_scanner: Optional[LimbScanner] = None,
                 extraction_method: Union[LimbExtractionMethods, str] = LimbExtractionMethods.EDGE_DETECTION,
                 interpolator: type = RegularGridInterpolator,
                 recenter: bool = True):
        """
        :param scene: The :class:`.Scene` object containing the target, light, and obscuring objects.
        :param camera: The :class:`.Camera` object containing the camera model and images to be utilized
        :param image_processing: The :class:`.ImageProcessing` object to be used to process the images
        :param limb_scanner: The :class:`.LimbScanner` object containing the limb scanning settings.
        :param extraction_method: The method to use to extract the observed limbs from the image.  Should be
                                  ``'LIMB_SCANNING'`` or ``'EDGE_DETECTION'``.  See :class:`.LimbExtractionMethods` for
                                  details.
        :param interpolator: The type of image interpolator to use if the extraction method is set to LIMB_SCANNING.
        :param recenter: A flag to estimate the center using the moment algorithm to get a fast rough estimate of the
                         center-of-figure
        """

        # store the scene and camera in the class instance using the super class's init method
        super().__init__(scene, camera, image_processing)

        # interpret the limb extraction method into the enum
        if isinstance(extraction_method, str):
            extraction_method = extraction_method.upper()

        self.extraction_method: LimbExtractionMethods = LimbExtractionMethods(extraction_method)
        """
        The method to use to extract observed limb points from the image.

        The valid options are provided in the :class:`LimbExtractionMethods` enumeration
        """

        self._limb_scanner: LimbScanner = limb_scanner
        """
        The limb scanning instance to use.
        """

        self.interpolator: type = interpolator
        """
        The type of interpolator to use for the image.  

        This is ignored if the :attr:`extraction_method` is not set to ``'LIMB_SCANNING'``.
        """

        self._edge_detection_limbs: List[NONEARRAY] = [None] * len(self.scene.target_objs)
        """
        The extracted limbs from the image in pixels before they have been paired to a target

        Until :meth:`estimate` is called this list will be filled with ``None``.
        """

        self.limbs_camera: List[NONEARRAY] = [None] * len(self.scene.target_objs)
        """
        The limb surface points with respect to the center of the target

        Until :meth:`estimate` is called this list will be filled with ``None``.

        Each element of this list corresponds to the same element in the :attr:`.Scene.target_objs` list.
        """

        self._image_interp: Optional[Callable] = None
        """
        The interpolator for the image to use.  

        This is set on the call to estimate
        """

        self._limbs_extracted: bool = False
        """
        This flag specifies where limbs have already be extracted from the current image or not.
        """

        self.recenter: bool = recenter
        """
        A flag specifying whether to locate the center of the target using a moment algorithm before beginning.

        If the a priori knowledge of the bearing to the target is poor (outside of the body) then this flag will help
        to correct the initial error.  See the :mod:`.moment_algorithm` module for details.
        """

        # the moment algorithm instance to use if recentering has been requested.
        self._moment_algorithm: MomentAlgorithm = MomentAlgorithm(scene, camera, image_processing,
                                                                  apply_phase_correction=False,
                                                                  use_apparent_area=True)
        """
        The moment algorithm instance to use to recenter if we are using limb scanning
        """


    def extract_and_pair_limbs(self, image: OpNavImage, target: SceneObject, target_ind: int) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract and pair limb points in an image to the surface point on a target that created it.

        For irregular bodies this is an approximate procedure that depends on the current estimate of the state vector.
        See :meth:`.Shape.find_limbs` for details.

        This technique extracts limbs in 2 ways.  If :attr:`extraction_method` is ``EDGE_DETECTION``, then all limbs are
        extracted from the image using :meth:`.ImageProcessing.identify_subpixel_limbs`.  These extracted limbs are then
        stored and paired to their corresponding targets based on the apparent diameter.  This only happens once per
        image since the extracted limb locations in the image are independent of the relative position of the target to
        the camera.  If :attr:`extraction_method` is ``LIMB_SCANNING`` then this will extract and pair the limbs for the
        requested target using :meth:`.LimbScanner.extract_limbs`.  This is performed ever iteration, as the extracted
        limb locations are dependent on the relative position of the target in the scene.

        For both techniques, the paired observed limb location in the image for the target are stored in the appropriate
        element of :attr:`observed_bearings` as a 2xn array of pixel locations.

        :param image: The image that the limbs are to be extracted from
        :param target:  The target that the extracted limbs are to be paired to
        :param target_ind:  The index of the target that the extracted limbs are to be paired to
        :return: The scan center, the scan center direction, the scan directions, the predicted limbs in the camera, and
                 the predicted limbs in the image.
        """

        # set the scan center
        scan_center = self.camera.model.project_onto_image(target.position,
                                                           temperature=image.temperature)
        scan_center_dir = target.position.ravel().copy()
        scan_center_dir /= np.linalg.norm(scan_center_dir)

        # Determine the illumination direction in the image
        line_of_sight_sun = self.camera.model.project_directions(self.scene.light_obj.position.ravel())

        if self.extraction_method == LimbExtractionMethods.EDGE_DETECTION:

            if not self._limbs_extracted:
                n_objs = len(self.scene.target_objs)
                # extract the limbs from the image
                self._edge_detection_limbs = self.image_processing.identify_subpixel_limbs(image, -line_of_sight_sun,
                                                                                           num_objs=n_objs)

                # match the limbs to each target
                self._match_limbs_to_targets(image.temperature)

                self._limbs_extracted = True

            extracted_limbs = self.observed_bearings[target_ind]

            # get the scan directions for each extracted limb point
            scan_dirs_pixels = extracted_limbs - scan_center.reshape(2, 1)

            scan_dirs_camera = self.camera.model.pixels_to_unit(scan_center.reshape(2, 1) + scan_dirs_pixels,
                                                                temperature=image.temperature)
            scan_dirs_camera /= scan_dirs_camera[2]
            scan_dirs_camera -= scan_center_dir.reshape(3, 1)/scan_center_dir[2]
            scan_dirs_camera /= np.linalg.norm(scan_dirs_camera, axis=0, keepdims=True)

            try:
                # find the corresponding limbs
                predicted_limbs = target.shape.find_limbs(scan_center_dir, scan_dirs_camera)

                # project them onto the image
                predicted_limbs_image = self.camera.model.project_onto_image(predicted_limbs,
                                                                             temperature=image.temperature)

            except ZeroDivisionError:
                predicted_limbs = np.zeros((3, extracted_limbs.shape[1]), dtype=np.float64)
                predicted_limbs_image = np.zeros(extracted_limbs.shape, dtype=np.float64)

        else:

            (predicted_limbs,
             predicted_limbs_image,
             self.observed_bearings[target_ind],
             scan_dirs_camera) = self._limb_scanner.extract_limbs(self._image_interp, image.temperature, target,
                                                                  scan_center, line_of_sight_sun)

        return scan_center, scan_center_dir, scan_dirs_camera, predicted_limbs, predicted_limbs_image

    def _match_limbs_to_targets(self, temperature: Real):
        """
        This matches the limb clumps returned by :meth:`.ImageProcessing.identify_subpixel_limbs` to the targets in
        :attr:`.Scene.target_objs`.

        The matching is done based on apparent size, therefore it is expected that the relative size of each target and
        the relative range to each target is mostly correct.  (i.e. if in real life target 1 is smaller but closer then
        target 2, then in the scene this should also be the case.  These can be wrong by a common scale factor, that is
        if target 2 is 50% larger than what is truth, then target 1 should also ~50% larger than truth.).

        The results are stored in :attr:`observed_bearings`.
        """

        apparent_diameters_observed = [np.linalg.norm(limbs.T.reshape((-1, 2, 1)) -
                                                      limbs.reshape((1, 2, -1)), axis=1).max(initial=None)
                                       for limbs in self._edge_detection_limbs]

        apparent_diameters_predicted = []
        for target in self.scene.target_objs:
            apparent_diameters_predicted.append(target.get_apparent_diameter(self.camera.model,
                                                                             temperature=temperature))

        sorted_diameters_observed = np.argsort(apparent_diameters_observed)
        sorted_diameters_predicted = np.argsort(apparent_diameters_predicted)

        for target_ind, limb_ind in zip(sorted_diameters_predicted, sorted_diameters_observed):
            self.observed_bearings[target_ind] = self._edge_detection_limbs[limb_ind]

    def estimate(self, image: OpNavImage, include_targets: Optional[List[bool]] = None):
        """
        This method identifies the position of each target in the camera frame using ellipse matching.

        This method first extracts limb observations from an image and matches them to the targets in the scene.  Then,
        for each target, the position is estimated from the limb observations.

        .. warning::
            Before calling this method be sure that the scene has been updated to correspond to the correct
            image time.  This method does not update the scene automatically.

        :param image: The image to locate the targets in
        :param include_targets: A list specifying whether to process the corresponding target in
                                :attr:`.Scene.target_objs` or ``None``.  If ``None`` then all targets are processed.
        """

        if self.extraction_method == LimbExtractionMethods.LIMB_SCANNING:
            self._image_interp = self.interpolator((np.arange(image.shape[0]), np.arange(image.shape[1])), image,
                                                   bounds_error=False, fill_value=None)

            # If we were requested to recenter using a moment algorithm then do it
            if self.recenter:
                print('recentering', flush=True)
                # Estimate the center using the moment algorithm to get a fast rough estimate of the cof
                self._moment_algorithm.estimate(image, include_targets=include_targets)

                for target_ind, target in self.target_generator(include_targets):
                    if np.isfinite(self._moment_algorithm.observed_bearings[target_ind]).all():
                        new_position = self.camera.model.pixels_to_unit(
                            self._moment_algorithm.observed_bearings[target_ind]
                        )
                        new_position *= np.linalg.norm(target.position)
                        target.change_position(new_position)

        # loop through each object in the scene
        for target_ind, target in self.target_generator(include_targets):

            relative_position = target.position.copy()
            self.computed_positions[target_ind] = relative_position

            # get the observed limbs for the current target
            (scan_center, scan_center_dir,
             scan_dirs_camera, self.limbs_camera[target_ind],
             self.computed_bearings[target_ind]) = self.extract_and_pair_limbs(image, target, target_ind)

            if self.observed_bearings[target_ind] is None:
                warnings.warn('unable to find any limbs for target {}'.format(target_ind))
                self.details[target_ind] = {'Failed': "Unable to find any limbs for target in the image"}
                continue

            # Drop any invalid limbs
            valid_test = ~np.isnan(self.observed_bearings[target_ind]).any(axis=0)

            self.observed_bearings[target_ind] = self.observed_bearings[target_ind][:, valid_test]

            # extract the shape from the object and see if it is an ellipsoid
            shape = target.shape

            if (not isinstance(shape, Ellipsoid)) and hasattr(shape, 'reference_ellipsoid'):
                warnings.warn("The primary shape is not an ellipsoid but it has a reference ellipsoid."
                              "We're going to use the reference ellipsoid but maybe you should actually use "
                              "limb_matching instead.")
                # if the object isn't an ellipsoid, see if it has a reference ellipsoid we can use
                shape = shape.reference_ellipsoid
            elif not isinstance(shape, Ellipsoid):
                warnings.warn('Invalid shape.  Unable to do ellipse matching')
                self.observed_bearings[target_ind] = np.array([np.nan, np.nan])
                self.observed_positions[target_ind] = np.array([np.nan, np.nan, np.nan])
                self.details[target_ind] = {'Failed': "The shape representing the target is not applicable to ellipse "
                                                      "matching"}
                continue

            # Extract the Q transformation matrix  (principal frame)
            q_matrix = np.diag(1 / shape.principal_axes)

            # Get the inverse transformation matrix  (principal frame)
            q_inv = np.diag(shape.principal_axes)

            # Now, we need to get the unit vectors in the direction of our measurements
            unit_vectors_camera = self.camera.model.pixels_to_unit(self.observed_bearings[target_ind],
                                                                   temperature=image.temperature)

            # Rotate the unit vectors into the principal axis frame
            unit_vectors_principal = shape.orientation.T @ unit_vectors_camera

            # Now, convert into the unit sphere space and make unit vectors
            right_sphere_cone_vectors = q_matrix @ unit_vectors_principal
            right_sphere_cone_size = np.linalg.norm(right_sphere_cone_vectors, axis=0, keepdims=True)
            right_sphere_cone_vectors /= right_sphere_cone_size

            # Now, we solve for the position in the sphere space
            location_sphere = np.linalg.lstsq(right_sphere_cone_vectors.T,
                                              np.ones(right_sphere_cone_vectors.shape[1]),
                                              rcond=None)[0]

            # Finally, get the position in the camera frame and store it
            self.observed_positions[target_ind] = (shape.orientation @ q_inv @ location_sphere /
                                                   np.sqrt(np.inner(location_sphere, location_sphere) - 1)).ravel()

            # Get the limb locations in the camera frame
            scan_center_dir = self.observed_positions[target_ind].copy()
            scan_center_dir /= np.linalg.norm(scan_center_dir)

            scan_dirs = unit_vectors_camera/unit_vectors_camera[2] - scan_center_dir.reshape(3, 1) / scan_center_dir[2]
            scan_dirs /= np.linalg.norm(scan_dirs[:2], axis=0, keepdims=True)

            self.limbs_camera[target_ind] = shape.find_limbs(scan_center_dir, scan_dirs,
                                                             shape.center - self.observed_positions[target_ind].ravel())

            self.computed_bearings[target_ind] = self.camera.model.project_onto_image(self.limbs_camera[target_ind],
                                                                                      temperature=image.temperature)

            target_centered_fixed_limb = self.limbs_camera[target_ind] - \
                                         self.observed_positions[target_ind].reshape(3, 1)
            target_centered_fixed_limb = target.orientation.matrix.T @ target_centered_fixed_limb

            # compute the covariance matrix
            delta_ns = location_sphere.reshape(3, 1) - right_sphere_cone_vectors

            decomp_matrix = (q_matrix@shape.orientation.T).reshape(1, 3, 3)
            meas_std = (self.observed_bearings[target_ind] - self.computed_bearings[target_ind]).std(axis=-1)
            model_jacobian = self.camera.model.compute_unit_vector_jacobian(self.observed_bearings[target_ind],
                                                                            temperature=image.temperature)
            meas_cov = model_jacobian@np.diag(meas_std**2)@model_jacobian.swapaxes(-1, -2)
            inf_eta = np.diag(right_sphere_cone_size.ravel()**2 *
                              (delta_ns.T.reshape(-1, 1, 3) @
                               decomp_matrix @
                               meas_cov @
                               decomp_matrix.swapaxes(-1, -2) @
                               delta_ns.reshape(-1, 3, 1)).squeeze()**(-1))

            cov_location_sphere = np.linalg.pinv(right_sphere_cone_vectors @
                                                 inf_eta @
                                                 right_sphere_cone_vectors.T)

            inner_m1 = location_sphere.T@location_sphere - 1
            transform_camera = (-shape.orientation @ q_inv @
                                (np.eye(3) - np.outer(location_sphere, location_sphere) / inner_m1) /
                                np.sqrt(inner_m1))

            covariance_camera = transform_camera@cov_location_sphere@transform_camera.T

            # store the details of the fit
            self.details[target_ind] = {"Surface Limb Points": target_centered_fixed_limb,
                                        "Covariance": covariance_camera}

    def reset(self):
        """
        This method resets the observed/computed attributes, the details attribute, and the limb attributes to have
        ``None``.

        This method is called by :class:`.RelativeOpNav` between images to ensure that data is not accidentally applied
        from one image to the next.
        """

        super().reset()

        self._edge_detection_limbs = [None] * len(self.scene.target_objs)
        self.limbs_camera = [None] * len(self.scene.target_objs)
        self._limbs_extracted = False
        self._image_interp = None
