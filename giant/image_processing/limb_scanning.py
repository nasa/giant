from dataclasses import dataclass, field

from typing import Literal, Callable, cast

import numpy as np
from numpy.typing import NDArray

from giant.point_spread_functions import PointSpreadFunction
from giant.point_spread_functions.gaussians import Gaussian
from giant.ray_tracer.illumination import IlluminationModel, McEwenIllumination
from giant.ray_tracer.scene import Scene, SceneObject
from giant.ray_tracer.rays import Rays
from giant.ray_tracer._typing import HasFindableLimbs
from giant.camera import Camera
from giant.utilities.options import UserOptions  

from giant.image_processing.peak_finders import parabolic_peak_finder_1d
from giant.image_processing.correlators import fft_correlator_1d

from giant.utilities.mixin_classes.user_option_configured import UserOptionConfigured

from giant._typing import DOUBLE_ARRAY, NONEARRAY
    
    
INTERPOLATOR_METHODS = Literal["linear", "nearest", "slinear", "cubic", "quintic", "pchip"]
   
    
@dataclass
class LimbScannerOptions(UserOptions):
    
    psf: PointSpreadFunction = field(default_factory=Gaussian)
    """
    The point spread function to apply to the predicted intensity lines.

    This should provide a :meth:`~.PointSpreadFunction.apply_1d` that accepts in a numpy array where each 
    row is an intensity line and returns the blurred intensity lines as a numpy array.
    """

    number_of_scan_lines: int = 51
    """
    The number of scan lines to generate/limb points to extract
    """

    scan_range: float = 3 * np.pi / 4
    r"""
    The extent about the illumination direction in radians in which to distribute the scan lines.

    The scan lines are distributed +/- scan_range/2 about the illumination direction.  This therefore should 
    generally be less than :math:`\frac{\pi}{2}` unless you are 100% certain that the phase angle is perfectly 0
    """

    number_of_sample_points: int = 501
    """
    The number of points to sample each scan line along for the extracted/predicted intensity lines
    """

    brdf: IlluminationModel = field(default_factory=McEwenIllumination)
    """
    The illumination function to use to render the predicted scan lines.
    """

    peak_finder: Callable[[DOUBLE_ARRAY], DOUBLE_ARRAY] = parabolic_peak_finder_1d
    """
    the callable to use to return the peak of the correlation lines.
    """
    
    interpolator_method: INTERPOLATOR_METHODS = "linear"
    """
    The method to use from scipy's RegularGridInterpolator.
    
    Generally linear is more than sufficient
    """


class LimbScanner(UserOptionConfigured[LimbScannerOptions], LimbScannerOptions):
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

    def __init__(self, scene: Scene, camera: Camera, options: None | LimbScannerOptions):
        r"""
        :param scene: The scene containing the target(s) and the light source
        :param camera: The camera containing the camera model
        :param options: The options structure to configure the class with
        """
        
        super().__init__(LimbScannerOptions, options=options)
        
        self.scene: Scene = scene
        """
        The scene containing the target(s) and the light source
        """

        self.camera: Camera = camera
        """
        The camera containing the camera model
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
                      camera_temperature: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        
        if not isinstance(target.shape, HasFindableLimbs):
            raise ValueError('the target must support finding the predicted limbs with a find_limbs method')

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

    def extract_limbs(self, image_interpolator: Callable[[NDArray], NDArray], camera_temperature: float, target: SceneObject,
                      scan_center: np.ndarray, line_of_sight_sun: np.ndarray) -> \
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        :param image_interpolator: A callable which returns the interpolated image values for provided [y,x] locations
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
        apparent_radius_pixels: float = np.linalg.norm(predicted_limbs_pixels - scan_center.reshape(2, 1), axis=0).max()

        search_dist = 2 * apparent_radius_pixels

        # Create an array of where we want to interpolate the image at/shoot rays through
        search_distance_array: DOUBLE_ARRAY = cast(DOUBLE_ARRAY, np.linspace(-search_dist, search_dist, self.number_of_sample_points))

        # Get the center of each scan line
        center = (self.number_of_sample_points - 1) // 2

        # Only take the middle of the predicted scan lines since we know the limb will lie in that region
        template_selection = self.number_of_sample_points // 4

        # Determine the deltas to apply to the limb locations
        search_deltas: DOUBLE_ARRAY = scan_dirs[:2].T.reshape((-1, 2, 1), order='F') * search_distance_array

        # Get the pixels that we are sampling in the image along each scan line
        search_points_image: DOUBLE_ARRAY = search_deltas + predicted_limbs_pixels.reshape((1, 2, -1), order='F')

        # Flatten everything to just 2d matrices instead of nd matrices
        sp_flat = np.rollaxis(search_points_image, 1).reshape(2, -1)

        # Select the template portion
        sp_template = search_points_image[..., center - template_selection:center + template_selection + 1]
        sp_flat_template = np.rollaxis(sp_template, 1).reshape(2, -1)

        # Compute the direction vector through each pixel we are sampling in the template
        direction_vectors = self.camera.model.pixels_to_unit(sp_flat_template, temperature=camera_temperature)

        # Build the rays we are going to trace to determine our predicted scan lines
        render_rays = Rays(np.zeros(3), direction_vectors)

        # Get the predicted scan line illumination inputs
        illum_inputs = cast(np.ndarray, self.scene.get_illumination_inputs(render_rays))

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

        distances = np.interp(self.correlation_peaks.ravel(), np.arange(self.number_of_sample_points), search_distance_array).reshape(self.correlation_peaks.shape)

        observed_limbs_pixels = distances.reshape(1, -1) * scan_dirs + predicted_limbs_pixels

        return predicted_limbs_camera, predicted_limbs_pixels, observed_limbs_pixels, scan_dirs_camera
