from abc import ABC

from dataclasses import dataclass

from typing import Sequence, Callable

import numpy as np

from scipy.interpolate import RegularGridInterpolator

from giant.relative_opnav.estimators.estimator_interface_abc import RelNavEstimator

from giant.image_processing.limb_edge_detection import LimbEdgeDetection, LimbEdgeDetectionOptions
from giant.image_processing.limb_scanning import LimbScanner, LimbScannerOptions

from giant.ray_tracer.scene import Scene, SceneObject
from giant.ray_tracer._typing import HasFindableLimbs
from giant.camera import Camera
from giant.image import OpNavImage

from giant.utilities.mixin_classes import UserOptionConfigured
from giant.utilities.options import UserOptions

from giant._typing import DOUBLE_ARRAY, ARRAY_LIKE


from enum import Enum

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


@dataclass
class LimbPairerOptions(UserOptions):
    """
    Options for the limb pairing portion whereby limb points in the image are paired with the 3d model
    """
    
    extraction_method: LimbExtractionMethods = LimbExtractionMethods.EDGE_DETECTION
    """
    The method to use to extract the observed limbs from the image.  Should be
    ``'LIMB_SCANNING'`` or ``'EDGE_DETECTION'``.  See :class:`.LimbExtractionMethods` for
    details.
    """
    
    limb_edge_detection_options: LimbEdgeDetectionOptions | None = None
    """
    The options to use to configure the limb edge detector (if being used)
    """
    
    limb_scanner_options: LimbScannerOptions | None = None
    """
    The options to use to configure the limb scanner (if being used)
    """
    

class LimbPairer(UserOptionConfigured[LimbPairerOptions], RelNavEstimator, LimbPairerOptions, ABC):
    
    def __init__(self, options_type: type[LimbPairerOptions], scene: Scene, camera: Camera, *args, options: LimbPairerOptions | None = None, **kwargs) -> None:
        super().__init__(options_type, scene, camera, *args, options=options, **kwargs)
        
        self.limb_scanner = LimbScanner(self.scene, self.camera, options=self.limb_scanner_options)
        """
        The limb scanner to use if the :attr:`extraction_method` is set to LIMB_SCANNING
        """
        
        self.limb_edge_detector = LimbEdgeDetection(options=self.limb_edge_detection_options)
        """
        The limb edge detector to use if the :attr:`extraction_method` is set to EDGE_DETECTION
        """
        
        self._edge_detection_limbs: Sequence[DOUBLE_ARRAY | None] = [None] * len(self.scene.target_objs)
        """
        The extracted limbs from the image in pixels before they have been paired to a target

        Until :meth:`estimate` is called this list will be filled with ``None``.
        """
        
        self._image_interp: Callable[[ARRAY_LIKE], np.ndarray] | None = None
        """
        The interpolator for the image to use.  

        This is set on the call to estimate if the extraction type is set to `LIMB_SCANNING`
        """

        self._limbs_extracted: bool = False
        """
        This flag specifies where limbs have already be extracted from the current image or not.
        """

        
    def extract_and_pair_limbs(self, image: OpNavImage, target: SceneObject, target_ind: int) \
            -> tuple[DOUBLE_ARRAY, DOUBLE_ARRAY, DOUBLE_ARRAY, DOUBLE_ARRAY, DOUBLE_ARRAY]:
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
        
        if not isinstance(target.shape, HasFindableLimbs):
            raise ValueError('the target needs to have findable limbs')

        # set the scan center
        scan_center = self.camera.model.project_onto_image(target.position,
                                                           temperature=image.temperature)
        scan_center_dir = target.position.ravel().copy()
        scan_center_dir /= np.linalg.norm(scan_center_dir)

        # Determine the illumination direction in the image
        if self.scene.light_obj is None:
            raise ValueError('light_obj must not be None at this point')
        lpos = getattr(self.scene.light_obj.shape, "position", self.scene.light_obj.position)
        line_of_sight_sun = self.camera.model.project_directions(lpos.ravel()-target.position.ravel())

        if self.extraction_method == LimbExtractionMethods.EDGE_DETECTION:

            if not self._limbs_extracted:
                n_objs = len(self.scene.target_objs)
                # extract the limbs from the image
                self._edge_detection_limbs = self.limb_edge_detector.identify_subpixel_limbs(image, -line_of_sight_sun, num_objs=n_objs)

                # match the limbs to each target
                self._match_limbs_to_targets(image.temperature)

                self._limbs_extracted = True

            extracted_limbs = self.observed_bearings[target_ind]
            if extracted_limbs is None:
                return (np.zeros((2, 0)),)*5 # type: ignore

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
            
            if self._image_interp is None:
                self._image_interp = RegularGridInterpolator((np.arange(image.shape[0]), np.arange(image.shape[1])), image,
                                                             bounds_error=False, fill_value=None, # type: ignore
                                                             method=self.limb_scanner.interpolator_method)

            (predicted_limbs,
             predicted_limbs_image,
             self.observed_bearings[target_ind],
             scan_dirs_camera) = self.limb_scanner.extract_limbs(self._image_interp, image.temperature, target,
                                                                 scan_center, line_of_sight_sun)

        return scan_center, scan_center_dir, scan_dirs_camera, predicted_limbs, predicted_limbs_image

    def _match_limbs_to_targets(self, temperature: float):
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
                                       for limbs in self._edge_detection_limbs if limbs is not None and limbs.size > 0]

        apparent_diameters_predicted = []
        for target in self.scene.target_objs:
            apparent_diameters_predicted.append(target.get_apparent_diameter(self.camera.model,
                                                                             temperature=temperature))

        sorted_diameters_observed = np.argsort(apparent_diameters_observed)
        sorted_diameters_predicted = np.argsort(apparent_diameters_predicted)

        for target_ind, limb_ind in zip(sorted_diameters_predicted, sorted_diameters_observed):
            self.observed_bearings[target_ind] = self._edge_detection_limbs[limb_ind]

        
    def reset(self):
        """
        This method resets the observed/computed attributes, the details attribute, and the limb attributes to have
        ``None``.

        This method is called by :class:`.RelativeOpNav` between images to ensure that data is not accidentally applied
        from one image to the next.
        """

        super().reset()

        self._edge_detection_limbs = [None] * len(self.scene.target_objs)
        self._limbs_extracted = False
        self._image_interp = None
        
        
        