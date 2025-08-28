


"""
This module provides a class which implements an unresolved center finding RelNav technique along with a new meta class
that adds concrete center-of-brightness to center-of-figure correction methods.

Description of the Technique
----------------------------

Unresolved center finding is applicable when you observe a target that is dominated by the point spread function of the
camera rather than by the geometry of the object.  Typically this occurs when the apparent diameter of the object in
the image is less than 5-10 pixels. Because these targets are dominated by the camera point spread function,
GIANT treats unresolved bodies the same way that stars are treated.  In fact, all of the same algorithms and functions
are used from the :mod:`.image_processing` module as are used for extracting potential star locations from images.

First, the area around the expected location of the target body is searched for all groupings of pixels that exceed a
specified threshold.  Then, as long as there is 1 and only 1 grouping of pixels above the threshold in the search
region, the sub-pixel center of brightness is extracted by using the specified fitting function (typically a 2D gaussian
or a moment algorithm.  This is all done in the :meth:`.ImageProcessing.locate_subpixel_poi_in_roi` method from the
image processing class.

This routine is generally used during early approach to an object, or for extracting Celestial Navigation (CelNav)
observations to known targets from the solar system.  It does not apply once the target's apparent diameter begins to
exceed 5-10 pixels and will begin failing at that point.

Tuning
------

The primary control for tuning this technique is through the tuning of the
:class:`.PointOfInterestFinder` clas.  There are a number of tuning parameters
for this class and we direct you to its documentation for more details.

In addition, there are a few tuning parameters for the class itself.
The search region is controlled by the :attr:`~.UnresolvedCenterFinding.search_distance` attribute.  This should be an
integer which specifies half of the square region to search around the predicted center, such that a 2*
:attr:`~.UnresolvedCenterFinding.search_distance` by 2* :attr:`~.UnresolvedCenterFinding.search_distance` pixels of the
image will be searched.

In addition, Whether the phase correction is applied or not is controlled by the boolean flag
:attr:`~.UnresolvedCenterFinding.apply_phase_correction`.  The phase correction computation can be controlled using the
:attr:`~.UnresolvedCenterFinding.phase_correction_type` and :attr:`~.UnresolvedCenterFinding.brdf` attributes.

Use
---

This class is not typically not used directly by the user, but instead is
called from the :class:`.RelativeOpNav` class using the technique name of ``unresolved``.  For more details on using
this class directly, refer to the following class documentation.  For more details on using this class through the
:class:`.RelativeOpNav` user interface refer to the :mod:`.relnav_class` documentation.
"""

import warnings

from enum import Enum, auto

from typing import List, Optional, Tuple, cast

from abc import ABC

import numpy as np

from dataclasses import dataclass, field

from giant.ray_tracer.rays import Rays
from giant.ray_tracer.scene import Scene, SceneObject
from giant.ray_tracer.illumination import IlluminationModel, McEwenIllumination
from giant.ray_tracer.shapes.triangle import Triangle32, Triangle64
from giant.ray_tracer._typing import Traceable
from giant.camera import Camera
from giant.image import OpNavImage
from giant.image_processing.point_source_finder import PointOfInterestFinder, PointOfInterestFinderOptions
from giant.utilities.options import UserOptions
from giant.utilities.mixin_classes.user_option_configured import UserOptionConfigured

from giant.relative_opnav.estimators.estimator_interface_abc import RelNavEstimator, RelNavObservablesType


class PhaseCorrectionType(Enum):
    """
    This enumeration provides the possible phase correction methods to use to convert the observed center-of-brightness
    to a center of figure observation.
    """

    SIMPLE = auto()
    """
    This method assumes a spherical target and computes the phase correction for a sphere 
    """

    RASTERED = auto()
    """
    This method assumes a tessellated object and renders it using rasterization to compute the phase correction.
    
    This will be much more accurate than the SIMPLE method for most tessellated targets, especially for ones that are 
    not roughly spherical in shape.  However, it is also much more computationally expensive and the newly realized 
    accuracy could be useless if the object is still very small in the FOV of the camera.
    """
    

@dataclass
class PhaseCorrectorOptions(UserOptions):
    
        phase_correction_type: PhaseCorrectionType  = PhaseCorrectionType.SIMPLE
        """
        The type of phase correction to use, if requested.

        See :class:`.PhaseCorrectionType` for details.
        """

        brdf: IlluminationModel = field(default_factory=McEwenIllumination)
        """
        The illumination model used to convert geometry into expected illumination.  

        This is only used if the ``RASTERED`` phase correction type is chosen and is ignored otherwise.
        """
        
        
class PhaseCorrector(RelNavEstimator, PhaseCorrectorOptions, ABC):
    """
    This class adds phase correction capabilities to RelNavEstimator.

    Phase correction is the process by which we attempt to move an observed center of brightness (the centroid of a
    bright patch in this case) to more closely resemble the location where the center of figure would be observed in the
    image.  This is done by correcting for the fact that the phase angle between the camera, the target, and the sun,
    can cause the observed center of brightness to be biased towards one side of the shape because only part of the
    target appears illuminated.

    Specifically, this class adds 2 new key word argument inputs and attributes :attr:`phase_correction_type`, and
    :attr:`brdf` which specify what phase correction method to use to compute the phase correction, as well as the BRDF
    that is used to generate the illumination information for each facet when the ``RASTERED`` phase correction method
    is used.  In addition, it provides the method :meth:`compute_phase_correction` which will return the phase
    correction based as a size 2 numpy array from the center of brightness to the center of figure in pixels using the
    current scene settings and the selected :attr:`phase_correction_type`.  It also defines 3 helper methods which are
    included for documentation purposes but are rarely interfaced with directly:
    :meth:`compute_line_of_sight_sun_image`, :meth:`simple_phase_correction`, and :meth:`rastered_phase_correction`.

    Generally this class is not used by the user and is instead used internally by the RelNav techniques that the user
    interacts with.  If you are trying to implement a new relnav technique that needs phase correction capabilities,
    then you can subclass this class (no need to also subclass :class:`.RelNavEstimator`) and then use the
    :meth:`compute_phase_correction` method when you need it.  For more information on defining a new RelNav technique,
    refer to the :mod:`.relative_opnav.estimators` documentation.
    """

    def __init__(self, scene: Scene, camera: Camera,
                 options: PhaseCorrectorOptions | None = None):
        """
        :param scene: The :class:`.Scene` object containing the target, light, and obscuring objects.
        :param camera: The :class:`.Camera` object containing the camera model and images to be utilized
        :param options: The options configuring the phase corrections
                                      enum values
        """

        super().__init__(scene, camera)

    def simple_phase_correction(self, target_ind: int, target: SceneObject, line_of_sight_sun_image: np.ndarray,
                                temperature: float) -> np.ndarray:
        """
        This method computes the simple phase correction assuming the target is a sphere.

        First, the apparent diameter of the target in pixels is computed using
        :meth:`.SceneObject.get_apparent_diameter`. Then the phase angle is computed using meth:`.Scene.phase_angle`
        for the target.  Finally, the correction magnitude is computed assuming a McEwen like scattering law.  The
        magnitude is then multiplied by the sun direction vector to compute the phase correction.

        This is based on the technique in https://www.aanda.org/articles/aa/pdf/2004/10/aah4644.pdf

        :param target_ind: the index of the target in the :attr:`.Scene.target_objs` list
        :param target: The target object itself to compute the phase correction for
        :param line_of_sight_sun_image: The unit vector from the sun to the target in the image
        :param temperature: the temperature of the camera when the image was captured
        :return: The phase correction as a length 2 numpy array from the center of brightness to the center of figure
        """

        # get the apparent radius of the target
        radius_pixels = target.get_apparent_diameter(self.camera.model,
                                                     temperature=temperature) / 2

        # get the phase angle between the camera, target, and sun
        phase = self.scene.phase_angle(target_ind)

        if phase < np.pi/9:
            lambertian_correction = radius_pixels*3/8*phase
            lommel_seeliger_correction = radius_pixels*radius_pixels/3
        else:
            lambertian_correction = radius_pixels*3*np.pi/16*(1+np.cos(phase))/((np.pi-phase)*(1/np.tan(phase)) + 1)
            lommel_seeliger_correction = (radius_pixels*2*(np.sin(phase)+(np.pi-phase)*np.cos(phase)) /
                                          (2*np.pi/np.tan(phase/2)-np.sin(phase/2*np.log(1/np.tan(phase/4)))))

        # use the average between lommel_seeliger and lambertian
        correction_mag = (lambertian_correction + lommel_seeliger_correction)/2

        # apply the correction along the sun direction vector
        correction = line_of_sight_sun_image * correction_mag

        return correction

    def rastered_phase_correction(self, target_ind: int, target: SceneObject, temperature: float):
        """
        This method computes the phase correction by raster rendering the target to determine the offset from the center
        of illumination to the center of figure.

        This method is only applicable to targets that are represented by tesselation, such as triangles or
        parallelograms.  It will in general be more accurate for tessellated bodies than the simple technique, it is
        also much more computationally efficient, and many times, especially when the target is still very small in the
        image, the added accuracy is overwhelmed by the uncertainty of identifying the center of brightness in the
        image.  In addition, if your shape model is very far off from the actual shape, then this will be just as
        inaccurate as using the simple technique, and in some cases perhaps more inaccurate.

        The specific steps to computing the correction using this technique are as follows.  First, the facets of the
        tesselation are each "rendered" assuming no occlusion or shadowing based solely on the incidence and view
        angles.  Then, the "center of brightness" is computed in the camera frame using a moment
        algorithm.  Finally, this center of brightness is projected onto the image, and the difference between it and
        the projected center of figure is the correction vector.

        :param target_ind: the index of the target in the :attr:`.Scene.target_objs` list
        :param target: The target object itself to compute the phase correction for
        :param temperature: the temperature of the camera when the image was captured
        :return: The phase correction as a length 2 numpy array from the center of brightness to the center of figure
        """

        # get the rasterized illumination
        brightness, centers, _ = self.scene.raster_render(target_ind, self.brdf)

        illuminated_facets = brightness > 0

        if isinstance(target.shape, (Triangle64, Triangle32)):
            # get the area
            illuminated_sides = target.shape.sides[illuminated_facets]
            # compute the norm of the cross product of the sides for the triangles divided by 2
            areas = np.linalg.norm(np.cross(illuminated_sides[..., 0], illuminated_sides[..., 1]), axis=-1)/2

            # compute the photo-center offset in the camera frame
            facet_brightness = areas*brightness[illuminated_facets]
            photo_center_offset = (facet_brightness*centers[illuminated_facets]).sum(axis=0)/facet_brightness.sum()
        else:
            # we probably shouldn't end up here
            photo_center_offset = (brightness[illuminated_facets]*centers[illuminated_facets]).sum(axis=0)/brightness[illuminated_facets].sum()

        # project the photo center into the camera
        image_photo_center = self.camera.model.project_onto_image(photo_center_offset, temperature=temperature)

        # compute and return the offset from the photo center to the center of figure
        return self.computed_bearings[target_ind] - image_photo_center

    def compute_line_of_sight_sun_image(self, target: SceneObject) -> np.ndarray:
        # get the line of sight from the sun in the image
        if self.scene.light_obj is None:
            raise ValueError('The light_obj must be specified by this point')
        lpos = getattr(self.scene.light_obj.shape, "position", self.scene.light_obj.position)
        line_of_sight_sun = target.position-self.scene.light_obj.position.ravel()
        line_of_sight_sun /= np.linalg.norm(line_of_sight_sun)
        line_of_sight_sun_image = self.camera.model.project_directions(line_of_sight_sun)

        return line_of_sight_sun_image

    def compute_phase_correction(self, target_ind: int, target: SceneObject, temperature: float,
                                 line_of_sight_sun_image: Optional[np.ndarray] = None):
        """
        The method computes the phase correction assuming a spherical target.

        The phase correction attempts to move the observed center-of-brightness measurement to be
        closer to what the actual observed center-of-figure of the object should be (akin to what you would receive from
        something like cross correlation).

        This is done by either assuming a spherical target, computing the apparent diameter of the assumed spherical
        target, and then computing the phase correction in the sun direction using a predefined model of phase shift for
        a spherical target, or by using rasterization to render the target and computing the shift in the center of
        brightness to the center of figure using the rasterized illumination data.  Which technique is used is set by
        the :attr:`phase_correction_type`.  The simple phase correction is performed using
        :meth:`simple_phase_correction` while the more accurate phase correction is performed using
        :meth:`rastered_phase_correction`.

        :param target_ind: The index of the target in the :attr:`.Scene.target_objs` attribute we are considering
        :param target: The actual target object we are considering.  The :attr:`.SceneObject.shape` attribute of this
                       object should be a :class:`.Surface` if the raster method is chosen.
        :param temperature: The temperature of the camera at the time the image was captured.  This is used for
                            projecting points into the camera
        :param line_of_sight_sun_image: The line of sight from the sun towards the target in the image.  This is
                                        essentially the slope of the line from the target to the sun projected into the
                                        image.  This should be a length 2 array or ``None``.  If ``None`` it will be
                                        computed from the scene if required.
        :return: The phase correction as a length 2 numpy array which goes from the observed center of brightness to
                 what the center of figure should be.
        """

        if self.phase_correction_type is PhaseCorrectionType.SIMPLE:
            if line_of_sight_sun_image is None:
                line_of_sight_sun_image = self.compute_line_of_sight_sun_image(target)
            return self.simple_phase_correction(target_ind, target, line_of_sight_sun_image, temperature)

        else:
            return self.rastered_phase_correction(target_ind, target, temperature)

@dataclass
class UnresolvedCenterFindingOptions(PhaseCorrectorOptions):
    """
    :param search_distance: The search radius to search around the predicted centers for the observed centers of
                            the target objects
    :param apply_phase_correction: A boolean flag specifying whether to apply the phase correction to the observed
                                    center of brightness to get closer to the center of figure based on the predicted
                                    apparent diameter of the object.
    :param phase_correction_type: The type of phase correction to use.  Should be one of the PhaseCorrectionType
                                    enum values
    :param brdf: The illumination model to use to compute the illumination values if the ``RASTERED`` phase
                    correction type is used.  If the ``RASTERED`` phase correction type is not used this is ignored.
                    If this is left as ``None`` and the ``Rastered`` phase correction type is used, this will default
                    to the McEwen Model, :class:`.McEwenIllumination
    """
    search_distance: int = 15
    """
    Half of the distance to search around the predicted centers for the observed centers of the target objects in 
    pixels.
    """

    apply_phase_correction: bool = False
    """
    A boolean flag specifying whether to apply the phase correction or not
    """
    
    point_of_interest_finder_options: PointOfInterestFinderOptions | None = None
    """
    The options to use to configure the point of interest finder
    """


class UnresolvedCenterFinding(UserOptionConfigured[UnresolvedCenterFindingOptions], PhaseCorrector, UnresolvedCenterFindingOptions):
    """
    This class implements GIANT's version of unresolved center finding for extracting bearing measurements to unresolved
    targets in an image.

    The class provides an interface to perform unresolved center finding for each
    target body that is predicted to be in an image.  It does this by looping through each target object contained
    in the :attr:`.Scene.target_objs` attribute.  For each of these targets the algorithm:

    #. Predicts the location of that target in the image using the a priori knowledge
    #. Searches the region around the predicted location defined by the :attr:`.search_distance` attribute for
       bright spots
    #. If only one bright spot exists in the region then it finds the sub-pixel center of the bright spot or, if
       more than one or no bright spots exist in the region it raises a warning and moves on to the next object.
    #. If requested, the observed center-of-brightness is corrected to be closer to what the observed center of
       figure should be using the phase angle, the illumination direction, and the predicted apparent diameter of
       the target in pixels.

    Steps 2 and 3 are both performed by the :meth:`.ImageProcessing.locate_subpixel_poi_in_roi` method, which is the
    same method used to extract potential star locations from an image in the :mod:`.stellar_opnav` package.  Therefore,
    all of the same settings are used to adjust the performance and the reader is directed to to the
    :meth:`.ImageProcessing.locate_subpixel_poi_in_roi` documentation for more information.

    The search region is controlled by the :attr:`search_distance` attribute.  This should be an integer which
    specifies half of the square region to search around the predicted center, such that a 2* :attr:`search_distance` by
    2* :attr:`search_distance` pixels of the image will be searched.  Whether the phase correction is applied or not is
    controlled by the boolean flag :attr:`apply_phase_correction`.

    When all of the required data has been successfully loaded into an instance of this class, the :meth:`estimate`
    method is used to extract the sub-pixel centers of the target bodies predicted to be in the requested image.  The
    results are stored into the :attr:`observed_bearings` attribute. In addition, the predicted location for each target
    is stored in the :attr:`computed_bearings` attribute. Finally, the details about the fit are stored as a
    dictionary in the appropriate element in the :attr:`details` attribute.  Specifically, these dictionaries will
    contain the following keys.

    ====================== =============================================================================================
    Key                    Description
    ====================== =============================================================================================
    ``'PSF'``              The fit PSF values.  Only available if successful.  Will be ``None`` if
                           :attr:`.ImageProcessing.save_psf` is ``False``
    ``'Phase Correction'`` The phase correction vector used to convert from center of brightness to center of figure.
                           This will only be available if the fit was successful.  If :attr:`apply_phase_correction` is
                           ``False`` then this will be an array of 0.
    ``'SNR'``              The peak signal to noise ratio of the detection.  This will only be set if the fit was
                           successful.  If :attr:`.ImageProcessing.return_stats` is ``False`` then this will be
                           ``None``.
    ``'Max Intensity'``    The intensity of the peak pixel used in the PSF fit.  This will only be set if the fit was
                           successful.
    ``'Failed'``           A message indicating why the fit failed.  This will only be present if the fit failed (so you
                           could do something like ``'Failed' in unresolved.details[target_ind]`` to check if something
                           failed.  The message should be a human readable description of what called the failure
    ``'Found Results'``    The points of interest that were found in the search region.  This is only present if the fit
                           failed because there were more than 1 point of interest in the search region.  The value to
                           this key is the return from :meth:`.ImageProcessing.locate_subpixel_poi_in_roi`
    ====================== =============================================================================================

    .. warning::
        Before calling the :meth:`.estimate` method be sure that the scene has been updated to correspond to the correct
        image time.  This class does not update the scene automatically.
    """

    technique = 'unresolved'
    """
    The name of the technique identifier in the :class:`.RelativeOpNav` class.
    """

    observable_type = [RelNavObservablesType.CENTER_FINDING]
    """
    The type of observables this technique generates.
    """

    def __init__(self, scene: Scene, camera: Camera, 
                options: Optional[UnresolvedCenterFindingOptions] = None):
        """
        :param scene: The :class:`.Scene` object containing the target, light, and obscuring objects.
        :param camera: The :class:`.Camera` object containing the camera model and images to be utilized
        :param image_processing: The :class:`.ImageProcessing` object to be used to process the images
        :param options: A dataclass specifying the options to set for this instance.
        """
        
        super().__init__(UnresolvedCenterFindingOptions, scene, camera, options=options)
        
        self.point_of_interest_finder = PointOfInterestFinder(self.point_of_interest_finder_options)
        """
        The instance of the point of interest finder to use when identifying the center of the uneresolved target
        """


    def estimate(self, image: OpNavImage, include_targets: Optional[List[bool]] = None):
        """
        This method extracts the observed sub-pixel centers for each requested target object from the supplied image.

        The method works by first predicting the center of the target objects, then searching for bright spots around
        the predicted centers, and finally identifying the subpixel centers of the bright spots.  For a more in depth
        discussion refer to the :class:.UnresolvedCenterFinding` documentation.  The results are stored into the
        :attr:`computed_bearings`, :attr:`observed_bearings`, and :attr:`details` attributes. If a target object
        cannot be matched to an observed bright spot then a warning is printed and NaN values are stored.

        .. warning::
            Before calling this method be sure that the scene has been updated to correspond to the correct
            image time.  This method does not update the scene automatically.

        :param image: The image the unresolved algorithm should be applied to as an OpNavImage
        :param include_targets: An argument specifying which targets should be processed for this image.  If ``None``
                                then all are processed (no, the irony is not lost on me...)
        """
        
        if self.scene.light_obj is None:
            raise ValueError('The light_obj cannot be None at this point')
        
        lpos = getattr(self.scene.light_obj.shape, "position", self.scene.light_obj.position)

        # process each requested target
        for target_ind, target in self.target_generator(include_targets):

            # store the relative position
            relative_position = target.position.ravel()

            # predict where the target should be
            predicted = self.camera.model.project_onto_image(relative_position, temperature=image.temperature)

            self.computed_bearings[target_ind] = predicted
            
            # check if this is being obscured by anything in the scene so we can't see it
            # also store the closest other object in the scene in the image

            # make the ray that starts at the target and points to the camera
            vis_ray = Rays(relative_position.ravel(),
                           -relative_position.ravel() / np.linalg.norm(relative_position))

            # trace a ray from the object to the sun.  If we strike something the object is shadowed
            shad_dir = lpos.ravel() - relative_position.ravel()
            shad_ray = Rays(relative_position.ravel(), shad_dir / np.linalg.norm(shad_dir))

            if self.scene.obscuring_objs is not None:
                possible_obscurers = self.scene.target_objs + self.scene.obscuring_objs
            else:
                possible_obscurers = self.scene.target_objs

            stop_processing: bool = False  # a flag specifying whether we need to stop because this is obscured

            # the closest distance from this target to any other object in the scene in pixels
            closest_other_distance: float = float(max(image.shape))*100

            for obscurer in possible_obscurers:
                if obscurer is target:
                    continue
                
                if not isinstance(obscurer.shape, Traceable):
                    raise ValueError('The obscurer must be traceable')

                # trace a ray from the target to the camera to see if we strike anything on the way
                ores = obscurer.shape.trace(vis_ray)

                if ores["check"].any():
                    print('Target is obscured', flush=True)
                    self.observed_bearings[target_ind] = np.array([np.nan, np.nan])
                    stop_processing = True
                    self.details[target_ind] = {'Failed': 'target is obscured'}
                    break

                # trace a ray from the target to the sun to see if we strike anything
                shad_res = obscurer.shape.trace(shad_ray)

                if shad_res["check"].any():
                    print('Target is shadowed', flush=True)
                    self.observed_bearings[target_ind] = np.array([np.nan, np.nan])
                    self.details[target_ind] = {'Failed': 'target is shadowed'}
                    stop_processing = True
                    break

                image_distance = float(np.linalg.norm(self.camera.model.project_onto_image(obscurer.position.ravel(),
                                                                                     temperature=image.temperature) -
                                                      predicted))

                closest_other_distance = min(closest_other_distance, image_distance)

            # if the target was obscured or shadowed skip it
            if stop_processing:
                continue

            if closest_other_distance < self.search_distance:
                warnings.warn(f"The target is with {closest_other_distance} pixels of another object in the scene but "
                              f"the search distance is {self.search_distance}")

            # determine the pixels we will search for the target in
            predicted_int = predicted.round().astype(np.int64)
            lr_search = predicted_int[0] + np.arange(-self.search_distance, self.search_distance)
            ud_search = predicted_int[1] + np.arange(-self.search_distance, self.search_distance)
            
            # check that we are within the field of view for our search region
            top_bound = ud_search < 0
            bottom_bound = ud_search >= image.shape[0]
            left_bound = lr_search < 0
            right_bound = lr_search >= image.shape[1]
            fov_test = top_bound | left_bound | bottom_bound | right_bound

            if fov_test.all():
                print('target is outside of FOV', flush=True)
                self.observed_bearings[target_ind] = np.array([np.nan, np.nan])
                self.details[target_ind] = {'Failed': 'target search region is completely outside of FOV'}
                continue

            # build the region of interest to search
            # noinspection PyTypeChecker
            roi: Tuple[np.ndarray, np.ndarray] = cast(tuple[np.ndarray, np.ndarray], np.meshgrid(lr_search[~fov_test], ud_search[~fov_test])[::-1])

            # use the star centroiding to find the pixel level location of the object
            # note that this will call denoise_image for us if it is turned on
            res = self.point_of_interest_finder(image, roi)

            # parse the output
            points = np.atleast_2d(res.centroids)

            if len(points) > 1:
                # TODO: consider picking the closest point to the a priori in this instance and throwing a warning
                warnings.warn("too many points of interest found in the search region for epoch: {0} target: {1}.\n"
                              "Please adjust your search distance/poi "
                              "size or manually specify the poi\n".format(image.observation_date, target_ind))
                self.observed_bearings[target_ind] = np.array([np.nan, np.nan])
                self.details[target_ind] = {'Failed': 'Too many points of interested found in search region',
                                            'found_results': res}
                continue

            elif len(points) == 0:
                warnings.warn("No points of interest found in the search region for epoch: {0} obj: {1}.\n"
                              "Please adjust your search distance/poi "
                              "size or manually specify the poi\n".format(image.observation_date,
                                                                          target_ind))
                self.observed_bearings[target_ind] = np.array([np.nan, np.nan])
                self.details[target_ind] = {'Failed': 'No points of interest found in search region'}

                continue

            elif np.isfinite(points).all():

                # store the found center
                self.observed_bearings[target_ind] = points.ravel()

                # if we are applying the phase correction, compute it
                if self.apply_phase_correction:
                    correction = self.compute_phase_correction(target_ind, target, image.temperature)

                    self.observed_bearings[target_ind] += correction
                else:
                    correction = np.zeros(2, dtype=np.float64)

                self.details[target_ind] = {'Point of Interest Results': res,
                                            'Phase Correction': correction,
                                            }

            else:
                warnings.warn('unable to locate subpixel center for epoch: {0} obj: {1}'.format(image.observation_date,
                                                                                                target_ind))
                self.details[target_ind] = {'Failed': "Bad PSF fit"}

                self.observed_bearings[target_ind] = np.array([np.nan, np.nan])
                continue
