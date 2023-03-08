# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This module provides a class which implements a moment based (center of illumination) center finding RelNav technique.

Description of the Technique
----------------------------

The moment algorithm is the technique that you typically use when your target begins to become resolved in your images,
but you still don't have an accurate shape model for doing a more advanced technique like :mod:`.limb_matching` or
:mod:`cross_correlation`.  Generally, this only is used for a short while when the target is between 5 and 100 pixels
in apparent diameter) as you attempt to build a shape model of the target to begin using the more advanced and more
accurate techniques, however, there is no hard limit on when you can and can't use this technique.  You can even use it
when the target is still unresolved or when the target is very large in the image, but in these cases (as in most cases)
there are much more accurate methods that can be used.

In order to extract the center finding observables from this method a few steps are followed.  First, we predict roughly
how many pixels we expect the illuminated portion our target to subtend based on the a priori scene knowledge and
assuming a spherical target.  We then use this predicted area to set the minimum number of connected pixels we are
going to consider a possible target in the image (this can be turned off using option :attr:`.use_apparent_area`.
We then segment the image into foreground/background objects using method :meth:`.segment_image` from image processing.
For each target in the image we are processing, we then identify the closest segmented object from the image to the
target and assume that this is the location of the target in the actual image (if you have multiple targets in an image
then it is somewhat important that your a priori scene is at least moderately accurate to ensure that this pairing works
correctly).  Finally, we take the foreground objects around the identified segment (to account for possible portions of
the target that may be separated from the main clump of illumination, such as along the limb) and compute the center of
illumination using a moment algorithm.  The center of illumination is then corrected for phase angle effects (if
requested) and the resulting center-of-figure measurements are stored.

Tuning
------

There are a few things that can be tuned for using this technique.  The first set is the tuning parameters for
segmenting an image into foreground/background objects from the :class:`.ImageProcessing` class.  These are

============================================= ==========================================================================
Parameter                                     Description
============================================= ==========================================================================
:attr:`.ImageProcessing.otsu_levels`          The number of levels to attempt to segments the histogram into using
                                              multi-level Otsu thresholding.
:attr:`.ImageProcessing.minimum_segment_area` The minimum size of a segment for it to be considered a foreground object.
                                              This can be determined automatically using the :attr:`use_apparent_area`
                                              flag of this class.
:attr:`.ImageProcessing.minimum_segment_dn`   The minimum DN value for a segment to be considered foreground.  This can
                                              be used to help separate background segments that are slightly brighter
                                              due to stray light or other noise issues.
============================================= ==========================================================================

For more details on using these attributes see the :meth:`.ImageProcessing.segment_image` documentation.

In addition, there are some tuning parameters on this class itself.  The first is the search radius.
The search radius is controlled by :attr:`search_distance` attribute.  This should be a number or ``None``.
If this is not ``None``, then the distance from the centroid of the nearest segment to the predicted target u
location must be less than this value.  Therefore, you should set this value to account for the expected
center-of-figure to center-of-brightness shift as well as the uncertainty in the a priori location of the target
in the scene, while being careful not to set too large of a value if there are multiple targets in the scene to
avoid ambiguity.  If this is ``None``, then the closest segment is always paired with the target (there is no
search region considered) unless the segment has already been paired to another target in the scene.

This technique can predict what the minimum segment area should be in the image using the predicted apparent areas
for each target.  This can be useful to automatically set the :attr:`.ImageProcessing.minimum_segment_area` based on
the targets and the a priori location in the camera frame.  Because this is just an approximation, a margin of
safety is included with :attr:`apparent_area_margin_of_safety`, which is used to shrink the predicted apparent area
to account for the assumptions about the spherical target and possible errors in the a priori scene information.
You can turn off this feature and just use the set minimum segment area by setting :attr:`use_apparent_area` to
``False``.

Whether the phase correction is applied or not is controlled by the boolean flag :attr:`apply_phase_correction`.
The information that is passed to the phase correction routines are controlled by the :attr:`phase_correction_type`
and :attr:`brdf` attributes.

Use
---

The class provided in this module is usually not used by the user directly, instead it is usually interfaced with
through the :class:`.RelativeOpNav` class using the identifier :attr:`~.RelativeOpNav.moment_algorithm`.  For more
details on using the :class:`.RelativeOpNav` interface, please refer to the :mod:`.relnav_class` documentation.  For
more details on using the technique class directly, as well as a description of the ``details`` dictionaries produced
by this technique, refer to the following class documentation.
"""

import warnings

from typing import Union, Optional, List, Dict, Any

import numpy as np

import cv2

from giant.point_spread_functions import Moment

from giant.camera import Camera
from giant.image import OpNavImage
from giant.image_processing import ImageProcessing
from giant.ray_tracer.scene import Scene
from giant.ray_tracer.illumination import IlluminationModel
from giant._typing import Real


from giant.relative_opnav.estimators.estimator_interface_abc import RelNavObservablesType
from giant.relative_opnav.estimators.unresolved import PhaseCorrector, PhaseCorrectionType



class MomentAlgorithm(PhaseCorrector):
    """
    This class implements GIANT's version of moment based center finding for extracting bearing measurements to resolved
    or or unresolved targets in an image.

    The class provides an interface to perform moment based center for each target body that is predicted to be in an
    image.  It does this by looping through each target object contained in the :attr:`.Scene.target_objs` attribute
    that is is requested.  For each of the targets, the algorithm:

    #. Predicts the location of the target in the image using the a priori knowledge of the scene
    #. Predicts the apparent area of the target in the scene assuming a spherical target.
    #. Segments the image into foreground/background objects using the smallest expected apparent area of all
       targets as the minimum segment area.  This is done using :meth:`.ImageProcessing.segment_image`
    #. Identifies the closest foreground segment to the predicted target location that is also within the user
       specified search radius.  If the closest segment is also the closest segment for another target in the image,
       then both targets are recorded as not found.  If no segments are within the search radius of the predicted
       target center then the target is marked as not found.
    #. Takes the foreground objects around the identified segment and finds the centroid of the illuminated areas
       using a moment algorithm to compute the observed center of brightness.
    #. If requested, corrects the observed center of brightness to the observed center of figure using the
       :meth:`.compute_phase_correction`.

    For more details on the image segmentation, along with possible tuning parameters, refer to the
    :meth:`.ImageProcessing.segment_image` documentation.

    The search radius is controlled by :attr:`.search_distance` attribute.  This should be a number or ``None``.
    If this is not ``None``, then the distance from the centroid of the nearest segment to the predicted target u
    location must be less than this value.  Therefore, you should set this value to account for the expected
    center-of-figure to center-of-brightness shift as well as the uncertainty in the a priori location of the target
    in the scene, while being careful not to set too large of a value if there are multiple targets in the scene to
    avoid ambiguity.  If this is ``None``, then the closest segment is always paired with the target (there is no
    search region considered) unless the segment has already been paired to another target in the scene.

    This technique can predict what the minimum segment area should be in the image using the predicted apparent areas
    for each target.  This can be useful to automatically set the :attr:`.ImageProcessing.minimum_segment_area` based on
    the targets and the a priori location in the camera frame.  Because this is just an approximation, a margin of
    safety is included with :attr:`apparent_area_margin_of_safety`, which is used to shrink the predicted apparent area
    to account for the assumptions about the spherical target and possible errors in the a priori scene information.
    You can turn off this feature and just use the set minimum segment area by setting :attr:`use_apparent_area` to
    ``False``.

    Whether the phase correction is applied or not is controlled by the boolean flag :attr:`apply_phase_correction`.
    The information that is passed to the phase correction routines are controlled by the :attr:`phase_correction_type`
    and :attr:`brdf` attributes.

    When all of the required data has been successfully loaded into an instance of this class, the :meth:`estimate`
    method is used to extract the observed centers of the target bodies predicted to be in the requested image.  The
    results are stored into the :attr:`observed_bearings` attribute. In addition, the predicted location for each target
    is stored in the :attr:`computed_bearings` attribute. Finally, the details about the fit are stored as a
    dictionary in the appropriate element in the :attr:`details` attribute.  Specifically, these dictionaries will
    contain the following keys.

    ====================== =============================================================================================
    Key                    Description
    ====================== =============================================================================================
    ``'Fit'``              The fit moment object.  Only available if successful.
    ``'Phase Correction'`` The phase correction vector used to convert from center of brightness to center of figure.
                           This will only be available if the fit was successful.  If :attr:`apply_phase_correction` is
                           ``False`` then this will be an array of 0.
    ``'Observed Area'``    The area (number of pixels that were considered foreground) observed for this target.
                           This is only available if the fit was successful.
    ``'Predicted Area'``   The area (number of pixels that were considered foreground) predicted for this target.
                           This is only available if the fit was successful.
    ``'Failed'``           A message indicating why the fit failed.  This will only be present if the fit failed (so you
                           could do something like ``'Failed' in moment_algorithm.details[target_ind]`` to check if
                           something failed.  The message should be a human readable description of what called the
                           failure.
    ``'Found Segments'``   All of the segments that were found in the image.  This is a tuple of all of the returned
                           values from :meth:`.ImageProcessing.segment_image`.  This is only included if the fit failed
                           for some reason.
    ====================== =============================================================================================

    .. warning::
        Before calling the :meth:`estimate` method be sure that the scene has been updated to correspond to the correct
        image time.  This class does not update the scene automatically.
    """

    technique: str = 'moment_algorithm'
    """
    The name of the technique identifier in the :class:`.RelativeOpNav` class.
    """

    observable_type: List[RelNavObservablesType] = [RelNavObservablesType.CENTER_FINDING]
    """
    The type of observables this technique generates.
    """

    def __init__(self, scene: Scene, camera: Camera, image_processing: ImageProcessing,
                 use_apparent_area: bool = True,
                 apparent_area_margin_of_safety: Real = 2, search_distance: Optional[int] = None,
                 apply_phase_correction: bool = True,
                 phase_correction_type: Union[PhaseCorrectionType, str] = PhaseCorrectionType.SIMPLE,
                 brdf: Optional[IlluminationModel] = None):
        """
        :param scene: The :class:`.Scene` object containing the target, light, and obscuring objects.
        :param camera: The :class:`.Camera` object containing the camera model and images to be utilized
        :param image_processing: The :class:`.ImageProcessing` object to be used to process the images
        :param use_apparent_area: A boolean flag specifying whether to predict the minimum apparent area we should
                                  consider when segmenting the image into foreground/background objects.
        :param apparent_area_margin_of_safety: The margin of safety we will use to decrease the predicted apparent area
                                               to account for errors in the a priori scene/shape model as well as errors
                                               introduced by assuming a spherical object.  The predicted apparent area
                                               will be divided by this number and then supplied as the
                                               :attr:`~.ImageProcessing.minimum_segment_area` attribute.  This should
                                               always be >= 1.
        :param search_distance: The search radius to search around the predicted centers for the observed centers of
                                the target objects.  This is used as a limit, so that if the closest segmented object to
                                a predicted target location is greater than this then the target is treated as not
                                found.  Additionally, if multiple segmented regions fall within this distance of the
                                target then we treat it as ambiguous and not found.
        :param apply_phase_correction: A boolean flag specifying whether to apply the phase correction to the observed
                                       center of brightness to get closer to the center of figure based on the predicted
                                       apparent diameter of the object.
        :param phase_correction_type: The type of phase correction to use.  Should be one of the PhaseCorrectionType
                                      enum values
        :param brdf: The illumination model to use to compute the illumination values if the ``RASTERED`` phase
                     correction type is used.  If the ``RASTERED`` phase correction type is not used this is ignored.
                     If this is left as ``None`` and the ``Rastered`` phase correction type is used, this will default
                     to the McEwen Model, :class:`.McEwenIllumination`.
        """
        super().__init__(scene, camera, image_processing, phase_correction_type=phase_correction_type, brdf=brdf)


        self.search_distance: Optional[int] = search_distance
        """
        Half of the distance to search around the predicted centers for the observed centers of the target objects in 
        pixels.
        
        This is also used to identify ambiguous target to segmented area pairings.  That is, if 2 segmented areas are 
        within this value of the predicted center of figure for a target, then that target is treated as not found and a
        warning is printed.
        
        If this is ``None`` then the closest segmented object from the image to the predicted center of figure of the 
        target in the image is always chosen.
        """

        self.apply_phase_correction: bool = apply_phase_correction
        """
        A boolean flag specifying whether to apply the phase correction or not
        """

        self.use_apparent_area: bool = use_apparent_area
        """
        A boolean flag specifying whether to use the predicted apparent area (number of pixels) of the illuminated 
        target in the image to threshold what is considered a foreground object in the image.
        """

        self.apparent_area_margin_of_safety: float = float(apparent_area_margin_of_safety)
        """
        The margin of safety used to decrease the predicted apparent area for each target.
        
        This value should always be >= 1, as the predicted area is divided by this to get the effective minimum apparent
        area for the targets.  This is included to account for errors in the a priori scene/shape model for the targets
        as well as the errors introduced by assuming spherical targets.  Since there is only one margin of safety for 
        all targets in a scene, you should set this based on the expected worst case for all of the targets.
        """

        self.details: List[Dict[str, Any]] = self.details
        """ 
        ====================== =============================================================================================
        Key                    Description
        ====================== =============================================================================================
        ``'Fit'``              The fit moment object.  Only available if successful.
        ``'Phase Correction'`` The phase correction vector used to convert from center of brightness to center of figure.
                               This will only be available if the fit was successful.  If :attr:`apply_phase_correction` is
                               ``False`` then this will be an array of 0.
        ``'Observed Area'``    The area (number of pixels that were considered foreground) observed for this target.
                               This is only available if the fit was successful.
        ``'Predicted Area'``   The area (number of pixels that were considered foreground) predicted for this target.
                               This is only available if the fit was successful.
        ``'Failed'``           A message indicating why the fit failed.  This will only be present if the fit failed (so you
                               could do something like ``'Failed' in moment_algorithm.details[target_ind]`` to check if
                               something failed.  The message should be a human readable description of what called the
                               failure.
        ``'Found Segments'``   All of the segments that were found in the image.  This is a tuple of all of the returned
                               values from :meth:`.ImageProcessing.segment_image`.  This is only included if the fit failed
                               for some reason.
        ====================== =============================================================================================
        """

    def estimate(self, image: OpNavImage, include_targets: Optional[List[bool]] = None):
        """
        This method extracts the observed center of figure for each requested target object from the supplied image.

        This method works by first predicting the location of the center-of-figure of the target objects in the image,
        then segmenting the current image into foreground/background objects using :meth:`.segment_image`, matching the
        expected targets with the segmented foreground objects using a nearest neighbor search, using a moment algorithm
        to compute a more accurate observed center-of-brightness for each target, and then finally correcting the
        observed center of brightness to the center of figure using a phase correction, if requested.  The results are
        stored into the :attr:`computed_bearings`, :attr:`observed_bearings`, and :attr:`details` attributes.  If a
        target object cannot be matched to an observed foreground object then a warning is printed and NaN values are
        stored. For a more in depth description of what is happening refer to the class documentation.

        .. warning::
            Before calling this method be sure that the scene has been updated to correspond to the correct
            image time.  This method does not update the scene automatically.

        :param image: The image the unresolved algorithm should be applied to as an OpNavImage
        :param include_targets: An argument specifying which targets should be processed for this image.  If ``None``
                                then all are processed (no, the irony is not lost on me...)
        """
        # store the original segment area from image processing in case we overwrite it
        original_segment_area = self.image_processing.minimum_segment_area

        # loop through the requested targets and compute their expected apparent area
        expected_areas = []
        for target_ind, target in self.target_generator(include_targets):
            # compute the phase angle
            phase = self.scene.phase_angle(target_ind)

            # predict the apparent radius in pixels
            apparent_diameter = target.get_apparent_diameter(self.camera.model, temperature=image.temperature)
            apparent_radius = apparent_diameter/2

            # compute the predicted area in pixels assuming a projected circle for the illuminated limb and an ellipse
            # for the terminator
            if phase <= np.pi/2:
                # if our phase angle is less than 90 degrees then we add half the terminator ellipse area to half the
                # limb circle area.
                expected_areas.append(np.pi*apparent_radius**2/2*(1+np.cos(phase)))
            else:
                # if our phase angle is greater than 90 degrees then we subtract half the terminator ellipse area from
                # half the limb circle area.
                expected_areas.append(np.pi*apparent_radius**2/2*(1-np.cos(phase)))

        # now if we are to use the minimum area for segmenting the image
        if self.use_apparent_area:
            # get the minimum area corrected by the margin of safety
            minimum_area = min(expected_areas)/self.apparent_area_margin_of_safety
            # set it to the appropriate attribute
            self.image_processing.minimum_segment_area = minimum_area

        # Now segment the image using Otsu/connected components
        segments, foreground, segment_stats, segment_centroids = self.image_processing.segment_image(image)

        used_segments = []

        # loop through each requested target
        for target_ind, target in self.target_generator(include_targets):

            # store the relative position
            relative_position = target.position.ravel()

            # predict where the target should be
            self.computed_bearings[target_ind] = self.camera.model.project_onto_image(relative_position,
                                                                                      temperature=image.temperature)

            # figure out which segment is closest
            closest_ind = None
            closest_distance = None

            for segment_ind, centroid in enumerate(segment_centroids):
                distance = np.linalg.norm(centroid - self.computed_bearings[target_ind])

                if closest_ind is None:
                    if self.search_distance is not None:
                        if distance < self.search_distance:
                            closest_ind = segment_ind
                            closest_distance = distance
                    else:
                        closest_ind = segment_ind
                        closest_distance = distance
                else:
                    if distance < closest_distance:
                        closest_ind = segment_ind
                        closest_distance = distance

            # check if nothing met the tolerance
            if closest_ind is None:
                warnings.warn(f'No segmented foreground objects are within the requested search region of target '
                              f'{target_ind}. There were {len(segment_centroids)} found in the image.'
                              f'Please consider adjusting your search region or changing the image '
                              f'processing settings.')

                # set the failure message in details
                self.details[target_ind] = {'Failed': f"This target is not with {self.search_distance} of any segmented"
                                                      f"area in the image.  There were {len(segment_centroids)} found "
                                                      f"in the image.",
                                            "Found Segments": (segments, foreground, segment_stats, segment_centroids)}
                continue

            # check if we already used this segment
            if closest_ind in used_segments:
                # TODO: consider using the predicted area/observed area ratio to break ties.
                other_target = used_segments.index(closest_ind)
                warnings.warn(f'Target {target_ind} is closest to segment {closest_ind} which is also the closest '
                              f'segment to target {other_target}.  This ambiguity cannot be broken.  Please consider '
                              f'adjusting the apriori scene conditions to be better or adjusting the image processing '
                              f'settings to find more or less segmented object')
                # set the other observed bearing to np.nan
                self.observed_bearings[other_target][:] = np.nan

                # set this observed bearing to np.nan
                self.observed_bearings[target_ind] = np.zeros(2, dtype=np.float64)+np.nan

                # set the other details to indicate the failure
                self.details[other_target] = {'Failed': f"This target and target {target_ind} are both closest to "
                                                        f"segment {closest_ind} at {segment_centroids[closest_ind]} "
                                                        f"leading to ambiguity.",
                                              "Found Segments": (segments, foreground, segment_stats,
                                                                 segment_centroids)}

                # set the details for this target to indicate the failure
                self.details[target_ind] = {'Failed': f"This target and target {other_target} are both closest to "
                                                      f"segment {closest_ind} at {segment_centroids[closest_ind]} "
                                                      f"leading to ambiguity.",
                                            "Found Segments": (segments, foreground, segment_stats, segment_centroids)}

                continue

            # store which segment was used
            used_segments.append(closest_ind)

            # get the observed centroid for the segmented area
            # extract the region around the blob frm the found segment.  Include some extra pixels to capture things
            # like the terminator.  Use a fudge factor of 1/tenth of the sqrt of the area with a minimum of 10 pixels
            fudge_factor = max(np.sqrt(segment_stats[closest_ind, cv2.CC_STAT_AREA])*0.1, 10)
            top_left = np.floor(segment_stats[closest_ind, [cv2.CC_STAT_TOP, cv2.CC_STAT_LEFT]] -
                                fudge_factor).astype(int)
            top_left = np.maximum(top_left, 0)
            bottom_right = np.ceil(top_left + segment_stats[closest_ind, [cv2.CC_STAT_HEIGHT, cv2.CC_STAT_WIDTH]] +
                                   2*fudge_factor).astype(int)  # no need to clip bottom right because the slice will

            use_image = np.zeros(image.shape, dtype=bool)
            use_image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = \
                foreground[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

            # get the x/y pixel location where we are including in centroiding
            y, x = np.where(use_image)

            # compute the center of the illumination of the blob (centroid)
            fit = Moment.fit(x.astype(np.float64), y.astype(np.float64), image[use_image].ravel().astype(np.float64))

            # store the location of the centroid, which is the observed center of brightness
            self.observed_bearings[target_ind] = fit.centroid

            # apply the phase correction if requested
            if self.apply_phase_correction:
                correction = self.compute_phase_correction(target_ind, target, image.temperature)

                self.observed_bearings += correction

            else:
                correction = np.zeros(2, dtype=np.float64)

            # store the details about the fit
            self.details[target_ind] = {'Fit': fit,
                                        'Phase Correction': correction,
                                        'Observed Area': use_image.sum(),
                                        'Predicted Area': expected_areas[target_ind]}

        # reset the image processing minimum segment area in case we messed with it
        self.image_processing.minimum_segment_area = original_segment_area
