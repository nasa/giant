"""
This module implements opportunistic feature matching between images or between an image and a template

Description of the Technique
----------------------------

Constraint matching, or opportunistic feature matching, refers to identifying the same "feature" in 
two different images, or between an image and a rendered template.

Between Images
^^^^^^^^^^^^^^

When doing constraint matching between 2 images, the observations are not tied to known points on the 
surface of the target.  This is thereore a less accurate technique (at least from an overal orbit 
determination perspective) than traditional terrain relative navigation, as the information we extract 
only pertains to the change in the location and orientation of the camera from one image to the next 
(and in general is just an estimate of the direction of motion due to scale ambiguities).  

That being said, this can still be a powerful measurement type, particularly when fused with other data sets
that can resolve the scale ambiguity.  Additionally, it can be used even at unmapped bodies, which can
dramatically reduce the time required to be able to operate at a new target.

Between and Image and a Template
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When doing constraint matching between an image and a template we can actually tie the observations back to
a known map of the body (whatever map the template was rendered from).  This means that the measurements can
generally be treated the same as regular TRN/SFN measurements, though, if your template comes from a global 
shape model, you should expect that the resulting feature locations are less accurate than would be from
a detailed map intended to make navigation maps.  Another difference from regular TRN/SFN is that you are
unlikely to observe the same exact "feature" multiple times in different images, rather, each observation 
will be slightly different.  This removes some information from the orbit determination process as each observation
is unique and mostly unrelated to any other observations (whereas in traditional TRN/SFN, we can do things like 
estimate both the spacecraft state and the feature locations since we receive multiple observations of the same
feature from different images).

The primary benefit to constraint matching in this case is again that you can begin doing a form of TRN much earlier 
in operations, before there has been time to make a detailed set of maps for navigation purposes.

Tuning
------

There are several tuning parameters that can impact the performance of constraint matching, as outlined in the 
:class:`ConstraintMatchingOptions` class. In general though, the most critical tuning parameters are the choice 
of the :attr:`~ConstraintMatchingOptions.feature_matcher` and the 
:attr:`~ConstraintMatchingOptions.max_time_difference`, with the `feature_matcher` being the more important of
the two.  In general, GIANT ships with several "hand tuned" feature matchers available, including SIFT and Orb.
These can perform well between a template and an image, or between two images where the illumincation conditions
are relatively similar (with SIFT generally outperforming Orb), but they will struggle with large changes in 
illumination conditions.  Optionally, you can install the open source implementation of RoMa (as described in
:mod:`.roma_matcher`) which is a machine learning based technique for matching images.  This model, even without
additional fine tuning, has show excellent performance even in challenging illumincation condition changes and
also outperforms the hand-tuned features in cases where they are well suited

Use
---

The class provided in this module is usually not used by the user directly, instead it is usually interfaced with
through the :class:`.RelativeOpNav` class using the identifier :attr:`~.RelativeOpNav.constraint_matching`.  For more
details on using the :class:`.RelativeOpNav` interface, please refer to the :mod:`.relnav_class` documentation.  For
more details on using the technique class directly, as well as a description of the ``details`` dictionaries produced
by this technique, refer to the following class documentation.

.. warning::
    While this technique is functional, it has undergone less development and testing than other GIANT techniques
    and there could therefore be some undiscovered bugs.  Additionally, the documentation needs a little more 
    massaging.  PRs are welcome...
"""
import datetime
import warnings

from dataclasses import dataclass, field
from typing import Optional, List, Union, Tuple, Sequence

import hashlib
import numpy as np

from giant.relative_opnav.estimators.estimator_interface_abc import RelNavEstimator, RelNavObservablesType
from giant.ray_tracer.rays import Rays, compute_rays
from giant.ray_tracer.scene import Scene, SceneObject, Shape
from giant.ray_tracer.illumination import IlluminationModel, McEwenIllumination
from giant.camera import Camera
from giant.image import OpNavImage
from giant.image_processing.feature_matchers import FeatureMatcher
import giant.image_processing.feature_matchers as feature_matchers

from giant.utilities.mixin_classes import UserOptionConfigured
from giant.utilities.options import UserOptions

from giant._typing import DOUBLE_ARRAY


@dataclass
class ConstraintMatchingOptions(UserOptions):
    """
    This dataclass serves as one way to control the settings for the :class:`.ConstraintMatching` class.

    You can set any of the options on an instance of this dataclass and pass it to the
    :class:`.ConstraintMatching` class at initialization (or through the method
    :meth:`.ConstraintMatching.apply_options`) to set the settings on the class. This class is the preferred way
    of setting options on the class due to ease of use in IDEs.
    """

    match_against_template: bool = True
    """
    A flag to match keypoints between the image and a rendered template.
    """

    match_across_images: bool = False
    """
    A flag to match keypoints across multiple images.
    """

    min_constraints: int = 5
    """
    Minimum number of matched constraints in order for the constraint matching to be considered successful.
    """

    compute_constraint_positions: bool = False
    """
    A flag to compute the constraint target-fixed positions.
    """

    brdf: IlluminationModel = field(default_factory=McEwenIllumination)
    """
    The illumination model that transforms the geometric ray tracing results (see :const:`.ILLUM_DTYPE`) into an
    intensity values. Typically this is one of the options from the :mod:`.illumination` module).
    """

    rays: Union[Optional[Rays], List[Optional[Rays]]] = None
    """
    The rays to use when rendering the template.  If ``None`` then the rays required to render the template will be 
    automatically computed.  Optionally, a list of :class:`.Rays` objects where each element corresponds to the rays to 
    use for the corresponding template in the :attr:`.Scene.target_objs` list.  Typically this should be left as 
    ``None``.
    """

    grid_size: int = 1
    """
    The subsampling to use per pixel when rendering the template.  This should be the number of sub-pixels per side of 
    a pixel (that is if grid_size=3 then subsampling will be in an equally spaced 3x3 grid -> 9 sub-pixels per pixel).  
    If ``rays`` is not None then this is ignored
    """

    template_overflow_bounds: int = -1
    """
    The number of pixels to render in the template that overflow outside of the camera field of view.  Set to a number 
    less than 0 to accept all overflow pixels in the template.  Set to a number greater than or equal to 0 to limit the 
    number of overflow pixels.
    """

    max_time_difference: Optional[datetime.timedelta] = None
    """
    Maximum time difference between image observation dates for keypoints to be matched between images. Set to a
    datetime.timedelta type.  If None, then a maximum time difference will not be applied.
    """
    
    feature_matcher: FeatureMatcher = field(default_factory=feature_matchers.SIFTKeypointMatcher)
    """
    The feature matcher instance to use
    """


class ConstraintMatching(UserOptionConfigured[ConstraintMatchingOptions], ConstraintMatchingOptions, RelNavEstimator):
    """
    This class implements constraint matching in GIANT.
    
    See the module documentation or the attribute and method documentation for more details.
    
    .. warning::
        While this technique is functional, it has undergone less development and testing than other GIANT techniques
        and there could therefore be some undiscovered bugs.  Additionally, the documentation needs a little more 
        massaging.  PRs are welcome...
    """

    observable_type  = [RelNavObservablesType.CONSTRAINT]
    """
    This technique generates CONSTRAINT bearing observables
    """

    generates_templates: bool = True
    """
    A flag specifying that this RelNav estimator generates and stores templates in the :attr:`templates` attribute.
    """

    def __init__(self, scene: Scene, camera: Camera, 
                 options: Optional[ConstraintMatchingOptions] = None):
        """
        :param scene: The scene describing the a priori locations of the targets and the light source.
        :param camera: The :class:`.Camera` object containing the camera model and images to be analyzed
        :param options: A dataclass specifying the options to set for this instance.
        """

        super().__init__(ConstraintMatchingOptions, scene, camera, options=options)

        self.constraint_ids = [[] for _ in range(len(self.scene.target_objs))]
        self.constraints = [{} for _ in range(len(self.scene.target_objs))]
        
        self.constraint_positions: list[DOUBLE_ARRAY | None] = [None] * len(self.scene.target_objs)



    def render(self, target_ind: int, target: SceneObject, temperature: float = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method returns the computed illumination values for the given target and the (sub)pixels that each
        illumination value corresponds to.

        The illumination values are computed by (a) determining the rays to trace through the scene (either user
        specified or by a call to :meth:`compute_rays`), (b) performing a single bounce ray trace through the scene
        using a call to :meth:`.Scene.get_illumination_inputs`, and then (c) converting the results of the ray trace
        into illumination values using :attr:`brdf`.

        :param target_ind: index into the :attr:`.Scene.target_objs` list of the target being rendering
        :param target: the :class:`.SceneObject` for the target being rendered
        :param temperature: The temperature of the camera at the time the scene is being rendered.
        :return: the computed illumination values for each ray and the pixel coordinates for each ray.
        """

        # determine the rays to trace through the scene
        if self.rays is None:
            # compute the rays required to render the requested target
            (rays, uv), _ = self.compute_rays(target, temperature=temperature)

        elif isinstance(self.rays, Rays):

            rays = self.rays
            # compute the pixel location of each ray
            uv = self.camera.model.project_onto_image(rays.start + rays.direction, temperature=temperature)

        elif self.rays[target_ind] is None:
            # compute the rays required to render the requested target
            (rays, uv), _ = self.compute_rays(target, temperature=temperature)

        else:

            rays: Rays = self.rays[target_ind]  # type: ignore

            # compute the pixel location of each ray
            uv = self.camera.model.project_onto_image(rays.start + rays.direction, temperature=temperature)

        print('Rendering {} rays'.format(rays.num_rays), flush=True)
        illum_inputs = self.scene.get_illumination_inputs(rays)

        # compute the illumination for each ray and return it along with the pixel location for each ray
        return self.brdf(illum_inputs), uv  # type: ignore

    def compute_rays(self, target: SceneObject,
                     temperature: float = 0) -> Tuple[Tuple[Rays, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        This method computes the required rays to render a given target based on the location of the target in the
        image.

        This method first determines which pixels to trace. If a circumscribing sphere is defined for the target,
        the edges of the sphere are used to compute the required pixels; otherwise, the bounding box is used. The pixels
        are checked to be sure they are contained in the image.  If the expected target location is completely outside
        of the image then it is relocated to be in the center of the image (for rendering purposes). The requested
        subsampling for each pixel is then applied, and the sub-pixels are then converted into rays originating at the
        camera origin using the :class:`.CameraModel`.

        :param target: The target that is being rendered
        :param temperature: The temperature of the camera at the time the target is being rendered
        :return: The rays to trace through the scene and the  the pixel coordinates for each ray as a tuple, plus
                 the bounds of the pixel coordinates
        """
        local_min, local_max = target.get_bounding_pixels(self.camera.model, temperature=temperature)

        # If the template size is too small, artificially increase it so we can get the PSF in there
        if ((local_max - local_min) < 5).any():
            local_max += 3
            local_min -= 3

        # Determine the actual bounds to use for the template so that it is within the image
        if self.template_overflow_bounds >= 0:
            min_inds = np.maximum(local_min, [-self.template_overflow_bounds,
                                              -self.template_overflow_bounds])

            max_inds = np.minimum(local_max, [self.camera.model.n_cols + self.template_overflow_bounds,
                                              self.camera.model.n_rows + self.template_overflow_bounds])
        else:
            min_inds = np.maximum(local_min, [0, 0])

            max_inds = np.minimum(local_max, [self.camera.model.n_cols, self.camera.model.n_rows])

        # If the min inds are greater than the max inds then they body is outside of the FOV, move it to the center of
        # the image as a work around and restart
        if np.any(min_inds > max_inds):

            warnings.warn("the predicted location of the body is outside of the FOV.  \n"
                          "We are moving the object to the middle of the FOV and attempting to render that way.\n")

            # set the target fully along the z-axis of the camera
            pos = target.position.copy()
            pos[-1] = np.linalg.norm(pos)
            pos[:2] = 0

            # change the target position
            target.change_position(pos)

            # recurse to compute the rays for the new target location
            return self.compute_rays(target, temperature=temperature)

        else:
            # Compute the rays for the template
            if self.template_overflow_bounds >= 0:
                return compute_rays(self.camera.model, (min_inds[1], max_inds[1]), (min_inds[0], max_inds[0]),
                                    grid_size=self.grid_size, temperature=temperature), (min_inds, max_inds)
            else:
                return compute_rays(self.camera.model, (local_min[1], local_max[1]), (local_min[0], local_max[0]),
                                    grid_size=self.grid_size, temperature=temperature), (local_min, local_max)

    def _generate_keys(self, values, length: int = 12) -> List[str]:

        keys = []

        if len(values.shape) == 1:
            n_vals = 1
        else:
            n_vals = len(values)

        for i in range(n_vals):
            if n_vals == 1:
                vals = values
            else:
                vals = values[i]

            hash_int = int(str(int(hashlib.sha256(np.round(vals, decimals=8).tobytes()).hexdigest(), 16))[:length])
            keys.append(hash_int)

        return keys

    def match_image_to_template(self, image, include_targets):
        """
        Matches keypoints between an image and a rendered template.
        
        :param image: The image to locate the targets in
        :param include_targets: A list specifying whether to process the corresponding target in
                                :attr:`.Scene.target_objs` or ``None``.  If ``None`` then all targets are processed.
        """

        for target_ind, target in self.target_generator(include_targets):

            # Do the ray trace and render the template
            illums, locs = self.render(target_ind, target, temperature=image.temperature)

            # Compute the expected location of the object in the image
            target_center_bearing = self.camera.model.project_onto_image(
                target.position,
                temperature=image.temperature
            ).ravel()

            # Get the expected bounds of the template in the image
            # noinspection PyArgumentList
            bounds = (locs.min(axis=1).round(), locs.max(axis=1).round())

            # Compute the size of the template in pixels
            template_size = (bounds[1] - bounds[0]) + 1

            # Initialize the template matrix
            template = np.zeros(template_size[::-1].astype(int))

            # Compute the subscripts into the template
            subs = (locs - bounds[0].reshape(2, 1)).round().astype(int)

            # use numpy fancy stuff to add the computed brightness for each ray to the appropriate pixels
            np.add.at(template, (subs[1], subs[0]), illums.flatten())

            # Apply the psf of the camera to the template
            if self.camera.psf is not None:
                self.templates[target_ind] = self.camera.psf(template)

            # Determine the center of the body in the template
            center_of_template = target_center_bearing - bounds[0]
            
            # match keypoints between image and rendered template
            matches = self.feature_matcher.match_images(image, template)
            
            self.templates[target_ind] = template

            n_constraints = len(matches)

            # check if matched keypoints were found
            if not np.isfinite(matches).all():
                warnings.warn("Unable to match keypoints between image and rendered template.")
                self.details[target_ind] = {'Failed': 'Unable to match keypoints between image and rendered template.'}
                self.observed_bearings[target_ind] = np.nan * np.ones(2, dtype=np.float64)
                continue

            # check if the correlation score was too low
            if n_constraints < self.min_constraints:
                warnings.warn(f"Number of detected constraints ({n_constraints}) is less than the minimum "
                              f"({self.min_constraints}).")

                self.details[target_ind] = {'Failed': 'Too few constraints detected.',
                                            "Number of Constraints": n_constraints}
                self.observed_bearings[target_ind] = np.nan * np.ones(2, dtype=np.float64)
                continue

            # store the observed constraint coordinates (from the image)
            self.observed_bearings[target_ind] = matches[:, 0, :].T  # image

            # store the computed constraint coordinates (from the template)
            self.computed_bearings[target_ind] = matches[:, 1, :].T + bounds[0].reshape(2, 1) # template

            # compute and store the constraint ids, keyed from the observed constraint coordinates
            self.constraint_ids[target_ind] = self._generate_keys(matches[:, 0, :])

            # store the details of the fit
            self.details[target_ind] = {"Number of Constraints": n_constraints,
                                        "Target Template Coordinates": center_of_template}

        return

    def match_keypoints_across_images(self, image: OpNavImage, include_targets: list[bool] | None = None):
        """
        Matches keypoints across different images.
        
        .. warning::
            This currently doesn't really work if you have multiple targets in an image/scene
        
        :param image: The image to locate the targets in
        :param include_targets: A list specifying whether to process the corresponding target in
                                :attr:`.Scene.target_objs` or ``None``.  If ``None`` then all targets are processed.
        """

        for target_ind, _ in self.target_generator(include_targets):

            self.constraint_ids[target_ind] = []

            for __, image2 in self.camera:

                # TODO: replace this check with more robust check for > 1 cameras
                if image.observation_date == image2.observation_date:  # images are the same
                    continue

                # TODO: replace this check with some better logic
                if image.observation_date > image2.observation_date:  # only process images in forward order
                    print('Assuming we already processed this image pair.')
                    continue

                if (self.max_time_difference is None or
                        abs(image.observation_date - image2.observation_date) < self.max_time_difference):

                    matches = self.feature_matcher.match_images(image, image2)

                    image_keypts = matches[:, 0, :].T
                    image2_keypts = matches[:, 1, :].T

                    for i in range(matches.shape[0]):

                        image_keypt = matches[i, 0, :].T
                        image2_keypt = matches[i, 1, :].T

                        constraint_id = self._generate_keys(image_keypt.T + np.random.rand(1))[0]

                        # TODO: Currently, constraints are not checked across images
                        if constraint_id not in self.constraint_ids[target_ind]:
                            self.constraint_ids[target_ind].append(constraint_id)
                            self.constraints[target_ind][constraint_id] = []

                        self.constraints[target_ind][constraint_id].append((
                                                image.observation_date,  np.hstack([image_keypt, [0]]),
                                                image2.observation_date, np.hstack([image2_keypt, [0]])))

                    self.observed_bearings[target_ind] = np.array((image_keypts, image2_keypts))

    def compute_constraint_position(self, image: OpNavImage, include_targets: list[bool] | None = None):
        """
        Trace from the camera to the target to estimate roughly the location that corresponds to each
        constraint on the target model.
        
        :param image: The image to locate the targets in
        :param include_targets: A list specifying whether to process the corresponding target in
                                :attr:`.Scene.target_objs` or ``None``.  If ``None`` then all targets are processed.
        """

        for target_ind, target in self.target_generator(include_targets):

            constraint_rays, _ = compute_rays(self.camera.model,
                                              self.computed_bearings[target_ind][1, :],  # type: ignore
                                              self.computed_bearings[target_ind][0, :],  # type: ignore
                                              temperature=image.temperature)

            assert isinstance(target.shape, Shape)
            intersects = target.shape.trace(constraint_rays)

            # TODO: Check if any intersects are False

            camera_to_body_rotation = target.orientation.inv() 

            intersect_positions_body_frame = (camera_to_body_rotation.matrix @
                                              intersects["intersect"].T).T

            camera_to_target_position_camera = target.position
            camera_to_target_body_frame = camera_to_body_rotation.matrix @ camera_to_target_position_camera

            self.constraint_positions[target_ind] = intersect_positions_body_frame - camera_to_target_body_frame

        return

    def estimate(self, image: OpNavImage, include_targets: Optional[List[bool]] = None):
        """
        Do the estimation according to the current settings
        
        .. warning::
            Before calling this method be sure that the scene has been updated to correspond to the correct
            image time.  This method does not update the scene automatically, even if the :attr:`scene` attribute is an
            :class:`.Scene` instance.

        :param image: The image to locate the targets in
        :param include_targets: A list specifying whether to process the corresponding target in
                                :attr:`.Scene.target_objs` or ``None``.  If ``None`` then all targets are processed.
        """

        if self.match_against_template is False and self.match_across_images is False:
            raise ValueError('At least one of the match_against_template or match_across_images flags must be True.')

        if self.match_against_template:
            self.generates_templates = True
            self.match_image_to_template(image, include_targets)

        if self.match_across_images:
            self.match_keypoints_across_images(image, include_targets)

        if self.compute_constraint_positions:  # raytrace the predicted keypoint coordinate to the shape to compute the 'target position'
            self.compute_constraint_position(image, include_targets)
