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
for the limb extraction routines.  There are 2 categories of routines you can choose from.  The first is edge
detection, where the limbs are extracted using only the image and the sun direction.  To tune the edge detection limb
extraction routines refer to the :class:`.LimbEdgeDetection` class.

The other option for limb extraction is limb scanning.  In limb scanning, predicted illumination values based on the
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

from typing import List, Optional,  cast

import numpy as np

from dataclasses import dataclass, field

from giant.relative_opnav.estimators.estimator_interface_abc import RelNavEstimator, RelNavObservablesType
from giant.relative_opnav.estimators.moment_algorithm import MomentAlgorithm, MomentAlgorithmOptions

from giant.ray_tracer.shapes import Ellipsoid

from giant.camera import Camera
from giant.image import OpNavImage
from giant.ray_tracer.scene import Scene
from giant.ray_tracer._typing import HasReferenceEllipsoid
from giant._typing import NONEARRAY, DOUBLE_ARRAY, ARRAY_LIKE
from giant.relative_opnav.estimators._limb_pairer import LimbPairer, LimbPairerOptions, LimbExtractionMethods



@dataclass
class EllipseMatchingOptions(LimbPairerOptions):
    """
    Options for configuring the EllipseMatching class
    
    Generally, the default options are what you want, which use subpixel edge detection to identify limbs in the image.
    """
    
    recenter: bool = True
    """
    A flag specifying whether to locate the center of the target using a moment algorithm before beginning.

    If the a priori knowledge of the bearing to the target is poor (outside of the body) then this flag will help
    to correct the initial error.  See the :mod:`.moment_algorithm` module for details.
    
    This is only used if the extraction method is set to LIMB_SCANNING as it does not affect the EDGE_DETECTION extraction method.
    """
   
    
class EllipseMatching(LimbPairer, RelNavEstimator, EllipseMatchingOptions):
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

    technique = 'ellipse_matching'
    """
    The name of the technique identifier in the :class:`.RelativeOpNav` class.
    """

    observable_type = [RelNavObservablesType.LIMB, RelNavObservablesType.RELATIVE_POSITION]
    """
    The type of observables this technique generates.
    """

    def __init__(self, scene: Scene, camera: Camera, 
                 options: Optional[EllipseMatchingOptions] = None):
        """
        :param scene: The :class:`.Scene` object containing the target, light, and obscuring objects.
        :param camera: The :class:`.Camera` object containing the camera model and images to be utilized
        :param image_processing: The :class:`.ImageProcessing` object to be used to process the images
        :param options: A dataclass specifying the options to set for this instance.
        """

        # store the scene and camera in the class instance using the super class's init method
        super().__init__(EllipseMatchingOptions, scene, camera, options=options)
        
        self.limbs_camera: List[NONEARRAY] = [None] * len(self.scene.target_objs)
        """
        The limb surface points with respect to the center of the target

        Until :meth:`estimate` is called this list will be filled with ``None``.

        Each element of this list corresponds to the same element in the :attr:`.Scene.target_objs` list.
        """


        moment_options = MomentAlgorithmOptions(apply_phase_correction=False, use_apparent_area=True)
        # the moment algorithm instance to use if recentering has been requested.
        self._moment_algorithm: MomentAlgorithm = MomentAlgorithm(scene, camera, options=moment_options)
        """
        The moment algorithm instance to use to recenter if we are using limb scanning
        """


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
            # If we were requested to recenter using a moment algorithm then do it
            if self.recenter:
                print('recentering', flush=True)
                # Estimate the center using the moment algorithm to get a fast rough estimate of the cof
                self._moment_algorithm.estimate(image, include_targets=include_targets)

                for target_ind, target in self.target_generator(include_targets):
                    target_observed = self._moment_algorithm.observed_bearings[target_ind]
                    if target_observed is not None and np.isfinite(target_observed).all():
                        new_position = self.camera.model.pixels_to_unit(target_observed, temperature=image.temperature)
                        new_position *= np.linalg.norm(target.position)
                        target.change_position(new_position)

        # loop through each object in the scene
        for target_ind, target in self.target_generator(include_targets):
            target_observed_bearings = self.observed_bearings[target_ind]
            if target_observed_bearings is None:
                raise ValueError('no observed limbs for target {target_ind}')

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
            valid_test = ~np.isnan(target_observed_bearings).any(axis=0)

            self.observed_bearings[target_ind] = target_observed_bearings[:, valid_test]

            # extract the shape from the object and see if it is an ellipsoid
            shape = target.shape
            

            if not isinstance(shape, Ellipsoid) and isinstance(shape, HasReferenceEllipsoid):
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
            unit_vectors_camera = self.camera.model.pixels_to_unit(target_observed_bearings,
                                                                   temperature=image.temperature)

            # Rotate the unit vectors into the principal axis frame
            unit_vectors_principal: DOUBLE_ARRAY = shape.orientation.T @ unit_vectors_camera

            # Now, convert into the unit sphere space and make unit vectors
            right_sphere_cone_vectors = q_matrix @ unit_vectors_principal
            right_sphere_cone_size: DOUBLE_ARRAY = np.linalg.norm(right_sphere_cone_vectors, axis=0, keepdims=True)
            right_sphere_cone_vectors /= right_sphere_cone_size

            # Now, we solve for the position in the sphere space
            location_sphere: DOUBLE_ARRAY = cast(DOUBLE_ARRAY,
                                                 np.linalg.lstsq(right_sphere_cone_vectors.T,
                                                                 np.ones(right_sphere_cone_vectors.shape[1]),
                                                                 rcond=None)[0])

            # Finally, get the position in the camera frame and store it
            target_observed_position: DOUBLE_ARRAY = (shape.orientation @ q_inv @ location_sphere /
                                                      np.sqrt(np.inner(location_sphere, location_sphere) - 1)).ravel()
            self.observed_positions[target_ind] = target_observed_position

            # Get the limb locations in the camera frame
            scan_center_dir = target_observed_position.copy()
            scan_center_dir /= np.linalg.norm(scan_center_dir)

            scan_dirs = unit_vectors_camera/unit_vectors_camera[2] - scan_center_dir.reshape(3, 1) / scan_center_dir[2]
            scan_dirs /= np.linalg.norm(scan_dirs[:2], axis=0, keepdims=True)

            target_limbs_camera = shape.find_limbs(scan_center_dir, scan_dirs,
                                                   shape.center - target_observed_position.ravel())

            self.limbs_camera[target_ind] = target_limbs_camera
            self.computed_bearings[target_ind] = self.camera.model.project_onto_image(target_limbs_camera,
                                                                                      temperature=image.temperature)

            target_centered_fixed_limb = target_limbs_camera - \
                                         target_observed_position.reshape(3, 1)
            target_centered_fixed_limb = target.orientation.matrix.T @ target_centered_fixed_limb

            # compute the covariance matrix
            delta_ns = location_sphere.reshape(3, 1) - right_sphere_cone_vectors

            decomp_matrix = (q_matrix@shape.orientation.T).reshape(1, 3, 3)
            meas_std = (target_observed_bearings - self.computed_bearings[target_ind]).std(axis=-1)
            model_jacobian = self.camera.model.compute_unit_vector_jacobian(target_observed_bearings,
                                                                            temperature=image.temperature)
            meas_cov = model_jacobian@np.diag(meas_std**2)@model_jacobian.swapaxes(-1, -2)
            inf_eta = np.diag((right_sphere_cone_size.ravel()**-2 *
                                (delta_ns.T.reshape(-1, 1, 3) @
                                decomp_matrix @
                                meas_cov @
                                decomp_matrix.swapaxes(-1, -2) @
                                delta_ns.T.reshape(-1, 3, 1)).squeeze())**(-1))

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

        self.limbs_camera = [None] * len(self.scene.target_objs)
