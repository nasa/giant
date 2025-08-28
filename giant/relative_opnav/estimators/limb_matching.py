


r"""
This module provides the capability to locate the relative position of any target body by matching the observed limb in
an image with the shape model of the target.

Description of the Technique
----------------------------

Limb matching is a form of OpNav which produces a full 3DOF relative position measurement between the target and the
camera.  It is a sister technique of ellipse matching, but extended to general bodies.  It does this by matching
observed limb points in an image to surface points on the shape model and then solving the PnP problem (essentially
triangulation). As such, this can be a very powerful measurement because it is less sensitive to errors in the a priori
knowledge of your range to the target than cross correlation, provides more information than just the bearing to the
target for processing in a filter, and is more computationally efficient.  That being said, the line-of-sight/bearing
component of the estimate is generally slightly less accurate than cross correlation (when there is good a priori
knowledge of the shape and the range to the target). This is because limb matching only makes use of the visible
limb, while cross correlation makes use of all of the visible target.

Because matching the observed limb to a surface point is not a well defined problem for general bodies (not ellipsoidal)
this technique is iterative.  It keeps pairing the observed limbs with the correct surface points as the relative
position between the target and the camera is refined.  In addition, the limb pairing process needs the a priori
bearing of the target to be fairly close to the actual location of the target in the image.  Therefore, the algorithm
generally proceeds as follows:

#. If requested, identify the center of the target in the image using a moment algorithm (:mod:`.moment_algorithm`) and
   move the target's a priori to be along the line of sight identified using the moment algorithm.
#. Identify the observed illuminate limb of the target in the image being processed using
   :meth:`.ImageProcessing.identify_subpixel_limbs` or :class:`.LimbScanner`
#. Pair the extracted limb points to possible surface points on the target shape using the current estimate of the state
#. Solve a linear least squares problem to update the state
#. Repeat steps 2-4 until convergence or maximum number of iterations exceeded

Further details on the algorithm can be found `here <https://bit.ly/3mQnB5J>`_.

.. note::

    This implements limb based OpNav for irregular bodies.  For regular bodies, like planets and moons, see
    :mod:`.ellipse_matching` which will be more efficient and accurate.

Typically this technique is used once the body is fully resolved in the image (around at least 50 pixels in apparent
diameter) and then can be used as long as the limb is visible in the image. For accurate results, this does require an
accurate shape model of the target, at least up to an unknown scale.  In addition, this technique can be sensitive to
errors in the knowledge of the relative orientation of the target frame to the image frame, therefore you need to have a
pretty good idea of its pole and spin state.  If you don't have these things then this technique may still work but with
degraded results.  For very irregular bodies (bodies that are not mostly convex) this technique may be more dependent on
at least a decent a priori relative state between the camera and the target, as if the initial limb pairing is very far
off it may never recover.

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

In addition, there are a few knobs that can be tweaked on the class itself.

========================================== =============================================================================
Parameter                                  Description
========================================== =============================================================================
:attr:`.LimbMatching.extraction_method`    Chooses the limb extraction method to be image processing or limb scanning.
:attr:`.LimbMatching.max_iters`            The maximum number of iterations to perform.
:attr:`.LimbMatching.recenter`             A flag specifying whether to use a moment algorithm to set the initial guess
                                           at the line of sight to the target or not.  If your a priori state knowledge
                                           is bad enough that the predicted location of the target is outside of the
                                           observed target in the image then you should set this to ``True``.
:attr:`.LimbMatching.discard_outliers`     A flag specifying whether to remove outliers each iteration step.  Generally
                                           this should be left to ``True``.
========================================== =============================================================================


Beyond this, you only need to ensure that you have a fairly accurate shape model of the target, the knowledge of the sun
direction in the image frame is good, and the knowledge of the rotation between the principal frame and the camera frame
is good.

Use
---

The class provided in this module is usually not used by the user directly, instead it is usually interfaced with
through the :class:`.RelativeOpNav` class using the identifier :attr:`~.RelativeOpNav.limb_matching`.  For more
details on using the :class:`.RelativeOpNav` interface, please refer to the :mod:`.relnav_class` documentation.  For
more details on using the technique class directly, as well as a description of the ``details`` dictionaries produced
by this technique, refer to the following class documentation.
"""

import warnings

from typing import Union, Optional, List
from dataclasses import dataclass

import numpy as np

from giant.relative_opnav.estimators.estimator_interface_abc import RelNavObservablesType, RelNavEstimator
from giant.utilities.outlier_identifier import get_outliers
from giant.image import OpNavImage
from giant.camera import Camera
from giant.ray_tracer.scene import SceneObject, Scene

from giant.relative_opnav.estimators._limb_pairer import LimbPairer, LimbPairerOptions
from giant.relative_opnav.estimators.moment_algorithm import MomentAlgorithm, MomentAlgorithmOptions

from giant._typing import NONEARRAY

@dataclass
class LimbMatchingOptions(LimbPairerOptions):
    """
    :param extraction_method: The method to use to extract the observed limbs from the image.  Should be
                                ``'LIMB_SCANNING'`` or ``'EDGE_DETECTION'``.  See :class:`.LimbExtractionMethods` for
                                details.
    :param limb_edge_detection_options: The options to use to configure the limb edge detector
    :param limb_scanner_options: The options to use to configure the limb scanner
    :param state_atol: the absolute tolerance state convergence criteria (np.abs(update) < state_atol).all())
    :param state_rtol: the relative tolerance state convergence criteria (np.abs(update)/state < state_rtol).all())
    :param residual_atol: the absolute tolerance residual convergence criteria
    :param residual_rtol: the relative tolerance residual convergence criteria
    :param max_iters: maximum number of iterations for iterative horizon relative navigation
    :param recenter: A flag to estimate the center using the moment algorithm to get a fast rough estimate of the
                        center-of-figure
    :param discard_outliers: A flag to use Median Absolute Deviation to find outliers and get rid of
                                them
    :param create_gif: A flag specifying whether to build a gif of the iterations.
    :param gif_file: the file to save the gif to, optionally with 2 positional format arguments for the image date
                        and target name being processed
    :param interpolator: The type of image interpolator to use if the extraction method is set to LIMB_SCANNING.
    """


    state_atol: float = 1e-6
    """
    The absolute tolerance state convergence criteria (np.abs(update) < state_atol).all())
    """

    state_rtol: float = 1e-4
    """
    The relative tolerance state convergence criteria (np.abs(update)/state < state_rtol).all())
    """

    residual_atol: float = 1e-10
    """
    The absolute tolerance convergence criteria for residuals 
    (abs(new_resid_ss - old_resid_ss) < residual_atol).all())
    """

    residual_rtol: float = 1e-4
    """
    The relative tolerance convergence criteria for residuals
    (abs(new_resid_ss - old_resid_ss)/old_resid_ss < residual_rtol).all())
    """

    max_iters: int = 10
    """
    The maximum number of iterations to attempt in the limb-matching algorithm.
    """

    recenter: bool = True
    """
    A flag to estimate the center using the moment algorithm to get a fast rough estimate of the
    center-of-figure
    """
    
    discard_outliers: bool = True
    """
    A flag specifying whether to attempt to remove outliers in the limb pairs each iteration. 
    
    For most targets this flag is strongly encouraged.
    """

    create_gif: bool = True
    """
    A flag specifying whether to create a gif of the iteration process for review.
    """

    gif_file: str = 'limb_match_summary_{}_{}.gif'
    """
    The file to save the gif to.
    
    This can optionally can include 2 format locators for the image date and target name to distinguish the gif 
    files from each other.  The image date will be supplied as the first argument to format and the target name will 
    be supplied as the second argument.
    """


class LimbMatching(LimbPairer, RelNavEstimator, LimbMatchingOptions):
    """
    This class implements GIANT's version of limb based OpNav for irregular bodies.

    The class provides an interface to perform limb based OpNav for each target body that is predicted to be in an
    image.  It does this by looping through each target object contained in the :attr:`.Scene.target_objs` attribute
    that is requested.  For each of the targets, the algorithm:

    #. Places the target along the line of sight identified from the image using the :mod:`.moment_algorithm` if
       requested
    #. Extracts observed limb points from the image and pairs them with the target based on the expected apparent
       diameter of the target and the extent of the identified limbs
    #. Identifies what points on the surface of the target likely correspond to the identified limb points in the image
    #. Computes the update to the relative position between the target and the camera that better aligns the observed
       limbs with the predicted limb points on the target surface.

    Steps 2-4 are repeated until convergence, divergence, or the maximum number of iteration steps are performed.

    In step 3, the paired image limb to surface points are filtered for outliers using the :func:`.get_outliers`
    function, if requested with the :attr:`discard_outliers` attribute.

    The convergence for the technique is controlled through the parameters :attr:`max_iters`, :attr:`state_rtol`,
    :attr:`state_atol`, :attr:`residual_rtol`, and :attr:`residual_atol`.  If the fit diverges or is unsuccessful for
    any reason, iteration will stop and the observed limb points and relative position will be set to NaN.

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
    ``'Jacobian'``              The Jacobian matrix from the last completed iteration.  Only available if successful.
    ``'Inlier Ratio'``          The ratio of inliers to outliers for the last completed iteration.  Only available if
                                successful.
    ``'Covariance'``            The 3x3 covariance matrix for the estimated relative position in the camera frame based
                                on the residuals.  This is only available if successful
    ``'Number of iterations'``  The number of iterations that the system converged in.  This is only available if
                                successful.
    ``'Surface Limb Points'``   The surface points that correspond to the limb points in the target fixed target
                                centered frame.
    ``'Failed'``                A message indicating why the fit failed.  This will only be present if the fit failed
                                (so you could do something like ``'Failed' in limb_matching.details[target_ind]`` to
                                check if something failed.  The message should be a human readable description of what
                                called the failure.
    ``'Prior Residuals'``       The sum of square of the residuals from the prior iteration.  This is only available if
                                the fit failed due to divergence.
    ``'Current Residuals'``     The sum of square of the residuals from the current iteration.  This is only available
                                if the fit failed due to divergence.
    =========================== ========================================================================================

    .. warning::
        Before calling the :meth:`estimate` method be sure that the scene has been updated to correspond to the correct
        image time.  This class does not update the scene automatically.
    """

    technique = 'limb_matching'
    """
    The name of the technique identifier in the :class:`.RelativeOpNav` class.
    """

    observable_type = [RelNavObservablesType.LIMB, RelNavObservablesType.RELATIVE_POSITION]
    """
    The type of observables this technique generates.
    """

    def __init__(self, scene: Scene, camera: Camera, options: Optional[LimbMatchingOptions] = None):
        """
        :param scene: The :class:`.Scene` object containing the target, light, and obscuring objects.
        :param camera: The :class:`.Camera` object containing the camera model and images to be utilized
        :param image_processing: The :class:`.ImageProcessing` object to be used to process the images
        :param options: A dataclass specifying the options to set for this instance.
        """
        
        super().__init__(LimbMatchingOptions, scene, camera, options=options)
        
        moment_options = MomentAlgorithmOptions(apply_phase_correction=False, use_apparent_area=True)
        # the moment algorithm instance to use if recentering has been requested.
        self._moment_algorithm: MomentAlgorithm = MomentAlgorithm(scene, camera, options=moment_options)
        """
        The moment algorithm instance to use to recenter if we are using limb scanning
        """
        
        self.limbs_camera: List[NONEARRAY] = [None] * len(self.scene.target_objs)
        """
        The limb surface points with respect to the center of the target

        Until :meth:`estimate` is called this list will be filled with ``None``.

        Each element of this list corresponds to the same element in the :attr:`.Scene.target_objs` list.
        """

        # these attributes are used for handling the gif generation and should not be modified by the user
        self._gif_ax = None
        self._gif_fig = None
        self._gif_writer = None
        self._gif_limbs_line = None
        
    def compute_jacobian(self, target_object: SceneObject, center: np.ndarray, center_direction: np.ndarray,
                         limb_points_image: np.ndarray, limb_points_camera: np.ndarray,
                         relative_position: np.ndarray, scan_vector: np.ndarray, temperature: float = 0) -> np.ndarray:
        r"""
        This method computes the linear change in the measurements (the distance between the predicted
        and observed limb points and the scan center) with respect to a change in the state vector.

        Mathematically the rows of the Jacobian matrix are given by:

        .. math::
            \frac{\partial d_i}{\partial \mathbf{s}}=\frac{\partial d_i}{\partial\mathbf{x}_i}
            \frac{\partial\mathbf{x}_i}{\partial\mathbf{p}_c}\frac{\partial\mathbf{p}_c}{\partial\mathbf{s}}

        where :math:`d_i` is the distance along the scan vector between the predicted limb pixel location and the scan
        center, :math:`\mathbf{s}` is the the state vector that is being estimated, :math:`\mathbf{x}_i` is the pixel
        location of the predicted limb point, and :math:`\mathbf{p}_c` is the predicted limb point in the camera frame.
        In addition

        .. math::
            \frac{d_i}{\mathbf{x}_i} = \frac{\mathbf{x}_i-\mathbf{c}}{\|\mathbf{x}_i-\mathbf{c}\|}

        is the linear change in the distance given a change in the limb pixel location where :math:`\mathbf{c}` is the
        scan center, :math:`\frac{\partial\mathbf{x}_i}{\partial\mathbf{p}_c}` is the linear change in the limb pixel
        location given a change in the limb camera frame vector which is defined by the camera model, and
        :math:`\frac{\partial\mathbf{p}_c}{\partial\mathbf{s}}` is the change in the limb camera frame vector given a
        change in the state vector, which is defined by the shape model.

        :param target_object: target object under consideration
        :param center: The pixel location of the center of the scan rays
        :param center_direction: the unit vector through the pixel location of the center of the scan rays in the camera
                                 frame
        :param limb_points_image: The predicted limb locations in the image in units of pixels
        :param limb_points_camera: The predicted limb vectors in the camera frame
        :param relative_position: The current estimate of the position vector from the camera to the target
        :param scan_vector: The unit vectors from the scan center to the limb points in the image.
        :param temperature: The temperature of the camera at the time the image was captured
        :return: The nxm jacobian matrix as a numpy array
        """

        # Compute how the predicted limbs change given a change in the state vector
        limb_jacobian = target_object.shape.compute_limb_jacobian(center_direction, scan_vector, limb_points_camera)  # type: ignore

        # Predict how the pixel locations change given a change in the limb points
        camera_jacobian = self.camera.model.compute_pixel_jacobian(limb_points_camera,
                                                                   temperature=temperature)

        # Compute the distance from the scan center to the limb points in units of pixels
        diff = limb_points_image - center.reshape(2, 1)
        distance = np.linalg.norm(diff, axis=0, keepdims=True)

        # Compute the predicted change in the distance given a change in the relative position from the camera to
        # the target
        jacobian = (diff / distance).T[..., np.newaxis, :] @ camera_jacobian @ limb_jacobian

        return np.vstack(jacobian)

    def _prepare_gif(self, image: OpNavImage, target: SceneObject, target_ind: int):
        """
        Set up the limb match summary gif.

        This prepares the figure and the gif writer and initializes some data.  It is only intended for internal use by
        the class itself.

        :param image: The image being processed
        :param target:  The target being processed
        :param target_ind:  The index of the target being processed
        """
        # since matplotlib can cause problems sometimes only import it if a gif was requested
        import matplotlib.pyplot as plt
        from matplotlib.animation import ImageMagickWriter

        # create the figure and set the layout to tight
        fig = plt.figure()
        fig.set_layout_engine('tight')

        # grab the primary axes
        ax = fig.add_subplot(111)

        # show the image in the axes
        ax.imshow(image, cmap='gray')

        # Show the predicted location of the limbs based on the a priori
        # pair the limbs
        _, __, ___, ____, predicted_limbs_image = self.extract_and_pair_limbs(image, target, target_ind)

        extracted_limbs = self.observed_bearings[target_ind]
        assert extracted_limbs is not None
        # Show the limb points found in the image
        ax.scatter(*extracted_limbs, color='blue', label='extracted limb points')

        # make the gif writer
        writer = ImageMagickWriter(fps=5)

        # determine the output file and prepare the writer
        out_file = self.gif_file.format(image.observation_date.isoformat().replace('-', '').replace(':', ''),
                                        target.name)

        writer.setup(fig=fig, outfile=out_file, dpi=100)

        # plot the line and save it for latter
        limbs_line = ax.scatter(*predicted_limbs_image, marker='.', color='red', label='predicted limb')

        # make a legend
        plt.legend()

        # zoom in on the region of interest
        # noinspection PyArgumentList
        bounds = extracted_limbs.min(axis=-1) - 10, extracted_limbs.max(axis=-1) + 10
        ax.set_xlim(bounds[0][0], bounds[1][0])
        ax.set_ylim(bounds[1][1], bounds[0][1])

        # store everything for later use
        self._gif_ax = ax
        self._gif_fig = fig
        self._gif_limbs_line = limbs_line
        self._gif_writer = writer

    def _update_gif(self, target_ind: int):
        """
        This captures a new frame in the GIF after updating the location of the predicted limb points

        :meth:`_prepare_gif` must have been called before this method.

        This is only intended for internal use by the class itself.

        :param target_ind: the index of the target being considered.
        """
        # update the location of the predicted limbs in the image
        assert self._gif_limbs_line is not None and self._gif_writer is not None
        bearings = self.computed_bearings[target_ind]
        assert bearings is not None
        self._gif_limbs_line.set_offsets(bearings.T)

        # add the frame to the gif
        self._gif_writer.grab_frame()

    def _finish_gif(self):
        """
        Finish writing the gif and clean up the matplotlib stuff.

        :meth:`_prepare_gif` must have been called before this method.

        This is only intended for internal use by the class itself.
        """
        import matplotlib.pyplot as plt
        assert self._gif_writer is not None

        # finish the gif
        self._gif_writer.finish()

        # close the figure
        plt.close(self._gif_fig)

        # reset everything to None
        self._gif_writer = None
        self._gif_fig = None
        self._gif_ax = None
        self._gif_limbs_line = None

    def estimate(self, image: OpNavImage, include_targets: Optional[List[bool]] = None):
        """
        This method identifies the position of each target in the camera frame using limb matching.

        This method first extracts limb observations from an image and matches them to the targets in the scene.  Then,
        for each target, the position is estimated from the limb observations by pairing the observed limb locations
        to possible surface locations on the target that could have produced the limb using the current estimate of the
        state (:meth:`pair_limbs`) and then updating the state vector based on the residuals between the extracted and
        predicted limbs in a least squares fashion.  This process is repeated until convergence or the maximum number of
        iterations are reached.

        Optionally, along the way, if the :attr:`create_gif` flag is set to ``True``, then this class will also create a
        gif showing how the predicted limb locations change for each iteration.

        .. warning::
            Before calling this method be sure that the scene has been updated to correspond to the correct
            image time.  This method does not update the scene automatically.

        :param image: The image the unresolved algorithm should be applied to as an OpNavImage
        :param include_targets: An argument specifying which targets should be processed for this image.  If ``None``
                                then all are processed (no, the irony is not lost on me...)
        """

        # If we were requested to recenter using a moment algorithm then do it
        if self.recenter:
            print('recentering', flush=True)
            # Estimate the center using the moment algorithm to get a fast rough estimate of the cof
            self._moment_algorithm.estimate(image, include_targets=include_targets)

        # Process each object in the scene
        for target_ind, target in self.target_generator(include_targets=include_targets):

            # Store the relative position between the object and the camera in the camera frame
            relative_position = target.position.copy()
            self.computed_positions[target_ind] = relative_position
            
            bearings = self._moment_algorithm.observed_bearings[target_ind]

            # recenter based on the moment algorithm estimates
            if self.recenter and bearings is not None and np.isfinite(bearings).all():
                new_position = self.camera.model.pixels_to_unit(bearings,
                                                                temperature=image.temperature)
                new_position *= np.linalg.norm(target.position)
                target.change_position(new_position)
                relative_position = target.position.copy()

            # predefine inliers for the angry inspector...
            inliers = np.ones(1, dtype=bool)

            stop_process = False
            jacobian = None
            residual_distances: np.ndarray = np.zeros(1)

            iter_num = 0

            residual_ss = np.finfo(np.float64).max

            # Iterate for the specified number of iterations
            for iter_num in range(self.max_iters):
                # extract and pair the limbs
                (scan_center, scan_center_dir,
                 scan_dirs_camera, self.limbs_camera[target_ind],
                 self.computed_bearings[target_ind]) = self.extract_and_pair_limbs(image, target, target_ind)

                if self.observed_bearings[target_ind] is None:
                    warnings.warn('unable to find any limbs for target {}'.format(target_ind))
                    self.details[target_ind] = {'Failed': "Unable to find any limbs for target in the image"}
                    stop_process = True
                    break

                # Drop any invalid limbs
                valid_test = (~np.isnan(self.observed_bearings[target_ind]).any(axis=0) | # type: ignore
                              ~np.isnan(self.computed_bearings[target_ind]).any(axis=0)) # type: ignore

                self.observed_bearings[target_ind] = self.observed_bearings[target_ind][:, valid_test] # type: ignore
                self.computed_bearings[target_ind] = self.computed_bearings[target_ind][:, valid_test] # type: ignore

                # Convert the extracted limb points in the image into unit vectors in the camera frame
                extracted_limbs_camera = self.camera.model.pixels_to_unit(self.observed_bearings[target_ind],  # type: ignore
                                                                          temperature=image.temperature)
                # Put everything onto the image plane at z=1
                extracted_limbs_camera /= extracted_limbs_camera[2]

                # Get the distance between the extracted limb points in the image
                # and the scan center location in the image
                observed_distances = np.linalg.norm(self.observed_bearings[target_ind] - scan_center.reshape(2, 1),  # type: ignore
                                                    axis=0)

                # If we are making a gif then update the predicted limb locations and grab the frame
                if self.create_gif:
                    if iter_num == 0:
                        self._prepare_gif(image, target, target_ind)
                    self._update_gif(target_ind)

                # Compute the residual distances
                residual_distances = observed_distances - np.linalg.norm(self.computed_bearings[target_ind] -  # type: ignore
                                                                         scan_center.reshape(2, 1), axis=0)

                # If we were told to reject outliers at each step
                if self.discard_outliers:
                    # Use Median Absolute Deviation to find outliers and get rid of them
                    inliers = ~get_outliers(residual_distances)

                    residual_distances = residual_distances[inliers]
                    self.computed_bearings[target_ind] = self.computed_bearings[target_ind][:, inliers]  # type: ignore
                    self.limbs_camera[target_ind] = self.limbs_camera[target_ind][:, inliers]  # type: ignore
                    scan_dirs_use = scan_dirs_camera[:, inliers]

                else:
                    inliers = np.ones(scan_dirs_camera.shape[-1], dtype=bool)
                    scan_dirs_use = scan_dirs_camera

                # Compute the Jacobian matrix based on the predicted/observed limb locations
                # noinspection PyTypeChecker
                jacobian = self.compute_jacobian(target, scan_center, scan_center_dir,
                                                 self.computed_bearings[target_ind],  # type: ignore
                                                 self.limbs_camera[target_ind],  # type: ignore
                                                 relative_position,
                                                 scan_dirs_use,
                                                 temperature=image.temperature)

                # Check where the jacobian is invalid
                nans = np.isnan(jacobian).any(axis=-1)

                # If the jacobian is invalid everywhere something probably went wrong, break out
                if nans.all():
                    print('invalid jacobian')
                    self.details[target_ind] = {'Failed': 'The fit failed because the computed Jacobian was all NaN'}
                    stop_process = True
                    if self.create_gif:
                        self._finish_gif()
                    break

                # Calculate the update to the state vector using LLS
                update = np.linalg.lstsq(jacobian[~nans], residual_distances.ravel()[~nans],
                                         rcond=None)[0]

                # If a large update was requested then only do part of the update to avoid over updating
                scale = 1
                while abs(update[-1] / scale) / abs(relative_position[-1]) >= 0.5:
                    scale *= 2

                # Apply the scaled update
                relative_position += update.ravel() / scale

                # Change the position of the object to the estimated position
                target.change_position(relative_position)

                residual_ss_new = (residual_distances**2).sum()

                # Check if we've converged, and if so break out
                if (np.abs(update.ravel() / relative_position.ravel()) < self.state_rtol).all():
                    print('converged in {} iterations'.format(iter_num))
                    break
                elif (np.abs(update.ravel()) < self.state_atol).all():
                    print('converged in {} iterations'.format(iter_num))
                    break
                elif abs(residual_ss_new - residual_ss) < self.residual_atol:
                    print('converged in {} iterations'.format(iter_num))
                    break
                elif abs(residual_ss_new - residual_ss)/residual_ss_new < self.residual_rtol:
                    print('converged in {} iterations'.format(iter_num))
                    break
                elif residual_ss < residual_ss_new:
                    warnings.warn(f'Estimation is diverging for target {target_ind} after {iter_num} iterations. '
                                  f'Divergence is {100*(residual_ss_new - residual_ss)/residual_ss}. '
                                  'Stopping iteration')
                    self.details[target_ind] = {'Failed': 'The fit failed because it was diverging.',
                                                'Prior Residuals': residual_ss,
                                                'Current Residuals': residual_ss_new}
                    stop_process = True
                    break

                residual_ss = residual_ss_new

            # Close out the writer if we were making a gif
            if self.create_gif:
                self._finish_gif()

            if stop_process:
                # we failed so we should set things to NaN
                self.observed_bearings[target_ind] = np.nan*self.computed_bearings[target_ind]  # type: ignore
                self.observed_positions[target_ind] = target.position.copy()*np.nan

            else:
                # Update the final set of limbs observed in the image that were used for the estimation
                self.observed_bearings[target_ind] = self.observed_bearings[target_ind][:, inliers]  # type: ignore


                # store the solved for state
                self.observed_positions[target_ind] = target.position.copy()

                # get the limb points in the target fixed target centered frame
                # noinspection PyUnresolvedReferences
                target_centered_fixed_limb = self.limbs_camera[target_ind] - target.position.reshape(3, 1)
                target_centered_fixed_limb = target.orientation.matrix.T@target_centered_fixed_limb

                # store the details of the fit
                self.details[target_ind] = {'Jacobian': jacobian,
                                            'Inlier Ratio': inliers.sum()/inliers.size,
                                            'Covariance': np.linalg.pinv(jacobian.T@jacobian)*residual_distances.var(),  # type: ignore
                                            'Number of Iterations': iter_num,
                                            "Surface Limb Points": target_centered_fixed_limb}

    def reset(self):
        """
        This method resets the observed/computed attributes, the details attribute, and the gif attributes to have
        ``None``.

        This method is called by :class:`.RelativeOpNav` between images to ensure that data is not accidentally applied
        from one image to the next.
        """

        super().reset()

        self._gif_ax = None
        self._gif_fig = None
        self._gif_writer = None
        self._gif_limbs_line = None
