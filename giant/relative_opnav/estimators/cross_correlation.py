# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


r"""
This module provides the capability to locate the center-of-figure of a target in an image using 2D cross-correlation.

Description of the Technique
----------------------------

Center-finding is the process by which the bearing to the center-of-figure of a target is identified in an image.
The most popular way of performing center finding for OpNav is through a process known as 2D cross-correlation.  In
cross-correlation, a template is correlated with the image to locate where the template and image match the most.  The
template typically represents what we expect the object to look like and the correlation is performed for every possible
alignment between the image and the template.  The location where the correlation score is highest is then said to be
the location of the template in the image.

Cross-correlation based center finding is extremely accurate when the shape model of the target is well known, the range
to the target is pretty well known, and the orientation between the target and the camera is pretty well known.  The
results degrade and can even become biased when any of these conditions are not met, particularly the range to the
target.  If you expect that your knowledge will not be sufficient in one of these areas it may be better to attempt
to use a different technique such as :mod:`.sfn` or :mod:`.limb_matching`.  Alternatively, you can use this method
to get initial results, then refine your knowledge with OD and shape modelling, and then reprocess the images to get
more accurate results.

In GIANT, the cross correlation algorithm is implemented as follows.

#. Render the template based on the a priori relative state between the camera and the target using a single bounce
   ray trace and the routines from :mod:`.ray_tracer`.
#. Perform 2D normalized cross correlation for every possible alignment between the center of the template and the
   image.
#. Locate the peak of the correlation surface (optionally locate the subpixel peak by fitting a 2D quadric to the
   correlation surface)
#. Correct the located peak based on the location of the center-of-figure of the target in the template to get the
   observed center-of-figure in the image.

Tuning
------

There aren't too many parameters that need to be tuned for successful cross correlation as long as you have a decent
shape model and decent a priori state information.  Beyond that, parameters you can tune are

========================================== =============================================================================
Parameter                                  Description
========================================== =============================================================================
:attr:`~XCorrCenterFinding.brdf`           The bidirectional reflectance distribution function used to compute the
                                           expected illumination of a ray based on the geometry of the scene.
:attr:`~XCorrCenterFinding.grid_size`      The size of the grid to use for subpixel sampling when rendering the template
:attr:`~XCorrCenterFinding.peak_finder`    The function to use to detect the peak of the correlation surface.
:attr:`~XCorrCenterFinding.blur`           A flag specifying whether to blur the correlation surface to decrease high
                                           frequency noise before identifying the peak.
:attr:`~XCorrCenterFinding.search_region`  An optional search region in pixels to restrict the area the peak of the
                                           correlation surface is search for around the a priori predicted center
:attr:`~XCorrCenterFinding.min_corr_score` The minimum correlation score to accept as a successful identification.
                                           Correlation scores range from -1 to 1, with 1 indicating perfect correlation.
========================================== =============================================================================

Of these options, most only make small changes to the results.  The 2 that can occasionally make large changes are
:attr:`~XCorrCenterFinding.search_region` and :attr:`~XCorrCenterFinding.blur`.  In general
:attr:`~XCorrCenterFinding.search_region` should be left at ``None`` which searches the whole image.  However, if your
object is nearly unresolved (<10 pixels across or so) and your a priori knowledge is pretty good, then it may be
beneficial to set this to a smallish number to ensure that you don't mistakenly correlate with a noise spike or bright
star.  The :attr:`~XCorrCenterFinding.blur` attribute can also be used to help avoid correlating with a star but in
general should be left as ``False`` unless the object is small (<7-10 pixels).  Finally, the
:attr:`~XCorrCenterFinding.min_corr_score` can generally be left at the default, though if you have a poor a priori
knowledge of either the shape model or the relative position of the object then you may need to decrease this some.

Use
---

The class provided in this module is usually not used by the user directly, instead it is usually interfaced with
through the :class:`.RelativeOpNav` class using the identifier :attr:`~.RelativeOpNav.cross_correlation`.  For more
details on using the :class:`.RelativeOpNav` interface, please refer to the :mod:`.relnav_class` documentation.  For
more details on using the technique class directly, as well as a description of the ``details`` dictionaries produced
by this technique, refer to the following class documentation.
"""


import warnings

from dataclasses import dataclass

from typing import Optional, Callable, List, Union, Tuple

import numpy as np

from giant.relative_opnav.estimators.estimator_interface_abc import RelNavEstimator, RelNavObservablesType
from giant.ray_tracer.rays import Rays, compute_rays
from giant.ray_tracer.scene import Scene, SceneObject
from giant.ray_tracer.illumination import IlluminationModel, McEwenIllumination
from giant.camera import Camera
from giant.image import OpNavImage
from giant.image_processing import ImageProcessing, quadric_peak_finder_2d
from giant._typing import Real


@dataclass
class XCorrCenterFindingOptions:
    """
    This dataclass serves as one way to control the settings for the :class:`.XCorrCenterFinding` class.

    You can set any of the options on an instance of this dataclass and pass it to the
    :class:`.XCorrCenterFinding` class at initialization (or through the method
    :meth:`.XCorrCenterFinding.apply_options`) to set the settings on the class. This class is the preferred way
    of setting options on the class due to ease of use in IDEs.
    """

    brdf: Optional[IlluminationModel] = None
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

    peak_finder:  Callable[[np.ndarray, bool], np.ndarray] = quadric_peak_finder_2d
    """
    The peak finder function to use. This should be a callable that takes in a 2D surface as a numpy array and returns 
    the (x,y) location of the peak of the surface.
    """

    min_corr_score: float = 0.3
    """
    The minimum correlation score to accept for something to be considered found in an image. The correlation score 
    is the Pearson Product Moment Coefficient between the image and the template. This should be a number between -1 
    and 1, and in nearly every cast a number between 0 and 1.  Setting this to -1 essentially turns the minimum 
    correlation score check off. 
    """

    blur: bool = True
    """
    A flag to perform a Gaussian blur on the correlation surface before locating the peak to remove high frequency noise
    """

    search_region: Optional[int] = None
    """
    The number of pixels to search around the a priori predicted center for the peak of the correlation surface.  If 
    ``None`` then searches the entire correlation surface.
    """


# TODO: Allow recentering with either the moment algorithm or using a raster rendered template.  If raster rendered
#  enable scale/rotation correction as well using Josh's technique
class XCorrCenterFinding(RelNavEstimator):
    """
    This class implements normalized cross correlation center finding for GIANT.

    All of the steps required for performing cross correlation are handled by this class, including the rendering of the
    template, the actual cross correlation, and the identification of the peak of the correlation surface.  This is all
    handled in the :meth:`estimate` method and is performed for each requested target.

    When all of the required data has been successfully loaded into an instance of this class, the :meth:`estimate`
    method is used to perform the estimation for the requested image.  The results are stored into the
    :attr:`observed_bearings` attribute for the observed center of figure locations. In addition, the predicted location
    for the center of figure for each target is stored in the :attr:`computed_bearings` attribute. Finally, the details
    about the fit are stored as a dictionary in the appropriate element in the :attr:`details` attribute.  Specifically,
    these dictionaries will contain the following keys.

    ================================= ==================================================================================
    Key                               Description
    ================================= ==================================================================================
    ``'Correlation Score'``           The correlation score at the peak of the correlation surface.  This is only
                                      available if the fit was successful.
    ``'Correlation Surface'``         The raw correlation surface as a 2D array.  Each pixel in the correlation surface
                                      represents the correlation score when the center of the template is lined up with
                                      the corresponding image pixel.  This is only available if the fit was successful.
    ``'Correlation Peak Location'``   The Location of the correlation peak before correcting it to find the location of
                                      the target center of figure. This is only available if the fit was successful.
    ``'Target Template Coordinates'`` The location of the center of figure of the target in the template.  This is only
                                      available if the fit was successful.
    ``'Failed'``                      A message indicating why the fit failed.  This will only be present if the fit
                                      failed (so you could do something like
                                      ``'Failed' in cross_correlation.details[target_ind]`` to
                                      check if something failed.  The message should be a human readable description of
                                      what caused the failure.
    ``'Max Correlation'``             The peak value of the correlation surface.  This is only available if the fit
                                      failed due to too low of a correlation score.
    ================================= ==================================================================================

    .. warning::
        Before calling the :meth:`estimate` method be sure that the scene has been updated to correspond to the correct
        image time.  This class does not update the scene automatically, even if the :attr:`scene` attribute is an
        :class:`.Scene` instance.
    """

    observable_type: List[RelNavObservablesType] = [RelNavObservablesType.CENTER_FINDING]
    """
    This technique generates CENTER-FINDING bearing observables to the center of figure of a target.
    """

    generates_templates: bool = True
    """
    A flag specifying that this RelNav estimator generates and stores templates in the :attr:`templates` attribute.
    """

    def __init__(self, scene: Scene, camera: Camera, image_processing: ImageProcessing,
                 options: Optional[XCorrCenterFindingOptions] = None,
                 brdf: Optional[IlluminationModel] = None, rays: Union[Optional[Rays], List[Rays]] = None,
                 grid_size: int = 1, peak_finder: Callable[[np.ndarray, bool], np.ndarray] = quadric_peak_finder_2d,
                 min_corr_score: float = 0.3, blur: bool = True, search_region: Optional[int] = None):
        """
        :param scene: The scene describing the a priori locations of the targets and the light source.
        :param camera: The :class:`.Camera` object containing the camera model and images to be analyzed
        :param image_processing: An instance of :class:`.ImageProcessing`.  This is used for denoising the image and for
                                 generating the correlation surface using :meth:`.denoise_image` and :meth:`correlate`
                                 methods respectively
        :param options: A dataclass specifying the options to set for this instance.  If provided it takes preference
                        over all key word arguments, therefore it is not recommended to mix methods.
        :param brdf: The illumination model that transforms the geometric ray tracing results (see
                     :const:`.ILLUM_DTYPE`) into a intensity values. Typically this is one of the options from the
                     :mod:`.illumination` module).
        :param rays: The rays to use when rendering the template.  If ``None`` then the rays required to render the
                     template will be automatically computed.  Optionally, a list of :class:`.Rays` objects where each
                     element corresponds to the rays to use for the corresponding template in the
                     :attr:`.Scene.target_objs` list.  Typically this should be left as ``None``.
        :param grid_size: The subsampling to use per pixel when rendering the template.  This should be the number of
                          sub-pixels per side of a pixel (that is if grid_size=3 then subsampling will be in an equally
                          spaced 3x3 grid -> 9 sub-pixels per pixel).  If ``rays`` is not None then this is ignored
        :param peak_finder: The peak finder function to use. This should be a callable that takes in a 2D surface as a
                            numpy array and returns the (x,y) location of the peak of the surface.
        :param min_corr_score: The minimum correlation score to accept for something to be considered found in an image.
                               The correlation score is the Pearson Product Moment Coefficient between the image and the
                               template. This should be a number between -1 and 1, and in nearly every cast a number
                               between 0 and 1.  Setting this to -1 essentially turns the minimum correlation score
                               check off.
        :param blur: A flag to perform a Gaussian blur on the correlation surface before locating the peak to remove
                     high frequency noise
        :param search_region: The number of pixels to search around the a priori predicted center for the peak of the
                              correlation surface.  If ``None`` then searches the entire correlation surface.
        """

        super().__init__(scene, camera, image_processing)

        self.rays: Union[Optional[Rays], List[Optional[Rays]]] = rays
        """
        The rays to trace to render the template, or ``None`` if the rays are to be computed automatically.
        """

        self.grid_size: int = grid_size
        """
        The size of each side of the subpixel grid to sample when rendering the template.  
        
        A ``grid_size`` of 3 would lead to a 3x3 sampling for each pixel or 9 rays per pixel.  
        
        See :func:`.compute_rays` for details.
        """

        if brdf is None:
            brdf = McEwenIllumination()

        self.brdf: IlluminationModel = brdf
        """
        A callable to translate illumination inputs (:data:`.ILLUM_DTYPE`) into intensity values in the template. 
        
        Typically this is one of the options from :mod:`.illumination`
        """

        self.peak_finder: Callable[[np.ndarray, bool], np.ndarray] = peak_finder
        """
        A callable to extract the peak location from a correlation surface.
        
        See :func:`.quadric_peak_finder_2d` for an example/details.
        """

        self.search_region: Optional[int] = search_region
        """
        The region to search around the a priori predicted center location for each target in the correlation surface 
        for a peak.
        
        If set to ``None`` then the entire correlation surface is searched.
        """

        self.blur: bool = blur
        """
        A flag specifying whether to apply a Gaussian blur to the correlation surface before attempting to find the peak
        to try and remove high-frequency noise.
        """

        self.min_corr_score: float = min_corr_score
        """
        The minimum correlation score to accept as a successfully located target.
        """

        # apply the options struct if provided
        if options is not None:
            self.apply_options(options)

    def apply_options(self, options: XCorrCenterFindingOptions):
        """
        This method applies the input options to the current instance.

        The input options should be an instance of :class:`.XcorrCenterFindingOptions`.

        When calling this method every setting will be updated, even ones you did not specifically set in the provided
        ``options`` input.  Any you did not specifically modify will be reset to the default value.  Typically the best
        way to change a single setting is through direct attribute access on this class, or by maintaining a copy of the
        original options structure used to initialize this class and then updating it before calling this method.

        :param options: The options to apply to the current instance
        """
        self.rays = options.rays
        self.grid_size = options.grid_size
        if options.brdf is None:
            self.brdf = McEwenIllumination()
        else:
            self.brdf = options.brdf
        self.peak_finder = options.peak_finder
        self.search_region = options.search_region
        self.blur = options.blur
        self.min_corr_score = options.min_corr_score

    def render(self, target_ind: int, target: SceneObject, temperature: Real = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method returns the computed illumination values for the given target and the (sub)pixels that each
        illumination value corresponds to

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

            rays = self.rays[target_ind]

            # compute the pixel location of each ray
            uv = self.camera.model.project_onto_image(rays.start + rays.direction, temperature=temperature)

        print('Rendering {} rays'.format(rays.num_rays), flush=True)
        illum_inputs = self.scene.get_illumination_inputs(rays)

        # compute the illumination for each ray and return it along with the pixel location for each ray
        return self.brdf(illum_inputs), uv

    def compute_rays(self, target: SceneObject,
                     temperature: Real = 0) -> Tuple[Tuple[Rays, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
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

        # If the template size is too small artificially increase it so we can get the PSF in there
        if ((local_max - local_min) < 5).any():
            local_max += 3
            local_min -= 3

        # Determine the actual bounds to use for the template so that it is within the image
        min_inds = np.maximum(local_min, [0, 0])

        max_inds = np.minimum(local_max, [self.camera.model.n_cols, self.camera.model.n_rows])

        # If the min inds are greater than the max inds then they body is outside of the FOV, move it to the center of
        # the image as a work around and restart
        if np.any(min_inds > max_inds):

            warnings.warn("the predicted location of the body is outside of the FOV.  \n"
                          "We are moving the object to the middle of the FOV and attempting to render that way\n")

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
            return compute_rays(self.camera.model, (local_min[1], local_max[1]), (local_min[0], local_max[0]),
                                grid_size=self.grid_size, temperature=temperature), (local_min, local_max)

    def estimate(self, image: OpNavImage, include_targets: Optional[List[bool]] = None):
        """
        This method identifies the center of each target in the image using cross correlation

        This method first does ray tracing to render a template and then computes the expected location of the object
        within the image. The expected bounds and size of the template are computed in pixels. The psf of the camera
        is then applied to the template and the center of the body is determined within the template. Noise is
        removed from the image and the image and template are correlated. The peaks of the correlation surface are
        determined in addition to residuals between the solved for center in the template and center in the image.

        .. warning::
            Before calling this method be sure that the scene has been updated to correspond to the correct
            image time.  This method does not update the scene automatically, even if the :attr:`scene` attribute is an
            :class:`.Scene` instance.

        :param image: The image to locate the targets in
        :param include_targets: A list specifying whether to process the corresponding target in
                                :attr:`.Scene.target_objs` or ``None``.  If ``None`` then all targets are processed.
        """

        # Make sure the image is single precision for use with opencv
        # noinspection PyTypeChecker
        image = image.astype(np.float32)  # type: OpNavImage

        for target_ind, target in self.target_generator(include_targets):

            # Do the ray trace and render the template
            illums, locs = self.render(target_ind, target, temperature=image.temperature)

            # Compute the expected location of the object in the image
            self.computed_bearings[target_ind] = self.camera.model.project_onto_image(
                target.position,
                temperature=image.temperature
            ).ravel()

            # Get the expected bounds of the template in the image
            # noinspection PyArgumentList
            bounds = (locs.min(axis=1).round(), locs.max(axis=1).round())

            # Compute the size of the template in pixels
            template_size = (bounds[1] - bounds[0]) + 1

            # Initialize the template matrix
            self.templates[target_ind] = np.zeros(template_size[::-1].astype(int))

            # Compute the subscripts into the template
            subs = (locs - bounds[0].reshape(2, 1)).round().astype(int)

            # use numpy fancy stuff to add the computed brightness for each ray to the appropriate pixels
            np.add.at(self.templates[target_ind], (subs[1], subs[0]), illums.flatten())

            # Apply the psf of the camera to the template
            if self.camera.psf is not None:
                self.templates[target_ind] = self.camera.psf(self.templates[target_ind])

            # Determine the center of the body in the template
            center_of_template = self.computed_bearings[target_ind] - bounds[0]

            if self.image_processing.denoise_flag:
                # Try to remove noise from the image
                # noinspection PyTypeChecker
                image = self.image_processing.denoise_image(image)
                # Not sure if we should also do the template here or not.  Probably
                # self.templates[target_ind] = self._image_processing.denoise_image(self.templates[target_ind])

            # Perform the correlation between the image and the template
            correlation_surface = self.image_processing.correlate(image, self.templates[target_ind])

            # Get the middle of the template
            temp_middle = np.floor(np.flipud(np.array(self.templates[target_ind].shape)) / 2)

            # Figure out the delta between the middle of the template and the center of the body in the template
            delta = temp_middle - center_of_template

            if self.search_region is not None:
                # If we want to restrict our search for the peak of the correlation surface around the a priori center

                # Compute the center of the search region
                search_start = np.round(self.computed_bearings[target_ind] + delta).astype(int)

                # Compute the beginning of the search region in pixels
                begin_roi = search_start - self.search_region
                begin_roi = np.maximum(begin_roi, 0)

                # Compute the end of the search region in pixels
                end_roi = search_start + self.search_region + 1
                end_roi = np.minimum(end_roi, correlation_surface.shape[::-1])

                # Extract the appropriate sub array from the correlation surface
                roi = correlation_surface[begin_roi[1]:end_roi[1], begin_roi[0]:end_roi[0]]

                # Find the peaks from the extracted correlation surface
                peaks = self.peak_finder(roi, self.blur).ravel()

                # Correct the peaks to correspond to the full image instead of the extracted surface and store them
                peaks += begin_roi

                peak_location = peaks

            else:
                # Find the peak of the correlation surface
                peak_location = self.peak_finder(correlation_surface, self.blur).ravel()

            # check if the correlation peak was found
            if not np.isfinite(peak_location).all():
                warnings.warn("Correlation peak not found.")
                self.details[target_ind] = {'Failed': 'Unable to find peak of correlation surface'}
                self.observed_bearings[target_ind] = np.nan*np.ones(2, dtype=np.float64)
                continue

            # check the peak score for the correlation surface
            correlation_peak_pix = np.round(peak_location).astype(int)
            correlation_score = correlation_surface[correlation_peak_pix[1], correlation_peak_pix[0]]

            # check if the correlation score was too low
            if correlation_score < self.min_corr_score:
                warnings.warn(f"Correlation peak too low. Peak of {correlation_score} is less "
                              f"than {self.min_corr_score}.")

                self.details[target_ind] = {'Failed': 'Correlation score too low',
                                            "Max Correlation": correlation_score}
                self.observed_bearings[target_ind] = np.nan * np.ones(2, dtype=np.float64)
                continue

            # Determine the solved for center of the body by correcting for the offset from the center of the template
            # to the center of the body
            self.observed_bearings[target_ind] = peak_location - delta

            # store the details of the fit
            self.details[target_ind] = {"Correlation Score": correlation_score,
                                        "Correlation Surface": correlation_surface,
                                        "Correlation Peak Location": peak_location,
                                        "Target Template Coordinates": center_of_template}
