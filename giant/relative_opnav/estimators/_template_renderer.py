


r"""
This module provides the capability to render a template from a scene for various relnav estimators
"""


import warnings

from dataclasses import dataclass, field

from typing import Optional, Callable, List, Union, Tuple

from abc import ABC

import numpy as np

from giant.ray_tracer.rays import Rays, compute_rays
from giant.ray_tracer.scene import Scene, SceneObject
from giant.ray_tracer.illumination import IlluminationModel, McEwenIllumination
from giant.relative_opnav.estimators.estimator_interface_abc import RelNavEstimator
from giant.camera import Camera
from giant.image import OpNavImage
from giant.utilities.options import UserOptions

from giant.utilities.mixin_classes import UserOptionConfigured

from giant._typing import DOUBLE_ARRAY


@dataclass
class TemplateRendererOptions(UserOptions):
    """
    This dataclass serves as one way to control the settings for the :class:`.XCorrCenterFinding` class.

    You can set any of the options on an instance of this dataclass and pass it to the
    :class:`.XCorrCenterFinding` class at initialization (or through the method
    :meth:`.XCorrCenterFinding.apply_options`) to set the settings on the class. This class is the preferred way
    of setting options on the class due to ease of use in IDEs.
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

    template_overflow_bounds : int = -1
    """
    The number of pixels to render in the template that overflow outside of the
    camera field of view.  Set to a number less than 0 to accept all overflow
    pixels in the template.  Set to a number greater than or equal to 0 to limit
    the number of overflow pixels.
    """

class TemplateRenderer(UserOptionConfigured[TemplateRendererOptions], TemplateRendererOptions, RelNavEstimator, ABC):
    """
    This class implements capabilites for generating template

    .. warning::
        Before calling the :meth:`estimate` method be sure that the scene has been updated to correspond to the correct
        image time.  This class does not update the scene automatically, even if the :attr:`scene` attribute is an
        :class:`.Scene` instance.
    """

    generates_templates: bool = True
    """
    A flag specifying that this RelNav estimator generates and stores templates in the :attr:`templates` attribute.
    """

    def __init__(self, scene: Scene, camera: Camera, 
                 options: Optional[TemplateRendererOptions] = None,
                 options_type: type[TemplateRendererOptions] = TemplateRendererOptions):
        """
        :param scene: The scene describing the a priori locations of the targets and the light source.
        :param camera: The :class:`.Camera` object containing the camera model and images to be analyzed
        :param image_processing: An instance of :class:`.ImageProcessing`.  This is used for denoising the image and for
                                 generating the correlation surface using :meth:`.denoise_image` and :meth:`correlate`
                                 methods respectively
        :param options: A dataclass specifying the options to set for this instance.

        """

        super().__init__(options_type, scene, camera, options=options)
        
    def render(self, target_ind: int, target: SceneObject, temperature: float = 0) -> Tuple[np.ndarray, np.ndarray]:
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

            rays: Rays = self.rays[target_ind] # type: ignore

            # compute the pixel location of each ray
            uv = self.camera.model.project_onto_image(rays.start + rays.direction, temperature=temperature)

        print('Rendering {} rays'.format(rays.num_rays), flush=True)
        illum_inputs = self.scene.get_illumination_inputs(rays)

        # compute the illumination for each ray and return it along with the pixel location for each ray
        return self.brdf(illum_inputs), uv # type: ignore

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

        # If the template size is too small artificially increase it so we can get the PSF in there
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
            if self.template_overflow_bounds >= 0:
                return compute_rays(self.camera.model, (min_inds[1], max_inds[1]), (min_inds[0], max_inds[0]),
                                    grid_size=self.grid_size, temperature=temperature), (min_inds, max_inds)
            else:
                return compute_rays(self.camera.model, (local_min[1], local_max[1]), (local_min[0], local_max[0]),
                                    grid_size=self.grid_size, temperature=temperature), (local_min, local_max)
                
    def prepare_template(self, target_ind: int, target: SceneObject, temperature: float) -> tuple[DOUBLE_ARRAY, tuple[DOUBLE_ARRAY, DOUBLE_ARRAY]]:
        """
        Renders the template for the specified target
        """
    
        illums, locs = self.render(target_ind, target, temperature=temperature)


        # Get the expected bounds of the template in the image
        # noinspection PyArgumentList
        bounds: tuple[DOUBLE_ARRAY, DOUBLE_ARRAY] = (locs.min(axis=1).round(), locs.max(axis=1).round())

        # Compute the size of the template in pixels
        template_size = (bounds[1] - bounds[0]) + 1

        # Initialize the template matrix
        template = np.zeros(template_size[::-1].astype(int), dtype=np.float64)

        # Compute the subscripts into the template
        subs = (locs - bounds[0].reshape(2, 1)).round().astype(int)

        # use numpy fancy stuff to add the computed brightness for each ray to the appropriate pixels
        np.add.at(template, (subs[1], subs[0]), illums.flatten())

        # Apply the psf of the camera to the template
        if self.camera.psf is not None:
            self.templates[target_ind] = self.camera.psf(template)
        
        return template, bounds
    
