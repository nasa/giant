


"""
This module provides the capability to perform coverage analysis for
remote sensing instruments given a notional ConOps.

Description
-----------

The coverage analysis works by taking the observation plan and then at each "imaging"
epoch in the observation plan, computing which facets of the shape model fall within
the field of view of the instrument (and optionally are illuminated by the sun).

This information is then stored, per image, per facet, and can then be queried to
check what percentage of the surface was observed according to some constraints.
Additionally, for SPC coverage, we can then take the information from the visibility
check, and use it to compute the shape from shading dilutions of precision (DOP) for each
facet, to get an idea of how good the coverage is for building shape models using SPC.

Use
---

## Preparation
To run coverage analysis we need to prepare the shape model.
This is done in 2 simple steps.

1. Ingest the shape model you wish to use for coverage analysis into the GIANT
   format using the `ingest_shape` command line utility provided by GIANT
2. run `python prepare_shape.py -t /path/to/kdtree.pickle -o /path/to/output.pickle`
   to precompute necessary information about the shape model for coverage analysis.
   (Depending on the size of the shape model, this can take a while to run, but
   only needs to be run once for shape model)

We also need to define our observation ConOps.
This is more detailed but roughly works through the following:

1. Provide trajectory information (usually through the form of NAIF SPICE SPK files
   but could also be done through a csv file or similar)
2. Provide pointing information (this can be done most simply through NAIF SPICE CK
   files, or can be done programmatically by defining a python function which takes
   in a `datetime` object and outputs a `giant.rotations.Rotation` object defining
   the rotation from the inertial frame to the "camera frame" or through a csv file)
3. Provide an imaging cadence. This can be done most simply from a file
   (like a list of image times or a csv file) or can be done programmatically

## Running the analysis
To get results for visibility and DOP, you will need to have a model for a camera,
and write a script to be able to create a :class:`.Coverage` instance.

Once initialized, you can run :meth:`.Coverage.compute_visibility` to get visibility
results, and then you can run :meth:`.Coverage.compute_dop` to get DOP results.

## Interpreting the results
The results for visibility (and optionally DOP) can be saved to numpy files as
dictionaries (or in some cases pickle files) that can later be used for plotting.
"""

import numpy as np
import time
from copy import deepcopy
from datetime import datetime
from multiprocessing import get_context
from typing import List, Dict, Union, Callable, Optional, Tuple, Sequence

from numpy.lib.recfunctions import append_fields
from numpy.typing import NDArray
from tqdm import tqdm

from giant.camera_models import CameraModel
from giant.ray_tracer.illumination import IlluminationModel, ILLUM_DTYPE
from giant.ray_tracer.kdtree import KDTree
from giant.ray_tracer.rays import Rays
from giant.ray_tracer.scene import Scene
from giant.ray_tracer.utilities import to_block

from giant.coverage.utilities.dop_computations import DOPComputations
from giant.coverage.typing import JACOBIAN_TYPE, DOP_TYPE


class Coverage:
    """
    This class provides coverage analysis capabilities for remote sensing instruments given
    an observation plan. It works by taking the observation plan and then at each 'imaging'
    epoch in the observation plan, computing which facets of the shape model fall within the
    field of view of the instrument (and optionally are illuminated by the sun). This
    information is then stored, per image, per facet, and can then be queried to check what
    percentage of the surface was observed according to some constraints. Additionally, for
    SPC coverage, we can then take the information from the visibility check, and use it to
    compute the shape from shading dilutions of precision (DOP) for each facet, to get an
    idea of how good the coverage is for building shape models using stereophotoclinometry
    (SPC).
    """

    def __init__(self, scene: Scene, imaging_times: Union[List[datetime], List[List[datetime]]],
                 brdf: IlluminationModel, camera_model: CameraModel, camera_position_function: Callable,
                 camera_orientation_function: Callable, sun_position_function: Callable,
                 ignore_indices: Optional[Sequence[int] | Sequence[Sequence[int] | NDArray[np.integer]] | NDArray[np.integer]] = None,
                 topography_variations: Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]] = None,
                 labels: Optional[List[str]] = None):
        """
        :param scene: A :class:`Scene` containing the target body to be analyzed. Only the
                      first target in the scene will be used and any others will be ignored.
                      The target must have a shape with facets and vertices (usually a 
                      KDTree or Triangles type).
        :param imaging_times: A list of datetimes for each observation. If using labels, each label would
                              have its own list of datetimes in this list
        :param brdf: An :class:`IlluminationModel` representing a BRDF used to compute the jacobian matrix of the
                          change in the illumination values given a change in the surface normal and/or albedo
        :param camera_model: A :class:`CameraModel` instance
        :param camera_position_function: A callable function that returns the location of the camera relative to the
                                         target in target body-fixed frame
        :param camera_orientation_function: A callable function that rotates the target body-fixed frame to the camera frame
        :param sun_position_function: A callable function that returns the location of the sun relative to the
                                      target in target body-fixed frame
        :param ignore_indices: An optional Sequence of sequence of indices to ignore from the shape model for each vertex on the target body's surface. 
                               generally this should be a list of list of ints where each element of the outer list correspond to a vertex of the shape
                               model and the inner list contains all triangles that triangle contributes to to ensure that the vertex is not self shadowed.
                               This can be gotten from :func:`.prepare_shape`
        :param topography_variations: An optional meshgrid of azimuth and elevation values representing permutations of
                                      surface normal vectors to search through for correlation with observation data
        :param labels: An optional list of string labels to assign to different lists of image times
        """

        self.scene = scene
        target_shape = scene.target_objs[0].shape
        if isinstance(target_shape, KDTree):
            target_shape = target_shape.surface
        
        vecs = getattr(target_shape, "vertices", None)
        facets = getattr(target_shape, "facets", None)
        if self.targetvecs is None or self.targetfacets is None:
            raise TypeError(
                'Target shape must be a type that includes vertices and facets (usually a KDTree or Triangle type)')
        self.targetvecs: NDArray[np.float64] = vecs # type: ignore
        self.targetfacets: NDArray[np.integer] = facets # type: ignore
        self.vec_viewed = np.zeros(max(self.targetvecs.shape), dtype=int)
        self.camera_model = camera_model
        self.camera_position_function = camera_position_function
        self.camera_orientation_function = camera_orientation_function
        self.imaging_times = imaging_times
        self.sun_position_function = sun_position_function

        self.visibility = None

        self.albedo_dop: list[DOP_TYPE] | dict[str, list[DOP_TYPE]] = []
        self.x_terrain_dop: list[DOP_TYPE] | dict[str, list[DOP_TYPE]] = []
        self.y_terrain_dop: list[DOP_TYPE] | dict[str, list[DOP_TYPE]] = []
        self.total_dop: list[DOP_TYPE] | dict[str, list[DOP_TYPE]] = []
        self.dop_jacobians: list[JACOBIAN_TYPE] | dict[str, list[JACOBIAN_TYPE]] = []
        self.observation_count: list[int] | dict[str, list[int]] = []
        self.ignore_indices = ignore_indices

        self.brdf = brdf

        if topography_variations is not None:

            self.topography_variations = topography_variations

        else:
            az_grid, elev_grid = np.meshgrid(np.linspace(0, 2 * np.pi, 25, endpoint=True),
                                             np.linspace(0, np.pi / 3, 12, endpoint=True)[1:])

            self.topography_variations = (az_grid, elev_grid)

        self.labels = labels

        self.facet_visibility = None
        self.facet_gsds = None
        self.gsds = None
        self.altitudes = None
        self.facet_altitudes = None

    def reduce_visibility_to_facet(self, do_gsds: bool = True) -> None:
        """
        This method serves to convert the visibility matrix of all surface vertices at all image times
        to a visibility matrix of all surface facets at all image times based on how each facet is
        characterized by the surface vertices.
        
        It will also compute the altitude for each facet at each observations.
        
        These computations are saved as :attr:`.facet_visibility`, :attr:`.facet_altitudes`,
        and :attr:`.facet_gsds` attributes of the :class:`.Coverage` instance,
        and nothing is returned.
        
        :param do_gsds: An optional flag to compute and save the ground sample distance to each facet
                        for every observation
        """
        assert self.visibility is not None, "you must call compute_visibility prior to this method"
        assert self.altitudes is not None, "you must call compute_visibility prior to this method"
        print('beginning reduction to faceted visibility')

        start = time.time()

        self.facet_visibility = np.empty((self.visibility.shape[0], len(self.targetfacets)),
                                         dtype=self.visibility.dtype)

        self.facet_altitudes = np.empty((self.visibility.shape[0], len(self.targetfacets)), dtype=np.float64)
        self.facet_altitudes[:] = np.nan
        print('facet visibility allocated in {} seconds'.format(time.time() - start))

        image_list = np.arange(self.visibility.shape[0])

        for idx, facet in tqdm(enumerate(self.targetfacets), total=len(self.targetfacets),
                               desc='reducing facets', unit=' facets', dynamic_ncols=True):
            bvis = self.visibility[:, facet]
            self.facet_visibility[:, idx] = bvis[image_list, bvis['visible'].argmax(axis=-1)]
            self.facet_altitudes[:, idx] = np.nanmean(self.altitudes[:, facet], axis=-1)

        print('reduction to faceted visibility done')

        if do_gsds:
            assert self.gsds is not None, "you must call compute_visibility with compute_gsd set to True prior to this method"
            self.facet_gsds = {}
            for lab, gsd in self.gsds.items():
                agsd = np.array(gsd)[:, self.targetfacets]
                self.facet_gsds[lab] = np.nanmax(agsd, axis=-1)

    def compute_visibility(self, check_fov: bool = True, compute_gsd: bool = True,
                           check_shadows: bool = True) -> None:
        """
        This method serves to determine which facets and vertices on the target body's surface
        are visible for each observation. In doing these computations, it will also evaluate the
        surface normal unit vector, illumination incidence and exidence unit vectors, and the albedo
        for each facet/vertex at each observation.
        
        These computations are saved as :attr:`.visibility`, :attr:`.altitudes`,
        and :attr:`.gsds` attributes of the :class:`.Coverage` instance,
        and nothing is returned.
        
        :param check_fov: An optional flag to consider whether facets/vertices are within the field of view of
                          the camera, and if they are not, it will filter out those observations of that facet/vertex
        :param compute_gsd: An optional flag to compute and save the ground sample distance to each facet and vertex
                            for every observation
        :param check_shadows: An optional flag to consider whether facets/vertices are shadowed, and if so to filter
                              out those observations of that facet/vertex
        """

        if self.ignore_indices is not None:
            rays = Rays(self.targetvecs, np.zeros(self.targetvecs.shape), ignore=to_block(self.ignore_indices))

        else:
            rays = Rays(self.targetvecs, np.zeros(self.targetvecs.shape))

        visibility = []
        gsds = {}
        alts = []
        if self.labels is None:
            nimages = len(self.imaging_times)
            gsds['all'] = []

            for idx, image_time in tqdm(enumerate(self.imaging_times),
                                        total=len(self.imaging_times),
                                        desc='processing images', unit=' images',
                                        dynamic_ncols=True):
                assert isinstance(image_time, datetime), "the imaging times must be a list of datetimes for non-labeled processing"
                vis, alt, gsd = self._compute_visibility_image(image_time, idx, nimages, rays, check_fov, compute_gsd,  
                                                               check_shadow=check_shadows)
                visibility.append(vis)
                gsds['all'].append(gsd)
                alts.append(alt)

        else:
            for label, time_group in zip(self.labels, self.imaging_times):
                assert isinstance(time_group, list), "The imaging times must be a list of lists of datetimes for labeled processing"
                gsds[label] = []
                print('processing {}'.format(label))

                nimages = len(time_group)

                for idx, image_time in tqdm(enumerate(time_group),
                                            total=len(time_group),
                                            desc='processing images', unit=' images',
                                            dynamic_ncols=True):
                    unlabeled_vis, alt, gsd = self._compute_visibility_image(image_time, idx, nimages, rays, check_fov,
                                                                             compute_gsd)
                    gsds[label].append(gsd)
                    visibility.append(append_fields(unlabeled_vis, 'label', np.array([label] * unlabeled_vis.size)))
                    alts.append(alt)

        self.visibility = np.array(visibility, dtype=ILLUM_DTYPE)
        self.gsds = gsds
        self.altitudes = np.array(alts)

        del visibility

        # reduce the visibility to be by facet instead of by vertex
        self.reduce_visibility_to_facet(do_gsds=compute_gsd)

        # add the normal vectors to the facet visibility
        try:
            assert self.facet_visibility is not None
            if (surface := getattr(self.scene.target_objs[0].shape, 'surface', None)) is not None:

                self.facet_visibility['normal'] = surface.normals.reshape(1, -1, 3)
            elif (normals := getattr(self.scene.target_objs[0].shape, "normals", None)) is not None:
                self.facet_visibility['normal'] = normals.reshape(1, -1, 3)

        except ValueError:
            pass

    def _compute_visibility_image(self, image_time: datetime, image_idx: int, n_images: int,
                                  rays: Rays, check_fov: bool, compute_gsd: bool,
                                  check_shadow: bool = True) \
            -> Tuple[NDArray, NDArray[np.float64], Optional[NDArray[np.float64]]]:
        """
        This method serves to evaluate the visibility of each vertex on the target shape model
        for a specific observation. It uses the GIANT scene ray tracer to determine the illumination
        geometry that would be required for the camera to observe each vertex. It then filters
        these initial results using flags to determine which vertices would actually
        be observable given shadowing and camera fov constraints.
        
        :param image_time: The time of the observation being considered
        :param image_idx: The index of the observation being considered
        :param n_images: The total number of observations
        :param rays: A :class:`Rays` instance representing lines of sight from each vertex to the camera
        :param check_fov: A flag to consider whether vertices are within the field of view of
                          the camera, and if they are not, it will filter out those observations of that vertex
        :param compute_gsd: A flag to compute and save the ground sample distance at each vertex
                            for every observation
        :param check_shadow: An optional flag to consider whether vertices are shadowed, and if so to filter
                             out those observations of that vertex
        
        :return: A tuple of the following data, each having the length of the number of
                 vertices for the shape model being used:\n
                 An array of visibility evaluations,\n
                 An array of surface altitudes, and\n
                 An array of ground sample distances
        """

        start = time.time()

        camera_model = self.camera_model(image_time) if callable(self.camera_model) else self.camera_model

        sun_position = self.sun_position_function(image_time)

        assert self.scene.light_obj is not None, "the light_obj must be specified to compute visibility"
        self.scene.light_obj.change_position(sun_position)

        camera_position = self.camera_position_function(image_time)

        surf2cam = camera_position.reshape(3, 1) - self.targetvecs.reshape(3, -1)
        surf2cam_distances = np.linalg.norm(surf2cam, axis=0, keepdims=True)
        surf2cam_dir = surf2cam / surf2cam_distances

        rays.direction = surf2cam_dir

        # trace through the scene to see if the rays hit anything
        initial_intersect = self.scene.trace(rays)

        if rays.num_rays == 1:
            if not initial_intersect['check']:
                shadow_rays = deepcopy(rays)
                shadow_start = shadow_rays.start

                shadow_dir = self.scene.light_obj.position.reshape(3, 1) - shadow_start.reshape(3, -1)
                shadow_dir /= np.linalg.norm(shadow_dir, axis=0, keepdims=True)
                shadow_rays.direction = shadow_dir

            else:
                illum_params = np.zeros((rays.num_rays,), dtype=ILLUM_DTYPE)

                surf2cam_distances[:] = np.nan
                return illum_params, surf2cam_distances.ravel(), None

        else:

            shadow_rays = rays[~initial_intersect['check']]
            shadow_start = shadow_rays.start

            shadow_dir = self.scene.light_obj.position.reshape(3, 1) - shadow_start.reshape(3, -1)
            shadow_dir /= np.linalg.norm(shadow_dir, axis=0, keepdims=True)
            shadow_rays.direction = shadow_dir

        if shadow_rays.num_rays > 1:

            illum_params = np.zeros((rays.num_rays,), dtype=ILLUM_DTYPE)

            check = ~initial_intersect['check'].copy()

            if check_shadow:
                shadow_check = self.scene.trace(shadow_rays)
                check[check] = ~shadow_check['check']
            else:
                shadow_check = np.array([(False,)] * check.sum(), dtype=[('check', bool)])
        else:
            illum_params = np.zeros((rays.num_rays,), dtype=ILLUM_DTYPE)
            check = ~initial_intersect['check'].copy()
            shadow_check = False

        if check_fov:
            rotation_body_fixed_to_camera = self.camera_orientation_function(image_time)

            in_fov = self.check_fov(-rotation_body_fixed_to_camera.matrix @ surf2cam[:, check],
                                    camera_model=camera_model)

            check[check] = in_fov

        if (check.sum() >= 1) and (shadow_rays.num_rays > 1):
            illum_params[check] = list(zip(-np.atleast_2d(shadow_rays[~shadow_check['check'].squeeze()].direction.T),  # type: ignore
                                           np.atleast_2d(rays[check].direction.T),
                                           np.atleast_2d(initial_intersect[check]['normal']),
                                           np.ones(check.sum(), dtype=np.float64),
                                           np.atleast_1d(check[check])))

        elif (check.sum() == 1) and (shadow_rays.num_rays == 1):
            if rays.num_rays == 1:
                illum_params[check] = (-shadow_rays.direction.ravel(), rays.direction.ravel(),
                                       initial_intersect[check]['normal'], 1., True)
            else:
                illum_params[check] = (-shadow_rays.direction.ravel(), rays[check].direction.ravel(),
                                       initial_intersect[check]['normal'], 1., True)

        else:
            print('warning, image {} has no visible targets'.format(image_idx + 1))

        illum_params[~check] = (None, None, None, None, False)

        self.vec_viewed += check

        if compute_gsd:

            gsd = np.zeros(illum_params.shape, dtype=np.float64) + np.nan

            ifov = float(camera_model.instantaneous_field_of_view().squeeze())

            gsd[check] = np.sin(ifov) * surf2cam_distances.ravel()[check]

            surf2cam_distances[..., ~check] = np.nan

            return illum_params, surf2cam_distances.ravel(), gsd

        else:
            surf2cam_distances[..., ~check] = np.nan
            return illum_params, surf2cam_distances.ravel(), None

    def compute_velocities(self, velocity_function: Callable) \
            -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        This method uses a given velocity function for the camera to compute the
        velocity and range of the camera relative to each surface element in the
        target body-fixed frame.
        
        :param velocity_function: A callable function that computes the relative velocity of
                                  each surface element to the camera in the camera frame
        
        :return: A tuple of the following data:\n
                 An array of camera velocities relative to each surface element in pixels/s,\n
                 An array of camera velocities relative to each surface element in km/s, and\n
                 An array of camera ranges relative to each surface element in km
        """

        assert self.visibility is not None, "you must call compute_visibility before compute_velocities"
        camera_velocities = np.ones((self.visibility.shape[0], 3), dtype=np.float64) * np.nan
        pixel_velocities = np.ones((self.visibility.shape[0], 2), dtype=np.float64) * np.nan
        ranges = np.ones((self.visibility.shape[0],), dtype=np.float64) * np.nan

        do_labels = self.labels is not None
        current_label = self.labels[0] if do_labels else None # type: ignore
        nlabel = 0
        aimage = 0

        image = 0
        for vis in self.visibility:

            if do_labels and (vis['label'][0] != current_label):
                image = 0
                nlabel += 1
                current_label = vis['label'][0]

            if vis['visible'].any():

                vecs = self.targetvecs[:, vis['visible']]

                if do_labels:
                    image_time = self.imaging_times[nlabel][image] # type: ignore
                else:
                    image_time = self.imaging_times[image]

                camera_model = self.camera_model(image_time) if callable(self.camera_model) else self.camera_model

                # rotation from body-fixed frame to camera frame
                body_fixed_to_camera_frame = self.camera_orientation_function(image_time).matrix

                # relative velocity between camera and targets in the camera frame
                cam_velocity = body_fixed_to_camera_frame @ velocity_function(image_time, vecs).reshape(3, -1)

                # relative position from camera to the target in the camera frame
                pos = -body_fixed_to_camera_frame @ (
                            self.camera_position_function(image_time).reshape(3, -1) - vecs.reshape(3, -1))

                # new position from camera to target in the camera frame one second later
                new_pos = pos + cam_velocity

                # velocity in units of pixels per second using finite differencing
                pix_velocity = (camera_model.project_onto_image(new_pos) -
                                camera_model.project_onto_image(pos))

                # store the maximum velocity experienced for each image both in units of km/s and pix/s
                max_pix_velo_idx = np.linalg.norm(pix_velocity, axis=0).argmax()

                if do_labels:
                    camera_velocities[aimage] = cam_velocity[:, np.linalg.norm(cam_velocity, axis=0).argmax()]
                    pixel_velocities[aimage] = pix_velocity[:, max_pix_velo_idx]
                    ranges[aimage] = np.linalg.norm(pos[:, max_pix_velo_idx])
                    aimage += 1

                else:
                    camera_velocities[image] = cam_velocity[:, np.linalg.norm(cam_velocity, axis=0).argmax()]
                    pixel_velocities[image] = pix_velocity[:, max_pix_velo_idx]
                    ranges[image] = np.linalg.norm(pos[:, max_pix_velo_idx])

            image += 1

        return camera_velocities, pixel_velocities, ranges

    def check_fov(self, vertices: NDArray[np.float64], camera_model: Optional[CameraModel] = None) -> NDArray[np.bool]:
        """
        This function determines which surface vertices are within the camera's FOV
        by projecting the vertices to a simulated image.

        :param vertices: An array of 3D vectors representing vertices on the target body's surface
        :param camera_model: A :class:`CameraModel` instance
        
        :return: An array of booleans representing which vertices are within the camera's FOV
        """

        if camera_model is None:
            camera_model = self.camera_model

        pixels = camera_model.project_onto_image(vertices)

        # verify this with John
        in_fov = (pixels[0] >= 0) & (pixels[1] >= 0) & \
                 (pixels[0] <= camera_model.n_cols - 1) & (pixels[1] <= camera_model.n_rows - 1)

        # the camera models break down at extremes so also spot check the angular FOV
        angles = (np.arccos(np.array([[0, 0, 1]]) @ vertices[:, in_fov] / np.linalg.norm(vertices[:, in_fov],
                                                                                         axis=0,
                                                                                         keepdims=True)) *
                  180 / np.pi).ravel()
        if camera_model.field_of_view == 0:
            # get the FOV computed
            camera_model.compute_field_of_view()
        in_fov[in_fov] = angles < 1.25 * camera_model.field_of_view

        return in_fov

    def determine_footprints(self) -> Union[List[NDArray[np.float64]], List[List[NDArray[np.float64]]]]:
        """
        This function determines a rectangular boundary footprint of the camera's field-of-view
        projected onto the target surface. If a corner of the FOV misses the target,
        that boundary point will surround the target, but if a corner of the FOV intersects the
        target, that boundary point will lie on the target's surface at the intersection point.
        
        :return: A list of 4 boundary points at each imaging time defining a rectangle that represents the
                 area on the target's surface that is visible to the camera in its current configuration.
        """

        def footprint_at_time(image_time: datetime):
            camera_model = self.camera_model(image_time) if callable(self.camera_model) else self.camera_model
            line_of_sight = camera_model.pixels_to_unit(np.array(
                [[0, 0, camera_model.n_cols, camera_model.n_cols],
                 [0, camera_model.n_rows, camera_model.n_rows, 0]]))
            rays = Rays([0, 0, 0], line_of_sight)
            # position of camera relative to body in body frame
            camera_position = self.camera_position_function(image_time)

            # from body frame to camera frame
            rotation_body_fixed_to_camera = self.camera_orientation_function(image_time)

            # put body in camera frame
            for obj in self.scene.target_objs:
                obj.change_orientation(rotation_body_fixed_to_camera)
                obj.change_position(-rotation_body_fixed_to_camera.matrix @ camera_position)

            checks = self.scene.trace(rays)

            footprint = (rotation_body_fixed_to_camera.matrix.T @ checks['intersect'].T + camera_position.reshape(3,
                                                                                                                  1)).T

            target_range = np.linalg.norm(self.scene.target_objs[0].position)

            footprint[~checks['check']] = (target_range * rotation_body_fixed_to_camera.matrix.T @
                                           rays[~checks['check']].direction.reshape(3, -1) +
                                           camera_position.reshape(3, 1)).T
            return footprint

        footprints = []

        for imaging_times in self.imaging_times:
            if self.labels is None:
                image_time = imaging_times
                assert isinstance(image_time, datetime)
                footprints.append(footprint_at_time(image_time))
            else:
                assert isinstance(imaging_times, list)
                label_footprints = [footprint_at_time(image_time) for image_time in imaging_times]
                footprints.append(label_footprints)

        return footprints

    def compute_dop(self) -> Tuple[Dict[str, List[int]], Dict[str, List[DOP_TYPE]], Dict[str, List[DOP_TYPE]],
                                   Dict[str, List[DOP_TYPE]], Dict[str, List[DOP_TYPE]]]:
        """
        This function computes the dilution of precision (DOP) metrics for all facets
        based off images where each facet visible as determined by the coverage analysis.

        The following parameters are computed for each facet separately and then
        organized into arrays containing the results for all facets ordered
        by the index of the corresponding facet:\n
            cnt -- number of usable observation images\n
            jac -- SCP jacobian corresponding to a change in illumination given\n
                   a change to the surface normal and/or albedo
            alb -- DOP of local terrain relative brightness\n
            xt  -- DOP of local terrain x slope, typically East/West\n
            yt  -- DOP of local terrain y slope, typically North/South\n
            tot -- RSS (L2-norm) of xt, yt, and alb DOP values\n
        
        The DOP values are dependent on the BRDF we are using to characterize
        surface illumination.
        
        :return: A tuple of DOP metrics (cnt, alb, xt, yt, tot) which are each wrapped
                 into their own dictionaries with keys denoted by an "all" label
                 and any other labels for imaging time intervals if applicable
        """

        # extract the local topography variations to consider
        az_grid, elev_grid = self.topography_variations

        mapping_obj = DOPComputations(az_grid, elev_grid, self.brdf)
        assert self.facet_visibility is not None, "compute_visibility and reduce_visibility_to_facet must be called before compute_dop"
        with get_context('spawn').Pool() as pool:  # spawn context pool is safer than default fork

            results = list(tqdm(pool.imap(mapping_obj.compute_target_dop_facet, self.facet_visibility.T),
                                total=len(self.facet_visibility.T), desc='processing facets', unit=' facets',
                                dynamic_ncols=True))

        cnt, jac, alb, xt, yt, tot = np.asarray(results, dtype=object).T

        if self.labels is not None:
            self.observation_count = {x: [dic[x] for dic in cnt.tolist()] for x in cnt.tolist()[0].keys()}
            self.dop_jacobians = {x: [dic[x] for dic in jac.tolist()] for x in jac.tolist()[0].keys()}
            self.albedo_dop = {x: [dic[x] for dic in alb.tolist()] for x in alb.tolist()[0].keys()}  # type: ignore
            self.x_terrain_dop = {x: [dic[x] for dic in xt.tolist()] for x in xt.tolist()[0].keys()}  # type: ignore
            self.y_terrain_dop = {x: [dic[x] for dic in yt.tolist()] for x in yt.tolist()[0].keys()}  # type: ignore
            self.total_dop = {x: [dic[x] for dic in tot.tolist()] for x in tot.tolist()[0].keys()}  # type: ignore
        else:
            self.observation_count = {'all': cnt.tolist()}
            self.dop_jacobians = {'all': jac.tolist()}
            self.albedo_dop = {'all': alb.tolist()}
            self.x_terrain_dop = {'all': xt.tolist()}
            self.y_terrain_dop = {'all': yt.tolist()}
            self.total_dop = {'all': tot.tolist()}

        return self.observation_count, self.x_terrain_dop, self.y_terrain_dop, self.albedo_dop, self.total_dop
    
