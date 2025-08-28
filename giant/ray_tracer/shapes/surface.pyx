# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This Cython module defines base classes for surface objects and their acceleration structures in GIANT.

Description
-----------

In GIANT a surface is an object which is represented by small, tessellated geometry primitives (like triangles).  It is
the most common way to represent an object in GIANT, since it can be used to represent arbitrary terrain, as well as
both small patches and full global shape models.  In general a user won't interact with this module, though, and instead
will used the predefined surfaces described in :mod:`.triangle`.  This module is reserved primarily for those
developing new surface geometry primitives or acceleration structures.

For those developing new surface geometry primitives, consider the :class:`.RawSurface`, :class:`.Surface64` and
:class:`.Surface32` classes below.  For those developing a new acceleration structure consider the :class:`.Surface`
class below.  For examples of how this is done refer to the :mod:`.triangle` and :mod:`.kdtree` modules.
"""


import numpy as np
cimport numpy as cnp
import cython
from cython.parallel import prange, parallel

cimport numpy as cnp
from libc.math cimport fabs

import pandas as pd

from giant.rotations import Rotation
from giant.ray_tracer.utilities import ref_ellipse, to_block
from giant.ray_tracer.rays import INTERSECT_DTYPE

from giant.ray_tracer.shapes.shape cimport Shape
from giant.ray_tracer.shapes.axis_aligned_bounding_box import AxisAlignedBoundingBox
from giant.ray_tracer.shapes.ellipsoid import Ellipsoid

import warnings


@cython.boundscheck(False)
def find_limbs_surface(Surface target, scan_center_dir, scan_dirs, observer_position=None, initial_step=None,
                       int max_iterations=25, double rtol=1e-12, double atol=1e-12):
    r"""
    find_limbs_surface(target, scan_center_dir, scan_dirs, observer_position=None, initial_step=None, max_iterations=25, rtol=1e-12, atol=1e-12)

    This helper function determines the limb points for a surface (visible edge of the surface) that would be
    visible for an observer located at ``observer_position`` looking toward ``scan_center_dir`` along the
    directions given by ``scan_dirs``.

    Typically it is assumed that the location of the observer is at the origin of the current frame and therefore
    ``observer_position`` can be left as ``None``.

    The limb for the surface is found iteratively by tracing rays from the observer to the surface.  First, the
    rays are traced along the scan center direction, which should beg guaranteed to strike the object.  Then, we
    adjust the direction of the rays so that they no longer intersect the surface using ``initial_step`` (or 2 times
    the largest principal axis of the reference ellipse if ``initial_step`` is ``None``).  We then proceed by
    tracing rays with directions half way between the left rays (guaranteed to strike the surface) and the right
    rays (guaranteed to not strike the surface) updating the left and right rays based on the result of the last
    trace.  This continues for a maximum of ``max_iterations`` or until the tolerances specified by ``rtol`` and
    ``atol`` are met for the change in the estimate of the limb location.  The returned limb location is the last
    ray intersect location that hit the surface for each ``scan_dirs``.

    The returned limbs are expressed as vectors from the observer to the limb point in the current frame as a 3xn
    numpy array.

    Note that this is specific to :class:`.RawSurface` surfaces and their subclasses (and is thus accelerated).  For a
    general purpose ``find_limbs`` for any traceable object see the :func:`.find_limbs` function in the utilities
    module.

    :param target: The target object that we are to find the limb points for as a :class:`.Surface`
    :type target: Surface
    :param scan_center_dir: the unit vector which the scan is to begin at in the current frame as a length 3 array
                            A ray cast along this unit vector from the ``observer_position`` should be guaranteed
                            to strike the surface and ideally should be towards the center of figure of the surface
    :type scan_center_dir: ARRAY_LIKE
    :param scan_dirs: the unit vectors along with the scan is to proceed as a 3xn array in the current frame where
                      each column represents a new limb point we wish to find (should be nearly orthogonal to the
                      ``scan_center_dir`` in most cases).
    :type scan_dirs: ARRAY_LIKE
    :param observer_position: The location of the observer in the current frame.  If ``None`` then it is assumed
                              the observer is at the origin of the current frame
    :type observer_position: Optional[ARRAY_LIKE]
    :param initial_step: The size of the initial step to take along the ``scan_dirs`` direction.  This should be
                         guaranteed to result in rays that do not strike the body.
    :type initial_step: float
    :param max_iterations: The maximum number of iteration steps to take when locating the limbs
    :type max_iterations: int
    :param rtol: The relative tolerance of the change in the limb location from one iteration to the next that
                 indicates convergence.
    :type rtol: float
    :param atol: the absolute tolerance of the change int he limb location from one iteration to the next that
                 indicates convergence.
    :return: the vectors from the observer to the limbs in the current frame as a 3xn array
    :rtype: numpy.ndarray
    """

    if not isinstance(target, Surface):
        raise ValueError("Invalid type for target")

    scan_dirs = np.array(scan_dirs).reshape(3, -1)

    if observer_position is not None:
        single_start = np.array(observer_position).reshape(3, 1)
    else:
        single_start = np.zeros((3, 1), dtype=np.float64)

    if hasattr(target, "rotation"):
        if target.rotation is not None:
            single_start = np.matmul(target.rotation.matrix, single_start)
            scan_center_dir = np.matmul(target.rotation.matrix, scan_center_dir)
            scan_dirs = np.matmul(target.rotation.matrix, scan_dirs)

    if hasattr(target, "position"):
        if target.position is not None:
            single_start += target.position.reshape(3, 1)

    cdef:
        int iter, ray_number, axis
        bint all_converged

        # rays for tracing
        const double[:, :] start = np.broadcast_to(single_start, (3, scan_dirs.shape[1]))
        double[:, :] left_dirs = np.array([scan_center_dir]*scan_dirs.shape[1], dtype=np.float64).T
        double[:, :] right_dirs = np.array([scan_center_dir]*scan_dirs.shape[1], dtype=np.float64).T
        double[:, :] trace_dirs = np.array([scan_center_dir]*scan_dirs.shape[1], dtype=np.float64).T
        double[:, :] inv_trace_dirs = np.array([1/scan_center_dir]*scan_dirs.shape[1], dtype=np.float64).T
        cnp.int64_t[:, :] ignore = -np.ones((scan_dirs.shape[1], 1), dtype=np.int64)

        # results for tracing
        cnp.uint8_t[:] hit = np.zeros((scan_dirs.shape[1]), dtype=np.uint8)
        double[:, :] intersect = np.zeros((scan_dirs.shape[1], 3), dtype=np.float64)
        double[:, :] normal = np.zeros((scan_dirs.shape[1], 3), dtype=np.float64)
        double[:] albedo = np.zeros((scan_dirs.shape[1]), dtype=np.float64)
        double[:] distance = np.zeros((scan_dirs.shape[1]), dtype=np.float64)+np.inf
        cnp.int64_t[:] facet = np.zeros((scan_dirs.shape[1]), dtype=np.int64)

        double[:, :] limb_locations = np.zeros((scan_dirs.shape[1], 3), dtype=np.float64)

        double cyinf = np.inf


    if initial_step is None:

        tlimbs = target.reference_ellipsoid.find_limbs(scan_center_dir, scan_dirs, observer_position=observer_position)
        tlimbs /= np.linalg.norm(tlimbs, axis=0, keepdims=True)

        center_limb_angle = np.arccos(tlimbs.T@scan_center_dir.ravel())

        scan_dir_angles = np.arccos(scan_dirs.T @ scan_center_dir.ravel())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            step = 2*np.nanmax(np.abs(np.sin(center_limb_angle)/np.sin(np.pi - center_limb_angle + scan_dir_angles)))

        if np.isnan(step):
            step = (2.0*target.reference_ellipsoid.principal_axes.max())
            step /= np.linalg.norm(target.reference_ellipsoid.center-single_start.ravel())

    else:
        step = initial_step

    # set the initial limb locations, which will be the intersect using the left rays, which should all be the same but
    # oh well
    target._trace(start, left_dirs, inv_trace_dirs, ignore, start.shape[1], True,
                  hit, limb_locations, normal, albedo, facet, distance)

    hit[:] = 0
    distance[:] = cyinf

    # set the right directions to start
    for ray_number in range(start.shape[1]):
        for axis in range(3):
            right_dirs[axis, ray_number] += step*scan_dirs[axis, ray_number]
            trace_dirs[axis, ray_number] += step/2.0*scan_dirs[axis, ray_number]
            inv_trace_dirs[axis, ray_number] = 1/trace_dirs[axis, ray_number]

    # loop through until convergence
    for iter in range(max_iterations):

        # trace the current rays we are checking
        target._trace(start, trace_dirs, inv_trace_dirs, ignore, start.shape[1], True,
                      hit, intersect, normal, albedo, facet, distance)

        # loop through and update what should now be our left and what should be our right rays
        all_converged = True
        for ray_number in range(start.shape[1]):
            if hit[ray_number]:
                # if we hit then this becomes the new left ray.  The new trace is halfway between it and the right ray
                # we also update the limb location
                for axis in range(3):
                    left_dirs[axis, ray_number] = trace_dirs[axis, ray_number]
                    trace_dirs[axis, ray_number] = (left_dirs[axis, ray_number] + right_dirs[axis, ray_number])/2
                    inv_trace_dirs[axis, ray_number] = 1/trace_dirs[axis, ray_number]
                    if (fabs(limb_locations[ray_number, axis] - intersect[ray_number, axis]) >
                        atol + rtol * fabs(intersect[ray_number, axis])):
                        all_converged = False
                    limb_locations[ray_number, axis] = intersect[ray_number, axis]
            else:
                # if we didn't hit then this becomes the new left ray.  The new trace is halfway between it
                # and the right ray
                for axis in range(3):
                    right_dirs[axis, ray_number] = trace_dirs[axis, ray_number]
                    trace_dirs[axis, ray_number] = (left_dirs[axis, ray_number] + right_dirs[axis, ray_number])/2
                    inv_trace_dirs[axis, ray_number] = 1/trace_dirs[axis, ray_number]
                    if (fabs(left_dirs[axis, ray_number] - right_dirs[axis, ray_number]) >
                        atol + rtol * fabs(left_dirs[axis, ray_number])):
                        all_converged = False

        if all_converged:
            break
        hit[:] = 0  # update the hit flag so we don't get messed up
        distance[:] = cyinf  # update the distance so that we don't shortcut

    # subtracts off the position already
    result = np.asarray(limb_locations).T - start

    # does the inverse rotation if we need to
    if hasattr(target, "rotation"):
        if target.rotation is not None:
            result = np.matmul(target.rotation.inv().matrix, result)

    return result


cdef class Surface(Shape):
    """
    This defines the basic interface expected of all surfaces in GIANT.

    In GIANT, a surface is considered anything that is represented by tesselation.  As such, it has a few distinguishing
    characteristics.  (1) limbs are found iteratively instead of analytically, (2) the limb jacobian is approximated,
    and (3) we have a :attr:`reference_ellipsoid` which is the best fit ellipsoid to the surface used for approximating
    that jacobian.  This class makes these distinctions explicit.

    You cannot directly use this class in GIANT (it doesn't even have an init method).  Instead, you should either use
    it for instance checks (all surfaces and surface acceleration structures inherit from this class) or you should
    subclass it in you adding a new surface.  When you do subclass you should no longer have to worry about implementing
    methods :meth:`find_limbs` or :meth:`compute_limb_jacobian` as they are already implemented for you.  In general you
    should only directly inherit from this if you are defining a new surface acceleration structure.  If you are
    defining a new surface in general you should instead from :class:`.Surface32` or :class:`.Surface64` which
    subsequently inherit from this class.
    """

    def find_limbs(self, scan_center_dir, scan_dirs, observer_position=None):
        """
        find_limbs(self, scan_center_dir, scan_dirs, observer_position=None)

        This method determines the limb points for a surface (visible edge of the surface) that would be
        visible for an observer located at ``observer_position`` looking toward ``scan_center_dir`` along the
        directions given by ``scan_dirs``.

        Typically it is assumed that the location of the observer is at the origin of the current frame and therefore
        ``observer_position`` can be left as ``None``.

        This method operates iteratively, as described by :func:`.find_limbs_surface`.  To have more control over the
        accuracy of the limb points you should use the :func:`.find_limbs_surface` function directly.

        The returned limbs are expressed as vectors from the observer to the limb point in the current frame as a 3xn
        numpy array.

        :param scan_center_dir: the unit vector which the scan is to begin at in the current frame as a length 3 array
                                A ray cast along this unit vector from the ``observer_position`` should be guaranteed
                                to strike the surface and ideally should be towards the center of figure of the surface
        :type scan_center_dir: ARRAY_LIKE
        :param scan_dirs: the unit vectors along with the scan is to proceed as a 3xn array in the current frame where
                          each column represents a new limb point we wish to find (should be nearly orthogonal to the
                          ``scan_center_dir`` in most cases).
        :type scan_dirs: ARRAY_LIKE
        :param observer_position: The location of the observer in the current frame.  If ``None`` then it is assumed
                                  the observer is at the origin of the current frame
        :type observer_position: Optional[ARRAY_LIKE]
        :return: the vectors from the observer to the limbs in the current frame as a 3xn array
        :rtype: numpy.ndarray
        """

        return find_limbs_surface(self, scan_center_dir, scan_dirs, observer_position)

    def compute_limb_jacobian(self, center_direction, scan_vectors, limb_points_camera, observer_position=None):
        """
        compute_limb_jacobian(self, scan_center_dir, scan_dirs, limb_points, observer_position=None)

        This method computes the linear change in the limb location given a change in the relative position between the
        surface and the observer.

        The limb Jacobian is approximated using the limb jacobian of the reference ellipsoid for the surface.  See
        :meth:`.Ellipsoid.compute_limb_jacobian` for details.  In addition, see paper
        https://seal.rpi.edu/sites/default/files/workshop/2018/papers-abstracts/Limb%20Based%20Optical%20Navigation%20for%20Irregular%20Bodies.pdf
        for why this is an OK approximation.

        :param scan_center_dir: the unit vector which the scan is to begin at in the current frame as a length 3 array
        :type scan_center_dir: np.ndarray
        :param scan_dirs: the unit vectors along with the scan is to proceed as a 3xn array in the current frame where
                          each column represents a new limb point we wish to find (should be nearly orthogonal to the
                          ``scan_center_dir`` in most cases).
        :type scan_dirs: np.ndarray
        :param limb_points: The vectors from the observer to the limb points in the current frame as a 3xn numpy array
                            where each column corresponds to the same column in the :attr:`scan_dirs` attribute.
        :type limb_points: np.ndarray
        :param observer_position: The location of the observer in the current frame.  If ``None`` then it is assumed
                                  the observer is at the origin of the current frame
        :type observer_position: Optional[np.ndarray]
        :return: The jacobian matrix as a nx3x3 array where each panel corresponds to the column in the ``limb_points``
                 input.
        """

        return self.reference_ellipsoid.compute_limb_jacobian(center_direction, scan_vectors, limb_points_camera,
                                                              observer_position)

    cdef void _compute_intersect(self, const double[:] start, const double[:] direction, const double[:] inv_direction,
                                 const cnp.int64_t[] ignore, const cnp.uint32_t num_ignore,
                                 cnp.uint8_t *hit, double[:] intersect, double[:] normal, double *albedo,
                                 cnp.int64_t *facet, double *hit_distance) noexcept nogil:
        """
        This C method is used to compute the intersect between a single ray and the surfaces contained in this object.

        The python version of this method :meth:`.compute_intersect` should be used unless working from Cython
        """

        pass

    def compute_intersect(self, ray):
        """
        compute_intersect(self, ray)

        This method computes the intersects between a single ray and the surface describe by this object.

        This is done by make a call to the efficient C implementation of the method.  This method packages everything in
        the way that the C method expects and restructures the results into the expect numpy structured array.

        In general, if you are tracing multiple rays you should use the :meth:`trace` method which is more optimized
        for multi ray tracing.

        :param ray: The ray to trace to the surce
        :type ray: Rays
        :return: a length 1 numpy array with a dtype of :attr:`.INTERSECT_DTYPE`
        :rtype: np.ndarray
        """

        # extract the components of the ray that we need
        start = ray.start.ravel()
        direction = ray.direction.ravel()
        inv_direction = ray.inv_direction.ravel()
        # prepare the ignore vector
        if ray.ignore is None:
            ignore = np.array([-1], dtype=np.int64)
        else:
            ignore = np.atleast_1d(ray.ignore).astype(np.int64).ravel()

        # initialize the output variables
        cdef:
            cnp.uint8_t hit = 0
            double[:] intersect = np.zeros(3)*np.nan
            double[:] normal = np.zeros(3)*np.nan
            double albedo = np.nan
            double distance = np.inf
            cnp.int64_t facet = -1
            cnp.int64_t[:] mignore = ignore


        # compute the intersect
        self._compute_intersect(start, direction, inv_direction, &mignore[0], ignore.size,
                                &hit, intersect, normal, &albedo, &facet, &distance)

        # put the result in the required structured array
        result = np.array([(hit, distance, intersect, normal, albedo, facet)], dtype=INTERSECT_DTYPE)

        return result

    @cython.boundscheck(False)
    cdef void _trace(self, const double[:, :] starts, const double[:, :] directions, const double[:, :] inv_directions,
                     const cnp.int64_t[:, :] ignore, const cnp.uint32_t num_rays, const bint omp,
                     cnp.uint8_t[:] hit, double[:, :] intersect, double[:, :] normal, double[:] albedo,
                     cnp.int64_t[:] facet, double[:] hit_distances) noexcept nogil:
        """
        This C method is used to intersect multiple rays with the surfaces in this object.

        This is done by making calls to the _compute_intersect C method for each ray provided.

        The python version of this function provides an easy user interface that avoids the need to allocate/deallocate
        arrays and should generally be used instead of calls to this C method.

        Unless specifically requested to the contrary, this method uses OpenMP to parallelize the tracing
        """

        cdef int ray
        cdef long long num_ignore = ignore.shape[1]

        if omp:

            with nogil, parallel():
                # if we are using parallel then drop the gil and do the trace in parallel
                for ray in prange(num_rays, schedule='dynamic'):

                    self._compute_intersect(starts[:, ray], directions[:, ray], inv_directions[:, ray], &ignore[ray, 0],
                                            num_ignore,
                                            &hit[ray], intersect[ray], normal[ray], &albedo[ray], &facet[ray],
                                            &hit_distances[ray])
        else:

            with nogil:
                for ray in range(num_rays):
                    # sequentially trace each ray
                    self._compute_intersect(starts[:, ray], directions[:, ray], inv_directions[:, ray], &ignore[ray, 0],
                                            num_ignore,
                                            &hit[ray], intersect[ray], normal[ray], &albedo[ray], &facet[ray],
                                            &hit_distances[ray])

    def trace(self, rays, omp=True):
        """
        trace(self, rays, omp=True)

        This python method provides an easy interface to trace a number of Rays through the surface.

        It packages all of the ray inputs and the required output arrays automatically and dispatches to the c version
        of the method for efficient computation.  It then packages the output into the expected structured array.

        Parallel processing can be turned off by setting the omp flag to False

        :param rays: The rays to trace to the surface
        :type rays: Rays
        :param omp: A boolean flag specifying whether to use parallel processing (``True``) or not
        :type omp: bool
        :return: A length n numpy array with a data type of :data:`.INTERSECT_DTYPE`
        :rtype: np.ndarray
        """

        # extract the components of the rays
        starts = rays.start.reshape(3, -1)
        directions = rays.direction.reshape(3, -1)
        inv_directions = rays.inv_direction.reshape(3, -1)

        # form the results structured array
        results = np.zeros(rays.num_rays, INTERSECT_DTYPE)

        # extract the required components of the results structured array
        hits = results["check"].astype(np.uint8)  # use integer dtype since cython doesn't like bool arrays
        intersects = results["intersect"]
        distances = results["distance"]
        normals = results["normal"]
        albedos = results["albedo"]
        facets = results["facet"]

        # initialize the output
        intersects[:] = np.nan
        distances[:] = np.inf
        normals[:] = np.nan
        albedos[:] = np.nan
        facets[:] = -1

        # fix the ignores so that they can be effectively used
        if rays.ignore is None:
            ignore = -np.ones((rays.num_rays, 1), dtype=np.int64)
        else:
            ignore = to_block(rays.ignore)

        # trace the rays
        self._trace(starts, directions, inv_directions, ignore, rays.num_rays, omp,
                    hits, intersects, normals, albedos, facets, distances)

        # re-add the check column since we needed to mess with the dtype
        results["check"] = hits

        return results


cdef class RawSurface(Surface):
    """
    __init__(self, vertices, albedos, facets, normals=None, compute_bounding_box=True, bounding_box=None, compute_reference_ellipsoid=True, reference_ellipsoid=None)

    This interface class serves as the backbone for surfaces in GIANT.

    This is simply included to reduce code duplication.  As such it provides the default :meth:`__init__`,
    :meth:`__reduce__`, :meth:`__eq__`, :meth:`merge`, :meth:`rotate`, :meth:`translate`, :meth:`compute_bounding_box`,
    and :meth:`compute_reference_ellipsoid` methods which are generally shared by all surfaces.  It additionally
    provides properties :attr:`facets`, :attr:`stacked_vertices`, and :attr:`num_faces` which are commonly shared.

    A user should never use this class directly, even in development.  Instead use :class:`.Surface32` or
    :class:`.Surface64`.
    """

    def __init__(self, vertices, albedos, facets, normals=None, compute_bounding_box=True,
                 bounding_box=None, compute_reference_ellipsoid=True, reference_ellipsoid=None):
        """
        :param vertices: The vertices for the surface as a n by 3 array
        :type vertices: ARRAY_LIKE
        :param albedos: The albedo(s) for the surface as either a scalar or a length n array
        :type albedos: Union[ARRAY_LIKE, float]
        :param facets: The connectivity map for the surface as a mxp array where m is the number of faces, p is the
                       number of vertices for each face, and each element is an index into the ``vertices`` array
        :type facets: ARRAY_LIKE
        :param normals: The normal vectors for each face of the surface.  If ``None`` then the normals will be computed
        :type normals: Optional[ARRAY_LIKE]
        :param compute_bounding_box: A flag specifying whether to compute the axis aligned bounding box for the surface
        :type compute_bounding_box: True
        :param bounding_box: The axis aligned bounding box for the surface.  This will be overwritten if
                             ``compute_bounding_box`` is ``True``
        :type compute_bounding_box: Optional[AxisAlignedBoundingBox]
        :param compute_reference_ellipsoid: A flag specifying whether to compute the reference ellipsoid for the
                                            surface.
        :type compute_reference_ellipsoid: bool
        :param reference_ellipsoid: The reference ellipsoid for the surface.  If ``compute_reference_ellipsoid`` is
                                    ``True`` this will be overwritten
        :type reference_ellipsoid: Optional[Ellipsoid]
        """


        self.vertices = vertices
        self.albedos = albedos
        self.facets = facets

        if normals is not None:
            self.normals = normals
        else:
            self.compute_normals()

        self.bounding_box = bounding_box
        if compute_bounding_box:
            self.compute_bounding_box()

        self.reference_ellipsoid = reference_ellipsoid

        if compute_reference_ellipsoid:
            self.compute_reference_ellipsoid()

    def __reduce__(self):
        """
        This method is used for pickling/unpickling the object
        """

        comp_bbox = False
        comp_ref_ellipsoid = False
        varray = np.asanyarray(self._vertices.base)

        if self._single_albedo:
            return self.__class__, (varray, self._albedo, self.facets, self.normals, comp_bbox,
                                    self.bounding_box, comp_ref_ellipsoid, self.reference_ellipsoid)
        else:
            return self.__class__, (varray, np.asanyarray(self._albedo_array.base), self.facets,
                                    self.normals, comp_bbox, self.bounding_box, comp_ref_ellipsoid,
                                    self.reference_ellipsoid)

    def __eq__(self, other):
        """
        __eq__(self, other)

        This method is used for unit testing to compare multiple surfaces

        :param other: The other surface to check equality against
        :type other: RawSurface
        :return: ``True`` if the surfaces are the same, ``False`` otherwise
        :rtype: bool
        """

        # check that all the vertices are the same
        if not np.allclose(self.vertices, other.vertices):

            return False

        if not np.array_equal(self.facets, other.facets):

            return False

        # check that all the albedos are the same
        if not np.allclose(self.albedos, other.albedos):

            return False

        # check that all the normals are the same
        if not np.allclose(self.normals, other.normals):

            return False

        # check that the bounding boxes are the same
        if self.bounding_box != other.bounding_box:
            return False

        # if we made it this far then the objects are probably the same
        return True

    def merge(self, other, compute_bounding_box=True, compute_reference_ellipsoid=True):
        """
        merge(self, other, compute_bounding_box=True, compute_reference_ellipsoid=True)

        This method is used to merge two surface collections together.  It only works if the two surfaces are the same

        This method returns a new object, it does not do an in-place merge.

        :param other: The other surface to be merged with this one
        :type other: RawSurface
        :param compute_bounding_box: A flag specifying whether to compute the bounding box for the merged shape
        :type compute_bounding_box: bool
        :param compute_reference_ellipsoid: A flag specifying whether to compute the reference ellipsoid for the merged
                                            shape
        :type compute_reference_ellipsoid: bool
        :return: A new surface resulting from combining self with other
        :rtype: RawSurface
        :raises ValueError: If the surface types are not the same
        """

        if isinstance(other, self.__class__):

            # merge the vertices
            verts = np.concatenate([self.vertices, other.vertices], axis=0)

            # merge the normal vectors
            normals = np.concatenate([self.normals, other.normals], axis=0)

            # merge the facets
            facets = np.concatenate([self.facets, other.facets+self.vertices.shape[0]])

            # TODO: add cases for when one is scalar but other isn't
            if np.isscalar(self.albedos) and np.isscalar(other.albedos):
                # merge the albedos
                if self.albedos == other.albedos:
                    # if the albedos are scalar and the same value just keep that value
                    albedos = self.albedos

                else:
                    # if the albedos are scalar and different then make them vectored
                    albedos = np.asarray([self.albedos]*self.num_faces + [other.albedos]*other.num_faces)

            elif np.isscalar(self.albedos):
                # if the albedos are scalar and different then make them vectored
                albedos = np.concatenate([[self.albedos]*self.num_faces, other.albedos])
            elif np.isscalar(other.albedos):
                # if the albedos are scalar and different then make them vectored
                albedos = np.concatenate([self.albedos, [other.albedos]*other.num_faces])
            else:
                # if both albedos are vectored merge the vectors
                albedos = np.concatenate([self.albedos, other.albedos])

            return self.__class__(verts, albedos, facets, normals=normals, compute_bounding_box=compute_bounding_box,
                                  compute_reference_ellipsoid=compute_reference_ellipsoid)

        else:
            raise ValueError('Cannot merge surfaces that are not the same type')

    @property
    def facets(self):
        """
        This property returns the facets for the surfaces(s) as a mxp ndarray where each face has p vertices and
        there are m faces overall.

        The facet elements are indices into the :attr:`vertices` array.  They must be positive.

        Each row of this array corresponds to the same row of the :attr:`normals` array
        """
        return np.asarray(self._facets)

    @facets.setter
    def facets(self, val):
        if isinstance(val, np.ndarray):
            if val.dtype == np.uint32:
                if val.shape[-1] == 3:
                    self._facets = val
                else:
                    self._facets = val.reshape(-1, 3)
            else:
                self._facets = val.reshape(-1, 3).astype(np.uint32)
        else:
            self._facets = np.asarray(val).reshape(-1, 3).astype(np.uint32)

    @property
    def albedos(self):
        """
        This property represents the albedo for the surface.

        The albedo can either be a scalar, if there is a single albedo for the entire array, or an array of length n
        if there is a different albedo value for each vertex.  Both are accessed through this property
        """
        return

    @albedos.setter
    def albedos(self, val):
        pass

    @property
    def stacked_vertices(self):
        """
        This property returns the vertices for the surfaces(s) as a mx3xp ndarray where each surface has p vertices and
        there are m surfaces overall.

        This returns self.vertices[self.facets].swapaxes(-1, -2).  Note that this will always produce a copy, so be
        cautious about using this property frequently
        """
        return self.vertices[self.facets].swapaxes(-1, -2)

    @property
    def vertices(self):
        """
        This property returns the vertices for the surfaces as a nx3 ndarray where there are n vertices.

        The :attr:`facets` array indexes into the row axis of this array
        """
        return

    @vertices.setter
    def vertices(self, val):
        pass

    @property
    def normals(self):
        """
        This property returns the normals for the surfaces(s) as a mx3 ndarray where there are m surfaces contained

        Each row of this array corresponds to the same row in the :attr:`facets` array.
        """
        return

    @normals.setter
    def normals(self, val):
        pass

    @property
    def num_faces(self):
        """
        The number of faces in this surface.

        This is the length of the first axis of the :attr:`facets` array.
        """

        return self._facets.shape[0]

    def rotate(self, rotation):
        """
        rotate(self, rotation)

        This method rotates the surface into a new frame.

        The rotation is applied to the :attr:`vertices`, :attr:`normals`, :attr:`bounding_box`, and
        :attr:`reference_ellipsoid`.

        :param rotation: an array like object or an Rotation object to rotate the surface by.  See the
                         :class:`.Rotation` documentation for details on valid inputs
        :type rotation: Union[Rotation, ARRAY_LIKE]
        """

        if isinstance(rotation, Rotation):
            # rotate the vertices
            self.vertices = (rotation.matrix @ self.vertices.T).T
            # rotate the normal vectors
            self.normals = (rotation.matrix @ self.normals.T).T

        else:
            # rotate the vertices
            self.vertices = (Rotation(rotation).matrix @ self.vertices.T).T
            # rotate the normal vectors
            self.normals = (Rotation(rotation).matrix @ self.normals.T).T

        if self.bounding_box:
            # rotate the bounding box
            self.bounding_box.rotate(rotation)

        if self.reference_ellipsoid:
            # rotate the reference ellipsoid
            self.reference_ellipsoid.rotate(rotation)

        return self

    def translate(self, translation):
        """
        translate(self, translation)

        This method translates the surface

        The translation is applied to the :attr:`vertices`, :attr:`bounding_box`, and :attr:`reference_ellipsoid`.

        :param translation: an array like object of size 3
        :type translation: ARRAY_LIKE
        """
        if np.size(translation) == 3:
            trans_array = np.asarray(translation)

            # translate the vertices
            self.vertices += trans_array.reshape(1, 3)

        else:
            raise ValueError("You have entered an improperly sized translation.\n"
                             "Only length 3 translations are allowed.\n"
                             "You entered {0}".format(np.size(translation)))

        if self.bounding_box:
            # translate the bounding box
            self.bounding_box.translate(translation)

        if self.reference_ellipsoid:
            # translate the reference ellipsoid
            self.reference_ellipsoid.translate(translation)

        return self

    def compute_bounding_box(self):
        """
        compute_bounding_box(self)

        This method computes the bounding box that contains all of the vertices.

        This is done by finding the minimum and maximum values of the vertices that are used in the surface according to
        :attr:`facets`.  The results are stored in :attr:`bounding_box`.
        """

        unique_verts = self.vertices[pd.unique(self.facets.ravel())]
        self.bounding_box = AxisAlignedBoundingBox(unique_verts.min(axis=0), unique_verts.max(axis=0))

    def compute_reference_ellipsoid(self):
        """
        compute_reference_ellipsoid(self)

        This method computes the reference ellipsoid that best represents the vertices.

        This is done by finding the best fit ellipsoid to the vertices that are used in the surface according to
        :attr:`facets`.  The results are stored in :attr:`reference_ellipsoid`.

        This is done by a dispatch to the :func:`.ref_ellipse` function.
        """

        unique_verts = self.vertices[pd.unique(self.facets.ravel())]
        self.reference_ellipsoid = ref_ellipse(unique_verts)


cdef class Surface64(RawSurface):
    """
    __init__(self, vertices, albedos, facets, normals=None, compute_bounding_box=True, bounding_box=None, compute_reference_ellipsoid=True, reference_ellipsoid=None)

    This class serves as the backbone for surfaces in GIANT represented using double precision.

    As such it provides the default :meth:`__init__`,
    :meth:`__reduce__`, :meth:`__eq__`, :meth:`merge`, :meth:`rotate`, :meth:`translate`, :meth:`compute_bounding_box`,
    and :meth:`compute_reference_ellipsoid` methods which are generally shared by all surfaces.  It additionally
    provides properties :attr:`facets`, :attr:`albedos`, :attr:`vertices`, :attr:`normals`, :attr:`stacked_vertices`,
    and :attr:`num_faces` which are commonly shared.

    You should only use this class when developing a new surface in GIANT using double precision vertices.  Beyond that
    this shouldn't even be used as an instance check.  For an example of how to implement a new surface, examine the
    :class:`.Triangle64` class code.  If you want to develop a new surface using single precision vertices then look
    at the :class:`.Surface32` and :class:`.Triangle32` classes.  Though you can technically initialize an instance of
    this class, there's not real reason to since the tracing interface is undefined making the instance mostly
    worthless.
    """

    @property
    def albedos(self):
        """
        This property represents the albedo for the surface.

        The albedo can either be a scalar, if there is a single albedo for the entire array, or an array of length n
        if there is a different albedo value for each vertex.  Both are accessed through this property
        """

        if self._single_albedo:
            return self._albedo

        else:
            return np.asarray(self._albedo_array)

    @albedos.setter
    def albedos(self, val):
        if np.isscalar(val):
            self._single_albedo = True
            self._albedo = float(val)
        else:
            self._single_albedo = False
            if isinstance(val, np.ndarray):
                if val.dtype == np.float64:
                    if val.ndim == 1:
                        self._albedo_array = val
                    else:
                        self._albedo_array = val.ravel()
                else:
                    self._albedo_array = val.ravel().astype(np.float64)
            else:

                self._albedo_array = np.asarray(val).ravel().astype(np.float64)

    @property
    def vertices(self):
        """
        This property returns the vertices for the surfaces as a nx3 ndarray where there are n vertices.

        The :attr:`facets` array indexes into the row axis of this array
        """
        return np.asarray(self._vertices)

    @vertices.setter
    def vertices(self, val):
        if isinstance(val, np.ndarray):
            if val.dtype == np.float64:
                if val.shape[-1] == 3:
                    self._vertices = val
                else:
                    self._vertices = val.reshape(-1, 3)
            else:
                if val.shape[-1] == 3:
                    self._vertices = val.astype(np.float64)
                else:
                    self._vertices = val.reshape(-1, 3).astype(np.float64)
        else:

            self._vertices = np.asarray(val).reshape(-1, 3).astype(np.float64)

    @property
    def normals(self):
        """
        This property returns the normals for the surfaces(s) as a mx3 ndarray where there are m surfaces contained

        Each row of this array corresponds to the same row in the :attr:`facets` array.
        """
        return np.asarray(self._normals)

    @normals.setter
    def normals(self, val):
        if isinstance(val, np.ndarray):
            if val.dtype == np.float64:
                if val.shape[-1] == 3:
                    self._normals = val
                else:
                    self._normals = val.reshape(-1, 3)
            else:
                if val.shape[-1] == 3:
                    self._normals = val.astype(np.float64)
                else:
                    self._normals = val.reshape(-1, 3).astype(np.float64)
        else:

            self._normals = np.asarray(val).reshape(-1, 3).astype(np.float64)

    def compute_normals(self):
        """
        compute_normals(self)

        This method computes the unit normal vectors for each facet.

        The normals are computed in parallel and are stored in the :attr:`normals` array.
        """

        pass


cdef class Surface32(RawSurface):
    """
    __init__(self, vertices, albedos, facets, normals=None, compute_bounding_box=True, bounding_box=None, compute_reference_ellipsoid=True, reference_ellipsoid=None)

    This class serves as the backbone for surfaces in GIANT represented using single precision.

    As such it provides the default :meth:`__init__`,
    :meth:`__reduce__`, :meth:`__eq__`, :meth:`merge`, :meth:`rotate`, :meth:`translate`, :meth:`compute_bounding_box`,
    and :meth:`compute_reference_ellipsoid` methods which are generally shared by all surfaces.  It additionally
    provides properties :attr:`facets`, :attr:`albedos`, :attr:`vertices`, :attr:`normals`, :attr:`stacked_vertices`,
    and :attr:`num_faces` which are commonly shared.

    You should only use this class when developing a new surface in GIANT using single precision vertices.  Beyond that
    this shouldn't even be used as an instance check.  For an example of how to implement a new surface, examine the
    :class:`.Triangle32` class code.  If you want to develop a new surface using single precision vertices then look
    at the :class:`.Surface32` and :class:`.Triangle32` classes.  Though you can technically initialize an instance of
    this class, there's not real reason to since the tracing interface is undefined making the instance mostly
    worthless.
    """

    @property
    def albedos(self):
        """
        This property represents the albedo for the surface.

        The albedo can either be a scalar, if there is a single albedo for the entire array, or an array of length n
        if there is a different albedo value for each vertex.  Both are accessed through this property
        """

        if self._single_albedo:
            return self._albedo

        else:
            return np.asarray(self._albedo_array)

    @albedos.setter
    def albedos(self, val):
        if np.isscalar(val):
            self._single_albedo = True
            self._albedo = float(val)
        else:
            self._single_albedo = False
            if isinstance(val, np.ndarray):
                if val.dtype == np.float32:
                    if val.ndim == 1:
                        self._albedo_array = val
                    else:
                        self._albedo_array = val.ravel()
                else:
                    self._albedo_array = val.ravel().astype(np.float32)
            else:

                self._albedo_array = np.asarray(val).ravel().astype(np.float32)

    @property
    def vertices(self):
        """
        This property returns the vertices for the surfaces as a nx3 ndarray where there are n vertices.

        The :attr:`facets` array indexes into the row axis of this array
        """
        return np.asarray(self._vertices)

    @vertices.setter
    def vertices(self, val):
        if isinstance(val, np.ndarray):
            if val.dtype == np.float32:
                if val.shape[-1] == 3:
                    self._vertices = val
                else:
                    self._vertices = val.reshape(-1, 3)
            else:
                if val.shape[-1] == 3:
                    self._vertices = val.astype(np.float32)
                else:
                    self._vertices = val.reshape(-1, 3).astype(np.float32)
        else:

            self._vertices = np.asarray(val).reshape(-1, 3).astype(np.float32)

    @property
    def normals(self):
        """
        This property returns the normals for the surfaces(s) as a mx3 ndarray where there are m surfaces contained

        Each row of this array corresponds to the same row in the :attr:`facets` array.
        """
        return np.asarray(self._normals)

    @normals.setter
    def normals(self, val):

        if isinstance(val, np.ndarray):
            if val.dtype == np.float32:
                if val.shape[-1] == 3:
                    self._normals = val
                else:
                    self._normals = val.reshape(-1, 3)
            else:
                if val.shape[-1] == 3:
                    self._normals = val.astype(np.float32)
                else:
                    self._normals = val.reshape(-1, 3).astype(np.float32)
        else:

            self._normals = np.asarray(val).reshape(-1, 3).astype(np.float32)
