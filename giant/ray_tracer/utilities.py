# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This module provides some basic utilities for working with the ray tracer in GIANT.

Use
---

In general you won't use the utilities provided herein directly, as they are dispersed throughout the ray tracer in the
areas where they are needed already.  That being said, everything in here should be useable with clear documentation if
you have some non-typical use.
"""

import copy
import datetime 
from typing import Tuple, Optional, Union, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from giant.ray_tracer.rays import Rays

from giant.ray_tracer.shapes.ellipsoid import Ellipsoid
from giant.ray_tracer.shapes.triangle import Triangle64, Triangle32
from giant.ray_tracer.shapes.shape import Shape
from giant.rotations import rotvec_to_rotmat
from giant._typing import ARRAY_LIKE


SPEED_OF_LIGHT = 299792.458  # km/sec
"""
The speed of light in kilometers per second
"""


def find_limbs(shape: Shape, scan_center_dir: ARRAY_LIKE, scan_dirs: ARRAY_LIKE,
               observer_position: Optional[ARRAY_LIKE] = None, initial_step: float = 1, max_iterations: int = 25,
               rtol: float = 1e-12, atol: float = 1e-12) -> np.ndarray:
    r"""
    This helper function determines the limb points for any traceable shape (visible edge of the shape) that would be
    visible for an observer located at ``observer_position`` looking toward ``scan_center_dir`` along the
    directions given by ``scan_dirs``.

    Typically it is assumed that the location of the observer is at the origin of the current frame and therefore
    ``observer_position`` can be left as ``None``.

    The limb for the surface is found iteratively by tracing rays from the observer to the shape.  First, the
    rays are traced along the scan center direction, which should beg guaranteed to strike the shape.  Then, we
    adjust the direction of the rays so that they no longer intersect the surface using ``initial_step``.
    We then proceed by tracing rays with directions half way between the left rays (guaranteed to strike the surface)
    and the right rays (guaranteed to not strike the surface) updating the left and right rays based on the result of
    the last trace.  This continues for a maximum of ``max_iterations`` or until the tolerances specified by ``rtol``
    and ``atol`` are met for the change in the estimate of the limb location.  The returned limb location is the last
    ray intersect location that hit the surface for each ``scan_dirs``.

    The returned limbs are expressed as vectors from the observer to the limb point in the current frame as a 3xn
    numpy array.

    :param shape: The target shape that we are to find the limb points for as a :class:`.Shape`
    :param scan_center_dir: the unit vector which the scan is to begin at in the current frame as a length 3 array
                            A ray cast along this unit vector from the ``observer_position`` should be guaranteed
                            to strike the surface and ideally should be towards the center of figure of the surface
    :param scan_dirs: the unit vectors along with the scan is to proceed as a 3xn array in the current frame where
                      each column represents a new limb point we wish to find (should be nearly orthogonal to the
                      ``scan_center_dir`` in most cases).
    :param observer_position: The location of the observer in the current frame.  If ``None`` then it is assumed
                              the observer is at the origin of the current frame
    :param initial_step: The size of the initial step to take along the ``scan_dirs`` direction.  This should be
                         guaranteed to result in rays that do not strike the body.
    :param max_iterations: The maximum number of iteration steps to take when locating the limbs
    :param rtol: The relative tolerance of the change in the limb location from one iteration to the next that
                 indicates convergence.
    :param atol: the absolute tolerance of the change int he limb location from one iteration to the next that
                 indicates convergence.
    :return: the vectors from the observer to the limbs in the current frame as a 3xn array
    """
    
    scan_dirs = np.asanyarray(scan_dirs)
    scan_center_dir = np.asanyarray(scan_center_dir)

    if observer_position is not None:
        single_start = np.array(observer_position).reshape(3, 1)
    else:
        single_start = np.zeros((3, 1), dtype=np.float64)

    start = np.broadcast_to(single_start, (3, scan_dirs.shape[1]))

    left_rays = Rays(start, np.array([scan_center_dir] * scan_dirs.shape[1]).T)

    right_rays = Rays(start, scan_center_dir.reshape(3, 1) + scan_dirs * initial_step)

    res: np.ndarray = shape.trace(left_rays)
    old_res = res

    for ind in range(max_iterations):

        trace_rays = copy.copy(left_rays)

        trace_rays.direction = left_rays.direction + (right_rays.direction - left_rays.direction) / 2

        res = shape.trace(trace_rays)

        keep_right = res["check"]

        left_rays.direction[:, keep_right] = trace_rays.direction[:, keep_right]
        right_rays.direction[:, ~keep_right] = trace_rays.direction[:, ~keep_right]

        converged = np.zeros(res.size, dtype=np.bool)

        converged[keep_right] = (np.abs(old_res[keep_right] - res[keep_right]) <=
                                 atol + rtol*np.abs(res[keep_right])).all(axis=0)
        converged[~keep_right] = (np.abs(right_rays.direction[:, ~keep_right] - left_rays.direction[:, ~keep_right]) <=
                                  atol + rtol*np.abs(left_rays.direction[:, ~keep_right])).all(axis=0)

        if converged.all():
            break

    final_rays = left_rays[~res["check"]]

    final_res = shape.trace(final_rays)

    res["intersect"][~res["check"]] = final_res["intersect"]

    return res["intersect"].T - single_start


def compute_com(tris: Union[Triangle64, Triangle32]) -> np.ndarray:
    """
    This function computes the center of mass assuming uniform density for a triangle tesselated surface.

    The center of mass is found by finding the center of volume (since with uniform density this is equivalent to the
    center of mass).  This is done by computing the volume of each tetrahedron formed by a face connected to the center
    of figure of the shape (note that therefore very irregular shapes cannot be analyzed by this function).  These
    volumes are then used in the usual center of mass equation (multiply the volumes times the volume centroids, take
    the sum, and divide by the sum of the volumes).

    :param tris: the triangular tessellated surface which we are to compute the uniform density center of mass for
    :return: the center of mass of the surface as a length 3 numpy array expressed in the current frame
    """

    # compute the volume of each pyramid

    # D = (x0-x).T@n where x0 is a vertex from the triangle (in this case vertex 0),
    # x is the point we want the surface for (in this case the origin),
    # and n is the normal vector
    sv = tris.stacked_vertices
    heights = np.abs((sv[..., 0]*tris.normals).sum(axis=-1))

    # determine the area of each triangle
    areas = np.linalg.norm(np.cross(tris.sides[..., 0], tris.sides[..., 1]).reshape(-1, 3), axis=-1)/2

    volumes = areas * heights/3

    # determine the centroid of each triangle
    tri_cents = sv.sum(axis=-1)/3

    # determine the centroid of each pyramid
    # this is 3/4 of the distance from the apex (origin) to the centeroid of the base
    pyr_cents = 3/4 * tri_cents

    # determine the centroid of the body
    com = (pyr_cents * volumes.reshape(-1, 1)).sum(axis=0)/volumes.sum()

    return com


def ref_ellipse(verts: np.ndarray) -> Ellipsoid:
    r"""
    This function finds the best fit ellipsoid to a set of vertices  (minimizing the algebraic distance residuals).

    This is done by solving the least squares equation

    .. math::

        \left[\begin{array}{ccccccccc} \mathbf{x}^2+\mathbf{y}^2-2\mathbf{z}^2 &
        \mathbf{x}^2 + \mathbf{z}^2 - 2\mathbf{y}^2 &
        2\mathbf{x}*\mathbf{y} &
        2\mathbf{x}*\mathbf{z} &
        2\mathbf{y}*\mathbf{z} &
        2\mathbf{x} &
        2\mathbf{y} &
        2\mathbf{z} &
        \mathbf{1} \end{array}\right]\left[\begin{array}{c} A\\ B \\ C \\ D \\ E \\ F \\ G \\ H \\ I\end{array}\right] =
        \mathbf{x}^2+\mathbf{y}^2+\mathbf{z}^2

    Given the solution, we then form the matrix

    .. math::

        \mathbf{A} = \left[\begin{array}{cccc} A+B+C-1 & C & D & F \\
        C & A-2B-1 & E & G \\
        D & E & B-2A-1 & H \\
        F & G & H & I \end{array}\right]

    We then have that the center of the ellipse is found by solving the 3x3 system

    .. math::

        -\mathbf{A}_{0:2,0:2}\mathbf{c} = \left[\begin{array}{ccc} F \\ G \\ H \end{array}\right]

    and the ellipsoid matrix is found according to

    .. math::

        \mathbf{T} = \left[\begin{array}{cccc} 1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        c_0 & c_1 & c_2 & 1\end{array}\right] \\
        \mathbf{R} = \mathbf{T}\mathbf{A}\mathbf{T}^T \\
        \mathbf{E} = -\frac{\mathbf{R}_{0:3,0:3}}{r_{3,3}}

    where :math:`\mathbf{E}` is the ellipsoid matrix.

    :param verts: The vertices to fit the ellipsoid to as a (n x 3) array of points
    :return: An :class:`.Ellipsoid` object that represents the best fit to the supplied vertices
    """

    x, y, z = verts.T

    x2 = x**2
    y2 = y**2
    z2 = z**2
    xy = x*y
    xz = x*z
    yz = y*z

    hmat = np.vstack([x2 + y2 - 2*z2,
                      x2 + z2 - 2*y2,
                      2*xy,
                      2*xz,
                      2*yz,
                      2*x,
                      2*y,
                      2*z,
                      np.ones(x.shape)]).T

    rhs = x2 + y2 + z2

    solu = np.linalg.lstsq(hmat, rhs, rcond=None)[0]

    coefs = np.zeros(10)
    coefs[0] = solu[:2].sum() - 1
    coefs[1] = solu[0] - 2 * solu[1] - 1
    coefs[2] = solu[1] - 2 * solu[0] - 1
    coefs[3:] = solu[2:]

    amat = np.array([[coefs[0], coefs[3], coefs[4], coefs[6]],
                     [coefs[3], coefs[1], coefs[5], coefs[7]],
                     [coefs[4], coefs[5], coefs[2], coefs[8]],
                     [coefs[6], coefs[7], coefs[8], coefs[9]]])

    center = -np.linalg.solve(amat[:3, :3], coefs[6:9])

    tmat = np.eye(4)
    tmat[-1, :3] = center

    rmat = tmat@amat@tmat.T

    ellipsmat = -rmat[:3, :3] / rmat[-1, -1]

    return Ellipsoid(center=center, ellipsoid_matrix=ellipsmat)


def to_block(vals: Sequence[Sequence[int] | NDArray[np.integer] | int] | NDArray[np.integer]) -> NDArray[np.integer]:
    """
    This helper function takes a list of lists/arrays and puts them all into a single contiguous array

    This is used when ray tracing rays that have ignore lists of different lengths.  It works by iterating through,
    determining the maximum length of the "rows", creating a new array of -1 with this many columns, and then assigning
    each row to this new array.

    For instance, an input of the form

    .. code::

        [[1, 2, 3], [0, 1], [7, 8, 3, 4]]

    will be converted to

    .. code::

        np.array([[1, 2, 3, -1], [0, 1, -1, -1], [7, 8, 3, 4]])

    :param vals: A ragged list of lists or arrays that is to be converted into a single contiguous array
    :return: the contiguous block
    """

    # check to see if we already have a contiguous array
    if isinstance(vals, np.ndarray):

        if vals.ndim == 2:

            return vals.astype(np.int64)

        else:
            return vals.reshape(-1, 1).astype(np.int64)

    # determine how many rows there are
    n_rows = len(vals)

    # make each row a 1d numpy array
    vals_one_d = list(map(np.atleast_1d, vals))

    # determine the maximum number of columns in any row
    n_cols = max(row.size for row in vals_one_d)

    # create an output array that is n_rows x n_cols
    out = -np.ones((n_rows, n_cols), dtype=np.int64)

    # copy each row into the appropriate columns in the output array
    for ind, row in enumerate(vals_one_d):
        out[ind, :row.size] = row

    return out


def compute_stats(tris: Union[Triangle64, Triangle32], mass: float) -> Tuple[np.ndarray, float, float, np.ndarray,
                                                                            np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute statistics on a shape tessellated using Triangles.

    The statistics computed include the center of mass, total volume, surface area, Inertia matrix, center of mass
    relative inertia matrix, moments of inertia, and rotation matrix to the inertia frame.  These are only valid for
    mostly-regular bodies, where the tetrahedrons formed by connecting each face to the center of figure of the shape
    is contained entirely within the shape.

    :param tris: The GIANT triangle objects to compute the statistics on
    :param mass: The mass of the object in kg
    :return: A tuple of the statistics in the order mentioned above
    """

    # compute the volume of each pyramid

    # D = (x0-x).T@n where x0 is a vertex from the triangle (in this case vertex 0),
    # x is the point we want the surface for (in this case the origin),
    # and n is the normal vector
    sv = tris.stacked_vertices
    heights = np.abs((sv[..., 0] * tris.normals).sum(axis=-1))

    # determine the area of each triangle
    areas = np.linalg.norm(np.cross(tris.sides[..., 0], tris.sides[..., 1]).reshape(-1, 3), axis=-1) / 2

    volumes = areas * heights / 3

    # determine the centroid of each triangle
    tri_cents = sv.sum(axis=-1) / 3

    # determine the centroid of each pyramid
    # this is 3/4 of the distance from the apex (origin) to the centeroid of the base
    pyr_cents = 3 / 4 * tri_cents

    # determine the centroid of the body
    volume = volumes.sum()
    com = (pyr_cents * volumes.reshape(-1, 1)).sum(axis=0) / volume

    # determine the density for the object
    density = mass / volume

    # determine the product of inertia for each facet
    vert0 = sv[..., 0].T
    vert1 = sv[..., 1].T
    vert2 = sv[..., 2].T

    # compute the outer products between the sides
    op00 = np.einsum('ij,jk->jik', vert0, vert0.T)
    op11 = np.einsum('ij,jk->jik', vert1, vert1.T)
    op22 = np.einsum('ij,jk->jik', vert2, vert2.T)
    op01 = np.einsum('ij,jk->jik', vert0, vert1.T)
    op10 = op01.swapaxes(-1, -2)
    op02 = np.einsum('ij,jk->jik', vert0, vert2.T)
    op20 = op02.swapaxes(-1, -2)
    op12 = np.einsum('ij,jk->jik', vert1, vert2.T)
    op21 = op12.swapaxes(-1, -2)

    i_product = (volumes.reshape(-1, 1, 1) * (2 * op00 + 2 * op11 + 2 * op22 + op01 +
                                              op10 + op02 + op20 + op12 + op21)).sum(axis=0) * density / 20

    # compute the moments of inertia
    ixx = i_product[[1, 2], [1, 2]].sum()
    iyy = i_product[[0, 2], [0, 2]].sum()
    izz = i_product[[0, 1], [0, 1]].sum()
    ixy = iyx = -i_product[0, 1]
    ixz = izx = -i_product[0, 2]
    iyz = izy = -i_product[1, 2]

    # form the inertia matrix
    i_matrix = np.array([[ixx, ixy, ixz], [iyx, iyy, iyz], [izx, izy, izz]])

    # determine the inertia matrix with respect to the center of mass
    i_matrix_com = i_matrix - mass * (np.diag([(com ** 2).sum()] * 3) - np.outer(com, com))

    # determine the moments and rotation matrix
    moments, rotation_matrix = np.linalg.eigh(i_matrix_com)

    return com, volume, areas.sum(), i_matrix, i_matrix_com, moments, rotation_matrix

def correct_stellar_aberration_fsp(camera_to_target_position_inertial: np.ndarray,
                                   camera_velocity_inertial: np.ndarray) -> np.ndarray:
    """
    Correct for stellar aberration using linear addition.

    Note that this only roughly corrects for the direction, it messes up the distance to the object, therefore you
    should favor the :func:`.correct_stellar_aberration` function which uses rotations and thus doesn't mess with the
    distance.

    Note that this assumes that the units for the input are all in kilometers and kilometers per secon.  If they are not
    you will get unexpected results.

    :param camera_to_target_position_inertial: The vector from the camera to the target in the inertial frame
    :param camera_velocity_inertial: The velocity of the camera in the inertial frame relative to the SSB
    :return: the vector from the camera to the target in the inertial frame corrected for stellar aberration
    """
    # this is only good for adjusting the unit vector. Don't use if you need anything involving range

    return (camera_to_target_position_inertial + np.linalg.norm(camera_to_target_position_inertial, axis=0) *
            camera_velocity_inertial / SPEED_OF_LIGHT)


def correct_stellar_aberration(camera_to_target_position_inertial: np.ndarray,
                               camera_velocity_inertial: np.ndarray) -> np.ndarray:
    """
    Correct for stellar aberration using rotations.

    This works by computing the rotation about the aberation axis and then applying this rotation to the vector from the
    camera to the target in the inertial frame.  This is accurate and doesn't mess up the distance to the target.  It
    should therefore always be preferred to :func:`.correct_stellar_aberration_fsp`

    Note that this assumes that the units for the input are all in kilometers and kilometers per secon.  If they are not
    you will get unexpected results.

    :param camera_to_target_position_inertial: The vector from the camera to the target in the inertial frame
    :param camera_velocity_inertial: The velocity of the camera in the inertial frame relative to the SSB
    :return: the vector from the camera to the target in the inertial frame corrected for stellar aberration
    """
    velocity_mag = np.linalg.norm(camera_velocity_inertial)

    if velocity_mag != 0:

        aberration_axis = np.cross(camera_to_target_position_inertial, camera_velocity_inertial / velocity_mag, axis=0)

        aberration_axis_magnitude = np.linalg.norm(aberration_axis, axis=0, keepdims=True)

        velocity_sin_angle = (aberration_axis_magnitude /
                              (np.linalg.norm(camera_to_target_position_inertial, axis=0, keepdims=True)))

        aberration_angle = np.arcsin(velocity_mag * velocity_sin_angle / SPEED_OF_LIGHT)

        aberration_axis /= aberration_axis_magnitude

        if (np.ndim(camera_to_target_position_inertial) > 1) and (np.shape(camera_to_target_position_inertial)[-1] > 1):

            return np.matmul(rotvec_to_rotmat(aberration_axis * aberration_angle),
                             camera_to_target_position_inertial.T.reshape(-1, 3, 1)).squeeze().T

        else:
            return np.matmul(rotvec_to_rotmat(aberration_axis * aberration_angle),
                             camera_to_target_position_inertial)

    else:
        return camera_to_target_position_inertial
    
    
def correct_light_time(target_location_inertial: Callable[[datetime.datetime], np.ndarray],
                       camera_location_inertial: np.ndarray,
                       time: datetime.datetime) -> np.ndarray:
    """
    Correct an inertial position to include the time of flight for light to travel between the target and the camera.

    This function iteratively calculates the time of flight of light between a target and a camera and then returns
    the relative vector between the camera and the target accounting for light time (the apparent relative vector) in
    inertial space.  This is done by passing a callable object for target location which accepts a python datetime
    object and returns the inertial location of the target at that time (this is usually a function wrapped around a
    call to spice).

    Note that this assumes that the units for the input are all in kilometers.  If they are not you will get unexpected
    results.

    :param target_location_inertial: A callable object which inputs a python datetime object and outputs the inertial
                                    location of the target at the given time
    :param camera_location_inertial: The location of the camera in inertial space at the time the image was captured
    :param time: The time the image was captured
    :return: The apparent vector from the target to the camera in inertial space
    """

    time_of_flight = 0

    camera_location_inertial = np.asarray(camera_location_inertial).ravel()

    for _ in range(10):

        target_location_reflect = np.asarray(
            target_location_inertial(time - datetime.timedelta(seconds=time_of_flight))
        ).ravel()

        time_of_flight_new = float(np.linalg.norm(camera_location_inertial - target_location_reflect) / SPEED_OF_LIGHT)

        if (time_of_flight_new - time_of_flight) < 1e-8:
            time_of_flight = time_of_flight_new
            break

        time_of_flight = time_of_flight_new

    return (target_location_inertial(time - datetime.timedelta(seconds=time_of_flight)).ravel() -
            camera_location_inertial)
