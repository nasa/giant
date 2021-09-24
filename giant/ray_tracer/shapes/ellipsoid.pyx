# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This cython module defines the :class:`.Ellipsoid` used for representing regular (well modelled by a tri-axial
ellipsoid) in GIANT.

Description
-----------

The :class:`.Ellipsoid` class is a fully functional and traceable object in GIANT, therefore it can be used in any of
the :mod:`.relative_opnav` techniques (with the exception of :mod:`.sfn`).  It represents the
object as a 3x3 matrix encoding of a triaxial ellipsoid (plus a 3 element center vector) and is very efficient for both
tracing and for finding limbs.  It is also well suited for use with the :mod:`.ellipse_matching` technique since that
technique assumes an ellipsoid.  Therefore, anytime your body is well represented by a triaxial ellipsoid you should
consider using this class.

For more details on how this class works, refer to the class documentation as follows.  Additionally, you can find great
information in the following paper and its references: https://ieeexplore.ieee.org/abstract/document/9326288 as well as
https://trs.jpl.nasa.gov/bitstream/handle/2014/41942/11-0589.pdf?sequence=1
both of which were used heavily in developing this class.
"""

from itertools import repeat

from numbers import Real, Complex

import numpy as np
import cython
from cython.parallel import prange, parallel

from scipy.linalg.cython_lapack cimport dgelsd, dgesv
from libc.math cimport sqrt, isnan

from giant.rotations import Rotation
from giant.ray_tracer.rays import INTERSECT_DTYPE
from giant.catalogues.utilities import unit_to_radec

from giant.ray_tracer.shapes.solid cimport Solid
from giant.ray_tracer.shapes.axis_aligned_bounding_box import AxisAlignedBoundingBox

from giant._typing import SCALAR_OR_ARRAY, ARRAY_LIKE


def quadratic_equation(a, b, c):
    r"""
    quadratic_equation(a, b, c)

    This helper function finds the routes of a quadratic equation with coefficients a, b, c

    The quadratic equation is given by

    .. math::

        \frac{-b\pm\sqrt{b^2-4ac}}{2a}

    for an equation of the form

    .. math::

        ax^2+bx+c=0

    This function is vectorized so it can accept either scalars or arrays as long as they are broadcastable.

    Note that this function will return Nan for places where the discriminant is negative unless the input is already
    complex.

    :param a: the a coefficient(s) of the equation(s)
    :type a: SCALAR_OR_ARRAY
    :param b: the b coefficient(s) of the equation(s)
    :type b: SCALAR_OR_ARRAY
    :param c: the c coefficient(s) of the equation(s)
    :type c: SCALAR_OR_ARRAY
    :return: the positive and negative roots of the equation as a tuple of floats or arrays.  The negative root is first
    :rtype: Union[Tuple[Union[Real, Complex], Union[Real, Complex]], Tuple[np.ndarray, np.ndarray]]
    """
    discriminant = np.sqrt(b*b - 4 * a * c)
    return (-b - discriminant) / (2 * a), (-b + discriminant) / (2 * a)


cdef class Ellipsoid(Solid):
    r"""
    __init__(self, center, principal_axes=None, orientation=None, ellipsoid_matrix=None, albedo_map=None, _bounding_box=None, _id=None)

    A shape for modelling spheres and triaxial ellipsoidal bodies.

    Most of the large objects in the solar system are generally ellipsoidal in shape.  Therefore, when rendering these
    objects it can be much more efficient to actually model them as a triaxial ellipsoid, which can generally be traced
    more efficiently than even a KDTree due to having only 1 surface to check.  Therefore, when you're dealing with a
    shape well modeled by a triaxial ellipsoid it is recommended that you use this class.

    This class works like other :class:`.Shape` classes in GIANT.  It provides :meth:`trace` and
    :meth:`compute_intersect` for calculating the intersect between the object and a ray. It provides methods
    :meth:`find_limbs` and :meth:`compute_limb_jacobian` to identify points along the limb of the target and how those
    points change when the location of the observer is changed.  In addition, it provides methods :meth:`rotate` and
    :meth:`translate` methods for moving the ellipsoid in the scene and an axis aligned bounding box in attribute
    :attr:`bounding_box`.

    In GIANT the triaxial ellipsoid is represented in matrix form, that is:

    .. math::

        \mathbf{A}_C = \mathbf{T}^{P}_{C}\left[\begin{array}{ccc} \frac{1}{a^2} & 0 & 0 \\
        0 & \frac{1}{b^2} & 0 \\
        0 & 0 & \frac{1}{c^2} \end{array}\right]\mathbf{T}_{P}^{C}

    Where :math:`\mathbf{A}_C` is the ellipsoid matrix in the current frame :math:`C`, :math:`\mathbf{T}^P_C` is the
    rotation matrix from the principal frame of the ellipsoid (:math:`P`, where the principal axes are aligned with the
    frame axes) to the current frame, and :math:`a\quad b\quad c` are the lengths of the principal axes of the
    ellipsoid.  From this class you can retrieve each component of the ellipsoid matrix.  :attr:`.ellipsoid_matrix`
    gives :math:`\mathbf{A}_C`, :attr:`.principal_axes` gives :math:`a\quad b\quad c\quad` as a length 3 array, and
    :attr:`orientation` gives :math:`\mathbf{T}_C^P` as a 3x3 rotation matrix.  In addition, the center of the ellipsoid
    in the current frame is in attribute :attr:`.center`.

    In addition to being its own object in a scene, the :class:`Ellipsoid` class is also used to represent the
    :attr:`.Surface.reference_ellipsoid` for a surface (used when doing limb based nav with irregular bodies in
    :mod:`.limb_matching` as well as the :attr:`.SceneObject.circumscribing_sphere` attribute of a scene object which
    is used to determine which pixels need traced for a specific object in the scene.

    To initialize this class, you can either provide the ellipsoid matrix itself (which will be decomposed into its
    components), the principal axes themselves (in which case it will be assumed the object is currently in its
    principle frame), the principal axes and the orientation, or all 3 (though note that if its all 3 we don't check
    to be sure they're consistent).  Also, note that any parameters beginning with an _ in the init method are primarily
    for pickling/unpickling the object and can mostly be ignored by the user.
    """

    def __init__(self, center=None, principal_axes=None, orientation=None, ellipsoid_matrix=None, albedo_map=None,
                 _bounding_box=None, _id=None):
        """
        :param center: The center of the ellipsoid in the current frame as a length 3 array
        :type center: Optional[ARRAY_LIKE]
        :param principal_axes: The length of each principal axis of the ellipsoid as a length 3 array
        :type principal_axes: Optional[ARRAY_LIKE]
        :param orientation: The rotation matrix from the principal frame for the ellipsoid to the current frame.
        :type orientation: Optional[ARRAY_LIKE]
        :param ellipsoid_matrix: The full 3x3 ellipsoid matrix as defined above
        :type ellipsoid_matrix: Optional[ARRAY_LIKE]
        :param albedo_map: The albedo map for the ellipsoid.  This should take the form of a callable object which takes
                           in an nx2 array of latitude/longitude in radians and returns a length n array of the albedo
                           values for each point.
        :type albedo_map: Optional[Callable[[np.ndarray], np.ndarray]]
        :param _bounding_box: The AxisAlignedBoundingBox for the ellipsoid.  This is typically just used for
                              pickling/unpickling
        :type _bounding_box: Optional[ARRAY_LIKE]
        :param _id: The unique id for the ellipsoid object.  This is typically only used for pickling/unpickling.
        :type _id: int
        """

        # initialize all of the arrays
        self._center = np.zeros(3, dtype=np.float64)
        self._principal_axes = np.zeros(3, dtype=np.float64)
        self._orientation = np.zeros((3, 3), dtype=np.float64)
        self._ellipsoid_matrix = np.zeros((3, 3), dtype=np.float64)

        # the albedo map
        self.albedo_map = albedo_map

        # the center is the center of the ellipsoid
        self.center = center

        # the principal axes are the 3 radii which define the triaxial ellipsoid. Typically these are ordered from
        # largest to smallest
        if principal_axes is not None:
            self._principal_axes = np.array(principal_axes, dtype=np.float64).ravel()

        # the orientation represents how the ellipsoid principal frame is aligned with the world frame
        if orientation is not None:
            self._orientation = np.array(orientation, dtype=np.float64).reshape(3, 3)

        # the ellipsoid matrix is orientation @ diag(1/principal_axes**2) @ orientation.T
        if ellipsoid_matrix is not None:
            self._ellipsoid_matrix = np.array(ellipsoid_matrix, dtype=np.float64).reshape(3, 3)

        # the axis aligned bounding box
        self.bounding_box = None

        # if the id has not been specified then just use the python id function to make an id
        if _id is not None:
            self.id = _id
        else:
            self.id = id(self)

        # store the bounding box
        if _bounding_box is not None:
            self.bounding_box = _bounding_box

        # check to see which information was given and fill out anything that is missing
        if (principal_axes is None) and (ellipsoid_matrix is not None):
            # if we were given the full ellipsoid matrix, compute the principal axes and orientation matrix
            # by finding the eignvalues and eigenvectors of the ellipsoid matrix

            self._principal_axes, self._orientation = np.linalg.eigh(ellipsoid_matrix)

            self._principal_axes = 1 / np.sqrt(np.abs(self.principal_axes)).ravel()

        elif (principal_axes is not None) and (ellipsoid_matrix is None):

            # if we were given the principal axes, form the ellipsoid matrix

            if orientation is not None:

                # if the orientation was provided then do the T@Ap@T.T multiplication
                self._ellipsoid_matrix = np.matmul(self.orientation,
                                                   np.matmul(np.diag(np.power(principal_axes.astype(np.float64),
                                                                              -2)), self.orientation.T))

            else:
                # otherwise assume the orientation is aligned with the world frame
                self._ellipsoid_matrix = np.diag(np.power(np.asarray(principal_axes).astype(np.float64), -2))

                self._orientation = np.eye(3).astype(np.float64)

        elif (principal_axes is None) and (ellipsoid_matrix is None):
            # if they gave us nothing raise an error

            raise ValueError('Either the principal_axes or the ellipsoid_matrix need to be specified.')

        # compute the AABB
        self.compute_bounding_box()

    def __reduce__(self):
        """
        This method is used to pickle/unpickle the object
        """

        return self.__class__, (self.center, self.principal_axes, self.orientation, self.ellipsoid_matrix,
                                self.albedo_map, self.bounding_box, self.id)

    @property
    def center(self):
        """
        The location of the center of th ellipsoid in the current frame as a length 3 numpy array.
        """

        return np.asarray(self._center)

    @center.setter
    def center(self, val):

        if val is not None:
            self._center = np.asarray(val).ravel().astype(np.float64)

    @property
    def principal_axes(self):
        """
        The lengths of the principal axes of the ellipsoid as a length 3 numpy array
        """
        return np.asarray(self._principal_axes)

    @property
    def orientation(self):
        """
        The rotation matrix from the principal axis frame to the current frame for the ellipsoid as a 3x3 numpy array.
        """
        return np.asarray(self._orientation)

    @property
    def ellipsoid_matrix(self):
        r"""
        The ellipsoid matrix in the current frame for the ellipsoid as a 3x3 numpy array.

        Mathematically this is given by:

        .. math::

            \mathbf{A}_C = \mathbf{T}^{P}_{C}\left[\begin{array}{ccc} \frac{1}{a^2} & 0 & 0 \\
            0 & \frac{1}{b^2} & 0 \\
            0 & 0 & \frac{1}{c^2} \end{array}\right]\mathbf{T}_{P}^{C}

        Where :math:`\mathbf{A}_C` is the ellipsoid matrix in the current frame :math:`C`, :math:`\mathbf{T}^P_C` is the
        rotation matrix from the principal frame of the ellipsoid (:math:`P`, where the principal axes are aligned with the
        frame axes) to the current frame, and :math:`a\quad b\quad c` are the lengths of the principal axes of the
        ellipsoid.
        """

        return np.asarray(self._ellipsoid_matrix)

    def compute_bounding_box(self):
        """
        compute_bounding_box(self)

        This method computes the AABB for the ellipsoid.

        The AABB is defined according to the positive and negative locations of the principal axes of the ellipsoid
        in the current frame (including the center offset).  The results are stored in the :attr:`bounding_box`
        attribute.
        """

        # get the positive and negative locations along the principal axes
        pos_bounds = self.orientation * np.diag(self.principal_axes) + self.center.reshape(3, 1)
        neg_bounds = -self.orientation * np.diag(self.principal_axes) + self.center.reshape(3, 1)

        # get the minimum and maximum bounds
        min_bounds = np.minimum(pos_bounds, neg_bounds).min(axis=-1)
        max_bounds = np.maximum(pos_bounds, neg_bounds).max(axis=-1)

        # form the AABB
        self.bounding_box = AxisAlignedBoundingBox(min_bounds, max_bounds)

    def intersect(self, rays):
        r"""
        intersect(self, rays)

        This method determines where the provides rays strike this tri-axial ellipsoid.

        The results are returned as a nx3 numpy array of floats with each row corresponding to the nearest intersect
        location between the corresponding ray and the ellipsoid.  Anywhere that the rays did not strike the ellipsoid
        the corresponding row is set to all NaN.

        The intersects are determined by solving the following quadratic equation for :math:`d`:

        .. math::

            ad^2+bd+c=0

        where

        .. math::

            a = \mathbf{d}^T\mathbf{A}_C\mathbf{d} \\
            b = 2\mathbf{d}^T\mathbf{A}_C\left(\mathbf{s}-\mathbf{c}\right) \\
            c = \left(\mathbf{s}-\mathbf{c}\right)^T\mathbf{A}_C\left(\mathbf{s}-\mathbf{c}\right)

        :math:`\mathbf{d}` is the direction of the ray, :math:`\mathbf{s}` is the start location for the ray, and
        :math:`\mathbf{c}` is the center of the ellipsoid in the current frame.  Solving this quadratic equation will
        give 2 distances.  If both are imaginary (or both are negative) then no intersect occurs.  Otherwise the
        smallest (positive) distance is taken as the closest intersect.

        Only the intersection point is returned for each ray.  To get all of the information (intersection point,
        surface normal, albedo, etc) see the :meth:`trace` or :meth:`compute_intersect` methods.

        :param rays: The rays to trace through the array
        :type rays: Rays
        :return: A nx3 array (where n is the number of rays) with the closest intersect between the corresponding ray
                 and the ellipsoid or NaN if no intersection occurs.
        :type: np.ndarray
        """

        # TODO: put this into C code so that it can be parallelized

        if rays.num_rays > 1:
            # if we have more than 1 ray reshape the center
            center = self.center.reshape(3, 1)

        else:
            center = self.center

        # multiply the direction vectors for the rays with the full blown ellipsoid matrix
        los_ellipsoid = np.matmul(rays.direction.T, self.ellipsoid_matrix)

        # compute the coefficients of the quadratic equation to check for intersection
        # a coefficient is d.T@A@d
        a_coef = (los_ellipsoid * rays.direction.T).sum(axis=-1)
        # b coefficient is 2*d.T@A@(s-c)
        b_coef = 2 * (los_ellipsoid * (rays.start-center).T).sum(axis=-1)
        # c coefficient is (s-c).T@A@(s-c)
        c_coef = ((rays.start-center).T * np.matmul(self.ellipsoid_matrix, rays.start-center).T).sum(axis=-1) - 1

        # compute the two possible intersections as the 2 roots of the quadratic equation
        dist_a, dist_b = quadratic_equation(a_coef, b_coef, c_coef)

        if rays.num_rays > 1:

            # initialize the output array
            result = np.empty((rays.num_rays, 3), dtype=np.float64)

            # find negative distances and nan distances
            test_neg_dist = (dist_a < 0) & (dist_b < 0)

            test_nan_dist = np.isnan(dist_a) & np.isnan(dist_b)

            # check for any rays that want to ignore this body
            if rays.ignore is not None:
                test_ignore = rays.ignore.ravel() == self.id
            else:
                test_ignore = np.zeros(rays.num_rays, dtype=bool)

            # set the result to nan where we didn't intersect
            result[test_nan_dist | test_neg_dist] = np.nan

            # check where the negative root is the right one
            dist_b_check = (dist_a < 0) & (dist_b >= 0)

            # compute the intersection point for these rays and store it
            result[dist_b_check] = (rays.start[:, dist_b_check] +
                                    dist_b[dist_b_check]*rays.direction[:, dist_b_check]).T

            # check where the positive root is the right one
            dist_a_check = (dist_b < 0) & (dist_a >= 0)

            result[dist_a_check] = (rays.start[:, dist_a_check] +
                                    dist_a[dist_a_check]*rays.direction[:, dist_a_check]).T

            # anywhere that isn't filled strikes the surface twice, only return the closer one
            unfilled_check = ~(dist_a_check | dist_b_check | test_neg_dist | test_nan_dist | test_ignore)

            result[unfilled_check] = (rays.start[:, unfilled_check] +
                                      np.minimum(dist_a[unfilled_check],
                                                 dist_b[unfilled_check])*rays.direction[:, unfilled_check]).T

            result[test_ignore] = np.nan

            return result
        else:

            if (dist_a < 0) and (dist_b < 0):
                # if the object is behind the ray return none
                return None

            elif np.isnan(dist_a) and np.isnan(dist_b):
                # if the object isn't intersected by the ray return none
                return None

            elif (dist_a < 0) and (dist_b >= 0):
                # if the ray only strikes the surface once choose the right one
                distance = dist_b

            elif (dist_a >= 0) and (dist_b < 0):
                # if the ray only strikes the surface once choose the right one
                distance = dist_a

            else:
                # otherwise, the object intersects the surface twice so choose the closer intersection
                distance = min(dist_a, dist_b)

            # return the intersection
            return rays.start + rays.direction * distance

    def compute_intersect(self, ray):
        """
        compute_intersect(self, ray)

        This method computes the intersect between a single ray and the ellipsoid, returning an :data:`.INTERSECT_DTYPE`
        numpy array with the intersect location, surface normal at the intersect, and surface albedo at the intersect.

        This method calls the :meth:`intersect` method first, and then computes the local normal vector and albedo
        at the intersection point, returning the results in the proper structured array format.

        While this method can handle multiple rays, it is better to reserve this for single ray checks for consistency
        with :class:`.Surface` objects.

        :param ray: The ray to perform the intersect check with
        :type ray: Rays
        :return: A numpy array with :data:`.INTERSECT_DTYPE` as the data type.
        :rtype: np.ndarray
        """

        intersects = self.intersect(ray)

        if intersects is not None:

            if intersects.ndim == 1:
                intersects = intersects.reshape(1, 3)

            body_centered_vecs = intersects - self.center.reshape(1, 3)

            normals = self.compute_normals(body_centered_vecs.T).T

            albedos = self.compute_albedos((self.orientation.T@body_centered_vecs.T))

            return np.array(list(zip(~(np.isnan(intersects).any(axis=1)), repeat(np.inf), intersects, normals, albedos,
                                     repeat(self.id))),
                            dtype=INTERSECT_DTYPE).squeeze()
        else:

            return np.array((False, np.inf, None, None, None, -1), dtype=INTERSECT_DTYPE)

    def trace(self, rays):
        """
        This method computes the intersect between rays and the ellipsoid, returning an :data:`.INTERSECT_DTYPE`
        numpy array with the intersect locations, surface normals at the intersects, and surface albedos at the
        intersects.

        This method calls the :meth:`intersect` method first, and then computes the local normal vector and albedo
        at the intersection points, returning the results in the proper structured array format.

        :param rays: The rays to perform the intersect check with
        :type rays: Rays
        :return: A numpy array with :data:`.INTERSECT_DTYPE` as the data type.
        :rtype: np.ndarray
        """

        return self.compute_intersect(rays)

    def compute_normals(self, locs):
        r"""
        compute_normals(self, locs)

        This method computes the local surface normal for a location on the surface of the ellipsoid.

        The input should be the body centered vectors in the current frame (that is the vector from the center of the
        body to a point on the surface).

        This method does not check that the point is on the surface, so if you provide a point that is not on the
        surface of the ellipsoid you will get undefined results.

        The surface normal vector for any point on the surface of a triaxial ellipsoid is defined as:

        .. math::

            \mathbf{n} = \mathbf{A}_C\mathbf{x}_C

        where :math:`\mathbf{x}_C` are the centered points in the current frame.  These are then converted to unit
        normal vectors.

        Typically a user won't directly use this method and instead will use the methods :meth:`compute_intersect` or
        :meth:`trace` to trace rays and get the normal vector at the intersect location.

        :param locs: The centered surface locations as a 3xn array, where each column corresponds to a surface location
        :type locs: np.ndarray
        :return: The unit normal vectors as a 3xn array
        :rtype: np.ndarray
        """
        # the normal vector is simply the product of A@locs
        normals = np.matmul(self.ellipsoid_matrix, locs)
        # make sure we have unit normal vectors
        normals /= np.linalg.norm(normals, axis=0)

        return normals

    def compute_albedos(self, body_centered_vecs):
        """
        compute_albedos(self, body_centered_vecs)

        This method computes the surface albedo for a location on the surface of the ellipsoid expressed in the
        principal frame of the ellipsoid.

        The input should be the body centered vectors in the principal frame (that is the vector from the center of the
        body to a point on the surface).

        This method does not check that the point is on the surface, so if you provide a point that is not on the
        surface of the ellipsoid you will get undefined results.

        The albedo is either returned as 1 (if no albedo map is attached to this ellipsoid) or is computed using the
        :attr:`albedo_map` attribute after converting the centered surface locations into longitude, latitude values in
        radians (longitude going from 0 to 2*np.pi).  Either way the result is returned as a numpy array of length n

        :param body_centered_vecs: a 3xn array of centered surface locations where each column represents a new location
        :type body_centered_vecs: np.ndarray
        :return: The surface albedo at each surface location provided as a lenght n numpy array
        :rtype: np.ndarray
        """

        if self.albedo_map is None:

            mult = body_centered_vecs.shape[1] if body_centered_vecs.ndim == 2 else 1

            return np.ones(mult)
        else:
            radec = np.array(unit_to_radec(np.array(body_centered_vecs)/
                                           np.linalg.norm(body_centered_vecs, axis=0, keepdims=True)))

            return np.array(self.albedo_map(radec.T))

    def rotate(self, rotation):
        r"""
        rotate(self, rotation)

        This method rotates the ellipsoid into a new frame.

        The rotation is applied both to the ellipsoid matrix

        .. math::

            \mathbf{A}_N = \mathbf{T}^C_N\mathbf{A}_C\mathbf{T}_C^N

        to the axis aligned bounding box, and to the center of the ellipsoid.

        :param rotation: The rotation by which to rotate the ellipsoid.
        :type rotation: Union[Rotation, ARRAY_LIKE]
        """

        # need to rotate the orientation, ellipsoid matrix, and center vector
        if isinstance(rotation, Rotation):
            self._orientation = np.matmul(rotation.matrix, self.orientation)
            self._ellipsoid_matrix = np.matmul(rotation.matrix, np.matmul(self.ellipsoid_matrix, rotation.matrix.T))
            self.center = rotation.matrix @ self.center

        else:
            rot = Rotation(rotation).matrix
            self._orientation = rot @ self.orientation
            self._ellipsoid_matrix = rot @ self.ellipsoid_matrix @ rot.T
            self.center = rot @ self.center

        # also need to rotate the AABB bounding box
        self.bounding_box.rotate(rotation)

    def translate(self, translation):
        """
        translate(self, translation)

        This method translates the ellipsoid center.

        The translation is applied to both the center of the ellipsoid and the axis aligned bounding box for the
        ellipsoid.

        :param translation: The length 3 translation
        :type translation: np.ndarray
        """

        # only the center and bounding box need to be translated
        self.center += np.asarray(translation).ravel()

        self.bounding_box.translate(translation)

    def find_limbs(self, scan_center_dir, scan_dirs, observer_position=None):
        r"""
        find_limbs(self, scan_center_dir, scan_dirs, observer_position=None)

        The method determines the limb points (visible edge of the ellipsoid) that would be an observed for an observer
        located at ``observer_position`` looking toward ``scan_center_dir`` along the directions given by ``scan_dirs``.

        Typically it is assumed that the location of the observer is at the origin of the current frame and therefore
        ``observer_position`` can be left as ``None``.

        The limb for the ellipsoid is found by first solving for the particular solution to the underdetermined system
        of equations

        .. math::

            \left[\begin{array}{c} \mathbf{s}_c^T\mathbf{A}_C \\
            \left(\mathbf{s}_c\times\mathbf{s}_d\right)^T\end{array}\right]\mathbf{p}_0 = \left[\begin{array}{c} -1 \\
            -\left(\mathbf{s}_c\times\mathbf{s}_d\right)^T\mathbf{r} \end{array}\right]

        where :math:`\mathbf{s}_c` is ``scan_center_dir``, :math:`\mathbf{s}_d` is ``scan_dirs``, and :math:`\mathbf{r}`
        is the vector from the observer to the center of the ellipsoid.  Once :math:`\mathbf{p}_0` is solved for, the
        limb can be found by solving the quadratic equation

        .. math::

            ad^2+bd+c=0

        where

        .. math::

            a = \mathbf{p}_h^T\mathbf{A}_C\mathbf{p}_h \\
            b = 2\mathbf{p}_0^T\mathbf{A}_C\mathbf{p}_h \\
            c = \mathbf{p}_0^T\mathbf{A}_C\mathbf{p}_0 - 1

        and :math:`\mathbf{p}_h=\mathbf{A}_C\mathbf{r}\times\left(\mathbf{s}_c\times\mathbf{s}_d\right)`.

        Given the 2 solutions to the quadratic equation, the 2 possible limb points are found by
        :math:`\mathbf{p}_0+d\mathbf{p}_h`.  The appropriate solution to use is the one which produces a positive dot
        product with the ``scan_dirs``.

        The returned limbs are expressed as vectors from the observer to the limb point in the current frame.

        :param scan_center_dir: the unit vector which the scan is to begin at in the current frame as a length 3 array
        :type scan_center_dir: np.ndarray
        :param scan_dirs: the unit vectors along with the scan is to proceed as a 3xn array in the current frame where
                          each column represents a new limb point we wish to find (should be nearly orthogonal to the
                          ``scan_center_dir`` in most cases).
        :type scan_dirs: np.ndarray
        :param observer_position: The location of the observer in the current frame.  If ``None`` then it is assumed
                                  the observer is at the origin of the current frame
        :type observer_position: Optional[np.ndarray]
        :return: the vectors from the observer to the limbs in the current frame as a 3xn array
        :rtype: np.ndarray
        """

        if observer_position is None:
            observer_position = np.zeros(3, dtype=np.float64)

        # declarations for increased speed
        cdef double[:] observer_to_ellipsoid = self.center - observer_position

        cdef int ind
        cdef int i, j

        cdef double[:, :] coefs = np.zeros((2, 3), dtype=np.float64, order='F')
        cdef double[3] rhs
        cdef double[3] line_dir

        cdef double[:, :] limbs = np.zeros((scan_dirs.shape[-1], 3), dtype=np.float64)

        cdef double a_coef, b_coef, c_coef, root, discriminant, rcond, cynan=np.nan

        cdef int  m, n, nrhs, lda, ldb, rank, lwork, info

        cdef double[1000] work
        cdef int[50] iwork
        cdef double[2] s
        cdef bint breakout

        # compute the cross product between the scan center directions and the scan direction vectors
        cdef double[:, :] crosses = np.cross(scan_center_dir.reshape(1, 3), scan_dirs.T).astype(np.float64)
        # set the value for the first element of the rhs to be -1
        rhs[0] = -1

        # initialize values for the least squares solution
        rcond = 7e-16
        m = 2
        n = 3
        nrhs = 1
        lda = m
        ldb = n
        lwork = 1000

        # TODO: figure out what to do when exactly nadir

        # TODO: consider parallelizing -- I think this would require breaking this into a sub function so that it has
        #  its own working arrays
        # for each scan_direction
        for ind in range(scan_dirs.shape[-1]):

            rhs[1] = 0

            # form the coefficient matrix and rhs vector defining the linear system that represents the line formed by
            # the intersection of the plane through the body center and the plane through the scan vector
            for j in range(3):

                # initialize the first row of the coefficient matrix to 0
                coefs[0, j] = 0

                # set the jth column of the first row of the coefficient matrix to be the inner product between the
                # observer_to_ellipsoid vector and the jth column of the ellipsoid matrix ( r_C^T@A_C )
                for i in range(3):

                    coefs[0, j] += observer_to_ellipsoid[i]*self.ellipsoid_matrix[i, j]

                # set the second row of the coefficient matrix to be the cross product between the center vector and
                # scan vector
                coefs[1, j] = crosses[ind, j]

                # set the last element of the rhs vector to be the negative inner product of the cross product between
                # the center vector and the scan vector and the observer_to_ellipsoid of the body ( -(c x s)^T@r_C )
                rhs[1] -= crosses[ind, j]*observer_to_ellipsoid[j]

            rhs[0] = -1
            rhs[2] = 0

            # get the direction for the line formed by the intersection of the plane through the body center and the
            # scan plane by taking the cross product of the normal vectors of the two planes
            line_dir[0] = coefs[0, 1]*crosses[ind, 2] - coefs[0, 2]*crosses[ind, 1]
            line_dir[1] = coefs[0, 2]*crosses[ind, 0] - coefs[0, 0]*crosses[ind, 2]
            line_dir[2] = coefs[0, 0]*crosses[ind, 1] - coefs[0, 1]*crosses[ind, 0]

            # find the particular solution to the underdetermined system of equations to determine a point on the line
            # formed by the intersection of the plane through the body center and the scan plane
            # line_start = np.linalg.lstsq(coefs, rhs[:2])[0]
            # make sure we don't have nans
            breakout = False
            for i in range(2):
                for j in range(3):
                    if isnan(coefs[i, j]):
                        breakout = True
                        break
                if breakout:
                    break

            if not breakout:
                for i in range(3):
                    if isnan(rhs[i]):
                        breakout = True

            if breakout:
                limbs[ind, :] = cynan
                continue

            dgelsd(&m, &n, &nrhs, &coefs[0, 0], &lda, &rhs[0], &ldb,
                   &s[0], &rcond, &rank, &work[0], &lwork, &iwork[0], &info)

            # now we can solve for the place where the line we just found pierces the ellipsoid

            # initialize the quadratic equation coefficients to 0
            a_coef = 0
            b_coef = 0
            c_coef = 0

            # compute the quadratic equation coefficients according to
            #   a = xh^T @ A_C @ xh
            #   b = 2*x0^T @ A_C @ xh
            #   c = x0^T @ A_C x0 - 1

            for i in range(3):
                for j in range(3):
                    temp = self.ellipsoid_matrix[j, i]
                    a_coef += line_dir[j]*temp*line_dir[i]
                    temp *= rhs[j]
                    b_coef += temp*line_dir[i]
                    c_coef += temp*rhs[i]

            b_coef *= 2
            c_coef -= 1

            # solve the quadratic equation for the distance

            # calculate the discriminant
            discriminant = sqrt(b_coef*b_coef - 4*a_coef*c_coef)

            # calculate the positive root
            root = (-b_coef + discriminant)/(2*a_coef)

            temp = 0

            # find the intersection point for this root and see if it has a positive dot product with the scan direction
            # vector
            for i in range(3):
                limbs[ind, i] = rhs[i] + root*line_dir[i]
                temp += limbs[ind, i]*scan_dirs[i, ind]

            # if the dot product is negative then use the negative root of the quadratic equation
            if temp < 0:
                root = (-b_coef - discriminant)/(2*a_coef)

                for i in range(3):
                    limbs[ind, i] = rhs[i] + root*line_dir[i]

        return np.asarray(limbs).T + np.asarray(observer_to_ellipsoid).reshape(3, 1)

    def compute_limb_jacobian(self, scan_center_dir, scan_dirs, limb_points, observer_position=None):
        r"""
        compute_limb_jacobian(self, scan_center_dir, scan_dirs, limb_points, observer_position=None)

        This method computes the linear change in the limb location given a change in the relative position between the
        ellipsoid and the observer.

        The limb Jacobian is defined mathematically as the solution to the system of equations given by

        .. math::

            \left[\begin{array}{c} 2\mathbf{p}_C^T\mathbf{A}_C \\ \mathbf{r}_C^T\mathbf{A}_C \\
            \left(\mathbf{s}_c \times\mathbf{s}_d\right)^T\end{array}\right](\mathbf{J} - \mathbf{I}) =
            \left[\begin{array}{c} \mathbf{0}_{1x3}\\-\mathbf{p}_C^T\mathbf{A}_C\\-\left(\mathbf{s}_c
            \times\mathbf{s}_d\right)^T \end{array} \right]

        where :math:`\mathbf{p}_C` is the vector from the center of the ellipsoid to the limb in current frame,
        :math:`\mathbf{r}_C` is the vector from the observer to the center of the ellipsoid in the current frame,
        :math:`\mathbf{s}_c` is the unit vector in the direction of the scan center in the current frame,
        and :math:`\mathbf{s}_d` is the unit vectors in the direction of the scan lines that correspond to limb
        :math:`\mathbf{p}_C`.  The Jacobian that is computed is
        :math:`\frac{\partial\mathbf{x}_C}{\partial\mathbf{r}_C}` where :math:`\mathbf{x}_C` is the vector from the
        observer to the limb in the current frame.

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
        if observer_position is not None:
            relative_position = self.center - np.array(observer_position).ravel()
        else:
            relative_position = self.center.ravel()

        # center the limb points at the ellipse center
        limb_points = limb_points - relative_position.reshape(3, 1)

        cdef double[:, :, :] coefs = np.rollaxis(
            np.zeros((3, 3, scan_dirs.shape[-1]), dtype=np.float64, order='F'), -1, 0)
        cdef double[:, :, :] rhs = np.rollaxis(
            np.zeros((3, 3, scan_dirs.shape[-1]), dtype=np.float64, order='F'), -1, 0)

        cdef double[:, :, :] jacobian = np.zeros((scan_dirs.shape[-1], 3, 3), dtype=np.float64)

        cdef int ind, i, j, jtype

        cdef double[:, :] crosses = np.cross(scan_center_dir.ravel(), scan_dirs.T).astype(np.float64)

        cdef double[:] posell = relative_position.ravel() @ self.ellipsoid_matrix

        cdef double[:, :] pointsell = limb_points.T @ self.ellipsoid_matrix

        cdef double[:, :] pointcrosses
        cdef double[:, :] poscrosses
        cdef int nlimbs = int(scan_dirs.shape[-1])

        cdef int n, nrhs, lda, ldb, info

        cdef int [:, :] ipiv = np.zeros((nlimbs, 3), dtype='i')

        # initialize the values for the lapack system solver
        n = nrhs = lda = ldb = 3

        # extract the number of limbs we are considering

        # if we're computing a "center" jacobian then compute the cross product between the limb vectors/position
        # vector and the scan vectors
        with cython.boundscheck(False):
            with nogil, parallel():
                for ind in prange(nlimbs, schedule='static'):

                    # form the coefficient matrix as
                    #    [2 p_C^T @ A_C]
                    #    [ r_C^T @ A_C ]
                    #    [  (c x s)^T  ]
                    for j in range(3):
                        for i in range(3):
                            coefs[ind, 0, j] += 2*pointsell[ind, j]

                        coefs[ind, 1, j] = posell[j]
                        coefs[ind, 2, j] = crosses[ind, j]

                    # form the rhs of the system of equations depending on the type
                    #   [ 0   0   0 ]
                    #   [ -p_C^T @ A_C ]
                    #   [  -(c x s)^T  ]
                    for j in range(3):
                        rhs[ind, 1, j] = -pointsell[ind, j]
                        rhs[ind, 2, j] = -crosses[ind, j]

                    # solve for the jacobian matrix for the current limb and store it
                    dgesv(&n, &nrhs, &coefs[ind, 0, 0], &lda, &ipiv[ind][0], &rhs[ind, 0, 0], &ldb, &info)
                    jacobian[ind] = rhs[ind]

                    for i in range(3):
                        jacobian[ind, i, i] += 1

        # return the Jacobians
        return np.asarray(jacobian)
