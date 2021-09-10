# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This cython module defines surfaces for GIANT tesselated using Triangles as the geometry primitive.

Description
-----------

The triangle is the most commonly used geometry primitive for tesselating a surface in the general rendering/modelling
community and GIANT is no different.  the vast majority of surfaces in GIANT are represented using the classes defined
in this module.  They represent surfaces as a collection of triangles (defined by an array of vertices, an array of
albedos, an array of normal vectors, and a facet map which specifies how the vertices are connected into Triangles)
which is a very memory efficient way to represent these structures (and is very similar to the format used for the
ubiquitous wavefront obj format).  There are 2 classes that implement this that are identical except for the precision
with which they store these values, :class:`.Triangle64` which stores the values using double precision and
:class:`.Triangle32` which store these values using single precision.  In many cases single precision is sufficient,
however, if you are dealing with very high resolution terrain of a large object, or are translating/rotating the terrain
frequently then you may need to consider using the double precision representation.  One way to get around this is to
use the :class:`.KDTree` acceleration structure (recommended in general anyway) since this never actually
translates/rotates the terrain, therefore avoiding loss of precision.

The triangle representations in this module are fully featured and can be used throughout GIANT for ray tracing,
rendering, and relative OpNav.  That being said, due to the nature of tesselation, they are typically not the most
efficient when it comes to ray tracing.  Therefore, we strongly recommend that you wrap these objects in the
acceleration structure provided by class :class:`.KDTree`, which dramatically accelerates the ray tracing performance
while still providing exact tracing results.

Use
---

In general users will rarely directly need to create instances of these classes, as GIANT provides tools the create them
from common formats in the scripts :mod:`.ingest_shape`, :mod:`.spc_to_feature_catalogue`, and :mod:`.tile_shape`.  If
you do need to use these classes directly the documentation below should help you (and examining the aforementioned
scripts as examples would also be helpful).
"""


from libc.float cimport DBL_MAX
from libc.math cimport fabs, sqrt

import numpy as np
cimport numpy as cnp

import cython
from cython.parallel import prange, parallel, threadid

from giant.ray_tracer.shapes.shapes cimport Surface32, Surface64


cdef void _solve_3x3sys(double[3][3] mat, double[3] rhs, double[3] solu) nogil:
    """
    This C function solves a 3x3 system of equations using cramer's rule because it is very fast for such a small
    problem.
    """

    # compute the determinant of the LHS matrix
    cdef double det = mat[0][0]*mat[1][1]*mat[2][2]-mat[0][0]*mat[1][2]*mat[2][1]-mat[0][1]*mat[1][0]*mat[2][2]+ \
      mat[0][1]*mat[1][2]*mat[2][0]+mat[0][2]*mat[1][0]*mat[2][1]-mat[0][2]*mat[1][1]*mat[2][0]

    # as long as this isn't a rank deficient matrix then compute the solved system
    if det != 0.:

        solu[0] = ((mat[0][1]*mat[1][2]-mat[0][2]*mat[1][1])*rhs[2]+
                   (mat[0][2]*mat[2][1]-mat[0][1]*mat[2][2])*rhs[1]+
                   (mat[1][1]*mat[2][2]-mat[1][2]*mat[2][1])*rhs[0])/det

        solu[1] = ((mat[0][2]*mat[1][0]-mat[0][0]*mat[1][2])*rhs[2]+
                   (mat[0][0]*mat[2][2]-mat[0][2]*mat[2][0])*rhs[1]+
                   (mat[1][2]*mat[2][0]-mat[1][0]*mat[2][2])*rhs[0])/det

        solu[2] = ((mat[0][0]*mat[1][1]-mat[0][1]*mat[1][0])*rhs[2]+
                   (mat[0][1]*mat[2][0]-mat[0][0]*mat[2][1])*rhs[1]+
                   (mat[1][0]*mat[2][1]-mat[1][1]*mat[2][0])*rhs[0])/det

    else:
        # if the matrix was rank deficient then return a big number, but why 1000???
        solu[1] = 1000.
        solu[2] = 1000.
        solu[3] = 1000.


cdef class Triangle64(Surface64):
    """
    __init__(self, vertices, albedos, facets, normals=None, compute_bounding_box=True, bounding_box=None, compute_reference_ellipsoid=True, reference_ellipsoid=None)

    This class represents surfaces as tessellated triangles, storing the vertices, albedos, and normal vectors using
    double precision.

    The surface is comprised of a :attr:`vertices` array, which is a nx3 double precision array specifying the vertices
    for the surface in the current frame, an :attr:`albedos` double precision nx3 array or scalar representing the
    albedo for each vertex (if a scalar, then every vertex has the same albedo), a :attr:`normals` mx3 double precision
    array representing the normal vector for each triangle, and a :attr:`facets` mx3 facet map which indexes into the
    :attr:`vertices` and :attr:`albedos` arrays to specify the triangles.  Note that here we assign the albedo to each
    vertex, not each face as is commonly done in other rendering programs.  When tracing, we then use linear
    interpolation to calculate the albedo at the intersect point based on the albedos of the vertices of the triangle we
    intersected.

    To use this class you must provide at minimum the :attr:`vertices`, :attr:`albedos`, and :attr:`facets`, which
    result in the :attr:`normals` being computed for you.  In addition, when initializing the class, you can provide
    (or have automatically computed) an axis aligned bounding box which contains all of the represented surface and a
    reference ellipsoid which describes the tri-axial ellipsoid which best represents the surface.  Both of these are
    optional though.

    Once initialized, you can use this class for ray tracing as well as for identifying limbs and limb jacobians.  That
    being said, you typically should accelerate these things using the :class:`.KDTree` class, which will result in much
    better performance.

    Note that beyond using double precision, this is identical to the :class:`.Triangle32` class.
    """

    @property
    def facets(self):
        """
        This property returns the facets for the surfaces(s) as a mx3 ndarray where each face has 3 vertices and
        there are m faces overall.

        The facet elements are indices into the :attr:`vertices` array.  They must be positive.

        Each row of this array corresponds to the same row of the :attr:`normals` array
        """
        return np.asarray(self._facets)

    @facets.setter
    def facets(self, val):
        try:
            # check that the last 2 shapes are 3x3
            if val.shape[-1] != 3:
                raise ValueError('The last axis of the vertices array must be 3 where each row is a vertex.')

            else:
                self._facets = val.astype(np.uint32)
        except (AttributeError, ValueError, TypeError) as e:
            self._facets = np.array(val).astype(np.uint32).reshape(-1, 3)

    @property
    def sides(self):
        """
        This property calculates the primary sides of the surface as a mx3x2 ndarray where each surface has 2 primary
        sides and there are m surfaces overall

        :return: The primary sides of the surface
        :rtype: np.ndarray
        """

        cdef cnp.uint32_t face, i
        cdef double[3] side1
        cdef double[3] side2
        sides = np.zeros((self.num_faces, 3, 2), dtype=np.float64)

        for face in range(self.num_faces):

            self._get_sides(face, side1, side2)

            for i in range(3):
                sides[face, i, 0] = side1[i]
                sides[face, i, 1] = side2[i]

        return sides

    @cython.boundscheck(False)
    cdef void _get_sides(self, cnp.uint32_t face, double[3] side1, double[3] side2) nogil:
        """
        This efficient c function computes the 2 primary sides for the requested face storing them into side1 and side2.
        """

        cdef int i

        for i in range(3):
            side1[i] = self._vertices[self._facets[face, 1], i] - self._vertices[self._facets[face, 0], i]
            side2[i] = self._vertices[self._facets[face, 2], i] - self._vertices[self._facets[face, 0], i]

    @cython.boundscheck(False)
    def compute_normals(self):
        """
        compute_normals(self)

        This method computes the unit normal vectors for each facet.

        The normal vector is the unit vector in the direction of the cross product of the side vector going from vertex
        0 to vertex 1 with the side vector going from vertex 0 to vertex 2.  The normals are computed in parallel and
        are stored in the :attr:`normals` array.
        """
        cdef unsigned long long i
        cdef cnp.uint32_t num_faces=self.num_faces
        cdef cnp.uint32_t ind
        cdef double[8][3] side1
        cdef double[8][3] side2
        cdef double[8] dist
        cdef int thread, num_threads=8
        # cross 0-1 with 0-2
        self._normals = np.zeros((self.num_faces, 3), dtype=np.float64)

        with nogil, parallel(num_threads=num_threads):
            for ind in prange(num_faces, schedule='guided'):
                thread = threadid()
                self._get_sides(ind, side1[thread], side2[thread])
                self._normals[ind, 0] = side1[thread][1] * side2[thread][2] - side2[thread][1] * side1[thread][2]
                self._normals[ind, 1] = -(side1[thread][0] * side2[thread][2] - side2[thread][0] * side1[thread][2])
                self._normals[ind, 2] = side1[thread][0] * side2[thread][1] - side2[thread][0] * side1[thread][1]
                dist[thread] = 0
                for i in range(3):
                    dist[thread] += self._normals[ind, i] ** 2

                dist[thread] = sqrt(dist[thread])
                for i in range(3):
                    self._normals[ind, i] /= dist[thread]

    @cython.boundscheck(False)
    cdef double _get_albedo(self, const double[3] rhs, const int face) nogil:
        """
        This C method determines the interpolated albedo for an intersection point.

        It works entirely in C so the GIL is not needed allowing it to be used in parallel.  The rhs input should be
        the coefficients of the barycentric coordinates corresponding to the point on the facet where the intersection
        occurs.

        The python version of this function properly calls to this function so the user does not need to worry about the
        specifics.
        """
        cdef int i
        cdef double alb
        if self._single_albedo:
            # if we only have a scalar albedo just return it
            return self._albedo

        else:
            # get the albedo values for each vertex belonging to the current facet
            alb = self._albedo_array[self._facets[face][0]]

            # compute the interpolated albedo using the barycentric coordinates.  This is a linear interpolation scheme
            for i in range(2):
                alb += rhs[i]*(self._albedo_array[self._facets[face][i+1]] - self._albedo_array[self._facets[face][0]])

            return alb

    def get_albedo(self, point, face_index):
        """
        get_albedo(self, point, face_index)

        This method computes the albedo for a given intersect point and intersect face.

        The albedo is computed using linear interpolation based on the barycentric coordinates of the intersect point.

        If the intersect ``point`` does not actually correspond to the ``face_index`` then the results will be
        undefined.

        Typically you do not need to worry about this method as the albedo is automatically computed when you use
        :meth:`trace` or :meth:`compute_intersect`.

        :param point:  The 3d intersection point(s) in the current frame as a 3xn array
        :type point: ARRAY_LIKE
        :param face_index: The index(ices) into the facet array giving the face(s) the intersect point(s) is(are) on
        :type face_index: Union[int, ARRAY_LIKE]
        :return: the albedo at the intersect point.
        :rtype: Union[float, numpy.ndarray]
        """

        cdef int i

        point = np.array(point).reshape(3, -1)

        try:
            n = len(face_index)

            results = []

            for i in range(n):

                # determine the barycentric coordinates
                bcoords = np.roll(
                    np.linalg.lstsq(np.vstack([self.vertices[self.facets[face_index[i]]].swapaxes(-1, -2),
                                               [1, 1, 1]]),
                                    np.concatenate([point[:, i], [1]]),
                                    rcond=None)[0].ravel().astype(np.float64), -1)

                results.append(self._get_alb(bcoords, int(face_index[i])))

            return np.asarray(results)

        except TypeError:

            bcoords = np.roll(np.linalg.lstsq(np.vstack([self.vertices[self.facets[face_index]].T, [1, 1, 1]]),
                                              np.concatenate([point.ravel(), [1]]),
                                              rcond=None)[0].ravel().astype(np.float64), -1)

            return self._get_alb(bcoords, int(face_index))

    def _get_alb(self, double[:] point, face_index):
        """
        This is a helper function to call to the c code.

        Use the :meth:`get_albedo` method instead
        """

        return self._get_albedo(&point[0], face_index)

    @cython.boundscheck(False)
    cdef void _compute_intersect(self, const double[:] start, const double[:] direction, const double[:] inv_direction,
                                 const cnp.int64_t[] ignore, const cnp.uint32_t num_ignore,
                                 cnp.uint8_t *hit, double[:] intersect, double[:] normal, double *albedo,
                                 cnp.int64_t *facet, double *hit_distance) nogil:
        """
        This C function checks if a ray intersects any of the surfaces contained in a given object.

        Users should see the python version :meth:`compute_intersect` or :meth:`trace` for more information

        This uses the Moller Trumbore method to perform the intersection
        https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
        """

        cdef int num_faces = self._facets.shape[0]
        cdef int face
        cdef int i
        cdef int j
        cdef bint ignore_face


        cdef double ldot

        cdef double[3][3] coef_mat
        cdef double[3] rhs
        cdef double[3] solu
        cdef double[3] side1, side2

        cdef double dist = DBL_MAX


        # loop through each face
        for face in range(num_faces):

            # check to see if we should be ignoring this face
            ignore_face = False
            for i in range(num_ignore):
                if face == ignore[i]:
                    ignore_face = True

            # if we are ignoring this face then move on to the next one
            if ignore_face:
                continue

            # check to see if the ray is parallel to the face
            ldot = 0.
            for i in range(3):  # dot product between the normal vector and the direction vector
                ldot += direction[i]*self._normals[face, i]

            if fabs(ldot) <= 1e-12:  # if so, move on
                continue

            # TODO: it may be faster to use the scalar triple products instead of solving the system
            # build the coefficient matrix to solve for the barycentric coordinates and distance along the ray
            self._get_sides(face, side1, side2)
            for i in range(3):
                coef_mat[i][0] = side1[i]
                coef_mat[i][1] = side2[i]
                coef_mat[i][2] = -direction[i]

            # form the right hand side to solve for the barycentric coordinates and distance along the ray
            for i in range(3):
                rhs[i] = start[i] - self._vertices[self._facets[face][0], i]

            # solve the system of equations
            _solve_3x3sys(coef_mat, rhs, solu)
            # lu_solve(coef_mat, rhs)

            # check to be sure that this is a valid intersection (bary centric coords are between 0 and 1, sum to less
            # than or equal to 1, and the distance is positive
            if (solu[0] >=0) & (solu[1] >= 0) & (solu[2] > 0) & ((solu[0] + solu[1]) <= 1):

                # check to see if we have already struct another facet at a closer distance
                if solu[2] < dist:
                    # if not then update the results
                    hit[0] = True
                    for i in range(3):
                        # compute the intersect as the ray start plus the ray direction times the distance
                        intersect[i] = start[i] + solu[2] * direction[i]
                        # store the normal vector for this face
                        normal[i] = self._normals[face, i]
                    # store which facet was struck
                    facet[0] = face
                    # compute the albedo using the barycentric coordinates
                    albedo[0] = self._get_albedo(solu, face)
                    # update the best distance so far
                    dist = solu[2]
                    hit_distance[0] = dist

cdef class Triangle32(Surface32):
    """
    __init__(self, vertices, albedos, facets, normals=None, compute_bounding_box=True, bounding_box=None, compute_reference_ellipsoid=True, reference_ellipsoid=None)

    This class represents surfaces as tessellated triangles, storing the vertices, albedos, and normal vectors using
    single precision.

    The surface is comprised of a :attr:`vertices` array, which is a nx3 single precision array specifying the vertices
    for the surface in the current frame, an :attr:`albedos` single precision nx3 array or scalar representing the
    albedo for each vertex (if a scalar, then every vertex has the same albedo), a :attr:`normals` mx3 single precision
    array representing the normal vector for each triangle, and a :attr:`facets` mx3 facet map which indexes into the
    :attr:`vertices` and :attr:`albedos` arrays to specify the triangles.  Note that here we assign the albedo to each
    vertex, not each face as is commonly done in other rendering programs.  When tracing, we then use linear
    interpolation to calculate the albedo at the intersect point based on the albedos of the vertices of the triangle we
    intersected.

    To use this class you must provide at minimum the :attr:`vertices`, :attr:`albedos`, and :attr:`facets`, which
    result in the :attr:`normals` being computed for you.  In addition, when initializing the class, you can provide
    (or have automatically computed) an axis aligned bounding box which contains all of the represented surface and a
    reference ellipsoid which describes the tri-axial ellipsoid which best represents the surface.  Both of these are
    optional though.

    Once initialized, you can use this class for ray tracing as well as for identifying limbs and limb jacobians.  That
    being said, you typically should accelerate these things using the :class:`.KDTree` class, which will result in much
    better performance.

    Note that beyond using single precision, this is identical to the :class:`.Triangle64` class.
    """

    @property
    def facets(self):
        """
        This property returns the facets for the surfaces(s) as a mx3 ndarray where each face has 3 vertices and
        there are m faces overall.

        The facet elements are indices into the :attr:`vertices` array.  They must be positive.

        Each row of this array corresponds to the same row of the :attr:`normals` array
        """
        return np.asarray(self._facets)

    @facets.setter
    def facets(self, val):
        try:
            # check that the last 2 shapes are 3x3
            if val.shape[-1] != 3:
                raise ValueError('The last axis of the vertices array must be 3 where each row is a vertex.')

            else:
                self._facets = val.astype(np.uint32)
        except (AttributeError, ValueError, TypeError) as e:
            self._facets = np.array(val).astype(np.uint32).reshape(-1, 3)

    @property
    def sides(self):
        """
        This property calculates the primary sides of the surface as a mx3x2 ndarray where each surface has 2 primary
        sides and there are m surfaces overall

        :return: The primary sides of the surface
        :rtype: np.ndarray
        """

        cdef long long face, i
        cdef float[3] side1
        cdef float[3] side2
        sides = np.zeros((self.num_faces, 3, 2), dtype=np.float32)

        for face in range(self.num_faces):

            self._get_sides(face, side1, side2)

            for i in range(3):
                sides[face, i, 0] = side1[i]
                sides[face, i, 1] = side2[i]

        return sides

    @cython.boundscheck(False)
    cdef void _get_sides(self, cnp.uint32_t face, float[3] side1, float[3] side2) nogil:
        """
        This efficient c function computes the 2 primary sides for the requested face storing them into side1 and side2.
        """

        cdef int i

        for i in range(3):
            side1[i] = self._vertices[self._facets[face, 1], i] - self._vertices[self._facets[face, 0], i]
            side2[i] = self._vertices[self._facets[face, 2], i] - self._vertices[self._facets[face, 0], i]

    @cython.boundscheck(False)
    def compute_normals(self):
        """
        compute_normals(self)

        This method computes the unit normal vectors for each facet.

        The normal vector is the unit vector in the direction of the cross product of the side vector going from vertex
        0 to vertex 1 with the side vector going from vertex 0 to vertex 2.  The normals are computed in parallel and
        are stored in the :attr:`normals` array.
        """
        cdef unsigned long long i
        cdef long long num_faces=self.num_faces
        cdef long long ind
        cdef float[8][3] side1
        cdef float[8][3] side2
        cdef float[8] dist
        cdef int thread, num_threads=8
        # cross 0-1 with 0-2
        self._normals = np.zeros((self.num_faces, 3), dtype=np.float32)

        with nogil, parallel(num_threads=num_threads):
            for ind in prange(num_faces, schedule='guided'):
                thread = threadid()
                self._get_sides(ind, side1[thread], side2[thread])
                self._normals[ind, 0] = side1[thread][1] * side2[thread][2] - side2[thread][1] * side1[thread][2]
                self._normals[ind, 1] = -(side1[thread][0] * side2[thread][2] - side2[thread][0] * side1[thread][2])
                self._normals[ind, 2] = side1[thread][0] * side2[thread][1] - side2[thread][0] * side1[thread][1]
                dist[thread] = 0
                for i in range(3):
                    dist[thread] += self._normals[ind, i] ** 2

                dist[thread] = sqrt(dist[thread])
                for i in range(3):
                    self._normals[ind, i] /= dist[thread]

    @cython.boundscheck(False)
    cdef float _get_albedo(self, const double[3] rhs, const int face) nogil:
        """
        This C method determines the interpolated albedo for an intersection point.

        It works entirely in C so the GIL is not needed allowing it to be used in parallel.  The rhs input should be
        the coefficients of the barycentric coordinates corresponding to the point on the facet where the intersection
        occurs.

        The python version of this function properly calls to this function so the user does not need to worry about the
        specifics.
        """
        cdef int i
        cdef float alb
        if self._single_albedo:
            # if we only have a scalar albedo just return it
            return self._albedo

        else:
            # get the albedo values for each vertex belonging to the current facet
            alb = self._albedo_array[self._facets[face][0]]

            # compute the interpolated albedo using the barycentric coordinates.  This is a linear interpolation scheme
            for i in range(2):
                alb += rhs[i]*(self._albedo_array[self._facets[face][i+1]] - self._albedo_array[self._facets[face][0]])

            return alb

    def get_albedo(self, point, face_index):
        """
        get_albedo(self, point, face_index)

        This method computes the albedo for a given intersect point and intersect face.

        The albedo is computed using linear interpolation based on the barycentric coordinates of the intersect point.

        If the intersect ``point`` does not actually correspond to the ``face_index`` then the results will be
        undefined.

        Typically you do not need to worry about this method as the albedo is automatically computed when you use
        :meth:`trace` or :meth:`compute_intersect`.

        :param point:  The 3d intersection point(s) in the current frame as a 3xn array
        :type point: ARRAY_LIKE
        :param face_index: The index(ices) into the facet array giving the face(s) the intersect point(s) is(are) on
        :type face_index: Union[int, ARRAY_LIKE]
        :return: the albedo at the intersect point.
        :rtype: float
        """

        cdef int i

        point = np.array(point).reshape(3, -1)

        try:
            n = len(face_index)

            results = []

            for i in range(n):

                # determine the barycentric coordinates
                bcoords = np.roll(np.linalg.lstsq(np.vstack([self.vertices[self.facets[face_index[i]]].swapaxes(-1, -2), [1, 1, 1]]),
                                                  np.concatenate([point[:, i], [1]]),
                                                  rcond=None)[0].ravel().astype(np.float64), -1)

                results.append(self._get_alb(bcoords, int(face_index[i])))

            return np.asarray(results)

        except TypeError:

            bcoords = np.roll(np.linalg.lstsq(np.vstack([self.vertices[self.facets[face_index]].T, [1, 1, 1]]),
                                              np.concatenate([point.ravel(), [1]]),
                                              rcond=None)[0].ravel().astype(np.float64), -1)

            return self._get_alb(bcoords, int(face_index))

    def _get_alb(self, double[:] point, face_index):
        """
        This is a helper function to call to the c code.

        Use the :meth:`get_albedo` method instead
        """

        return self._get_albedo(&point[0], face_index)

    @cython.boundscheck(False)
    cdef void _compute_intersect(self, const double[:] start, const double[:] direction, const double[:] inv_direction,
                                 const cnp.int64_t[] ignore, const cnp.uint32_t num_ignore,
                                 cnp.uint8_t *hit, double[:] intersect, double[:] normal, double *albedo,
                                 cnp.int64_t *facet, double *hit_distance) nogil:
        """
        This C function checks if a ray intersects any of the surfaces contained in a given object.

        Users should see the python version :meth:`compute_intersect` or :meth:`trace` for more information

        This uses the Moller Trumbore method to perform the intersection
        https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
        """

        cdef int num_faces = self._facets.shape[0]
        cdef int face
        cdef int i
        cdef int j
        cdef bint ignore_face


        cdef double ldot

        cdef double[3][3] coef_mat
        cdef double[3] rhs
        cdef double[3] solu
        cdef float[3] side1, side2

        cdef double dist = DBL_MAX


        # loop through each face
        for face in range(num_faces):

            # check to see if we should be ignoring this face
            ignore_face = False
            for i in range(num_ignore):
                if face == ignore[i]:
                    ignore_face = True

            # if we are ignoring this face then move on to the next one
            if ignore_face:
                continue

            # check to see if the ray is parallel to the face
            ldot = 0.
            for i in range(3):  # dot product between the normal vector and the direction vector
                ldot += direction[i]*self._normals[face, i]

            if fabs(ldot) <= 1e-12:  # if so, move on
                continue

            # TODO: it may be faster to use the scalar triple products instead of solving the system
            # build the coefficient matrix to solve for the barycentric coordinates and distance along the ray
            self._get_sides(face, side1, side2)
            for i in range(3):
                coef_mat[i][0] = side1[i]
                coef_mat[i][1] = side2[i]
                coef_mat[i][2] = -direction[i]

            # form the right hand side to solve for the barycentric coordinates and distance along the ray
            for i in range(3):
                rhs[i] = start[i] - self._vertices[self._facets[face][0], i]

            # solve the system of equations
            _solve_3x3sys(coef_mat, rhs, solu)
            # lu_solve(coef_mat, rhs)

            # check to be sure that this is a valid intersection (bary centric coords are between 0 and 1, sum to less
            # than or equal to 1, and the distance is positive
            if (solu[0] >=0) & (solu[1] >= 0) & (solu[2] > 0) & ((solu[0] + solu[1]) <= 1):

                # check to see if we have already struct another facet at a closer distance
                if solu[2] < dist:
                    # if not then update the results
                    hit[0] = True
                    for i in range(3):
                        # compute the intersect as the ray start plus the ray direction times the distance
                        intersect[i] = start[i] + solu[2] * direction[i]
                        # store the normal vector for this face
                        normal[i] = self._normals[face, i]
                    # store which facet was struck
                    facet[0] = face
                    # compute the albedo using the barycentric coordinates
                    albedo[0] = self._get_albedo(solu, face)
                    # update the best distance so far
                    dist = solu[2]
                    hit_distance[0] = dist
