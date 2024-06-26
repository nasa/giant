# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This cython module provides the ability to accelerate ray tracing of :class:`.RawSurface` objects in GIANT.

Description
-----------

One of the most common ways to represent a 3D object is a tessellated surface made up of small, planar geometry
primitives (usually triangles).  This allows great freedom in specifying the shape of arbitrary objects at various
ground sample distances, however, because it usually takes many geometry primitives (on the order of tens of
thousands to millions) to represent objects with sufficient accuracy, ray tracing these surfaces can quickly become
prohibitively expensive (tracing thousands of rays against many geometry primitives requires a ton of work).  Therefore,
we need a way to accelerate the ray tracing of surfaces by limiting the number of geometry primitives we need to check
for each ray.  This is done through the :class:`.KDTree`.

A KDTree (or k-dimensional tree) is an acceleration structure where geometry primitives are split into smaller and
smaller groups as you traverse down the tree. These groups are represented by :class:`.KDNode`, where each node
contains either "left" and "right" children nodes (in which case it is called a branch node), or a small collection of
geometry primitives (in which case it is called a leaf node), plus an axis aligned bounding box which fully contains the
children nodes (if a branch node) or the geometry primitives (if a leaf node).  In order to trace this structure then,
we first trace the bounding box of a node (which is much more efficient than tracing geometry primitives), which
tells us if we need to further consider the contents of the node of not.  If the ray does intersect the bounding box
then we proceed to trace the left and right children nodes (if this is a branch) or the geometry primitives (if this
is a leaf) contained in the node.  By doing this, we limit the number of ray-geometry primitive intersection tests that
we need to perform dramatically (many times to as low as <100), which saves a lot of computational efficiency (again,
since the ray-axis aligned bounding box check is so much more efficient than the ray-geometry primitive checks).

Use
---

Since the :class:`.KDTree` is simply an acceleration structure, you can use it anywhere in GIANT that you would use a
:class:`.Surface` (in fact, the :class:`.KDTree` inherits from the :class:`.Surface` to emphasize this point).  To do
this, simply build your tree (by giving it the :class:`.RawSurface` that you want to accelerate and then calling the
:meth:`.KDTree.build`), and then use the resulting tree as your target geometry.  The :class:`.KDTree` implements all
of the required methods for the :class:`.Surface`, including :meth:`.KDTree.trace`, :meth:`.KDTree.compute_intersect`,
:meth:`.KDTree.find_limbs`, :meth:`.KDTree.compute_limb_jacobian`, :meth:`.KDTree.rotate`, and
:meth:`.KDTree.translate`.

Because it can take a while to build the tree initially, especially for particularly large surfaces, it is recommended
that you save the tree to a file after building it the first time, and then for future use load from the file.  You can
do this using the :meth:`.KDTree.save` and :meth:`.KDTree.load` methods, or using the typical pickle interface from
Python.

It will likely be rare that you directly create a :class:`.KDTree` in your scripts and software when using GIANT, since
GIANT provides scripts that already build them for you and save the results to file, including :mod:`.ingest_shape`,
:mod:`.tile_shape`, and :mod:`.spc_to_feature_catalogue`.  We therefore recommend that you look at these first to see
if they meet your needs.
"""

import numpy as np
cimport numpy as cnp

import pandas as pd

import pickle

import copy

from typing import Union, Tuple, Optional, Callable

import cython
from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, free

from giant.ray_tracer.shapes.axis_aligned_bounding_box import AxisAlignedBoundingBox
from giant.ray_tracer.shapes.surface import RawSurface, find_limbs_surface, Surface

from giant._typing import ARRAY_LIKE, PATH
from giant.rotations import Rotation


_CID = 0
"""
This is used to give a unique identifier to each node
"""


cdef class KDNode:
    """
    __init__(self, surface=None, _left=None, _right=None, _bounding_box=None, _order=0, _id=None, _id_order=None, _centers=None)

    A class to represent a node in a :class:`.KDTree`

    A node represents a spatial subdivision of the geometry that is contained in the scene.  It is comprised of an axis
    aligned bounding box, specifying the extent of the geometry contained with the node and its children, along with
    references to its children (if any), the geometry contained locally in the node (if any), and a unique identity for
    the node in the tree.

    Nodes are not typically used directly by the user, instead you should interact with the :class:`.KDTree`, which acts
    as a container/manager for all of the nodes that make up a tree.

    Each node is described by a unique identifier :attr:`id`.  Typically, this identifier is encoded into a large
    integer of the following form

    .. code-block:: none

        xxxxxxxxyyyyyyyy
        |--id--||facet#|
        |---id_order---|

    where :attr:`id` is the id of the node, ``facet#`` is reserved to specify a facet number within the node with a length
    of :attr:`order` digits, and there are a total of :attr:`id_order` digits used to represent the ids. This form is
    used for checking that we are not intersecting the same triangle again (due to finite precision issues) when we are
    doing a bounce or other tracing that starts already on the surface.  For more details on how this works refer to
    :meth:`.Scene.trace`.
    """

    def __init__(self, surface=None, _left=None, _right=None, _bounding_box=None, _order=0, _id=None, _id_order=None,
                 _centers=None):
        """
        :param surface: the surfaces that are to be stored in the node and its children
        :type surface: Optional[Surface]
        :param _left: The left child of this node.  This is only used for pickling/unpickling purposes and is typically
                      not used by the user
        :type _left: Optional[KDNode]
        :param _right: The right child of this node.  This is only used for pickling/unpickling purposes and is
                       typically not used by the user
        :type _right: Optional[KDNode]
        :param _bounding_box: The bounding box for this node.  This is only used for pickling/unpickling purposes and is
                              typically not used by the user
        :type _bounding_box: Optional[AxisAlignedBoundingBox]
        :param _order: The order for this node (the maximum number of faces in the geometry for any node in the current
                       :class:`.KDTree`).  This is only used for pickling/unpickling purposes and is typically not used
                       by the user
        :type _order: Optional[int]
        :param _id: A unique identifier for this node.  This is only used for picling/unpickling purposes and is
                    typically not used by the user.
        :type _id: Optional[int]
        :param _id_order: The order for this node (the maximum number of digits in any ID for any node in the current
                          :class:`.KDTree`).  This is only used for pickling/unpickling purposes and is typically not
                          used by the user.
        :type _id_order: Optional[int]
        :param _centers: The centers of the surfaces that are being current split.  This is not typically used by the
                         user.
        :type _centers: np.ndarray
        """

        # store the left and right children (if we are unpacking) or initialize the left and right children to None
        self.left = _left
        self.right = _right

        self.bounding_box = _bounding_box
        """
        The axis aligned bounding box specifying the extend of space covered by this node.

        The bounding box contains both any geometry primitives contained in the node, as well as the bounding boxes of
        all children nodes, therefore if a ray does not intersect the bounding box of the node, it will not intersect
        any of its dependents (children or geometry)
        """

        # If we have surface, set the has_surface flag to true and store the surface
        if isinstance(surface, RawSurface):
            self.has_surface = True
            self.surface = surface
        else:
            self.has_surface = False

        # if we have surface and we aren't unpacking compute the bounding box for the node
        if (surface is not None) and (self.bounding_box is None):
            self.compute_bounding_box()

        # store the order for the node (if we are unpacking) or initialize it to 1
        self._order = _order

        # if the unique id for this node is not specified (we are not unpacking) then assign a unique id
        if _id is None:
            global _CID
            # assign the unique id
            self.id = _CID
            # increment the unique id counter
            _CID += 1

        else:
            self.id = _id

        # specify the order of the id, which is the order of the largest id assigned to any node in the parent tree
        if _id_order is None:
            # if we aren't unpacking this node then initialize the id order to 1
            self.id_order = 1
        else:
            # if we are unpacking this node then store the saved id order
            self.id_order = _id_order

        self._centers = _centers

    def __reduce__(self):
        """
        __reduce__(self)

        This method returns the class and the arguments to the class constructor for pickling
        """

        return self.__class__, (self.surface, self.left, self.right,
                                self.bounding_box, self.order, self.id, self.id_order)

    @property
    def order(self):
        """
        The order represents the number of digits required to represent of the maximum number of the geometry primitives
        contained in any relative nodes (nodes belonging to the same tree).
        """
        return self._order

    @order.setter
    def order(self, val):

        # when we set the order we must propagate this change to all the children, since all related nodes need to
        # have the same order

        self._order = val
        if self.left is not None:
            self.left.order = val

        if self.right is not None:
            self.right.order = val

    @cython.boundscheck(False)
    cpdef compute_bounding_box(self):
        """
        compute_bounding_box(self)

        This method computes the axis aligned bounding box for the node based off of the provided geometry primitives.
        It is guaranteed to encapsulate all of the primitives in the node (including those contained in sub-nodes).

        The result is stored in the :attr:`bounding_box` attribute.
        """

        cdef:
            long long facet
            double[:] max_sides = -np.array([np.inf, np.inf, np.inf])
            double[:] min_sides = np.array([np.inf, np.inf, np.inf])
            cnp.uint32_t[:] unique_facets
            size_t axis, facet_number

        if self.has_surface:

            unique_facets = pd.unique(self.surface.facets.ravel())

            for facet_number in range(unique_facets.shape[0]):
                facet = unique_facets[facet_number]
                for axis in range(3):
                    max_sides[axis] = max(max_sides[axis], self.surface._vertices[facet, axis])
                    min_sides[axis] = min(min_sides[axis], self.surface._vertices[facet, axis])

            # form and store the bounding box
            self.bounding_box = AxisAlignedBoundingBox(min_sides, max_sides)

        else:
            # something went wrong if we got here and we should probably throw an error instead
            self.bounding_box = AxisAlignedBoundingBox([0, 0, 0], [0, 0, 0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def split(self, force=False, flip=False, print_progress=True):
        """
        split(self, force=False, flip=False, print_progress=True)

        This method is used to "grow" the tree.  It takes the geometry primitives contained in this node, and
        distributes them to the left and right using the median value of the center of each facet along the axis with
        the widest separation.

        Normally this method won't split once there are less than 10 geometry primitives contained in the node.  This
        can be over ridden by setting the force flag to True, which will split until there is only 1 geometry primitive
        in the node.

        The results are stored in the :attr:`left` and :attr:`right` attributes of the current node as well as returned
        as a tuple.  The local surface is also removed from this node since the primitives are distributed to the
        child nodes.

        Typically a user won't directly use this method and instead will use :meth:`.KDTree.build`.

        :param force: A flag specifying whether to force splitting even past once there are less than 10 facets in the
                      node
        :type force: bool
        :param flip: A flag specifying whether to flip which side of the split gets the "equals" in the comparison.  If
                     ``True`` then the right side gets the equals.  If ``False`` then the left side.
        :type flip: bool
        :param print_progress: A flag specifying whether to print the progress of the splitting as we go.  This prints a
                               a lot to the screen but at least lets you know that things are not stuck somewhere
        :type print_progress: bool
        :return: A tuple of the new left and right nodes (in order) or a tuple of None, None if we have already split as
                 far as possible
        :rtype: Union[Tuple[KDNode, KDNode], Tuple[None, None]]
        """

        cdef size_t i, j, k, n_shapes
        cdef double[:, :] _centers
        cdef double[:, :] _npverts

        if self.has_surface:
            # check to be sure we actually have geometry primitives

            n_shapes = self.surface.num_faces
            if n_shapes == 1:
                return None, None

            if print_progress:
                print('splitting {} shapes'.format(n_shapes))

            if (not force) and (self.surface.num_faces <= 10):  # if we have split as much as possible
                # the calling function will know what to do with the Nones
                return None, None

            normals = self.surface.normals  # type: np.ndarray

            if self._centers is None:
                # find the median of the vertices to use as the split point
                npverts = self.surface.vertices
                _npverts = npverts.astype(np.float64)

                centers = np.zeros((self.surface.num_faces, 3), dtype=np.float64)
                _centers = centers
                with nogil, parallel():
                    for i in prange(n_shapes, schedule='dynamic'):
                        for j in range(3):
                            for k in range(3):
                                _centers[i, j] += _npverts[self.surface._facets[i, k], j] / 3
            else:
                centers = self._centers.base

            split_axis = (centers.max(axis=0) - centers.min(axis=0)).argmax()
            median = np.median(centers[:, split_axis])

            if flip:
                left_test = centers[:, split_axis] < median
            else:
                left_test = centers[:, split_axis] <= median

            # need to use the "hidden" attributes here to ensure that pickling happens correctly
            vertices = np.asanyarray(self.surface._vertices.base)
            if self.surface._single_albedo:
                albedos = self.surface._albedo
            else:
                albedos = np.asanyarray(self.surface._albedo_array.base)

            if np.any(left_test):
                left_surface = type(self.surface)(vertices, albedos, self.surface.facets[left_test],
                                                  normals=normals[left_test], compute_bounding_box=False,
                                                  compute_reference_ellipsoid=False)
            else:
                return None, None

            right_test = ~left_test

            if np.any(right_test):
                right_surface = type(self.surface)(vertices, albedos, self.surface.facets[right_test],
                                                   normals=normals[right_test], compute_bounding_box=False,
                                                   compute_reference_ellipsoid=False)
            else:
                return None, None

            # generate and store the new nodes -- bounding boxes for the shapes are calculated here
            self.left = KDNode(left_surface, _centers=centers[left_test])
            self.right = KDNode(right_surface, _centers=centers[right_test])

            # delete the local copy of shapes since its not needed anymore and takes up a lot of space
            self.surface = None
            self.has_surface = False
            self._centers = None


            return self.left, self.right
        else:
            return None, None


    @cython.boundscheck(False)
    cdef void _compute_intersect(self, const double[:] start, const double[:] direction, const double[:] inv_direction,
                                  const cnp.int64_t[] ignore, const cnp.uint32_t num_ignore, cnp.int64_t[] shape_ignore,
                                  cnp.uint8_t *hit, double[:] intersect, double[:] normal, double *albedo,
                                  cnp.int64_t *facet, double* previous_hit_distance) noexcept nogil:
        """
        This method is used to compute the intersect of a single ray with this node.  First the intersect is checked on
        the bounding box for the node.  If this successfully intersects and the intersect distance is less than the
        previous_hit_distance, then the ray is traced through either the
        geometry primitives (if this is a leaf node) or the children nodes (if this is a branch node).  Regardless,
        The results of the trace are collected and returned in the hit, intersect, normal, albedo, facet, and
        previous_hit_distance pointers.
        
        This method makes no calls back to python so it can be run without the GIL, allowing for parallelization
        """

        cdef:
            # for the bounding box
            cnp.uint8_t bb_check = 0

            # for the left node
            cnp.uint8_t left_hit = 0
            double[3] left_intersect
            double[3] left_normal
            double left_albedo = 0.
            cnp.int64_t left_facet = -1
            double left_distance = previous_hit_distance[0]

            # for the right node
            cnp.uint8_t right_hit = 0
            double[3] right_intersect
            double[3] right_normal
            double right_albedo = 0.
            cnp.int64_t right_facet = -1
            double right_distance = previous_hit_distance[0]

            # for storing the initial values while we're working
            double[3] initial_intersect
            double[3] initial_normal

            # this is used to determine if the ray has requested to ignore a primitive contained in this node
            cnp.uint32_t sizer = <cnp.uint32_t>(10 ** (self._order + 1))

            # for iterating
            size_t i

            # for the distances from the bounding box intersection check
            double near_distance
            double far_distance


        # copy over the existing intersect value so we can reset it if we don't strike anything
        for i in range(3):

            initial_intersect[i] = intersect[i]
            initial_normal[i] = normal[i]

        # check to see if the ray intersects the bounding box
        self.bounding_box._compute_intersect(start, inv_direction, &bb_check, &near_distance, &far_distance)

        if bb_check and (near_distance <= previous_hit_distance[0]):  # if the ray intersects the bounding box and we haven't
            if self.has_surface:  # if this is a leaf node

                # check to see if we are to ignore certain faces from within this node
                for i in range(num_ignore):

                    # if the ignore under consideration is from this node, store it
                    if (ignore[i] // sizer) == self.id:

                        shape_ignore[i] = ignore[i] % sizer

                    else:
                        shape_ignore[i] = -1

                # check for intersect with the shapes
                self.surface._compute_intersect(start, direction, inv_direction, shape_ignore, num_ignore,
                                               hit, intersect, normal, albedo, facet, previous_hit_distance)
                facet[0] += self.id*<cnp.int64_t>(10**(self._order+1))

            else:
                # check for intersect with the left node
                self.left._compute_intersect(start, direction, inv_direction, ignore, num_ignore, shape_ignore,
                                              &left_hit, intersect, normal, &left_albedo, &left_facet, &left_distance)

                # copy over the results of the intersect from the trace
                for i in range(3):
                    left_intersect[i] = intersect[i]
                    left_normal[i] = normal[i]

                # check for intersect with the right node
                self.right._compute_intersect(start, direction, inv_direction, ignore, num_ignore, shape_ignore,
                                              &right_hit, intersect, normal, &right_albedo, &right_facet,
                                              &right_distance)

                # copy over the results of the intersect from the trace
                for i in range(3):
                    right_intersect[i] = intersect[i]
                    right_normal[i] = normal[i]
                    intersect[i] = initial_intersect[i]
                    normal[i] = initial_normal[i]

                # if both sides hit
                if left_hit and right_hit:

                    # keep the shorter distance
                    if left_distance <= right_distance:

                        hit[0] = left_hit
                        albedo[0] = left_albedo
                        facet[0] = left_facet
                        for i in range(3):
                            intersect[i] = left_intersect[i]
                            normal[i] = left_normal[i]

                        previous_hit_distance[0] = left_distance

                    else:
                        hit[0] = right_hit
                        albedo[0] = right_albedo
                        facet[0] = right_facet
                        for i in range(3):
                            intersect[i] = right_intersect[i]
                            normal[i] = right_normal[i]

                        previous_hit_distance[0] = right_distance

                elif left_hit:
                    # if only the left hit keep it
                    hit[0] = left_hit
                    albedo[0] = left_albedo
                    facet[0] = left_facet
                    for i in range(3):
                        intersect[i] = left_intersect[i]
                        normal[i] = left_normal[i]
                    previous_hit_distance[0] = left_distance

                elif right_hit:
                    # if only the right hit keep it
                    hit[0] = right_hit
                    albedo[0] = right_albedo
                    facet[0] = right_facet
                    for i in range(3):
                        intersect[i] = right_intersect[i]
                        normal[i] = right_normal[i]
                    previous_hit_distance[0] = right_distance

    def __eq__(self, other):
        """
        __eq__(self, other)

        Used for unit testing.  checks that important attributes are equivalent

        Specifically this checks the bounding box, the shapes, the children nodes, and the order are all the same.

        :param other: The other node to compare
        :type other: KDNode
        :return: True if the nodes are equivalent, false otherwise
        :rtype: bool
        """

        # check that the bounding boxes are the same
        bbox_check = (self.bounding_box == other.bounding_box)

        if not bbox_check:
            return False

        # check that both are leafs or branches
        if self.has_surface != other.has_surface:
            return False

        # check that the left is the same
        if self.left != other.left:
            return False

        # check that the right is the same
        if self.right != other.right:
            return False

        # check that the order is the same
        if self.order != other.order:

            return False

        if self.has_surface:
            if self.surface != other.surface:
                return False

        # if we've made it here then everything is the same
        return True

    def translate(self, translation):
        """
        translate(self, translation)

        Translate this node and any child nodes by ``translation``.

        The bounding box, the child nodes, and the shapes are all translated.

        This is probably not needed anymore since the KDTree applies translations and rotations to the rays for
        efficiency.

        :param translation: a size 3 array by which to translate the node
        :type translation: ARRAY_LIKE
        """

        # make sure the input is an array
        translation = np.array(translation)

        # translate the bounding box
        self.bounding_box.translate(translation)

        # descend down the tree to the left and right
        if self.left is not None:
            self.left.translate(translation)

        if self.right is not None:
            self.right.translate(translation)

        # if this is a leaf node then translate the leaves
        if self.has_surface:

            self.surface.translate(translation)

    def rotate(self, rotation):
        """
        rotate(self, rotation)

        Rotate this node and any child nodes by ``rotation``

        The bounding box, the child nodes, and the shapes are all rotated.

        This is probably not needed anymore since the KDTree applies translations and rotations to the rays for
        efficiency.

        :param rotation: The rotation by which to rotate the node
        :type rotation: Rotation
        """

        # rotate the bounding box
        self.bounding_box.rotate(rotation)

        # descend down the tree to the left and the right
        if self.left is not None:
            self.left.rotate(rotation)

        if self.right is not None:
            self.right.rotate(rotation)

        # if this is a leaf node then rotate the leaves
        if self.has_surface:

            self.surface.rotate(rotation)


cdef class KDTree(Surface):
    """
    __init__(self, surface, max_depth=10, _root=None, _rotation=None, _position=None, _bounding_box=None, _reference_ellipsoid=None)

    A KD Tree representation for accelerated tracing of surface objects with many geometry primitives using axis aligned
    bounding box acceleration.

    A KD Tree is essentially a modifier on a :class:`.Surface` which makes it much more computationally efficient to
    trace without altering the results from the trace at all (that is tracing a :class:`.Surface` and a
    :class:`.KDTree` built for that surface will produce equivalent results.  The acceleration is performed by dividing
    the surface into local sub surfaces which much fewer primitives contained in them.  Each sub surface is then
    encapsulated by an :class:`.AxisAlignedBoundingBox` specifying the extent of the geometry contained in it. Each one
    of these subsurfaces is contained in a :class:`.KDNode`.

    The nodes are built hierarchically, so that there is one large node at the top (the :attr:`root` of the tree) and
    then exponentially divide as you descend the tree until you reach the nodes that contain actual geometry.  This
    structure allows very efficient tracing by dramatically decreasing the number of geometry primitives that must be
    traced for each ray.  The wikipedia article on kd trees provides a decent description of how they work
    (https://en.wikipedia.org/wiki/K-d_tree) though our specific implementation varies slightly from what is described
    there.

    Conceptually, if you don't care about the details, the tree can be treated just like any other surface in GIANT.
    You can use it as a target for relative opnav, or trace rays through it however you see fit. Simply provide your
    :class:`.Surface`, call the :meth:`build` method, and then use the results in a :class:`.SceneObject`.  The only
    real setting you may need to worry about is the :attr:`max_depth` which specifies
    how many node levels you want to make before tracing the remaining geometry.  Typically you want to set this so that
    there are between 10-100 remaining geometry primitives inside of the final leaf nodes in the tree (this information
    is provided to you if you use the :mod:`.ingest_shape` script). If you have too many geometry primitives remaining
    then your tree will be inefficient.  If you have too few geometry primitives remaining your tree will also be
    inefficient (it is sometimes a case of trial and error to find the optimal number).  Just remember that when
    adjusting this number that it is exponential.
    """

    def __init__(self, surface, max_depth=10, _root=None, _rotation=None, _position=None, _bounding_box=None,
                 _reference_ellipsoid=None):
        """
        :param surface: the surface that is to be stored in the tree.  This must be a :class:`.RawSurface` subclass
        :type surface: Optional[Surface]
        :param max_depth: The maximum depth when building the tree (resulting in ``2**max_depth`` nodes)
        :type max_depth: int
        :param _root: The root node for the tree.  This is only used for pickling/unpickling purposes and is typically
                      not used by the user.
        :type _root: Optional[KDNode]
        :param _rotation: The current orientation from the world frame to the local tree frame as a :class:`.Rotation`.
                          This is only used for pickling/unpickling purposes and is typically not used by the user.
        :type _rotation: Optional[Rotation]
        :param _position: The current location of the center of the tree in the world frame.
                          This is only used for pickling/unpickling purposes and is typically not used by the user.
        :type _position: Optional[numpy.ndarray]
        :param _bounding_box: The bounding box for the tree.  This is only used for pickling/unpickling purposes and
                              is typically not used by the user.
        :type _bounding_box: Optional[AxisAlignedBoundingBox]
        :param _reference_ellipsoid: The reference for the tree.  This is only used for pickling/unpickling purposes and
                                     is typically not used by the user.
        :type _reference_ellipsoid: Optional[Ellipsoid]
        """

        self.root = _root
        """
        The root node of the tree.
        """

        if isinstance(surface, RawSurface):
            self.surface = surface
            """
            The surface that will be stored in this tree
            """

            self.reference_ellipsoid = surface.reference_ellipsoid
            """
            The circumscribing ellipsoid for the surface contained in this tree (used to determine how big the shape is)
            """

            self.bounding_box = surface.bounding_box

        elif surface is None:
            pass

        else:
            raise ValueError("surface must be a surface instance.")


        self.max_depth = max_depth
        """
        The maximum number of times to branch (split) when building the tree
        """

        self._rotation = _rotation
        """
        The rotation vector to rotate into the tree's frame
        """

        self._position = _position
        """
        The position vector to translate to the origin of the tree's frame
        """

        if _bounding_box is not None:
            self.bounding_box = _bounding_box
        if _reference_ellipsoid is not None:
            self.reference_ellipsoid = _reference_ellipsoid

    def __reduce__(self):
        """
        Used to package the tree for pickling/unpickling.
        """

        return self.__class__, (self.surface, self.max_depth, self.root, self._rotation, self.position,
                                self.bounding_box, self.reference_ellipsoid)

    @property
    def position(self):
        """
        The current location of the center of the tree in the world frame as a numpy array.

        This is used to translate the rays into the tree frame (using the opposite) which is typically more
        computationally more efficient than translating the entire tree
        """

        if self._position is None:
            return None
        else:
            return np.asarray(self._position)

    @property
    def rotation(self):
        """
        The current orientation of the tree with respect to the world frame as a :class:`.Rotation`

        This is used to rotate the rays into the tree frame (using the inverse) which is typically more
        computationally more efficient than rotating the entire tree
        """
        return self._rotation

    def build(self, force=True, print_progress=True):
        """
        build(self, force=True, print_progress=True)

        This method performs the branching of the tree down to the maximum depth.

        Essentially this method forms the root node from the surfaces provided to the tree at initialization.  The
        split method is then called on the root node, and the subsequent children until the maximum depth is reached or
        the minimum number of geometry primitives in each node is passed.

        The force argument can be used to continue splitting nodes even when there are less than 10 geometry primitives
        contained in the node.  It is passed to the :meth:`KDNode.split` method.

        The maximum depth is controlled through :attr:`max_depth`.  Typically this should be set so that the number of
        geometry primitives in each leaf node is between 10-100.

        :param force: A flag specifying that we should build the tree even when there are less than 10 geometry
                      primitives in the current level of nodes
        :type force: bool
        :param print_progress:  A flag specifying that we should print out the progress in building the tree.  This
                                helps you be confident the build is continuing but can slow things down because a lot of
                                text is printed to the screen.
        :type print_progress: bool
        """

        # form the root node
        self.root = KDNode(surface=self.surface)

        # final_nodes = 0

        # initialize the list of nodes that have been created
        nodes = [[self.root]]

        # for each depth call the split method
        flip = False
        for depth in range(1, self.max_depth):
            # initialize the list to store the nodes of the current level
            current_nodes = []

            # loop through the nodes in the previous depth
            for node in nodes[depth - 1]:

                # call their split method passing the force argument
                split_nodes = node.split(force=force, flip=flip, print_progress=print_progress)

                # if we successfully split then append these nodes to the list of current nodes
                if (split_nodes[0] is not None) and (split_nodes[1] is not None):

                    current_nodes.extend(split_nodes)
            flip = ~flip
            # if we successfully split
            if current_nodes:

                # append the nodes from this level to the master list of nodes
                nodes.append(current_nodes)

                # final_nodes = depth

            else:
                # other wise we have split as far as possible and it doesn't make sense to continue deeper
                break

        # get the leaf nodes from the list of nodes
        leaves = []
        for level in nodes:
            for node in level:
                if (node is not None) and node.has_surface:
                    leaves.append(node)

        # determine the order as the order of the maximum number of faces in any of the leaf nodes +2
        order = np.int64(np.log10(np.max([node.surface.num_faces for node in leaves])))
        # store the order of the tree in the root node, which will propagate to all children nodes
        self.root.order = order

        global _CID

        # store the id order as the order of the _CID variable at this time (which is a count of the total number of
        # nodes created since the kdtree module was imported
        self.root.id_order = np.int64(np.log10(_CID))

        self.bounding_box = copy.deepcopy(self.root.bounding_box)

    cdef void _compute_intersect(self, const double[:] start, const double[:] direction, const double[:] inv_direction,
                                 const cnp.int64_t[] ignore, const cnp.uint32_t num_ignore,
                                 cnp.uint8_t *hit, double[:] intersect, double[:] normal, double *albedo,
                                 cnp.int64_t *facet, double *hit_distance) noexcept nogil:
        """
        This C method is used to compute the intersect between a single ray and the surfaces contained in this object.

        The python version of this method :meth:`.compute_intersect` should be used unless working from Cython
        """

        cdef:
            cnp.int64_t* shape_ignore_data = <cnp.int64_t*> malloc(num_ignore * sizeof(cnp.int64_t))

        self.root._compute_intersect(start, direction, inv_direction, ignore,
                                     num_ignore, shape_ignore_data,
                                     hit, intersect, normal, albedo, facet,
                                     hit_distance)

        free(shape_ignore_data)

    def compute_intersect(self, ray):
        """
        compute_intersect(self, ray)

        This method computes the intersects between a single ray and the surface describe by this object.

        This is done by make a call to the efficient C implementation of the method.  This method packages everything in
        the way that the C method expects and restructures the results into the expect numpy structured array.

        This method also translates/rotates the ray into the tree frame first, for efficiency in tracing.

        In general, if you are tracing multiple rays you should use the :meth:`trace` method which is more optimized
        for multi ray tracing.

        :param ray: The ray to trace to the surce
        :type ray: Rays
        :return: a length 1 numpy array with a dtype of :attr:`.INTERSECT_DTYPE`
        :rtype: np.ndarray
        """

        # if there is a rotation or translation into the tree's frame then copy the rays so that we don't mess with the
        # originals
        if (self._rotation is not None) or (self._position is not None):

            ray = copy.copy(ray)

        # apply any rotations so the ray are expressed in the tree frame
        if self._rotation is not None:
            ray.rotate(self._rotation)

        # apply any translation so the ray are expressed in the tree frame
        if self._position is not None:
            ray.translate(np.asarray(self._position))

        results =  super().compute_intersect(ray)

        hits = results['check']

        # if we translated the rays then un-translate the results
        if self._position is not None:

            results["intersect"][hits] -= self._position

        # if we rotated the rays then un-rotate the results
        if self._rotation is not None:

            intersects = results[hits]["intersect"]
            normals = results[hits]["normal"]

            results["intersect"][hits] = (self._rotation.inv().matrix @
                                          intersects.squeeze().T).T.reshape(intersects.shape)
            results["normal"][hits] = (self._rotation.inv().matrix @
                                       normals.squeeze().T).T.reshape(normals.shape)

        return results

    def trace(self, rays, omp=True):
        """
        trace(self, rays, omp=True)

        This python method provides an easy interface to trace a number of Rays through the surface.

        It packages all of the ray inputs and the required output arrays automatically and dispatches to the c version
        of the method for efficient computation.  It then packages the output into the expected structured array.

        This method also translates/rotates the rays into the tree frame first, for efficiency in tracing.

        Parallel processing can be turned off by setting the omp flag to False

        :param rays: The rays to trace to the surface
        :type rays: Rays
        :param omp: A boolean flag specifying whether to use parallel processing (``True``) or not
        :type omp: bool
        :return: A length n numpy array with a data type of :data:`.INTERSECT_DTYPE`
        :rtype: np.ndarray
        """

        # if there is a rotation or translation into the tree's frame then copy the rays so that we don't mess with the
        # originals
        if (self._rotation is not None) or (self._position is not None):

            rays = copy.copy(rays)

        # apply any rotations so the rays are expressed in the tree frame
        if self._rotation is not None:
            rays.rotate(self._rotation)

        # apply any translation so the rays are expressed in the tree frame
        if self._position is not None:
            rays.translate(np.asarray(self._position))

        results = super().trace(rays, omp)

        hits = results['check']

        # if we translated the rays then un-translate the results
        if self._position is not None:

            results["intersect"][hits] -= self._position

        # if we rotated the rays then un-rotate the results
        if self._rotation is not None:

            intersects = results[hits]["intersect"]
            normals = results[hits]["normal"]

            results["intersect"][hits] = (self._rotation.inv().matrix @
                                          intersects.squeeze().T).T.reshape(intersects.shape)
            results["normal"][hits] = (self._rotation.inv().matrix @
                                       normals.squeeze().T).T.reshape(normals.shape)

        return results

    def translate(self, translation):
        """
        translate(self, translation)

        Translate the tree by translation.

        The tree is not actually translated, but its current location is stored so that we can translate rays when we
        trace them through the tree which is usually more efficient.

        :param translation: A size 3 array that the tree is to be translated by
        :type translation: ARRAY_LIKE
        """

        # make sure the input is a float array
        translation = np.asarray(translation, dtype=np.float64).ravel()

        # translate the reference ellipsoid
        if self.reference_ellipsoid:
            self.reference_ellipsoid.translate(translation)

        if self.bounding_box:
            self.bounding_box.translate(translation)

        # rotate the translation into the tree frame
        if self._rotation is not None:

            translation = self._rotation.matrix @ translation

        # apply the translation on top of any existing translation.
        # note that we will be translating rays into the tree frame, so we actually want the negative translation stored
        if self._position is not None:
            self._position -= translation
        else:
            self._position = -translation

    def rotate(self, rotation):
        """
        rotate(self, rotation)

        Rotate the tree by ``rotation``.

        The tree is not actually rotated, but its current orientation is stored so that we can rotate rays when we
        trace them through the tree which is usually more efficient.

        :param rotation: The rotation with which to rotate the tree
        :type rotation: Union[Rotation, ARRAY_LIKE]
        """

        if not isinstance(rotation, Rotation):
            rotation = Rotation(rotation)

        # store the rotation as the inverse rotation since we will actually rotate rays into the tree frame instead of
        # rotating the tree
        if self._rotation is None:
            self._rotation = rotation.inv()

        else:
            # if there was an existing rotation then apply the new one to the right
            self._rotation = self._rotation*rotation.inv()

        if self.reference_ellipsoid:
            # apply the rotation to the reference ellipsoid
            self.reference_ellipsoid.rotate(rotation)

        if self.bounding_box:
            self.bounding_box.rotate(rotation)

    def save(self, filename):
        """
        save(self, filename)

        Use this method to dump the tree to a pickle file.  You can also do this yourself in the usual way
        (nothing special happens)

        .. warning::

            This method saves some results to python pickle files.  Pickle files can be used to execute arbitrary
            code, so you should never open one from an untrusted source.

        :param filename: The name of the file to save the tree to
        :type filename: PATH
        """

        with open(filename, 'wb')as pick_file:

            pickle.dump(self, pick_file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        """
        load(self, filename)

        Load a tree from a pickle file and overwrite this instance with that tree.

        This is useful to store an "empty" tree in a scene, and then load the actual tree in only when you need it.

        .. warning::

            This method reads some results from python pickle files.  Pickle files can be used to execute arbitrary
            code, so you should never open one from an untrusted source.

        :param filename: The name of the file to load the tree from
        :type filename: PATH
        """

        with open(filename, 'rb') as pick_file:

            loaded = pickle.load(pick_file)

        # overwrite the important things
        self.max_depth = loaded.max_depth
        self.root = loaded.root
        self.surface = loaded.surface
        self.reference_ellipsoid = loaded.reference_ellipsoid
        self.bounding_box = loaded.bounding_box
        self._rotation = loaded.rotation
        self._position = loaded.position

    @property
    def order(self):
        """
        The order for the tree, which is the same as the :attr:`~.KDNode.order` of the :attr:`root` plus the
        :attr:`.id_order` of the :attr:`root`.

        Essentially this number represents the number of digits required to uniquely identify every geometry primitive
        contained in the tree.
        """

        try:
            return self.root.order + 1 + self.root.id_order
        except IndexError:
            return None


cpdef list get_ignore_inds(KDNode node, size_t vertex_id):
    """
    get_ignore_inds(node, vertex_id)

    This python/C helper function can be used to transverse a tree and find the ignore indices that should be used
    to ignore all facets sharing a particular vertex.

    This is primarily used for coverage analysis when we are tracing starting at a vertex of a shape and want to make
    sure that the vertex doesn't shadow itself.  Essentially we transverse the tree and identify the unique ID for any
    geometry primitives that contain the supplied vertex.  This is not super fast unfortunately.

    :param node: the current node we are looking through
    :type node: KDNode
    :param vertex_id: The row index of the vertex we are looking for in the original vertices array
    :type vertex_id: int
    :return: A numpy array containing the ID for each geometry primitive that uses the supplied vertex.  This can be
             passed as the :attr:`.Rays.ignore` attribute for a ray to ignore that primitive when tracing
    """

    # define the variables we will be using
    cdef:
        size_t fint, vint
        list inds = []
        bint lcheck

    # if this is a leaf node then check to see if the vertex is contained in it
    if node.has_surface:
        for fint in range(node.surface._facets.shape[0]):

            # loop through each vector in the current shape
            for vint in range(3):

                # check if the vertex is the same we're looking for
                lcheck = node.surface._facets[fint, vint] == vertex_id

                if lcheck:
                    # store the id of this shape in the output list
                    inds.append(fint + node.id*(10 ** (node._order + 1)))
                    break

        return inds
    else:
        # if we are at a branch node then recurse to the left and right
        left_inds = get_ignore_inds(node.left, vertex_id)
        right_inds = get_ignore_inds(node.right, vertex_id)

        # concatenate all of the results into a single array and return it
        return left_inds+right_inds


cpdef cnp.ndarray get_facet_vertices(KDNode node, size_t facet_id):
    """
    get_facet(node, facet_id)

    This python/C helper function can be used to transverse a tree and find the vertices for a specified facet id.

    This is primarily used for debugging when we want to identify which triangle we intersected with a ray.
    Essentially we transverse the tree and identify the geometry primitive that the ID belongs to and then return its
    vertices.

    :param node: the current node we are looking through
    :type node: KDNode
    :param facet_id: The unique ID for the facet we are looking for
    :type vertex_id: int
    :return: A numpy array containing the vertices as the columns of a 3xn matrix.
    """

    # define the variables we will be using
    cdef:
        size_t desired_facet_id
        cnp.uint32_t sizer

    # if this is a leaf node then check to see if the vertex is contained in it
    if node.has_surface:
        sizer = <cnp.uint32_t>(10 ** (node._order + 1))
        if (facet_id // sizer) == node.id:
            desired_facet_id = facet_id % sizer
            return node.surface.vertices[node.surface.facets[desired_facet_id]]

        else:
            return None
    else:
        # if we are at a branch node then recurse to the left and right
        left_check = get_facet_vertices(node.left, facet_id)
        if left_check is None:
            return get_facet_vertices(node.right, facet_id)
        else:
            return left_check


def describe_tree(tree: KDTree):
    """
    describe_tree(tree)

    This function describes the results of building a tree.

    For each level of the tree, the total number of nodes are counted.  Then the leaf nodes on that level are counted,
    and the min/max/mean number of facets for all of the leaf nodes on that level are computed.  This is then printed to
    the screen in tabular format.

    :param tree: the tree we are to describe
    :type tree: KDTree
    """

    def traverse_node(func: Callable[[KDNode, int], None], node: KDNode, ndepth: int = 1):

        if node.left is not None:
            traverse_node(func, node.left, ndepth + 1)

        if node.right is not None:
            traverse_node(func, node.right, ndepth + 1)

        if (node.left is None) and (node.right is None):
            func(node, ndepth)

    leaf_nodes = {}

    def leaf_collector(node: KDNode, cdepth: int):

        dlist = leaf_nodes.get(cdepth)

        if dlist is None:
            dlist = []
            leaf_nodes[cdepth] = dlist

        dlist.append(node)

    traverse_node(leaf_collector, tree.root)

    max_depth = max(leaf_nodes.keys())
    print('max depth: {}'.format(max_depth))

    for depth in range(1, max_depth+1):

        print('depth {}:'.format(depth))
        lnodes = leaf_nodes.get(depth, [])
        nnodes = len(lnodes)
        print('\t {} leaf nodes'.format(nnodes))
        if lnodes:
            nshapes = list(map(lambda x: x.surface.num_faces, lnodes))
            print('\t max faces per node {}'.format(max(nshapes)))
            print('\t min faces per node {}'.format(min(nshapes)))
            print('\t mean faces per node {}'.format(np.mean(nshapes)))

    print('Tree order: {}'.format(tree.order))
