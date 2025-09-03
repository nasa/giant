


r"""
This module implements the idea of a surface feature and surface feature catalog for GIANT.

Detailed Description
--------------------

In GIANT, a surface feature refers to a small path of surface from a body that is treated as an individual target to
identify in an image. A feature catalog is then a collection of these features that is used for tracking the features
together and determining which are visible/worthy to search for in a given image.  These capabilities are implemented in
the :class:`.SurfaceFeature`, :class:`.FeatureCatalog`, and :class:`.VisibleFeatureFinder` respectively.

In more detail, a :class:`.SurfaceFeature` in GIANT is essentially a wrapper around a DEM modeled as a traceable object
(a :class:`.KDTree` or :class:`.Shape`) that also contains some basic information about the DEM, including the normal
vector of the best fit plane through the DEM data, the center of the DEM, and the average ground sample distance of the
DEM.  Additionally, the :class:`.SurfaceFeature` class provides functionality for storing the DEM information itself on
disk until it is needed as well as managing at which point it can be unloaded from memory, which can be important for
long running processes.

The :class:`.FeatureCatalog` then stores a list of these features, as well as some numpy arrays which combine all of
the normal vectors, bounding box vertices, and locations into single contiguous arrays for computational efficiency.
These numpy arrays also get rotated/translated whenever the feature catalog gets rotated/translated, which means that
they are always in the current frame (which normally ends up being the camera frame). Additionally, this class provides
an option for filtering the features that are included when a call to the ``trace`` method is made.

Finally, the :class:`.VisibleFeatureFinder` operates on a feature catalog to identify which features are visible in an
image based on the current knowledge of the scene at the time of the image.  This filtering considers things like the
incidence and reflection angles, the ratio of the camera ground sample distance to the feature ground sample distance,
and the percentage of the feature predicted to actually be within the field of view of the camera.

For more in-depth details on each of these classes we encourage you to consider their documentation.

Lazy Loading/Unloading
----------------------

One of the key things about surface features are that they are usually pretty high resolution patches of surface (that
is patches of surface with small ground sample distances).  We can get away with these small ground sample distances
because the surface area each feature covers is also small, which means the size of any given feature is small.  Despite
this, we frequently have enough patches to globally cover the surface of the body we are imaging, and many times have
features that overlap each other so we actually have enough features to cover the surface 2-3 times over.  Beyond that,
we also typically have global coverage of the surface with features at varying ground sample distances.  All of this
combines to mean that feature catalogs are typically huge, particularly when compared with a typical global model for
a body.  In many cases, because of this size, it is infeasible or even impossible to store the entire feature catalog
in memory at once, even on modern systems.

To overcome this issue, we have implemented a lazy loading/unloading scheme for feature catalogs and surface features
in GIANT.  The way this scheme works is when we create a :class:`.SurfaceFeature`, instead of providing the DEM data
that represents the feature, we instead provide an absolute file path to a pickle file which contains just the DEM data
for the feature.  When doing this, the size of each :class:`.SurfaceFeature` object in the :class:`.FeatureCatalog` is
very small.  Then, when we actually need the DEM data for the feature, we load it into memory from the referenced file.
This dramatically decreases the memory footprint of our process, making it feasible to operate through a whole feature
catalog without using hundreds of GB of memory.

Now, this lazy loading is great at the start, but if we have a long running process that keeps using more and more
features we will eventually end up loading the entire feature catalog into memory anyway, which would ultimately
defeat the purpose of the lazy loading.  Therefore, we also provide an unloading mechanism, where the "time" since the
feature was last used (time here being the number of images since we last used the feature) is used to determine if we
should remove the feature from memory.  Additionally, we provide a check one the percentage of the total system memory
that the process is using and if we are over a certain percentage, we start unloading features regardless of when they
were last used.

The combination of the loading and unloading makes surface feature navigation possible in GIANT even on systems with
modest amounts of memory available to the system.  If you are working on a system with huge amounts of memory, you can
bypass these features, but for most people, even with large amounts of memory, we encourage leaving them on.  For more
details on how you can tune each of these capabilities, refer to the :class:`.FeatureCatalog` and
:class:`.SurfaceFeature` documentation.

One thing to note about using an absolute path to the file containing the DEM information for each feature is that it
makes feature catalogs brittle.  That is, unless the directory structure between 2 systems is the same, you cannot
directly transfer a feature catalog built on one system to another.  To help with doing this, we provide the methods
:meth:`.FeatureCatalog.update_feature_paths` and :meth:`.SurfaceFeature.update_path` which can be used to specify a
a new path for the files containing the surface feature DEMs.

Use
---

Generally a user will have minimal direct interaction with the classes in this module, as all of the interaction is
handled either by the :class:`.SurfaceFeatureNavigation` class for doing navigation or the
:mod:`.spc_to_feature_catalog` and :mod:`.tile_shape` scripts for importing/building a feature catalog.  If you do
need to interact directly with the classes in this documentation we encourage you to consult the class documentation
directly and to look at the use of these classes in the mentioned class/scripts for examples/insight.

"""


from copy import deepcopy, copy


import pickle

import os

from pathlib import Path

from dataclasses import dataclass

from typing import Optional, List, Union, Dict, Tuple, Callable, cast

import numpy as np

import psutil

from giant.rotations import Rotation

from giant.ray_tracer.shapes import AxisAlignedBoundingBox, Shape
from giant.ray_tracer.kdtree import KDTree
from giant.ray_tracer.scene import Scene
from giant.ray_tracer.rays import INTERSECT_DTYPE
from giant.ray_tracer.rays import Rays
from giant.camera_models.camera_model import CameraModel

from giant._typing import NONENUM, PATH, ARRAY_LIKE


_PROCESS: psutil.Process = psutil.Process()
"""
This module attribute contains the Process object for the current process at import time.

This is used to query memory use statistics for the system to use in the lazy load/unload evaluations.
"""


class FeatureCatalog:
    """
    This class represents a collection of :class:`.SurfaceFeatures` for use in GIANT relative OpNav.

    In GIANT, a surface feature is used to represent a small patch of surface which we look for in an image to generate
    a bearing measurement to in the process of :class:`.SurfaceFeatureNavigation`.  Because we normally have many
    features in order to globally cover the surface, we need a special manager to handle all of these features, rather
    than making each one a :class:`.SceneObject` in a scene.  That is the purpose of this class.

    Essentially this class works as its own mini scene (in fact much of the code was copied from the :class:`.Scene`
    but without a :attr:`.Scene.light_obj`.  Somewhat confusingly though, this class is meant to be wrapped in a
    :class:`.SceneObject` and then stored as a :attr:`.Scene.target_obj` for processing.  To the user (and to the
    :class:`.Scene`) this will look like any other traceable object in GIANT for the most part.

    Internally, the features are filtered with the :attr:`include_features` attribute, which is a list of integer
    indices into the :attr:`features` list of which features to include while tracing.  Normally, however, a user won't
    interact with this attribute, which is instead handled by the :class:`.SurfaceFeatureNavigation` class.

    This class also collects some information about each feature into the :attr:`feature_normals`,
    :attr:`feature_locations`, and :attr:`feature_bounds` attributes, which are used to easily determine which features
    are visible for a given image in conjunction with the :class:`.VisibleFeatureFinder` class and the
    :class:`.SurfaceFeatureNavigation` class.

    Similar to a :class:`.KDTree`, when something requests to rotate or translate the feature catalog through the
    :meth:`rotate` and :meth:`translate` methods, the features themselves are not actually moved.  Instead the
    rotation/translation is stored and is used to rotate/translate the rays into the original feature catalog frame
    before ray tracing for performance reasons.  These methods do update the :attr:`feature_normals`,
    :attr:`feature_locations`, and :attr:`feature_bounds` attributes though.

    One of the keys of the :class:`.SurfaceFeature` class is that is provides a mechanism for lazy loading/unloading of
    the DEM data itself from memory.  This class provides 2 easy properties to change the control of this lazy
    load/unload through the :attr:`stale_count_unload_threshold` and :attr:`memory_percent_unload_threshold` which can
    be used to change the corresponding settings for all features in the catalog.

    Creating a feature catalog is a difficult process which has largely be automated into the
    :mod:`.spc_to_feature_catalog` and :mod:`.tile_shape` scripts, which we encourage you to consider at least as
    examples if you are building your own.  If you are making your own, once you have your list of
    :class:`.SurfaceFeature` objects, if you are using the lazy load/unload functionality, you should also provide a
    corresponding list of dictionaries which contain the bounding box vertices under key ``'bounds'`` and the feature
    DEM order under key ``'order'``.  Again, for an example of how to do this consider the :mod:`.tile_shape` script.
    """

    def __init__(self, features: List['SurfaceFeature'],
                 map_info: Optional[List[Dict[str, Union[int, np.ndarray]]]] = None):


        self.features: List['SurfaceFeature'] = features
        """
        The list of surface features contained in the catalog.
        
        Each element should be a :class:`.SurfaceFeature` object which describes the surface feature.
        """

        self._order: int = -1
        """
        This specifies the largest order of any of the shapes used to represent the features.
                
        This is used to determine which ignore indices apply to the features in this catalog, for cases where multiple
        targets are included in a :class:`.Scene`.  In general a user does not need to worry about this number and 
        should not modify it themselves
        """
        
        self._id_order: np.int64 = np.int64(np.log10(len(features)))
        """
        This specifies the number of digits required to represent all of the features in this catalog.
                
        This is used to determine which ignore indices apply to the features in this catalog, for cases where multiple
        targets are included in a :class:`.Scene`.  In general a user does not need to worry about this number and 
        should not modify it themselves
        """

        # temp variables for getting the bounds/locations/normals
        bounds = []
        normals = []
        locations = []

        for find, feature in enumerate(features):
            
            normals.append(feature.normal)
            locations.append(feature.body_fixed_center)
            if not feature.loaded and (map_info is not None):
                self._order = max(self._order, cast(int, map_info[find]['order']))
                bounds.append(map_info[find]['bounds'])

            else:
                assert feature.bounding_box is not None
                bounds.append(feature.bounding_box.vertices)
                if (forder := getattr(feature.shape, "order", None)) is not None:

                    self._order = max(self._order, forder)

                elif (fnum_faces := getattr(feature.shape, "num_faces", None)) is not None:

                    self._order = max(self._order, int(np.log10(fnum_faces)))

        if not normals:
            normals = [[]]
        self.feature_normals = np.vstack(normals)
        """
        The normal vectors of the best fit plane for each feature expressed in the current external frame as a nx3 
        numpy array.
        
        Each row of this matrix corresponds to the same index into the :attr:`features` attribute.
        
        These vectors are rotated whenever the :meth:`rotate` method is called, therefore they should always be 
        expressed in the internal frame (for instance in the camera frame), not in the base catalog frame.
        """

        self.feature_bounds: np.ndarray = np.array(bounds)
        """
        The vertices of the bounding box of every feature as a nx3x8 numpy array.

        Each slice along the first axis of this matrix corresponds to the same index into the :attr:`features` 
        attribute.

        These vectors are rotated whenever the :meth:`rotate` method is called and translated whenever the 
        :meth:`translate` method is called, therefore they should always be 
        expressed in the internal frame (for instance in the camera frame, not in the base catalog frame).
        """

        if not locations:
            locations = [[]]
        self.feature_locations: np.ndarray = np.vstack(locations)
        """
        The vectors to the center of each feature expressed in the current external frame as a nx3 numpy array.

        Each row of this matrix corresponds to the same index into the :attr:`features` attribute.

        These vectors are rotated whenever the :meth:`rotate` method is called and translated whenever the 
        :meth:`translate` method is called, therefore they should always be 
        expressed in the internal frame (for instance in the camera frame, not in the base catalog frame).
        """

        self.include_features: Optional[List[int]] = None
        """
        A list of features to include when tracing this feature catalog.
        
        The list should contain the indices into the :attr:`features` list of the features that you want to trace.  
        
        If this is not specified, it is assumed that you want to trace all of the features in the scene and special 
        handling is performed to see which are actually needed, so that only those are loaded into memory, as described 
        in the :meth:`trace` method.
        """

        self._rotation: Optional[Rotation] = None
        """
        The rotation that goes from the external frame to the original catalog frame.
        
        Note that this is the inverse of the rotation applied through the :meth:`rotate` method, and it is 
        multiplicatively updated, not overwritten.
        """

        self._position: Optional[np.ndarray] = None
        """
        The translation vector that goes from the external frame to the original catalog frame.
        
        Note that this is the negative of the translation applied through the :meth:`translate` method, rotated into the 
        original catalog frame and added to any existing position.
        """

        self._feature_bounding_boxes: Dict[int, AxisAlignedBoundingBox] = {}
        """
        A dictionary mapping feature index to :class:`.AxisAlignedBoundingBox` expressed in the original feature frame.
        
        This is only used when :attr:`include_features` is ``None`` and the :meth:`trace` method is called.
        """

        self._stale_count_unload_threshold: Optional[int] = None
        """
        The current stale count unload threshold for all of the features in the catalog.

        If this is ``None`` on a call to the :attr:`stale_count_unload_threshold` property, we'll just grab the value 
        from the first feature in the list.  If we set to the :attr:`stale_count_unload_threshold` property well 
        update this and all of the features in the catalog.
        """

        self._memory_percent_unload_threshold: NONENUM = None
        """
        The current memory percent unload threshold for all of the features in the catalog.
        
        If this is ``None`` on a call to the :attr:`memory_percent_unload_threshold` property, we'll just grab the value 
        from the first feature in the list.  If we set to the :attr:`memory_percent_unload_threshold` property well 
        update this and all of the features in the catalog.
        """

        self._feature_finder: Optional[Callable[[CameraModel, Scene, float], List[int]]] = None
        """
        This is used to determine the visible features in the catalog.
        
        It will typically be an instance of the :class:`.VisibleFeatureFinder`.  
        
        We do not set this at initialization because it is typically set at run time and is frequently changed
        """
        
    @property
    def order(self) -> int:
        """
        This specifies the number of digits reserved for specifying facet ids for features in this catalog
                
        This is used to determine which ignore indices apply to the features in this catalog, for cases where multiple
        targets are included in a :class:`.Scene`.  In general a user does not need to worry about this number and 
        should not modify it themselves
        """
        
        return self._order + int(self._id_order) + 1

    @property
    def stale_count_unload_threshold(self) -> int:
        """
        The number of times :meth:`not_found` must be called since the last :meth:`found` was called for a feature to
        be unloaded from memory.

        Setting to this property will change this value for all features contained in the catalog.
        """

        if self._stale_count_unload_threshold is None:
            self._stale_count_unload_threshold = self.features[0].stale_count_unload_threshold

        return self._stale_count_unload_threshold

    @stale_count_unload_threshold.setter
    def stale_count_unload_threshold(self, value: int):
        self._stale_count_unload_threshold = value

        for feature in self.features:
            feature.stale_count_unload_threshold = value

    @property
    def feature_finder(self) -> Callable[[CameraModel, Scene, float], List[int]]:
        """
        This property returns the feature finder for this class, which is a callable that takes in a camera model,
        scene, and temperature and returns a list of indices into the :attr:`features` list and related.

        Typically this is an instance of :class:`.VisibleFeatureFinder`
        """
        if self._feature_finder is None:
            self._feature_finder = VisibleFeatureFinder(self)

        return self._feature_finder

    @feature_finder.setter
    def feature_finder(self, val: Callable[[CameraModel, Scene, float], List[int]]):
        self._feature_finder = val

    @property
    def memory_percent_unload_threshold(self) -> float:
        """
        The memory percentage used by the current process at which point we begin unloading features that are not found
        regardless of how long its been since a feature was used.

        If you trust your system to handle swap appropriately you can set this to some value greater than 100 which will
        effectively disable this check.

        If you plan to run multiple instances of GIANT/SFN at the same time then you should probably set this value
        lower so that you limit the resources they are fighting over.

        Setting to this property will change this value for all features contained in the catalog.
        """

        if self._memory_percent_unload_threshold is None:
            self._memory_percent_unload_threshold = self.features[0].memory_percent_unload_threshold

        return self._memory_percent_unload_threshold

    @memory_percent_unload_threshold.setter
    def memory_percent_unload_threshold(self, value: int):
        self._memory_percent_unload_threshold = value

        for feature in self.features:
            feature.memory_percent_unload_threshold = value

    def update_feature_paths(self, new_path: PATH):
        """
        This method goes through and updates the directory structure for each feature contained in the catalog.

        Updates are made through a call to :meth:`.SurfaceFeature.update_path`.  With the update, we are just changing
        the directory structure, not the file itself.  Therefore the input should end with a director, not a file, and
        the existing file name for each feature will be joined to this directory structure.

        For any features which do not make use of the lazy loading/unloading capabilities nothing will happen.

        :param new_path: The new directory structure to traverse to find the DEM file
        """

        for feature in self.features:

            feature.update_path(new_path)

    @classmethod
    def __init_from_pickle__(cls, features: List['SurfaceFeature'],
                             bounds: np.ndarray, normals: np.ndarray, locations: np.ndarray,
                             rotation: Rotation, position: np.ndarray,
                             feature_bounding_boxes: Dict[int, AxisAlignedBoundingBox],
                             include_features: Optional[List[int]], order: int,
                             id_order: Optional[int] = None) -> 'FeatureCatalog':
        """
        This class method is used to initialize the class from pickle, instead of you usual init method.

        Don't use this yourself!

        :param features: The list of features
        :param bounds: The vertices of the bounding boxes of the features as a nx3x18 array in the current frame
        :param normals: The normal vectors of the surface features in the current frame
        :param locations: The center of the features in the current frame
        :param rotation: The rotation from the external frame to the internal frame
        :param position: The translation from the external frame to the internal frame
        :param feature_bounding_boxes: A dictionary mapping feature index to an AxisAlignedBoundingBox in the original
                                       feature frame.  Usually this is empty
        :param include_features: A list of integers specifying which features to consider when tracing
        :param order: The order of the identity for this feature catalog, used to identify whether ignore indices
                      apply to it.
        :return: An initialized version of the class
        """

        out = cls([])

        out.features = features
        out.feature_bounds = bounds
        out.feature_normals = normals
        out.feature_locations = locations
        out._position = position
        out._rotation = rotation
        out._feature_bounding_boxes = feature_bounding_boxes
        out.include_features = include_features
        out._order = order
        if id_order is None:
            out._id_order = np.int64(np.log10(len(features)))
        else:
            out._id_order = np.int64(id_order)

        return out

    def __reduce__(self) -> Tuple[Callable, tuple]:
        """
        Used to control how this class is pickled/unpickled
        """

        return self.__init_from_pickle__, (self.features, self.feature_bounds, self.feature_normals,
                                           self.feature_locations, self._rotation, self._position,
                                           self._feature_bounding_boxes, self.include_features, self._order, self._id_order)

    @property
    def bounding_box(self) -> AxisAlignedBoundingBox:
        """
        The approximate axis aligned bounding box of all of the features in the catalog.

        This is found by finding the minimum/maximum bound of the bounding box vertices (:attr:`feature_bounds`)
        rotated/translated into the current frame.  As such, it is only a rough estimate.
        """

        # noinspection PyArgumentList
        return AxisAlignedBoundingBox(self.feature_bounds.min(axis=(0, -1)), self.feature_bounds.max(axis=(0, -1)))

    def rotate(self, rotation: Union[Rotation, ARRAY_LIKE]):
        """
        Rotates the feature catalog.

        The rotation is applied to the :attr:`feature_bounds`, :attr:`feature_normals`, and :attr:`feature_locations`.
        It is not applied to the DEM data for each feature, instead, when rays are traced into the scene they are
        rotated into the base frame for the feature catalog before tracing.  This is generally more efficient because
        there are normally much fewer rays to trace than DEM points to rotate.

        :param rotation: The rotation to apply
        """

        if not isinstance(rotation, Rotation):
            rotation = Rotation(rotation)

        self.feature_normals = (rotation.matrix @ self.feature_normals.T).T

        self.feature_bounds = rotation.matrix @ self.feature_bounds

        self.feature_locations = (rotation.matrix @ self.feature_locations.T).T

        if self._rotation is None:
            self._rotation = rotation.inv()

        else:
            self._rotation = self._rotation * rotation.inv()

    def translate(self, translation: ARRAY_LIKE):
        """
        Translates the feature catalog.

        The translation is applied to the :attr:`feature_bounds` and :attr:`feature_locations`.  It is not applied to
        the DEM data for each feature, instead, when rays are traced into the scene they are translated into the base
        frame for the feature catalog before tracing.  This is generally more efficient because there are normally
        much fewer rays to trace than DEM points to rotate.

        :param translation: The translation to apply
        """

        translation = np.asarray(translation, dtype=np.float64).ravel()

        self.feature_bounds += translation.reshape(1, 3, 1)

        self.feature_locations += translation.reshape(1, 3)

        if self._rotation is not None:
            translation = self._rotation.matrix @ translation

        if self._position is not None:

            self._position -= translation

        else:
            self._position = -translation

    @staticmethod
    def get_first(total_results: np.ndarray, traced_rays: Rays) -> np.ndarray:
        """
        This static method identifies the first intersection for each ray when there is more than 1 feature intersected.

        Each feature in the scene is responsible for identifying the first intersection with itself for each ray.
        This method is then responsible for identifying which feature was struck first.

        This method works by considering the intersection for each ray with each feature, and then finding the
        the intersection with the minimum distance between the ray and the camera.  The result is a 1D array with
        dtype of :attr:`.INTERSECT_DTYPE`.

        A user will almost never use this method directly, as it is automatically called by the :meth:`trace` method.

        :param total_results: The first intersection for each ray with each feature in the catalog
        :param traced_rays: The rays that these results pertain to
        :return: The shrunk array specifying the first intersection for each array
        """

        if total_results.shape[0] == 1:
            return total_results[0]

        nan_check = cast(np.ndarray, total_results["check"].any(axis=0).squeeze())

        if not np.any(nan_check):
            return total_results[0]

        min_ind = np.zeros(total_results.shape[1], dtype=int)

        min_ind[nan_check] = np.nanargmin(np.linalg.norm(traced_rays[nan_check].start.T -
                                                         total_results["intersect"][:, nan_check], axis=-1), axis=0)

        return total_results[min_ind, np.arange(min_ind.size)]

    def trace(self, rays: Rays) -> np.ndarray:
        """
        This method traces rays through the feature catalog, optionally filtering which features are included traced
        through the :attr:`include_features` attribute.

        The rays are first rotated/translated into the base frame of the feature catalog and are then traced through
        each active feature to look for intersections.  Only the first (shortest distance) intersection for each ray is
        returned.  The results are returned as a numpy array with type :attr:`.INTERSECT_DTYPE`.

        If the :attr:`include_features` attribute is set to ``None``, then this method will attempt to smartly only load
        features where the Rays intersect the bounding box of the feature, before "lazy loading" the feature.  This is
        typically only used in the case when you are tracing the full feature catalog to render a high resolution
        image.

        :param rays: The rays to trace through the feature catalog
        :return: A numpy array specifying where each ray intersected with type :attr:`.INTERSECT_DTYPE`.
        """

        # if we haven't specified which features to include then we include all of them but intersect
        # the bounding box for the feature first before trying to load it to save memory.
        if self.include_features is None:
            self.include_features = list(range(len(self.features)))
            check_bbox = True
        else:
            check_bbox = False

        # rotate/translate the rays into the local frame defined for the feature catalog
        if (self._rotation is not None) or (self._position is not None):
            rays_local = copy(rays)

            if self._rotation is not None:
                rays_local.rotate(self._rotation)

            if self._position is not None:
                rays_local.translate(self._position)

        else:
            rays_local = rays

        res = []

        # loop through each included feature
        for feature_index in self.include_features:
            feature: SurfaceFeature = self.features[feature_index] 

            # at this point we are rendering the whole feature catalog, but we don't want to have to load everything
            # into memory if we don't need it so we trace the bounding box first to ensure that the feature is
            # intersected before loading it
            if check_bbox:
                # in case we have an old feature catalog that doesn't had the bbox attribute yet
                if not hasattr(self, "_feature_bounding_boxes"):
                    self._feature_bounding_boxes = {}

                # check to see if we already have this bounding box
                bbox: Optional[AxisAlignedBoundingBox] = self._feature_bounding_boxes.get(feature_index) 
                if bbox is None:
                    # figure out what the original bounds are in the body fixed frame without loading the shape
                    bounds = self.feature_bounds[feature_index].copy()
                    if self._rotation is not None:
                        bounds = self._rotation.matrix@bounds
                    if self._position is not None:
                        bounds += self._position.reshape(3, 1)

                    # make the AABB
                    bbox = AxisAlignedBoundingBox(bounds.min(axis=1),
                                                  bounds.max(axis=1))

                    # store it for future use
                    self._feature_bounding_boxes[feature_index] = bbox

                # check if the bounding box is hit by any of the rays
                bbox_results = cast(np.ndarray, bbox.trace(rays_local))

                if not bbox_results.any():
                    # if not notify the feature it wasn't found and move to the next feature
                    feature.not_found()
                    continue
                else:
                    # if it was notify the feature it was found
                    feature.found()

            # get a copy of the ignore indices to modify which ones count for the current feature
            original_ignore_inds = deepcopy(rays_local.ignore)
            ignore_inds = rays_local.ignore

            # modify the ignore inds for ones that apply to this feature
            if ignore_inds is not None:
                if isinstance(ignore_inds, (float, int)):
                    ignore_inds = np.array([ignore_inds]*rays_local.num_rays)
                else:
                    ignore_inds = np.asanyarray(ignore_inds)

                ignore_inds[ignore_inds // (10 ** (self._order + 1)) != feature_index] = -1
                ignore_inds[ignore_inds // (10 ** (self._order + 1)) == feature_index] %= 10 ** (self._order + 1)

            # trace the feature
            feature_results = feature.trace(rays_local)

            # reset the original ignore inds
            rays_local.ignore = original_ignore_inds

            # update the id for the intersect face based on the current feature index
            feature_results["facet"][feature_results["check"]] += feature_index * (10 ** (self._order + 1))

            # store the results
            res.append(feature_results)

        # now figure out with intersection was first for any ray with multiple intersects
        res = np.asarray(res, dtype=INTERSECT_DTYPE)

        res = self.get_first(res, rays_local)

        # now rotate/translate the result back into the frame the rays started in
        if self._position is not None:
            res["intersect"][res["check"]] -= self._position

        if self._rotation is not None:
            intersects = res[res["check"]]["intersect"]
            normals = res[res["check"]]["normal"]

            res["intersect"][res["check"]] = (self._rotation.inv().matrix @
                                              intersects.squeeze().T).T.reshape(intersects.shape)
            res["normal"][res["check"]] = (self._rotation.inv().matrix @
                                           normals.squeeze().T).T.reshape(normals.shape)

        # reset the include feature list
        if check_bbox:
            self.include_features = None

        return res


class SurfaceFeature:
    """
    This class represents a surface feature in GIANT.

    In GIANT, a surface feature is defined a small DEM of a patch of surface combined with a name, the average ground
    sample distance of the DEM in kilometers, and a best fit plane through the topography of the DEM.  The DEM itself is
    represented by a traceable object from the :mod:`.ray_tracer` module (typically either a :class:`.KDTree` or a
    :class:`.Surface` subclass) and is stored in the :attr:`shape` attribute. The best fit plane is represent by a
    normal vector, expressed in the body-fixed frame, and the location of the center of the plane, expressed in the
    body-fixed frame, which are stored in the :attr:`normal` and :attr:`body_fixed_center` attributes respectively.
    Finally, the name and the ground sample distance are stored in the :attr:`name` and :attr:`ground_sample_distance`
    attributes respectively.  These are all typically set at initialization of an instance of a surface feature.  We
    note here that typically all of these attributes stay in the original frame in which they are defined unless you are
    manually messing with things.  This is important for exporting your results to orbit determination software, which
    typically needs to know feature locations in a body fixed frame.

    Surface features are normally created through an external program, like SPC, or by tiling a very high resolution
    global shape model.  Both the process of ingesting a set of SPC features (called Maplets) and tiling a high
    resolution global shape model, are available in the :mod:`.spc_to_feature_catalog` and :mod:`.tile_shape` scripts
    respectively, therefore, it is rare that you will manually create surface features using this class.

    As discussed above, when doing surface feature navigation, the feature catalog is normally very large, as we
    typically globally tile a surface at very small ground sample distances with significant overlap between each tile.
    This means that it is usually infeasible to hold an entire feature catalog in memory at once.  To alleviate this
    issue, this class provides a lazy loading/unloading mechanism that only keeps the actual shape information (which
    is by far the biggest memory hog) in memory when it is needed.  With this mechanism, the DEM shape data is only
    loaded once something trying to access the :attr:`.shape` attribute of an instance of this class.  When this
    happens, the class will check if the DEM information has already been loaded into memory, and if not, it will load
    it automatically.  Then, after a specified number of images (controlled by the :attr:`stale_count_unload_threshold`
    attribute) have been processed which do not need the DEM information, it will automatically unload the information
    from memory. Additionally, features can be unloaded from memory every time their not found if the memory footprint
    of the current process as a percent of the total system memory exceeds the threshold specified in the
    :attr:`memory_percent_unload_threshold`.  This also can be controlled for all features in a
    :class:`.FeatureCatalog` through the :attr:`.FeatureCatalog.memory_percent_unload_threshold` property.
    This loading/unloading is managed by calls to the :meth:`found` and :meth:`not_found` methods of this class.

    The automatic loading and unloading of data is generally pretty invisible to the user outside of log messages, as it
    all happens automatically in the :class:`.SurfaceFeatureNavigation` and :class:`.FindVisibleFeatures` classes.  That
    being said, you may need to consider tuning how quickly things are unloaded from memory, which can easily be set for
    all features in a catalog through the :attr:`.FeatureCatalog.stale_count_unload_threshold` attribute.  Typically
    you want to set this sufficiently large enough that you are not frequently loading/unloading the same features over
    and over again, but small enough that you don't overwhelm the memory capabilities of your filter.  On modern solid
    state hard drives, the read speeds are generally fast enough that you can set this number fairly low, even if you
    end up loading/unloading more than absolutely necessary.  On older hard drives with slower read speeds you will
    generally what to try to make this as large as possible without consistently forcing your system to use swap.  It
    can take some experimentation to find the sweet spot, but we do want to stress that ultimately you are not affect
    the results you are generating here, just the speed at which those results can be generated.

    To use this automatic loading/unloading, when initializing the class, instead of the providing the traceable DEM
    data to the shape argument, instead provide a string or Path object that points to a pickle file containing the DEM
    data as the first object in the file.

    Generally you will not interact with surface features directly all that frequently, and instead will interact with a
    catalog of surface features through the :class:`.FeatureCatalog` class.

    .. warning::

        When using the lazy load/unload feature of this class pickle files are read.  While in general these pickle
        files have been created by GIANT and are completely safe, if the pickle files are somehow compromised or you
        received them from an untrusted source they could be used to execute arbitrary code on your system, therefore
        you should carefully verify that your pickle files have not been tampered with before using them.
    """

    def __init__(self, shape: Union[PATH, KDTree, Shape], normal: ARRAY_LIKE, body_fixed_center: ARRAY_LIKE,
                 name: str, ground_sample_distance: Optional[float] = None, stale_count_unload_threshold: int = 10,
                 memory_percent_unload_threshold: float = 90):
        """
        :param shape: The shape object that represents the DEM topography for the feature as a :class:`.KDTree` or
                      :class:`.Shape`, or the path to the file containing the shape object as a ``str`` or ``Path``.
        :param normal: The normal vector for the best fit plane to the DEM topography in the body-fixed frame
        :param body_fixed_center: The center of the best fit plane to the DEM topography in the body-fixed frame
        :param name: The name of the feature
        :param ground_sample_distance: The average ground sample distance of the DEM topography in kilometers.
        :param stale_count_unload_threshold: The number of times a feature must be marked as :meth:`not_found` for it
                                             to be unloaded from memory.
        :param memory_percent_unload_threshold: The size of the memory footprint of the current process as the percent
                                                of the total system memory before features are unloaded from memory.
        """

        self._shape: Optional[Union[KDTree, Shape]] = None
        """
        This private attribute is used to store the actual shape data or ``None`` if the shape data has not been loaded.
        
        To always access the shape data, use the :attr:`shape` property instead.
        """

        self._shape_file: Optional[PATH] = None
        """
        This private attribute is use to store the path to the file containing the DEM data as a pickle file.
        
        This should generally be an absolute path.
        """

        if isinstance(shape, (KDTree, Shape)):
            self._shape = shape
        else:
            self._shape_file = shape

        self.normal: np.ndarray = np.array(normal).ravel()
        """
        The normal vector of the best fit plane through the DEM expressed in the body-fixed frame as a length 3 numpy 
        array.
        """

        self.body_fixed_center: np.ndarray = np.array(body_fixed_center).ravel()
        """
        The vector from the center of the body to the middle of the best fit plane of the DEM expressed in the 
        body-fixed frame as a length 3 numpy array.
        """

        self.name: str = name
        """
        The name of the feature as a string.
        """

        self.ground_sample_distance: Optional[float] = ground_sample_distance
        """
        The average ground sample distance of the feature DEM in units of kilometers or ``None``.
        
        If ``None`` then scaling filtering will not be performed for this feature in the :class:`VisibleFeatureFinder`
        class.
        """

        self.n_not_found: int = 0
        """
        The number of times :meth:`not_found` has been called since the last time :meth:`found` was called.
        
        This is used to determine when the feature should be unloaded from memory.  Generally you should treat this as
        read only.
        """

        self.stale_count_unload_threshold: int = stale_count_unload_threshold
        """
        The number of times :meth:`not_found` must be called since the last :meth:`found` was called for the feature to
        be unloaded from memory.
        """

        self.memory_percent_unload_threshold: float = memory_percent_unload_threshold
        """
        The memory percentage used by the current process at which point we begin unloading features that are not found 
        regardless of how long its been since a feature was used.
        
        If you trust your system to handle swap appropriately you can set this to some value greater than 100 which will
        effectively disable this check.
        
        If you plan to run multiple instances of GIANT/SFN at the same time then you should probably set this value 
        lower so that you limit the resources they are fighting over.
        """

    @property
    def shape(self) -> Union[Shape, KDTree]:
        """
        This property gives the traceable object that represents the DEM.

        This will always return a traceable object (either a :class:`.KDTree` or a :class:`.Shape`, but if the DEM data
        hasn't been loaded from disk yet there may be a slight delay while it is retrieved.
        """
        if not self.loaded:
            self.load()
            
        assert self._shape is not None
        return self._shape

    @property
    def bounding_box(self) -> Optional[AxisAlignedBoundingBox]:
        """
        This property returns the bounding box for the feature by retrieving the bounding box from the traceable shape
        object.

        If for some reason the traceable shape object does not have a ``bounding_box`` attribute, ``None`` is returned.

        Note that this property will access the Shape property, so if the DEM shape data has not yet been loaded from
        disk it will be loaded and there may be a slight delay.
        """

        try:
            return deepcopy(self.shape.bounding_box)

        except AttributeError:
            return None

    @property
    def loaded(self) -> bool:
        """
        This property simply returns whether the shape data is actively loaded into memory or not.
        """
        return self._shape is not None

    def load(self):
        """
        This method loads the shape information from disk.

        Typically a user will not use this method directly as it is automatically called by the :attr:`shape` and
        :attr:`bounding_box` properties when required.

        :raises ValueError: If the shape file information is not available
        :raises IOError: If the shape file is not available
        """
        if self._shape_file is None:
            raise ValueError('Shape file not available')
        with open(self._shape_file, 'rb') as f:
            self._shape = pickle.load(f)
        print(f"{self.name} successfully Loaded...", flush=True)

    def update_path(self, new_path: PATH):
        """
        This updates the absolute directory structure to the file containing the DEM data for this shape.

        This only changes the directory structure, not the file itself, therefore it extracts the actual file name from
        the current path and appends it to the new path (that is, new_path should simply be a directory structure, not
        a full file path).  If you want to change the full file you can use :meth:`update_file`.

        If this feature is not a lazy load feature then this method will do nothing.
        :param new_path: The new directory structure to navigate to find the file
        """

        if isinstance(self._shape_file, str):
            self._shape_file = os.path.join(new_path, os.path.basename(self._shape_file))
        elif isinstance(self._shape_file, Path):
            self._shape_file = Path(new_path) / self._shape_file.name

    def update_file(self, new_file: PATH):
        """
        This updates the pointer to the file containing the DEM data for this feature.

        If you are just updating the directory structure and not the file itself see the :meth:`update_path` method.

        :param new_file: The file containing the DEM data for this feature.
        """

        self._shape_file = new_file

    def found(self):
        """
        This method is called when the feature is found in an image and will be processed.

        It resets the :attr:`n_not_found` counter to 0.

        Typically a user will not use this method directly as it is automatically called by the
        :class:`.VisualFeatureFinder` class.
        """
        self.n_not_found = 0

    def not_found(self):
        """
        This method is called when the feature is not found in an image.

        It increments the :attr:`n_not_found` counter by 1.  If the :attr:`n_not_found` counter is then greater than the
        :attr:`stale_count_unload_threshold`, the shape data will be unloaded from memory.  Additionally, if the memory
        footprint of the current process is using more than :attr:`memory_percent_unload_threshold` then, regardless of
        how long its been since this feature was last used, the shape data will be unloaded from memory.

        Typically a user will not use this method directly as it is automatically called by the
        :class:`.VisualFeatureFinder` class.

        If the instance of this class was provided the shape object directly, instead of the file containing the shape
        data then this method will not try to unload, though it will still increase the counter for posterity.
        """

        if self.loaded:
            # this is just for old models before we implemented the counter
            if not hasattr(self, 'n_not_found'):
                self.n_not_found = 1
            else:
                self.n_not_found += 1

            # make sure this is a lazy load/unload feature
            if self._shape_file is not None:
                if self.n_not_found > self.stale_count_unload_threshold:

                    print("unloading feature {} because it hasn't been used recently.".format(self.name))

                    self._shape = None
                    self.n_not_found = 0

                elif _PROCESS.memory_percent() > self.memory_percent_unload_threshold:
                    print("unloading feature {} because memory usage is at {}%.".format(self.name,
                                                                                        _PROCESS.memory_percent()))

                    self._shape = None
                    self.n_not_found = 0

    def trace(self, rays: Rays) -> np.ndarray:
        """
        This method traces the provided rays through the feature DEM.

        The trace it handled by the :meth:`~.Shape.trace` method of the DEM object directly.  Because we retrieve this
        method through the :attr:`shape` attribute, if the shape is not already in memory, it will be loaded.
        :param rays: the rays to trace against the DEM shape object.
        :return: A numpy array of type :attr:`INTERSECT_DTYPE` specifying the results of the ray trace.
        """

        return self.shape.trace(rays)


@dataclass
class VisibleFeatureFinderOptions:
    """
    This dataclass serves as one way to control the settings for the :class:`.VisibleFeatureFinder` at initialization.

    You can set any of the options on an instance of this dataclass and pass it to the :class:`.VisibleFeatureFinder` at
    initialization (or through the method :meth:`.VisibleFeatureFinder.apply_options`) to set the settings
    on the class.  This class is the preferred way of setting options on the class due to ease of use in IDEs.
    """

    target_index: Optional[int] = None
    """
    The index into the target list tor the scene object that contains the feature catalog, or ``None`` to 
    automatically deduce the index
    """

    feature_list: Optional[List[str]] = None
    """
    A list of feature names to test against (useful for filtering if you only want to use a subset of features
    """

    off_boresight_angle_maximum: NONENUM = None
    """
    The maximum angle between the boresight and the line of sight to a feature in degrees.  This is useful to avoid 
    overflows in the other checks by throwing things out that are way outside the field of view.  The default (if 
    left as ``None``) is ``1.5*camera_model.field_of_view``
    """

    gsd_scaling: float = 3
    """
    The ratio allowed between the ground sample distance of the camera and the ground sample distance of the feature.
    """

    reflectance_angle_maximum: float = 70
    """
    The maximum reflectance angle (angle between the line of sight vector and the feature normal vector) in degrees.
    """

    incident_angle_maximum: float = 70
    """
    The maximum incident angle (angle between the incoming light vector and the feature normal vector) in degrees.
    """

    percent_in_fov: float = 50
    """
    The percentage of the feature that is in the field of view based on a bounding box test. This should be between 0 
    and 100
    """


class VisibleFeatureFinder:
    """
    This class creates a callable which is used to filter features that are visible in an image.

    The features are filtered in a number of ways.  First, they can be directly filtered by name using the
    :attr:`feature_list` attribute. Next, the features are filter based on the angle between the line of sight vector to
    the feature and the camera boresight vector, using the :attr:`off_boresight_angle_maximum` attribute.  Then,
    the features are filtered based on the reflection angle (angle between the line of sight vector and feature normal
    vector, using the :attr:`reflectance_angle_maximum` attribute.  Then, the features are filtered based on the
    incidence angle (angle between the sun direction vector and the feature normal vector) using the
    :attr:`incident_angle_maximum` attribute.  The the features are filtered based on the ratio of the camera GSD at the
    feature location and the GSD of the feature, using the :attr:`gsd_scaling` attribute.  Finally the features are
    filtered based on the percentage of the feature that is in the FOV of the camera, using the :attr:`percent_in_fov`
    attribute.  For each of these filtering passes, the only the features that met the preceding filters are considered
    for efficiency.

    After initializing this class with the appropriate data, you can generate a list of the visible feature indices
    (index into the :attr:`.FeatureCatalog.features` list and related) by calling the result and providing the
    temperature of the camera.  This assumes that the scene/feature catalog/light source have been appropriately
    placed in the camera frame already, so typically you should ensure that you provide a reference (not a copy) of the
    feature catalog and the scene.

    Typically a user will not interact directly with this class and instead it will be managed by the
    :class:`.SurfaceFeatureNavigation` class.  If you do want to use it manually, provide the appropriate inputs
    to the class constructor, update the scene to place everything in the camera frame at the time you want to identify
    visible features, and then call the instance of this class providing the camera temperature at the time you want to
    identify the visible features.  The resulting list of indices can be used to index into the
    :attr:`.FeatureCatalog.features` list and related.

    To specify the settings for this class, you can either use keyword arguments or the
    :class:`.VisibleFeatureFinderOptions` dataclass, which is the preferred method.  It is not recommended to mix
    methods as this can lead to unexpected results
    """

    def __init__(self, feature_catalog: FeatureCatalog,
                 options: Optional[VisibleFeatureFinderOptions] = None,
                 feature_list: Optional[List[str]] = None,
                 off_boresight_angle_maximum: NONENUM = None, gsd_scaling: float = 3,
                 reflectance_angle_maximum: float = 70, incident_angle_maximum: float = 70, percent_in_fov: float = 50):
        """
        :param feature_catalog: The feature catalog that specifies the features we care considering
        :param options: A dataclass specifying the options to set for this instance.  If provided it takes preference
                        over all key word arguments, therefore it is not recommended to mix methods.
        :param feature_list: A list of feature names to test against (useful for filtering if you only want to use a
                             subset of features
        :param off_boresight_angle_maximum: The maximum angle between the boresight and the line of sight to a feature
                                            in degrees.  This is useful to avoid
                                            overflows in the other checks by throwing things out that are way outside
                                            the field of view.  The default (if left as ``None``) is
                                            ``1.5*camera_model.field_of_view``
        :param gsd_scaling: The ratio allowed between the ground sample distance of the camera and the ground sample
                            distance of the feature.
        :param reflectance_angle_maximum: The maximum reflectance angle (angle between the line of sight vector and the
                                          feature normal vector) in degrees.
        :param incident_angle_maximum: The maximum incident angle (angle between the incoming light vector and the
                                       feature normal vector) in degrees.
        :param percent_in_fov: The percentage of the feature that is in the field of view based on a bounding box test.
                               This should be between 0 and 100
        """

        self.feature_catalog: FeatureCatalog = feature_catalog
        """
        The catalog of features we are looking through.
        """

        self.feature_list: Optional[List[str]] = feature_list
        """
        This is used to filter features by name.

        You can use this list to process only a subset of features (which are still filtered through the other processes
        of this class) which may be useful if you have a large model with many features.  If this is a list of strings, 
        the only features who's :attr:`.SurfaceFeature.name` attribute are contained in this list are considered (note
        that the name must match exactly for this to work).

        If this is left as ``None`` then all features in the feature catalog are considered.
        """

        # compute the maximum fov extent based on the camera field of view
        if off_boresight_angle_maximum is not None:
            off_boresight_angle_maximum = float(off_boresight_angle_maximum)

        self.off_boresight_angle_maximum: Optional[float] = off_boresight_angle_maximum
        r"""
        The maximum angle between the feature location and the camera boresight in degrees.

        This check is used to avoid features that are far outside of the camera FOV to avoid overflows in the projection
        of the feature bounding box vectors onto the image that can occasionally incorrectly label the feature as being
        within the field of view.

        The off boresight angle is computed as

        .. math::

            \theta = \cos^{-1}\left(\frac{\left[\begin{array}{ccc}{0 & 0 & 1}\end{array}\right]\mathbf{x}_{iC}}
            {\|\mathbf{x}_{iC}\|}\right)

        where :math:`theta` is the view angle in degrees, :math:`\cos^{-1}` is the arc cosine in degrees, 
        :math:`\mathbf{x}_{iC}` is the vector from the camera center to the :math:`i^{th}` feature in the camera frame 
        from :attr:`.FeatureCatalog.feature_locations`, and :math:`\|\bullet\|` is the 2 norm of the vector.  
        Features are marked as possibly visible if :math:`\theta` is less than this attribute.

        Typically this angle should be set to a multiple of the half diagonal field of view of the camera, which is what
        the default for this parameter is (if left as ``None``).  The maximum value for this should be 180 degrees 
        (which is unlikely to ever be reached).
        """

        self.gsd_scaling: float = gsd_scaling
        r"""
        The ratio between the camera ground sample distance and the ground sample distance of the feature itself that 
        is allowed for a feature to be considered visible.

        This should be a value greater than or equal to 1.

        For a feature to be considered visible it must meet the requirement of

        .. math::

            \frac{1}{s} <= \frac{g_c}{g_f} <= s

        where :math:`s` is the ``gsd_scaling``, :math:`g_c` is the ground sample distance of the camera computed using 
        :meth:`.CameraModel.compute_ground_sample_distance`, and :math:`g_f` is the ground sample distance of the 
        feature stored as :attr:`.SurfaceFeature.ground_sample_distance`
        """

        self.reflectance_angle_maximum: float = float(reflectance_angle_maximum)
        r"""
        The maximum reflectance angle in degrees for a feature to be considered visible.

        The reflectance angle is defined as the angle between the normal vector for the feature and the unit vector 
        from the feature to the camera center.  It is computed as

        .. math::

            \gamma_r = \cos^{-1}\left(\hat{\mathbf{x}}_r^T\hat{\mathbf{n}}_i\right)

        where :math:`\gamma_r` is the reflectance angle in degrees, :math:`\cos^{-1}` is the arc cosine in degrees,
        :math:`\hat{\mathbf{x}}_{ri}^T` is the unit vector from the :math:`i^{th}` feature to the camera center 
        expressed in the camera frame (computed using :attr:`.FeatureCatalog`.feature_locations`), and 
        :math:`\hat{\mathbf{n}}_i` is the unit normal vector for the :math:`i^{th}` feature in the camera frame.

        Features are marked as possibly visible if :math:`\gamma_r` is less than this attribute.

        Typically, since most features are well approximated by a flat plate, this angle should not exceed 90 degrees as 
        an absolute maximum, which would imply you are viewing the feature completely from the side.
        """

        self.incident_angle_maximum: float = float(incident_angle_maximum)
        r"""
        The maximum incident angle in degrees for a feature to be considered visible.

        The incident angle is defined as the angle between the normal vector for the feature and the unit vector 
        from the feature to the sun.  It is computed as

        .. math::

            \gamma_i = \cos^{-1}\left(\hat{\mathbf{x}}_{ij}^T\hat{\mathbf{n}}_j\right)

        where :math:`\gamma_i` is the incident angle in degrees, :math:`\cos^{-1}` is the arc cosine in degrees,
        :math:`\hat{\mathbf{x}}_{ij}^T` is the unit vector from the :math:`j^{th}` feature to the sun 
        expressed in the camera frame (computed using :attr:`.SceneObject.position` of :attr:`.Scene.light_obj`), and 
        :math:`\hat{\mathbf{n}}_j` is the unit normal vector for the :math:`j^{th}` feature in the camera frame.

        Features are marked as possibly visible if :math:`\gamma_i` is less than this attribute.

        Typically, since most features are well approximated by a flat plate, this angle should not exceed 90 degrees 
        as an absolute maximum, which would imply the feature is illuminated completely from the side.
        """

        self.percent_in_fov: float = float(percent_in_fov)
        """
        The percentage of the predicted feature feature_bounds in the image plane that falls within the FOV of the 
        camera.
        
        This should be a number <= 100.  
        
        The actual percentage of the feature contained in the FOV is computed by 
        
        #. projecting the bounding box vertices of the feature onto the image using 
           :meth:`.CameraModel.project_onto_image` and the :attr:`.FeatureCatalog.feature_bounds` attribute.
        #. determining the axis aligned bounding box in the image by finding the min and max pixels of the projected 
           points
        #. determining the overlap between the AABB of the feature in the image and the AABB of the image 
           ``(0->n_cols, 0->n_rows)``
        #. computing the percent as the overlap area divided by the area of the AABB of the feature in the image.
        
        If this computed percentage is greater than or equal to this attribute, then the feature is possibly visible 
        (depending on further checks).  If it is less than this attribute then the feature is not considered visible and
        no more checks are performed.
        """

        # apply the options from the options structure
        if options is not None:
            self.apply_options(options)

        self._feature_gsds = np.fromiter((f.ground_sample_distance for f in self.feature_catalog.features),
                                         np.float64, count=len(self.feature_catalog.features))
        """
        This private attribute stores the GSD for each feature as a numpy array to make logical indexing easier.
        """

        self._feature_names = np.array([f.name for f in self.feature_catalog.features])
        """
        This private attribute stores the name for each feature as a numpy array to make logical indexing easier.
        """

    def apply_options(self, options: VisibleFeatureFinderOptions):
        """
        This method applies the input options to the current instance.

        The input options should be an instance of :class:`.VisibleFeatureFinderOptions`.

        When calling this method every setting will be updated, even ones you did not specifically set in the provided
        ``options`` input.  Any you did not specifically modify will be reset to the default value.  Typically the best
        way to change a single setting is through direct attribute access on this class, or by maintaining a copy of the
        original options structure used to initialize this class and then updating it before calling this method.

        :param options: The options to apply to the current instance
        """

        self.feature_list = options.feature_list

        if options.off_boresight_angle_maximum is None:
            # compute the maximum fov extent based on the camera field of view
            self.off_boresight_angle_maximum = None
        else:
            self.off_boresight_angle_maximum = float(options.off_boresight_angle_maximum)

        self.gsd_scaling = options.gsd_scaling

        self.reflectance_angle_maximum = float(options.reflectance_angle_maximum)

        self.incident_angle_maximum = float(options.incident_angle_maximum)

        self.percent_in_fov = float(options.percent_in_fov)

    def __call__(self, camera_model: CameraModel, scene: Scene, temperature: float = 0) -> List[int]:
        """
        The call method of this class determines which features are currently visible based off of the current scene
        setup and the provided filter inputs stored in the attributes of this class.

        The visible features are returned as a list of integers that index into the :attr:`.FeatureCatalog.features`
        list and related.

        If a feature is identified as not visible for the current settings, its :attr:`.SurfaceFeature.not_found` method
        is called so we can evaluate if the DEM data should be unloaded.  If a feature is identified as found, its
        :attr:`.SurfaceFeature.found` method is called so the DEM data can be loaded if need be.

        :param temperature: The temperature of the camera at the current time.  Used for projection of points
        :return: A list of integers that specify which features are visible according to the current state of the
                 :attr:`scene`.
        """

        # figure out which target to consider
        target_use = None

        for target in scene.target_objs:
            if target.shape is self.feature_catalog:
                target_use = target

        if target_use is None:
            raise ValueError('Unable to determine which target holds the feature catalog.  Please provide a scene'
                             'which contains the feature catalog in it.')

        # set the off boresight angle maximum if it hasn't been set yet
        if self.off_boresight_angle_maximum is None:
            self.off_boresight_angle_maximum = max(1.5 * camera_model.field_of_view, 180.0)  # restrict to <= 180

        # assume that the boresight of the camera goes through the center of the pixel array
        boresight_pixel = [(camera_model.n_cols-1)/2, (camera_model.n_rows-1)/2]
        boresight_vector = camera_model.pixels_to_unit(boresight_pixel, temperature=temperature)


        # get the unit vectors from the camera to the features
        reflectance_vectors = (self.feature_catalog.feature_locations /
                               np.linalg.norm(self.feature_catalog.feature_locations,
                                              axis=-1, keepdims=True))

        # first check the feature list
        if self.feature_list is not None:
            visible_feature_bool = np.isin(self._feature_names, self.feature_list)
        else:
            visible_feature_bool = np.ones(len(self._feature_names), dtype=bool)

        if not visible_feature_bool.any():
            for feature in self.feature_catalog.features:
                feature.not_found()
            return []

        # second check the boresight vector offset
        visible_feature_bool[visible_feature_bool] = ((reflectance_vectors[visible_feature_bool] @ boresight_vector) >=
                                                      np.cos(np.deg2rad(self.off_boresight_angle_maximum)))
        
        if not visible_feature_bool.any():
            for feature in self.feature_catalog.features:
                feature.not_found()
            return []

        # now check the reflectance angle
        visible_feature_bool[visible_feature_bool] = (
                (-reflectance_vectors[visible_feature_bool] *
                 self.feature_catalog.feature_normals[visible_feature_bool]).sum(axis=-1) >=
                np.cos(np.deg2rad(self.reflectance_angle_maximum)))
        
        if not visible_feature_bool.any():
            for feature in self.feature_catalog.features:
                feature.not_found()
            return []

        # now check the incidence angle
        assert scene.light_obj is not None
        sun_direction = scene.light_obj.position.ravel() - target_use.position.ravel()
        sun_direction /= np.linalg.norm(sun_direction)
        visible_feature_bool[visible_feature_bool] = (
                (self.feature_catalog.feature_normals[visible_feature_bool] @ sun_direction) >=
                np.cos(np.deg2rad(self.incident_angle_maximum))
        )

        # now check the ground sample distance
        gsd_ratio = cast(np.ndarray, camera_model.compute_ground_sample_distance(
            self.feature_catalog.feature_locations[visible_feature_bool].T,
            target_normal=self.feature_catalog.feature_normals[visible_feature_bool].T,
            temperature=temperature
        )).squeeze() / self._feature_gsds[visible_feature_bool]

        # need to mark anything as NAN as valid here because it means we don't have a GSD for that feature
        visible_feature_bool[visible_feature_bool] = np.isnan(gsd_ratio) | ((gsd_ratio <= self.gsd_scaling) &
                                                                            (gsd_ratio >= 1/self.gsd_scaling))
        
        if not visible_feature_bool.any():
            for feature in self.feature_catalog.features:
                feature.not_found()
            return []

        feature_image_bounds = camera_model.project_onto_image(
            np.hstack(self.feature_catalog.feature_bounds[visible_feature_bool]), temperature=temperature  # type: ignore
        ).reshape((2, visible_feature_bool.sum(), -1)).swapaxes(0, 1)

        feature_image_bounds_min = feature_image_bounds.min(axis=-1)
        feature_image_bounds_max = feature_image_bounds.max(axis=-1)

        max_bounds = [camera_model.n_cols, camera_model.n_rows]

        interior_check_min = ((feature_image_bounds_min >= 0) &
                              (feature_image_bounds_min <= max_bounds)).all(axis=-1)
        interior_check_max = ((feature_image_bounds_max >= 0) &
                              (feature_image_bounds_max <= max_bounds)).all(axis=-1)

        # figure out features that are fully visible
        temp_visible_features = interior_check_min & interior_check_max

        # check features that are partially visible
        partially_v_features = (interior_check_min | interior_check_max) & ~temp_visible_features

        # determine the overlap of partially visible features
        overlap = np.maximum(0, np.minimum(feature_image_bounds_max[partially_v_features], max_bounds) -
                             np.maximum(feature_image_bounds_min[partially_v_features], 0))

        area = np.prod(overlap, axis=-1)

        percent_overlap = area/np.prod(feature_image_bounds_max[partially_v_features] -
                                       feature_image_bounds_min[partially_v_features], axis=-1)

        temp_visible_features[partially_v_features] = percent_overlap >= self.percent_in_fov/100.0

        # now update the boolean list
        visible_feature_bool[visible_feature_bool] = temp_visible_features

        # now walk through the list and call found/not_found respectively
        visible_features = []
        for find, feature in enumerate(self.feature_catalog.features):
            if visible_feature_bool[find]:
                visible_features.append(find)
                feature.found()
            else:
                feature.not_found()

        return visible_features
