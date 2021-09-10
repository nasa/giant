# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This module provides scene functionality for rendering in GIANT.

Description
-----------

In GIANT, a scene is used to describe the location and orientation of objects with respect to each other and some
defined frame (usually the camera frame) at a given moment in time.  This then facilitates rendering multiple objects
in a scene, doing single bounce ray tracing to a light source, and other similar tasks. Additionally, the scene makes it
easy to tie functions that specify the location/orientation of an object at given times so that they can be used to
automatically place things in the scene at a requested image time.

Use
---

To use the scene in GIANT you simply create :class:`.SceneObject` instances for any targets and the light source for
your scene (currently only a single light source is allowed).  You then create a :class:`.Scene` around these objects.
This then gives you the ability to call the most commonly used methods of the scene :meth:`.Scene.trace` to trace rays
through all of the targets in the scene and :meth:`.Scene.get_illumination_inputs` to do a single bounce ray trace and
create the inputs required to estimate the intensity for each rendered ray using :mod:`.illumination`.

In general, besides initializing your :class:`.Scene` and :class:`.SceneObject` instances, you won't interact directly
with the scene classes much, as this is done for you in the rest of GIANT.
"""

import datetime
from typing import Callable, Optional, Any, List
import warnings
from typing import Tuple, Union
from enum import Enum
import copy

import numpy as np

from giant.ray_tracer.rays import Rays
from giant.rotations import Rotation, quaternion_inverse, rotvec_to_rotmat
from giant.ray_tracer.shapes import Point, Surface, Ellipsoid, Shape
from giant.ray_tracer.kdtree import KDTree
from giant._typing import Real, ARRAY_LIKE
from giant.camera_models import CameraModel
from giant.image import OpNavImage
from giant.ray_tracer.illumination import IlluminationModel, ILLUM_DTYPE
from giant.ray_tracer.rays import INTERSECT_DTYPE
from giant.ray_tracer.utilities import to_block


SPEED_OF_LIGHT = 299792.458  # km/sec
"""
The speed of light in kilometers per second
"""

_SCAN_DIRS = np.array([[1, 0, -1, 0], [0, 1, 0, -1], [0, 0, 0, 0]])
"""
A set of scan direction vectors for identifying the bounds of the body if supplied with a circumscribing sphere
"""


class CorrectionsType(Enum):
    """
    This enumeration provides options for the different corrections that can be used when calculating the apparent
    position of an object in a scene
    """

    NONE = "none"
    """
    Perform not corrections
    """

    LTPS = "lt+s"
    """
    Perform light time and stellar aberration corrections (default)
    """

    LT = "lt"
    """
    Perform only light time corrections.
    
    Note that this is not recommended because stellar aberration undoes much of the angular change imparted by light
    time only corrections.
    """

    S = "s"
    """
    Perform only stellar aberration corrections.

    Note that this is not recommended because light time undoes much of the angular change imparted by stellar 
    aberration only corrections.
    """


class SceneObject:
    """
    This class provides a quick and easy interface for changing the position and orientation of various objects.

    Essentially, this class adds position and orientation attributes to shapes, rays, KDTrees, and other objects that
    can be translated and rotated.  It also adds optional orientation and position functions, which can be used to
    automatically set the position/orientation for this object at a given time using :meth:`place`.  This makes it much
    easier to perform frame transformations as the current frame's orientation and origin are stored with the object.
    The position and orientation are specified with respect to the
    default fixed frame for the object, which is usually [0, 0, 0] and eye(3) respectively.

    This class also provides 5 methods for updating the objects contained.

    * :meth:`change_position` - changes the position of the origin of the frame the object is expressed in.  This works
                                by first resetting the origin to 0 by subtracting off the previous origin, and then
                                setting the new origin at the specified location.  This is good when you are completely
                                changing the frame the object is expressed in.
    * :meth:`translate` - This is similar to :func:`change_position` but it does not first reset the origin to 0.  This
                          is good for making updates to the frame the object is expressed in.
    * :meth:`change_orientation` - Changes the orientation of the frame the object is expressed in by first
                                   resetting the orientation to be the identity matrix by rotating by the inverse
                                   transformation of the previous orientation, and then rotating to the new orientation
                                   specified.  This is good for when you are completely changing the frame the object is
                                   expressed in.
    * :meth:`rotate` - This is similar to :func:`change_orientation` but it does not first reset the orientation to the
                       identity.  This method is good for making updates to the frame the object is expressed in.
    * :meth:`place` - This automatically updates the orienation and position of the object based on the provided
                      :attr:`orientation_function` and :attr:`position_function`.  If either of these are still ``None``
                      then this will print a warning to the screen and do nothing

    Note that this class does not intelligently relate frames to each other, that must be done by the user.

    The following shows an example use of the SceneObj type.  First, we want to change the frame we have the rays
    expressed in.

    Start by doing your imports and setting up your ray in the camera frame:

        >>> import giant.ray_tracer.rays as g_rays
        >>> import giant.ray_tracer.scene as g_scene
        >>> from giant.rotations import Rotation
        >>> import numpy
        >>> ray = g_rays.Rays([0, 0, 0], [-1, 0, 0])

    Now, lets define a position and orientation function.  Here for demo purposes we'll just return constant values,
    though most commonly these are generated as spice function calls

        >>> def position_function(date):
        ...     return numpy.array([-5, 6, 7])
        >>> def orientation_function(date):
        ...     return Rotation([0.5, 2, -3.2])

    Now we can create our SceneObj around the ray:

        >>> scene_obj = g_scene.SceneObject(ray, position_function=position_function,
        ...                                 orientation_function=orientation_function)

    Note here, that we are defining the default frame for the ray to be the current frame that it is in (the camera
    frame).  This means that whenever we want to **change** the frame (not update) for the rays we need to specify the
    new origin and orientation with respect to the camera frame.  Let's change our rays to some new frame:

        >>> new_loc = [1, 2, 3]  # the new origin location expressed in the original frame
        >>> new_orientation = [4, 5, 6]  # the new orientation of the new frame with respect to the original frame

    First, we'll update the origin of the frame (we are assuming that new_loc is expressed in the camera frame here):

        >>> scene_obj.change_position(new_loc)
        >>> print(scene_obj.shape.start)
        [ 1.  2.  3.]
        >>> print(scene_obj.shape.direction)
        [ -1.  0.  0.]

    Now, we'll change the orientation:

        >>> scene_obj.change_orientation(new_orientation)
        >>> print(scene_obj.shape.start)
        [ 1.98283723  2.55366624  1.88338665]
        >>> print(scene_obj.shape.direction)
        [ 0.42296095 -0.05284171 -0.90460588]

    Now lets say we want to update the current frame our ray is expressed in:

        >>> update_pos = [0.05, -1, 0.2]  # where we want to move the current origin to expressed in the current frame
        >>> scene_obj.translate(update_pos)
        >>> print(scene_obj.shape.start)
        [ 2.03283723  1.55366624  2.08338665]
        >>> print(scene_obj.shape.direction)
        [ 0.42296095 -0.05284171 -0.90460588

    Note how this was added to the current frame, not reset back from the original frame:

        >>> update_orientation = [0.001, 0.001, 0.001]
        >>> scene_obj.rotate(update_orientation)
        >>> print(scene_obj.shape.start)
        [ 2.0323073   1.55371729  2.08386553]
        >>> print(scene_obj.shape.direction)
        [ 0.42381181 -0.05416946 -0.90412898]

    Now, lets use the position function and orientation function we provided to update the rays position/orientation
    automatically

        >>> scene_obj.place(datetime.datetime.utcnow())
        >>> print(scene_obj.shape.start)
        [-5.  6.  7.]
        >>> print(scene_obj.shape.direction)
        [ 0.75609837 -0.64204013 -0.12688471]

    And finally, lets say that we want to return our ray to the initial frame:

        >>> scene_obj.change_position([0, 0, 0])
        >>> scene_obj.change_orientation(numpy.eye(3))
        >>> print(scene_obj.shape.start)
        [ 0.  0.  0.]
        >>> print(scene_obj.shape.direction)
        [ -1.  0.  0.]

    Typically when using the :attr:`position_function` and :attr:`orientation_function` these should give the position
    of the object relative to the solar system bary center in the inertial frame and the rotation from the object fixed
    frame to the inertial frame respectively.  This will then pair well with updating the scene to put everying in the
    camera frame if you also put the inertial camera position and rotation from the intertial frame to the camera frame
    on the :class:`.OpNavImage` as is typical.  You can conceivably work differently than this but it is then up to you
    to ensure all of your definitions are consistent.

    In addition to making it easy to move objects around in the scene, the :class:`.SceneObject` class also provides
    some useful methods for getting information about the object in the current scene.  This includes
    :meth:`get_bounding_pixels` which determines the extent of the object in the image (this only works once the object
    has been rotated/translated into the camera frame) and :meth:`get_apparent_diamter` which predicts the apparent
    diameter of the object in pixels in the image (this also only works once the object
    has been rotated/translated into the camera frame).
    """

    def __init__(self, shape: Shape,
                 current_position: Optional[ARRAY_LIKE] = None,
                 current_orientation: Optional[Union[Rotation, ARRAY_LIKE]] = None,
                 name: str = 'object',
                 position_function: Optional[Callable[[datetime], np.ndarray]] = None,
                 orientation_function: Optional[Callable[[datetime], Rotation]] = None,
                 corrections: Optional[CorrectionsType] = CorrectionsType.LTPS):
        """
        :param shape: The shape that represents the object.  This is typically a subclass of :class:`.Shape`, but can be
                      anything so long as it implements ``translate``, ``rotate``, and ``trace`` methods.
        :param current_position: The current position of the object in the current frame.  If ``None`` then this will be
                                 assumed to be the origin.
        :param current_orientation: The current orientation of the object in the current frame.  If ``None`` then this
                                    will be assumed to be the identity rotation
        :param name: An identifying name for the object.  This is simply for logging/readability purposes
        :param position_function: A function which accepts a python datetime object and returns a 3 element array giving
                                  the position of the object at the requested time.  usually this should return the
                                  position of the object with respect to the solar system bary center in the inertial
                                  frame.  While this is not required, it is strongly encouraged in most cases
        :param orientation_function: A function which accepts a python datetime object and returns a :class:`.Rotation`
                                     that gives the orientation of the object at the requested time.  Usually this
                                     should return the rotation from the object fixed frame to the inertial frame.
                                     While this is not required, it is strongly encouraged in most cases
        :param corrections: What corrections to apply when calculating the apparent location of the object in the camera
                            frame.  This should either be ``None`` for no corrections, or one of the enums from
                            :data:`.CorrectionsType`, most typically :attr:`.CorrectionsTyps.LTPS` which applies light
                            time and stellar aberration corrections.  This is used by :class:`.Scene` and is only used
                            when the :attr:`position_function` and :attr:`orientation_function` are not ``None``.
        """

        self.position_function: Optional[Callable[[datetime], np.ndarray]] = position_function
        """
        A function which accepts a python datetime object and returns a 3 element array giving
        the position of the object at the requested time.  
        
        Usually this should return the position of the object with respect to the solar system bary center in the 
        inertial frame.  While this is not required, it is strongly encouraged in most cases
        """

        self.orientation_function: Optional[Callable[[datetime], Rotation]] = orientation_function
        """
       A function which accepts a python datetime object and returns a :class:`.Rotation`
       that gives the orientation of the object at the requested time.  
       
       Usually this should return the rotation from the object fixed frame to the inertial frame.
       While this is not required, it is strongly encouraged in most cases
       """

        self.corrections: Optional[CorrectionsType] = corrections
        """
        What corrections to apply when calculating the apparent location of the object in the camera frame.  
        
        This should either be ``None`` for no corrections, or one of the enums from
        :data:`.CorrectionsType`, most typically :attr:`.CorrectionsTyps.LTPS` which applies light
        time and stellar aberration corrections.  This is used by :class:`.Scene` and is only used
        when the :attr:`position_function` and :attr:`orientation_function` are not ``None``. 
        """

        self._shape = None
        self.shape = shape

        self._position = None
        if current_position is not None:
            self.position = current_position
        else:
            self.position = np.zeros(3, dtype=np.float64)

        self._orientation = None
        if current_orientation is not None:
            self.orientation = current_orientation
        else:
            self.orientation = Rotation([0, 0, 0, 1])

        self.name = name
        """
        The name of the object, used for logging purposes and readability.
        """

    @property
    def shape(self) -> Union[Shape, Any]:
        """
        This is the shape of interest.

        It is usually a :class:`.Shape` or object but the only requirement is that it have
        translate and rotate methods.  Ideally it should also have a trace method, though this isn't checked.
        """

        return self._shape

    @shape.setter
    def shape(self, val):

        if hasattr(val, "translate") and hasattr(val, "rotate"):
            self._shape = val

        else:
            raise AttributeError("The object must have translate and rotate attributes\n")

    @property
    def position(self) -> np.ndarray:
        """
        This is the current position of this object as a flat length 3 array
        """

        return self._position

    @position.setter
    def position(self, val: ARRAY_LIKE):

        if np.size(val) == 3:

            self._position = np.asarray(val).ravel().astype(np.float64)

        else:

            raise ValueError('The position array must be of length three')

    @property
    def orientation(self) -> Rotation:
        """
        This is the current orientation of the frame this object is expressed in as a :class:`.Rotation` object
        """

        return self._orientation

    @orientation.setter
    def orientation(self, val: Union[Rotation, ARRAY_LIKE]):

        if isinstance(val, Rotation):

            self._orientation = val

        else:

            # error checking is handled by Rotation
            self._orientation = Rotation(val)

    def change_position(self, new_position: ARRAY_LIKE):
        """
        Change the location of the object in the current frame.

        This is done by first subtracting off the old position then adding this value.

        To update the location instead of changing it see the :meth:`translate` method.

        :param new_position: The new location for the object in the frame
        """

        old_position = self.position

        # error checking is handled by the position property setter
        self.position = new_position

        self.shape.translate(-old_position)

        self.shape.translate(self.position)

    def change_orientation(self, new_orientation: Union[Rotation, ARRAY_LIKE]):
        """
        Change the orientation of the frame the object is expressed in.

        This is done by first applying the inverse rotation of the current orientation to get back to the base frame,
        and then applying the new orientation.  Note that we also update the orientation for the position vector in the
        same way.

        :param new_orientation: the new frame orientation to express the object and location in
        """

        previous_orientation = copy.copy(self.orientation)

        if isinstance(new_orientation, Rotation):
            self.orientation = new_orientation
        else:
            self.orientation = Rotation(new_orientation)

        self.shape.rotate(previous_orientation.inv())

        self.shape.rotate(new_orientation)

        # update the orientation of the position
        self.position = np.matmul(quaternion_inverse(previous_orientation).matrix, self.position)

        self.position = np.matmul(self.orientation.matrix, self.position)

    def translate(self, translation: ARRAY_LIKE):
        """
        Update the location of the object in the current frame by adding a vector to the current location.

        :param translation: The vector to add to the current vector
        """

        if np.size(translation) == 3:
            trans_array = np.asarray(translation).astype(np.float64)

            self.position += trans_array.ravel()

            self.shape.translate(trans_array)

        else:

            raise ValueError("You have entered an improperly sized translation.\n"
                             "Only length 3 translations are allowed.\n"
                             "You entered a translation of length {0}".format(len(translation)))

    def rotate(self, rotation: Union[Rotation, ARRAY_LIKE]):
        """
        Update the orientation of the frame the object and its location are expressed in.

        :param rotation: how we are to rotate the current frame/location
        """

        self.shape.rotate(rotation)

        if not isinstance(rotation, Rotation):
            rotation = Rotation(rotation)

        self.orientation.rotate(rotation)

        # update the orientation of the position
        self.position = np.matmul(rotation.matrix, self.position)

    def get_bounding_pixels(self, model: CameraModel, image: int = 0,
                            temperature: Real = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the bounding pixels for the given camera model and the current location.

        The bounding pixels are the pixel bounds which the scene predicts the target should be completely contained in.
        They are computed either by using the :attr:`.circumscribing_sphere` bounding sphere of the object if it is not
        ``None``, by make a circumscribing sphere if the object is an ellipsoid, or by projecting the bounding box
        vertices onto the image.  These are typically used to determine which pixels to trace for rendering the object

        This method assumes that the object has already been placed in the camera frame (the frame centered at the focus
        of the camera) with the z axis pointing perpendicular to the image plane.  If any of the assumptions are not met
        this method will return nonsense.

        :param model: The camera model that relates points in the 3d world to points on the image plane
        :param image: the index of the image that is being projected onto.  Can normally be ignored
        :param temperature: The temperature of the camera at the time we are trying to get the apparent diameter
        :return: The minimum (upper left) and maximum (lower right) pixel bounds that should contain the object based
                 off of the scene as a tuple of length 2 numpy arrays (min, max).  Note that these are inclusive bounds
        """

        if isinstance(self._shape, Point):
            center = model.project_onto_image(self._position.ravel(), temperature=temperature, image=image).ravel()
            return center-1, center+1

        elif isinstance(self._shape, Ellipsoid):
            cicum_sphere = Ellipsoid(self._position.ravel(),
                                     principal_axes=np.array([self._shape.principal_axes.max()] * 3))

            limbs = cicum_sphere.find_limbs(self._position.ravel()/np.linalg.norm(self._position), _SCAN_DIRS)

            image_locs = model.project_onto_image(limbs, image=image,
                                                  temperature=temperature)

        elif (hasattr(self._shape, 'circumscribing_sphere') and
              (getattr(self._shape, 'circumscribing_sphere') is not None)):
            limbs = self._shape.circumscribing_sphere.find_limbs(self._position.ravel() /
                                                                 np.linalg.norm(self._position),
                                                                 _SCAN_DIRS)

            image_locs = model.project_onto_image(limbs, image=image,
                                                  temperature=temperature)

        else:
            image_locs = model.project_onto_image(self._shape.bounding_box.vertices, image=image,
                                                  temperature=temperature)

        # noinspection PyArgumentList
        return image_locs.min(axis=-1), image_locs.max(axis=-1)

    def get_apparent_diameter(self, model: CameraModel, image: int = 0, temperature: Real = 0.0) -> float:
        """
        Computes the apparent diameter for the given camera model and the current location.

        The apparent diameter is the be guess at how large the object should appear in the image based on the current
        scene and assuming a spherical approximation for the target. It is computed either by making a reference sphere
        by taking the mean of the principal axes of the :attr:`.Surface.reference_ellipsoid` best fit ellipsoid of the
        object if it is not ``None``, by making a reference sphere by taking the mean of the principal axes if the
        object is an ellipsoid, or by maxing a reference sphere using the average radius of the bounding box vertices of
        the target.

        This method assumes that the object has already be placed in the camera frame (the frame centered at the focus
        of the camera) with the z axis pointing perpendicular to the image plane.  If any of the assumptions are not met
        this method will return nonsense.

        :param model: The camera model that relates points in the 3d world to points on the image plane
        :param image: the index of the image that is being projected onto.  Can normally be ignored
        :param temperature: The temperature of the camera at the time we are trying to get the apparent diameter
        :return: The approximate apparent diameter of the object in pixels.  Note that if the contained object is a
                 :class:`.Point` this will always return 0.
        """

        if isinstance(self._shape, Point):
            return 0.0

        elif isinstance(self._shape, Ellipsoid):

            ref_sphere = Ellipsoid(self._position.ravel(),
                                   principal_axes=np.array([self._shape.principal_axes.mean()] * 3))

        elif getattr(self._shape, 'reference_ellipsoid') is not None:
            ref_sphere = Ellipsoid(self._position.ravel(),
                                   principal_axes=np.array([self._shape.reference_ellipsoid.principal_axes.mean()] * 3))

        else:
            # we really should never get here, but if we do, this is probably a poor approximation..
            ref_sphere = Ellipsoid(self._position.ravel(),
                                   np.array([(self._shape.bounding_box.max_sides -
                                              self._shape.bounding_box.min_sides).mean() / 2] * 3))

        # get the limbs from the reference sphere
        limbs = ref_sphere.find_limbs(self.position.ravel() / np.linalg.norm(self.position),
                                      _SCAN_DIRS)

        # project the limbs onto the image
        image_locs = model.project_onto_image(limbs, image=image,
                                              temperature=temperature)

        # get the norm of the extent of the projected limbs to be the apparent diameter
        return np.linalg.norm(image_locs.T.reshape(-1, 2, 1) - image_locs.reshape(1, 2, -1), axis=1).max(initial=0)

    def place(self, date: datetime):
        """
        Place the object using the :attr:`orientation_function` and :attr:`position_function` at the requested date.

        This is done by calling the :attr:`orientation_function` and :attr:`position_function` with the input date and
        then calling methods :meth:`change_orientation` and :meth:`change_position` in order with the results from the
        function calls.  If either of the attributes are still None this method will print a warning and do nothing.

        :param date: The date we are to place the object at
        """

        if self.orientation_function is not None:
            new_orientation = self.orientation_function(date)

            self.change_orientation(new_orientation)
        else:
            warnings.warn("Attempted to place a SceneObject without an orientation_function")
        if self.position_function is not None:
            new_position = self.position_function(date)
            self.change_position(new_position)
        else:
            warnings.warn("Attempted to place a SceneObject without a position_function")


class Scene:
    """
    This is a container for :class:`SceneObject` instances that provides an easy interface for tracing and rendering.

    This is most useful when you have multiple objects that you want to trace at once, as the scene will trace all of
    the objects for you, and choose the first intersection for each ray between all of the objects (if desired).

    This is also useful because it can calculate the apparent location of objects in the scene (if they have the
    :attr:`.position_function` and :attr:`.orientation_function` attributes defined) while applying corrections for
    light time and stellar aberration.  This can be done for all objects in a scene using method :meth:`update` or for
    individual object using :meth:`calculate_apparent_position`.  Any objects that do not define the mentioned
    attributes will likely not be placed correctly in the scene when using these methods and thus warnings will be
    printed.
    """

    def __init__(self, target_objs: Optional[Union[List[SceneObject], SceneObject]] = None,
                 light_obj: Optional[SceneObject] = None,
                 obscuring_objs: Optional[List[SceneObject]] = None):
        """
        :param target_objs: The objects that are to be traced/rendered in a scene as a list of :class:`.SceneObject`.
        :param light_obj: The light object.  This is just used to track the position of the light, therefore it is
                          typically just a wrapper around a :class:`.Point`
        :param obscuring_objs: A list of objects that shouldn't be rendered but may be used externally from the scene
                               to identify whether targets are visible or not.
        """

        self.order: int = -1
        """
        The number of digits that are required to represent the unique id of all objects contained in the scene
        
        This is used to comprise the facet number in the :data:`.INTERSECT_DTYPE` when tracing the scene such that the 
        resulting id is
        
        .. code::
        
            [-target index-][---surface id---]
                            [------order-----]
                            
        where ``target index`` is the index into the :attr:`target_objs` list, ``surface id`` is the ID returned by the 
        surface that the ray struck encoded in ``order`` digits (zero padded on the left).  This is also used when 
        tracing to determine whether the :attr:`Rays.ignore` attribute applies to the object currently being traced.
        
        Typically users shouldn't need to worry about this too much since it is primarily handled entirely internal to 
        the class
        """

        self._target_objs = None
        self.target_objs = target_objs

        self._obscuring_objs = None
        self.obscuring_objs = obscuring_objs

        self._light_obj = None
        self.light_obj = light_obj


    @property
    def target_objs(self) -> Optional[List[SceneObject]]:
        """
        A list of objects to be tracked/rendered in the scene.

        This must be set before a call to :meth:`get_illumination_inputs`
        """
        return self._target_objs

    @target_objs.setter
    def target_objs(self, val: Optional[Union[SceneObject, List[SceneObject]]]):

        if isinstance(val, (np.ndarray, list, tuple)):

            self._target_objs = list(val)

        elif isinstance(val, SceneObject):
            self._target_objs = [val]

        elif val is None:
            self._target_objs = []

        else:
            raise ValueError("The target_objs property must be set as a list or SceneObject.\n")

        # update the order
        for obj in self.target_objs:

            if hasattr(obj.shape, "order"):

                self.order = max(self.order, obj.shape.order + 1)

            elif hasattr(obj.shape, "num_faces"):

                self.order = max(self.order, int(np.log10(obj.shape.num_faces)))

            elif hasattr(obj.shape, "id"):
                self.order = max(self.order, int(np.log10(obj.shape.id)) + 1)

    @property
    def obscuring_objs(self) -> Optional[List[SceneObject]]:
        """
        A list of objects to be kept up to date with the scene but which are not actually used in the scene
        """
        return self._obscuring_objs

    @obscuring_objs.setter
    def obscuring_objs(self, val: Optional[Union[SceneObject, List[SceneObject]]]):

        if isinstance(val, (np.ndarray, list, tuple)):

            self._obscuring_objs = list(val)

        elif isinstance(val, SceneObject):
            self._obscuring_objs = [val]

        elif val is None:
            self._obscuring_objs = []

        else:
            raise ValueError("The target_objs property must be set as a list or SceneObject.\n")

    @property
    def light_obj(self) -> Optional[SceneObject]:
        """
        An object describing the location of the light source in the scene.

        This must be set before a call to :meth:`get_illumination_inputs`
        """

        return self._light_obj

    @light_obj.setter
    def light_obj(self, val):

        if isinstance(val, SceneObject):
            self._light_obj = val

        elif val is None:
            self._light_obj = val

        else:
            raise ValueError("The light_obj property must be set as a SceneObject or None.\n")

    def phase_angle(self, target_index: int) -> float:
        r"""
        This method computes the phase angle between the observer, the target at ``target_index`` and the
        :attr:`light_obj`.

        The phase angle is define as the interior angle between the vector from the target to the light
        and the vector from the target to the observer.  The phase angle is computed as the arccos of the
        dot product of the unit vectors in these two directions, and is returned with units of radians.

        This method assumes that the scene has already been put in the observer frame,
        that is, the observer is located at the origin and the :attr:`light_obj` and the target
        are also defined with respect to that origin.

        The phase angle will always be between 0 and :math:`\pi` radians (0-180 degrees)

        :param target_index: the index into the :attr:`target_objs` list for which to compute the phase angle for
        :return: the phase angle in radians
        """

        # get the unit vector from the target to the light
        line_of_sight_light = self.light_obj.position.ravel() - self._target_objs[target_index].position.ravel()
        line_of_sight_light /= np.linalg.norm(line_of_sight_light)

        # get the unit vector from the target to the observer
        line_of_sight_observer = (-self._target_objs[target_index].position.ravel() /
                                  np.linalg.norm(self._target_objs[target_index].position))

        return np.arccos(line_of_sight_light@line_of_sight_observer)

    def trace(self, trace_rays: Rays) -> np.ndarray:
        """
        Trace trace_rays through the current scene and return the intersections with the objects in the scene for each
        ray (optionally only the first intersection for each ray).

        This method iterates through each object in the :attr:`target_objs` property and traces the ray through that
        object using the object's trace method.  The results from the tracing of each object are then reduced to only
        the first intersection for each ray.

        This method handles determining whether things are to be ignored for each target, as well as updating the
        ``facet`` component of the return to have the appropriate id (encoding the target number at the head of the id).

        This method returns a numpy array of shape (n,) where n is the number of rays traced with dtype
        :data:`.INTERSECT_DTYPE`.

        :param trace_rays:  The rays to be traced through the scene
        :return: a numpy structured array of the intersections for each ray.
        """

        results = []

        for ind, target in enumerate(self.target_objs):

            ignore_inds = copy.deepcopy(trace_rays.ignore)

            if ignore_inds is not None:

                ignore_inds = to_block(ignore_inds)

                ignore_inds[ignore_inds // (10 ** (self.order + 1)) != ind] = -1
                ignore_inds[ignore_inds // (10 ** (self.order + 1)) == ind] = (
                    ignore_inds[ignore_inds // (10 ** (self.order + 1)) == ind].astype(np.longfloat) %
                    (10 ** (self.order + 1))
                ).astype(np.int64)

                ray_use = copy.deepcopy(trace_rays)

                ray_use.ignore = ignore_inds

            else:
                ray_use = trace_rays

            object_results = target.shape.trace(ray_use)

            object_results["facet"][object_results["check"]] += ind * (10 ** (self.order + 1))

            results.append(object_results)

        results = np.asarray(results, dtype=INTERSECT_DTYPE)

        return self.get_first(results, trace_rays)

    def get_illumination_inputs(self, trace_rays: Rays,
                                return_intersects: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        This method returns the required inputs for an illumination function to compute the illumination for each ray.

        This is done by performing a single bounce ray trace against all objects in the scene.  First, the rays as
        provided are traced into the scene, returning the first intersect with anything in :attr:`target_objs`.  Then,
        if the ray actually struck something, we trace from the intersect point toward the :attr:`light_obj` to see if
        the place we struck was shadowed or not.  Presuming it was not shadowed, the geometry of the single bounce ray
        trace is encoded into a numpy structured array with dtype :data:`.ILLUM_DTYPE` which can then be passed to the
        classes from the :mod:`.illumination` module to compute the intensity for each ray.

        If requested, this method also returns the results of the initial intersect (before the shadow bounce) as a
        structured numpy array with dtype :data:`.INTERSECT_DTYPE` which can be useful for determining if a ray didn't
        see anything because it was shadowed or because it didn't strike anything.  Both returns will be shape (n,)
        where n is the number of rays traced.

        :param trace_rays: The rays we are to compute the illumination inputs for
        :param return_intersects: A flag specifying whether the results of the initial trace should also be returned
        :return: A numpy array with shape (n,) and data type :data:`.ILLUM_DTYPE` and optionally a numpy array with
                 shape (n,) and data type :data:`.INTERSECT_DTYPE`
        """

        if (self.light_obj is None) or (not self.target_objs):
            raise ValueError("both light_obj and target_objs must be set to get the illumination inputs")

        # trace through the scene to see if the rays hit anything
        initial_intersect = self.trace(trace_rays)

        if not initial_intersect["check"].any():
            illum_params = np.zeros((trace_rays.num_rays,), dtype=ILLUM_DTYPE)
            illum_params[:] = (None, None, None, None, False)

            if return_intersects:
                return illum_params, initial_intersect
            else:
                return illum_params

        shadow_start = initial_intersect[initial_intersect["check"]]["intersect"]

        shadow_dir = self.light_obj.shape.location.flatten() - shadow_start

        shadow_dir /= np.linalg.norm(shadow_dir, axis=-1, keepdims=True)

        shadow_start = shadow_start + shadow_dir * (1e-15*np.linalg.norm(shadow_start, axis=-1, keepdims=True))

        shadow_rays = Rays(shadow_start.T, shadow_dir.T,
                           ignore=initial_intersect[initial_intersect["check"]]["facet"])

        shadow_check = self.trace(shadow_rays)

        illum_params = np.zeros((trace_rays.num_rays,), dtype=ILLUM_DTYPE)

        check = np.atleast_1d(initial_intersect["check"].copy())

        check[check] = ~shadow_check["check"]

        if shadow_rays.num_rays == 1:

            if trace_rays.num_rays == 1:
                if check[0]:
                    illum_params[0] = (-np.atleast_2d(shadow_rays.direction.T),
                                       -np.atleast_2d(trace_rays.direction.T),
                                       np.atleast_2d(initial_intersect[check]["normal"]),
                                       np.atleast_1d(initial_intersect[check]["albedo"]),
                                       np.atleast_1d(check))

            else:
                illum_params[check] = list(zip(-np.atleast_2d(shadow_rays.direction.T),
                                               -np.atleast_2d(trace_rays[check].direction.T),
                                               np.atleast_2d(initial_intersect[check]["normal"]),
                                               np.atleast_1d(initial_intersect[check]["albedo"]),
                                               np.atleast_1d(check[check])))

        else:
            illum_params[check] = list(zip(-np.atleast_2d(shadow_rays[~shadow_check["check"].squeeze()].direction.T),
                                           -np.atleast_2d(trace_rays[check].direction.T),
                                           np.atleast_2d(initial_intersect[check]["normal"]),
                                           np.atleast_1d(initial_intersect[check]["albedo"]),
                                           np.atleast_1d(check[check])))

        illum_params[~check] = (None, None, None, None, False)

        if return_intersects:
            return illum_params, initial_intersect
        else:
            return illum_params

    @staticmethod
    def get_first(res: np.ndarray, traced_rays: Rays) -> np.ndarray:
        """
        This static method identifies the first intersection for each ray when there are more than 1 object in the
        scene.

        Each object in the scene is responsible for identifying the first intersection with itself for each ray.
        The scene is then responsible for identify which object was struck first which is handled by this method.

        This method works by considering the intersection for each ray with each object, and then finding the
        the intersection with the minimum distance between the ray and the camera.

        :param res: The first intersection for each ray with each :attr:`target_objs` in the scene as a numpy array with
                    dtype :data:`.INTERSECT_DTYPE` and shape (k,n) where k is ``len(self.target_objs)`` and n is the
                    number of rays
        :param traced_rays: The rays that these results pertain to
        :return: numpy array of shape (n) with the first intersect for each ray and data type :data:`.INTERSECT_DTYPE`
        """

        if res.shape[0] == 1:
            return res[0]

        # nan_check = ~np.isnan(res["albedo"]).all(axis=0).squeeze()
        nan_check = res["check"].any(axis=0).squeeze()

        if not np.any(nan_check):
            return res[0]

        min_ind = np.zeros(res.shape[1], dtype=int)

        min_ind[nan_check] = np.nanargmin(np.linalg.norm(traced_rays[nan_check].start.T -
                                                         res["intersect"][:, nan_check], axis=-1), axis=0)

        return res[min_ind, np.arange(min_ind.size)]

    def raster_render(self, target_ind: int,
                      illumination_model: Optional[IlluminationModel] = None) -> Tuple[np.ndarray,
                                                                                       np.ndarray,
                                                                                       np.ndarray]:
        """
        Computes the geometry for rendering a target using rasterization, not ray tracing.

        This is done by simply considering the geometry for each facet of a target based solely on the
        incidence/exidence/normal vectors for the facet.  This therefore does no occlusion or shadowing.

        This assumes that the scene has already been placed in the camera frame.

        The return is the numpy array with a dtype of :attr:`.ILLUM_DTYPE` that can be provided to an
        :mod:`.illumination` model and the center of each facet.

        This is experimental and probably shouldn't be used much.

        :raises ValueError: if the target is not represented by a tesselation (Surface)

        :param target_ind: The target index we are to render using rasterization
        :param illumination_model: The model we are using to convert the geometry into illumination values.  If this is
                                   not None then the illumination values are returned, rather than the illumination
                                   inputs.
        :return: Either the illumination inputs, the center of each facet, and the vertices of each facet in the camera
                 frame if ``illumination_model`` is ``None`` or the illumination values, the center of each facet
                 and the vertices of each facet in the camera frame if ``illumination_model`` is not ``None``.
        """

        target = self._target_objs[target_ind]

        if isinstance(target.shape, KDTree):
            shape = target.shape.shapes

        elif not isinstance(target.shape, Surface):
            raise ValueError('Unable to rasterize targets that are not a surface')
        else:
            shape = target.shape

        # grab the normal vectors from the triangles in the camera frame
        normals = shape.normals @ target.orientation.matrix.T

        facets_camera = target.orientation.matrix@shape.stacked_vertices + target.position.reshape(1, 3, 1)

        # get the centers for each triangle
        centers = facets_camera.mean(axis=-1)

        # get the incidence vectors to each facet
        incidence = centers - self._light_obj.position.reshape(1, 3)
        incidence /= np.linalg.norm(incidence, axis=-1, keepdims=True)

        # get the exidence vectors from each facet
        exidence = centers
        exidence /= np.linalg.norm(centers, axis=-1, keepdims=True)

        illum_inp = np.empty(centers.shape[0], dtype=ILLUM_DTYPE)

        illum_inp['exidence'] = -exidence
        illum_inp['incidence'] = incidence
        illum_inp['albedo'] = 1  # todo: use the albedo for the centers?
        illum_inp['normal'] = normals
        illum_inp['visible'] = True

        if illumination_model is None:

            return illum_inp, centers, facets_camera

        illum_values = illumination_model(illum_inp)

        return illum_values, centers, facets_camera

    def update(self, image: OpNavImage, corrections: Optional[CorrectionsType] = None):
        """
        This method changes the scene to reflect the time specified by image optionally using corrections of
        "LT" (light time), "S" (stellar aberration", or "LTPS" (light time and aberration).

        :param image: The image from the camera you want to update the scene for as an :class:`.OpNavImage`
        :param corrections: A flag specifying which corrections to use or ``None`` to use the corrections specified for
                            each object in the scene.
        """

        # place the targets in the camera frame, optionally correcting for light time and stellar aberration
        for target in self.target_objs:
            if corrections is None:
                use_corrections = getattr(target, 'corrections', CorrectionsType.LTPS)
            else:
                use_corrections = corrections
            self.calculate_apparent_position(target, image, use_corrections)
        if self.obscuring_objs is not None:
            for obs in self.obscuring_objs:
                if corrections is None:
                    use_corrections = getattr(obs, 'corrections', CorrectionsType.LTPS)
                else:
                    use_corrections = corrections
                self.calculate_apparent_position(obs, image, use_corrections)

        if corrections is None:
            use_corrections = getattr(self.light_obj, 'corrections', CorrectionsType.LTPS)
        else:
            use_corrections = corrections
        self.calculate_apparent_position(self.light_obj, image, use_corrections)

    @staticmethod
    def calculate_apparent_position(target: SceneObject, image: OpNavImage,
                                    corrections: Optional[CorrectionsType]):
        """
        This method calculates the apparent position of objects in the scene in the camera frame optionally correcting
        for "LT" (light time), "S" (stellar aberration", or "LTPS" (light time and aberration).  Corrections can only be
        applied to scene objects with position and orientation functions defined

        The target must be either specified in the inertial frame at the time of the image already or have
        a position function that places the object inertially when given a datetime

        Note that this method will update the target location in place to be in the camera frame instead of the inertial
        frame.

        :param target: The target to place in the camera frame.  Generally should be an SceneObject
        :param image: The index of the image from the camera you want to place to object in
        :param corrections: The corrections to be applied when calculating the apparent position (only applied to
                            scene objects with position and orientation functions defined)
        """

        # place the target inertially at the time specified for the image
        date = image.observation_date

        try:
            # target.place(observation_date)
            target.change_orientation(target.orientation_function(date))

            if (corrections is None) or (CorrectionsType.NONE == corrections):

                camera_to_target_inertial = target.position_function(date) - image.position

            elif CorrectionsType.LTPS == corrections:

                camera_to_target_inertial = correct_light_time(target.position_function,
                                                               image.position, date)

                camera_to_target_inertial = correct_stellar_aberration(
                    camera_to_target_inertial, image.velocity.ravel()
                )

            elif CorrectionsType.LT == corrections:

                camera_to_target_inertial = correct_light_time(target.position_function,
                                                               image.position, date)

            elif CorrectionsType.S == corrections.lower():

                camera_to_target_inertial = image.position - target.position_function(date)

                camera_to_target_inertial = correct_stellar_aberration(camera_to_target_inertial,
                                                                       image.velocity)

            else:
                warnings.warn("not sure what you entered for corrections but we don't know how to do it so we're not"
                              "doing anything")
                camera_to_target_inertial = target.position_function(date) - image.position

        except AttributeError:
            warnings.warn('You have specified a standard SceneObject which cannot be automatically placed\n'
                          'We will assume that you know what you are doing and you have correctly placed this object\n'
                          'for the current image but remember that you must update the inertial location of this \n'
                          'object yourself whenever you change an image')

            camera_to_target_inertial = target.position - image.position

        # change the position of the target in inertial space
        target.change_position(camera_to_target_inertial)

        # rotate the target into the camera frame
        target.rotate(image.rotation_inertial_to_camera)


def correct_light_time(target_location_inertial: Callable[[datetime], np.ndarray],
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

        time_of_flight_new = np.linalg.norm(camera_location_inertial - target_location_reflect) / SPEED_OF_LIGHT

        if np.all((time_of_flight_new - time_of_flight) < 1e-8):
            time_of_flight = time_of_flight_new
            break

        time_of_flight = time_of_flight_new

    return (target_location_inertial(time - datetime.timedelta(seconds=time_of_flight)).ravel() -
            camera_location_inertial)


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
