# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
The opnav_class module provides an OpNav object that serves as the foundation for other high-level user
interface objects throughout GIANT.

Essentially, the OpNav class serves as a container for both a :class:`.Camera` and
:class:`.ImageProcessing` instance, and then provides aliases (in the way of properties) to be able to access a few
of the attributes of these instances directly from the OpNav class instance.

Example
_______

In general, the OpNav class is not used directly in any setups.  Instead, it is used as the super class for other high
level user interface classes, such as :class:`.StellarOpNav` and :class:`.RelativeOpNav`.  For instance, say we want to
create a new high-level interface class called MyAwesomeNewOpNav.  If we subclass the OpNav class when creating this
new class then we automatically get a ``camera`` attribute, a ``image_processing`` attribute, and a few aliases to the
attributes of the camera and image processing instances

    >>> from giant.opnav_class import OpNav
    >>> from giant.camera import Camera
    >>> class MyAwesomeNewOpNav(OpNav):
    ...     def __init__(self, camera, image_processing, image_processing_kwargs):
    ...         super().__init__(camera, image_processing=image_processing,
    ...                          image_processing_kwargs=image_processing_kwargs)
    ...         self.new_attribute = 2
    ...
    >>> inst = MyAwesomeNewOpNav(Camera())
    >>> hasattr(inst, 'camera')
    True
    >>> hasattr(inst, 'image_processing')
    True
"""

from typing import Callable, Union, Iterable, Optional
import warnings

from giant._typing import ARRAY_LIKE_2D, PATH
from giant.image_processing import ImageProcessing
from giant.camera import Camera
from giant.camera_models import CameraModel


class OpNav:
    """
    This serves as a container for :class:`.Camera` and :class:`.ImageProcessing` instances and provides aliases to
    quickly access their attributes from an instance of this class.

    This class is rarely used as is, and instead is used as a super class for new OpNav user interfaces.
    """

    def __init__(self, camera: Camera,
                 image_processing: Optional[ImageProcessing] = None, image_processing_kwargs: Optional[dict] = None):
        """
        :param camera: An instance of :class:`.Camera` that is to have OpNav performed on it
        :param image_processing: An already initialized instance of :class:`.ImageProcessing` (or a subclass).  If not
                                 ``None`` then ``image_processing_kwargs`` are ignored.
        :param image_processing_kwargs: The keyword arguments to pass to the :class:`.ImageProcessing` class
                                        constructor.  These are ignored if argument ``image_processing`` is not ``None``
        """

        self._camera = None
        self.camera = camera

        if image_processing is None:
            if image_processing_kwargs is not None:
                self._image_processing = ImageProcessing(**image_processing_kwargs)
            else:
                self._image_processing = ImageProcessing()
        else:
            self._image_processing = image_processing

        # store the initial image processing key_word_arguments
        self._initial_image_processing_kwargs = image_processing_kwargs

    def __repr__(self) -> str:

        ip_dict = {}
        for key, value in self._image_processing.__dict__.items():
            if not key.startswith("_"):
                ip_dict[key] = value

        return (self.__module__ + "." + self.__class__.__name__ +
                "(" + repr(self._camera) + ", image_processing_kwargs=" + str(ip_dict) + ")")

    def __str__(self) -> str:
        ip_dict = {}
        for key, value in self._image_processing.__dict__.items():
            if isinstance(value, Callable):
                value = value.__module__ + "." + value.__name__

            if not key.startswith("_"):
                ip_dict[key] = value

        return (self.__module__ + "." + self.__class__.__name__ +
                "(" + str(self._camera) + ", image_processing_kwargs=" + str(ip_dict) + ")")

    # ____________________________________________Camera Properties____________________________________________
    @property
    def camera(self) -> Camera:
        """
        The camera instance to perform OpNav on.

        This should be an instance of the :class:`.Camera` class or one of its subclasses.

        See the :class:`.Camera` class documentation for more details
        """

        return self._camera

    @camera.setter
    def camera(self, val):
        if isinstance(val, Camera):
            self._camera = val
        else:
            warnings.warn("The camera should probably be an object that subclasses the Camera class\n"
                          "We'll assume you know what you're doing for now but "
                          "see the Camera documentation for details")
            self._camera = val

    def add_images(self, data: Union[Iterable[Union[PATH, ARRAY_LIKE_2D]], PATH, ARRAY_LIKE_2D],
                   parse_data: bool = True, preprocessor: bool = True):
        """
        This method adds new images to be processed.

        Generally this is an alias to the :meth:`.Camera.add_images` method.  In some implementations, however, this
        method adds some functionality to the original method as well. (such as in the :class:`.StellarOpNav` class)

        See :meth:`.Camera.add_images` for a description of the valid input for `data`

        :param data:  The image data to be stored in the :attr:`.images` list
        :param parse_data:  A flag to specify whether to attempt to parse the metadata automatically for the images
        :param preprocessor: A flag to specify whether to run the preprocessor after loading an image.
        """

        self.camera.add_images(data, parse_data=parse_data, preprocessor=preprocessor)

    @property
    def model(self) -> CameraModel:
        """
        This alias returns the current camera model from the camera attribute.

        It is provided for convenience since the camera model is used frequently.
        """

        return self._camera.model

    @model.setter
    def model(self, val: CameraModel):
        self._camera.model = val

    # ____________________________________________ImageProcessing Aliases____________________________________________

    @property
    def image_processing(self) -> ImageProcessing:
        """
        The ImageProcessing instance to use when doing image processing on the images

        This must be an instance of the :class:`.ImageProcessing` class.

        See the :class:`.ImageProcessing` class documentation for more details
        """

        return self._image_processing

    @image_processing.setter
    def image_processing(self, val):
        if isinstance(val, ImageProcessing):
            self._image_processing = val
        else:
            warnings.warn("The image_processing object should probably subclass the ImageProcessing class\n"
                          "We'll assume you know what you're doing for now, but "
                          "see the ImageProcessing documentation for details")
            self._image_processing = val

    # ____________________________________________METHODS____________________________________________

    def reset_image_processing(self):
        """
        This method replaces the existing image processing instance with a new instance
        using the initial ``image_processing_kwargs`` argument passed to the constructor.

        A new instance of the object is created, therefore there is no backwards reference whatsoever to the state
        before a call to this method.
        """

        if self._initial_image_processing_kwargs is not None:
            self._image_processing = ImageProcessing(**self._initial_image_processing_kwargs)
        else:
            self._image_processing = ImageProcessing()

    def update_image_processing(self, image_processing_update: Optional[dict] = None):
        """
        This method updates the attributes of the :attr:`image_processing` instance.

        See the :class:`.ImageProcessing` class for accepted attribute values.

        If a supplied attribute is not found in the :attr:`image_processing` attribute then this will print a warning
        and ignore the attribute. Any attributes that are not supplied are left alone.

        :param image_processing_update: A dictionary of attribute->value pairs to update the attributes of the
                                        :attr:`image_processing` attribute with.
        """

        if image_processing_update is not None:
            for key, val in image_processing_update.items():
                if hasattr(self._image_processing, key):
                    setattr(self._image_processing, key, val)
                else:
                    warnings.warn("The attribute {0} was not found.\n"
                                  "Cannot update ImageProcessing instance".format(key))
