


"""
The opnav_class module provides an OpNav object that serves as the foundation for other high-level user
interface objects throughout GIANT.

Essentially, the OpNav class serves as a container for a :class:`.Camera` and then provides aliases (in the way of 
properties) to be able to access a few of the attributes of this instances directly from the OpNav class instance.

Example
_______

In general, the OpNav class is not used directly in any setups.  Instead, it is used as the super class for other high
level user interface classes, such as :class:`.StellarOpNav` and :class:`.RelativeOpNav`.  For instance, say we want to
create a new high-level interface class called MyAwesomeNewOpNav.  If we subclass the OpNav class when creating this
new class then we automatically get a ``camera`` attribute, and a few aliases to the
attributes of the camera instance

    >>> from giant.opnav_class import OpNav
    >>> from giant.camera import Camera
    >>> class MyAwesomeNewOpNav(OpNav):
    ...     def __init__(self, camera):
    ...         super().__init__(camera)
    ...         self.new_attribute = 2
    ...
    >>> inst = MyAwesomeNewOpNav(Camera())
    >>> hasattr(inst, 'camera')
    True
    >>> hasattr(inst, 'model')
    True
"""

from typing import Iterable
import warnings

import numpy as np

from giant._typing import PATH
from giant.camera import Camera
from giant.camera_models import CameraModel

from giant.utilities.mixin_classes import AttributePrinting, AttributeEqualityComparison


class OpNav(AttributePrinting, AttributeEqualityComparison):
    """
    This serves as a container for :class:`.Camera` instances and provides aliases to
    quickly access their attributes from an instance of this class.

    This class is rarely used as is, and instead is used as a super class for new OpNav user interfaces.
    """

    def __init__(self, camera: Camera):
        """
        :param camera: An instance of :class:`.Camera` that is to have OpNav performed on it
        """

        self._camera = self._validate_camera(camera)

    # ____________________________________________Camera Properties____________________________________________
    @property
    def camera(self) -> Camera:
        """
        The camera instance to perform OpNav on.

        This should be an instance of the :class:`.Camera` class or one of its subclasses.

        See the :class:`.Camera` class documentation for more details
        """
        return self._camera
    
    @staticmethod
    def _validate_camera(val: Camera) -> Camera:
        if not isinstance(val, Camera):
            warnings.warn("The camera should probably be an object that subclasses the Camera class\n"
                          "We'll assume you know what you're doing for now but "
                          "see the Camera documentation for details")
        return val

    @camera.setter
    def camera(self, val: Camera):
            
        self._camera = self._validate_camera(val)

    def add_images(self, data: Iterable[PATH | np.ndarray] | PATH | np.ndarray,
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

        return self.camera.model

    @model.setter
    def model(self, val: CameraModel):
        self.camera.model = val

