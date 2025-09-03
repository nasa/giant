


"""
This package provides classes and functions for creating/using geometric camera models in GIANT.

In GIANT, a camera model refers to a class that provides a collection of methods for mapping 3D points and directions
expressed in the camera frame to 2D points in an image, for mapping 2D points in an image to 3D directions in
the camera frame, and provides jacobian matrices for those processes.

These objects are used extensively throughout GIANT and are one of the building blocks of optical navigation.
The modules in this package provide a number of the most commonly used camera models for optical navigation, including
the :mod:`.pinhole_model`, the :mod:`.brown_model`, the :mod:`.owen_model`, and the :mod:`.opencv_model`.  In addition,
the :mod:`.camera_model` module provides an abstract base class and instructions for constructing your own custom camera
models.

If you are just starting out, we recommend that you begin with one of the provided camera models as these are adequate
for almost all cameras and are generally easy to initialize if you have some basic knowledge about the camera itself.
Refer to the documentation for each module to get more details about the models.

While all of the classes and functions in this package are defined in the sub-modules discussed above, they are imported
into the package to make access easier; therefore, you can do::

    >>> from giant.camera_models import BrownModel, save, load

or::

    >>> from giant.camera_models import OwenModel

"""

from giant.camera_models.camera_model import CameraModel, ReturnShape, save, load
from giant.camera_models.pinhole_model import PinholeModel
from giant.camera_models.owen_model import OwenModel
from giant.camera_models.brown_model import BrownModel
from giant.camera_models.opencv_model import OpenCVModel
from giant.camera_models.fisheye_model import FisheyeModel
from giant.camera_models.split_camera import SplitCamera

__all__ = ['PinholeModel', 'OwenModel', 'BrownModel', 'OpenCVModel', 'ReturnShape', 'save', 'load', 'CameraModel', 'FisheyeModel', 'SplitCamera']
