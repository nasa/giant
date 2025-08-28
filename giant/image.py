# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This module provides the OpNavImage class, which is the primary image type used by GIANT.

The OpNavImage class is a container to store both an image itself (raw DN values) as well as metadata about the image
that is required in various routines.  The metadata includes information like the observation_date and time the image
was taken, the name of the camera used to capture the image, the position, velocity, and orientation at the time the
image was captured, among other things (see the :class:`OpNavImage` documentation for more thorough details).

The :class:`OpNavImage` class is a subclass of the numpy 
`ndarray <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_ class, with the image data itself
being stored in the array (so that you can use all of the usual numpy indexing/ufuncs as if it was just an ndarray) and
the metadata being stored as extra attributes.  As such, the image data can be any type (float, int, etc) but be aware
that many of the GIANT routines cast the images to float type internally.

In general, you should not use this class directly, but rather subclass it so that you can automatically parse the 
metadata for an image instead of having to manually specify it each time (by overriding the NotImplemented 
:meth:`.parse_data` method).  This also allows you to specify custom image loading routines if your project uses a
non-default image format (by overriding the :meth:`.load_image` method).  As an example, check out the
:ref:`getting started <getting-started>` page.
"""

from pathlib import Path

from typing import Union, Optional, Any, Self, cast

import os

from enum import Enum

import numpy as np
import cv2
import astropy.io.fits as pf

from giant._typing import ARRAY_LIKE_2D, ARRAY_LIKE, PATH, DatetimeLike
from giant.rotations import Rotation


class ExposureType(Enum):
    """
    This enumeration provides options for the different ways an image can be be classified in GIANT
    """

    SHORT = "short"
    """
    A short exposure image to be used for relative navigation (center finding, limbs, SFN, etc).
    """

    LONG = "long"
    """
    A long exposure image to be used for star based navigation (attitude estimation and camera calibration).
    """

    DUAL = "dual"
    """
    A dual purpose image to be used for both star based and relative navigation
    """


# noinspection PyAttributeOutsideInit
class OpNavImage(np.ndarray):
    """
    This is a subclass of a numpy array for images which adds various parameters to the ndarray class necessary for the
    GIANT algorithms as well as some helper methods for loading in an image.

    The OpNavImage class is primarily a numpy ndarray which stores the illumination values of an image.  In addition
    to the illumination data (which is used as you would normally use a numpy array) this class has some extra
    attributes which are used throughout the GIANT routines.  These attributes are metadata for the image, including
    the location the image was taken, the observation_date the image was taken, the camera used the take the image,
    the spacecraft hosting the camera used to take the image, the attitude of the camera at the time the image was
    taken, the velocity of the camera at the time the image was taken, the file the image was loaded from, and the
    length of exposure used to generate the image.

    The OpNavImage class also provides helper methods for loading the data from an image file.  There is the
    :meth:`.load_image` static method which will read in a number of standard image formats and return a numpy array
    of the illumination data.  There is also the :meth:`parse_data` method which attempts to extract pertinent
    information about an image to fill the metadata of the OpNavImage class.  The :meth:`.parse_data` method is
    not implemented here and most be implemented by the user if it is desired to be used (by subclassing the OpNavImage
    class).  It is possible to use the OpNavImage class without subclassing and defining the :meth:`parse_data` method
    by manually specifying the required metadata in the class initialization and setting the **parse_data** flag to
    False in the class initialization, but in general this is not recommended.

    You can initialize this class by either passing in a path to the image file name (*recommended*) or by passing in
    an array-like object of the illumination data.  The metadata can be specified as keyword arguments to the class
    initialization or can be loaded by overriding the :meth:`parse_data` method.
    
    Note that if you have overridden the :meth:`parse_data` method, specified ``parse_data=True``, and specify one 
    of the other optional inputs, what you have specified manually will overwrite anything filled by :meth:`parse_data`
    """

    def __new__(cls, data: Union[PATH, ARRAY_LIKE_2D],
                observation_date: Union[DatetimeLike, None] = None,
                rotation_inertial_to_camera: Union[Rotation, ARRAY_LIKE, None] = None,
                temperature: Optional[float] = None, position: Union[ARRAY_LIKE, None] = None,
                velocity: Union[ARRAY_LIKE, None] = None, exposure_type: Union[ExposureType, str, None] = None,
                saturation: Optional[float] = None, file: Union[PATH, None] = None,
                parse_data: bool = False, exposure: Optional[float] = None,
                dark_pixels: Optional[ARRAY_LIKE] = None, instrument: Optional[str] = None, spacecraft: Optional[str] = None,
                target: Optional[str] = None, pointing_post_fit: bool = False) -> Self:
        """
        :param data: The image data to be formed into an OpNavImage either as a path to an image file or the
                     illumination data directly
        :param observation_date: The observation_date the image was captured
        :param rotation_inertial_to_camera: the rotation to go from the inertial frame to the camera frame at the time
                                            the image was taken
        :param temperature: The temperature of the camera when the image was captured
        :param position: the inertial position of the camera at the time the image was taken
        :param velocity: The inertial velocity of the camera at the time the image was taken
        :param exposure_type: The type of exposure for the image ('short' or 'long')
        :param saturation: The saturation level for the image
        :param file:  The file the illumination data came from.  Generally required if parse_data is to be used and the
                      data was entered as an array like value
        :param parse_data: A flag whether to try the parse_data method.  The parse_data method must be defined by the
                           user.
        :param exposure: The exposure time used to capture the image.  This isn't actually used in GIANT (in favor of
                         :attr:`exposure_type` attribute) so it is provided for convenience and for manual inspection
        :param dark_pixels: An array of dark pixels to be used in estimating the noise level of the image (this
                            generally refers to a set of pixels that are active but specifically not exposed to light
        :param instrument: The camera used to capture the image.  This is not used internally by GIANT and is provided
                           for convenience and for manual inspection
        :param spacecraft: The spacecraft hosting the camera. This is not used internally by GIANT and is provided
                           for convenience and for manual inspection
        :param target: The target that the camera is pointed towards. This is not used internally by GIANT and is
                       provided for convenience and for manual inspection
        :param pointing_post_fit: A flag specifying whether the attitude for this image has been estimated (True) or not
        """

        if isinstance(data, (str, Path)):
            image_data = cls.load_image(data).view(cls)

            image_data.file = data

        else:
            image_data = np.asarray(data).view(cls)
            image_data.file = None

        # initialize all the fields
        image_data._observation_date = None
        image_data._rotation_inertial_to_camera = Rotation()
        image_data._position = np.zeros(3, dtype=np.float64)
        image_data._velocity = np.zeros(3, dtype=np.float64)
        image_data._exposure_type = None
        image_data._saturation = np.finfo(np.float64).max
        image_data._temperature = 0
        image_data.exposure = None
        image_data.dark_pixels = None
        image_data.instrument = None
        image_data.spacecraft = None
        image_data.target = None
        image_data.pointing_post_fit = pointing_post_fit

        if file is not None:
            image_data.file = file

        # parse the data if requested
        if parse_data:
            image_data.parse_data()

        # overwrite any data we've already loaded/parsed with what the User has specified
        if observation_date is not None:
            image_data.observation_date = observation_date
        if instrument is not None:
            image_data.instrument = instrument
        if spacecraft is not None:
            image_data.spacecraft = spacecraft
        if rotation_inertial_to_camera is not None:
            image_data.rotation_inertial_to_camera = rotation_inertial_to_camera
        if position is not None:
            image_data.position = position
        if velocity is not None:
            image_data.velocity = velocity
        if exposure is not None:
            image_data.exposure = exposure
        if exposure_type is not None:
            image_data.exposure_type = exposure_type
        if saturation is not None:
            image_data.saturation = saturation
        if dark_pixels is not None:
            image_data.dark_pixels = dark_pixels
        if temperature is not None:
            image_data.temperature = temperature
        if target is not None:
            image_data.target = target

        return image_data

    def __reduce__(self) -> tuple[type[np.ndarray], tuple[np.ndarray], dict[str, Any]]:

        return self.__class__, (self.view(np.ndarray),), self.__dict__

    def __setstate__(self, state: dict, *args, **kwargs) -> None:

        # super().__setstate__(state, *args, **kwargs)

        self.__dict__.update(state)

    def __array_finalize__(self, obj: Optional[np.ndarray]) -> None:

        if obj is None:
            return

        self.file = getattr(obj, 'file', None)
        try:
            self.observation_date = getattr(obj, 'observation_date', None)
        except AssertionError:
            self.observation_date = None
        self.instrument = getattr(obj, 'instrument', None)
        self.spacecraft = getattr(obj, 'spacecraft', None)
        self.rotation_inertial_to_camera = getattr(obj, 'rotation_inertial_to_camera', Rotation())
        self.position = getattr(obj, 'position', np.zeros(3, dtype=np.float64))
        self.velocity = getattr(obj, 'velocity', np.zeros(3, dtype=np.float64))
        self.dark_pixels = getattr(obj, 'dark_pixels', None)
        self.exposure = getattr(obj, 'exposure', None)
        self.exposure_type = getattr(obj, 'exposure_type', None)
        self.saturation = getattr(obj, 'saturation', float(np.finfo(np.float64).max))
        self.temperature = getattr(obj, 'temperature', 0)
        self.target = getattr(obj, 'target', None)
        self.pointing_post_fit = getattr(obj, 'pointing_post_fit', False)

    def __repr__(self) -> str:

        data = super().__repr__()
        odict = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                try:
                    odict[key] = value
                except AssertionError:
                    odict[key] = None

        return (self.__module__ + "." + self.__class__.__name__ + "(" + data +
                ', '.join(['{}={!r}'.format(k, v) for k, v in odict.items()]) + ")")

    def __str__(self) -> str:
        data = super().__str__()
        odict = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                try:
                    odict[key] = value
                except AssertionError:
                    odict[key] = None

        return (self.__module__ + "." + self.__class__.__name__ + "(" + data +
                ', '.join(['{}={!s}'.format(k, v) for k, v in odict.items()]) + ")")

    @property
    def observation_date(self) -> DatetimeLike:
        """
        The observation_date specifies when the image was captured (normally set to the middle of the exposure period).

        This is used for tagging observations with timestamps, updating attitude knowledge in short exposure images
        using long exposure images, and updating a scene to how it is expected to be at the time an image is captured.

        Typically this attribute is a python datetime object, however, you can make it a different object if you want
        as long as the different object implements ``isoformat``, ``__add__``, and ``__sub__`` methods.  You can also
        set this attribute to ``None`` but this will break some functionality in GIANT so it is not recommended.  
        """

        assert self._observation_date is not None, "observation_date cannot be None at this point"
        return self._observation_date

    @observation_date.setter
    def observation_date(self, val: Optional[DatetimeLike]):

        if val is None:
            self._observation_date = val
        elif isinstance(val, DatetimeLike):
            self._observation_date = val
        else:
            raise ValueError("We can't use the value you set for observation_date.  Please consider using a datetime "
                             "object, or if you really know what you're doing you can directly set the "
                             "_observation_date attribute")

    @property
    def rotation_inertial_to_camera(self) -> Rotation:
        """
        The rotation_inertial_to_camera attribute encodes the rotation to transform from the inertial frame to the
        camera frame at the time of the image.

        This is used extensively throughout GIANT.  It is updated when using stars to estimate an updated attitude, when
        doing relative navigation to predict where points in the scene project to points in the image, and also in
        relative navigation to predict where points in the image project to in inertial space.

        This attribute should be set to a :class:`.Rotation` object, or something that the Rotation object can
        interpret. When you set this value, it will be converted to an :class:`.Rotation` object. 
        """

        return self._rotation_inertial_to_camera

    @rotation_inertial_to_camera.setter
    def rotation_inertial_to_camera(self, val: Rotation | ARRAY_LIKE):

        self._rotation_inertial_to_camera = Rotation(val)

    @property
    def velocity(self) -> np.ndarray:
        """
        The velocity attribute encodes the inertial velocity of the camera at the time the image was captured.

        This must be the inertial velocity with respect to the solar system barycenter and is used when
        to compute the stellar aberration correction to stars and targets.  To ignore stellar aberration you can set
        this to the zero vector.

        This attribute should be set to a length 3 array like object and it will be converted into a double numpy
        ndarray.  If you try to set this value to None then it will be reset to a vector of zeros.
        """

        return self._velocity

    @velocity.setter
    def velocity(self, val: ARRAY_LIKE | None):

        if val is not None:
            self._velocity = np.array(val, dtype=np.float64).ravel()
        else:
            self._velocity = np.zeros(3, dtype=np.float64)

    @property
    def position(self) -> np.ndarray:
        """
        The position attribute encodes the inertial position of the camera at the time the image was captured.

        Typically this is the inertial position from the solar system barycenter to the spacecraft and is used when
        updating an :class:`.Scene` to place objects in the camera frame at the time of the image.  You can
        optionally put this in another frame or with another central body as long as you know what you are doing and
        understand how the :class:`.Scene` works.

        This attribute should be set to a length 3 array like object and it will be converted into a double numpy
        ndarray.  If you try to set this value to None then it will be reset to a vector of zeros.
        """

        return self._position

    @position.setter
    def position(self, val: ARRAY_LIKE | None):

        if val is not None:
            self._position = np.array(val, dtype=np.float64).ravel()
        else:
            self._position = np.zeros(3, dtype=np.float64)

    @property
    def exposure_type(self) -> Union[ExposureType, None]:
        """
        The exposure type specifies what type of processing to use on this image.

        ``short`` exposure images are used for relative navigation like center finding.
        ``long`` exposure images are used for star based navigation like attitude estimation.
        ``dual`` exposure images are used for both star based and relative navigation.

        this property should be set to an :class:`.ExposureType` value or a string.  
        
        If it is set to `None` then the exposure type will be defaulted to ``dual``
        """

        return self._exposure_type

    @exposure_type.setter
    def exposure_type(self, val: ExposureType | str | None):

        if val is None:
            self._exposure_type = ExposureType.DUAL
        else:
            if isinstance(val, str):
                val = val.lower()

            self._exposure_type = ExposureType(val)

    @property
    def temperature(self) -> float:
        """
        The temperature of the camera at the time the image was captured

        This property is used by the camera model to apply temperature dependent focal length changes.  It should be
        a real number and convertible to a float by using the ``float`` function.  
        
        If you set this to None it will default to a temperature of 0
        """

        return self._temperature

    @temperature.setter
    def temperature(self, val: float | None):

        if val is None:
            self._temperature = 0.0
        else:
            try:
                self._temperature = float(val)
            except ValueError:
                raise ValueError('Unable to convert {} to a float.  The temperature must be a number that is '
                                 'convertible to a float'.format(val))

    @property
    def saturation(self) -> float:
        """
        The saturation value of the camera.

        This attribute is used when determining if a pixel is saturated or not in image processing.  It may be set to a
        very high number to effectively ignore the check.
        
        If set to None, this defaults to the maximum double value
        """

        return self._saturation

    @saturation.setter
    def saturation(self, val: float | None):

        if val is None:
            self._saturation = float(np.finfo(np.float64).max)
        else:
            try:
                _ = 0 < val
                self._saturation = val
            except TypeError:
                raise TypeError("The saturation must be a number that supports comparisons with numbers")

    def parse_data(self, *args):
        """
        This method should fill in the metadata for an OpNavImage.

        This method must be implemented by the user.
        
        :raises: NotImplementedError
        """
        raise NotImplementedError('This method needs to be implemented by the user.')

    @staticmethod
    def load_image(image_path: PATH) -> np.ndarray:
        """
        This method reads in a number of standard image formats using OpenCV and pyfits and converts it to grayscale
        if it is in color.

        :param image_path: The path to the image file to be read.
        :return: The illumination data from the image file
        """
        cv_ext = ['.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2',
                  '.png', '.webp', '.pbm', '.pgm', '.ppm', '.sr', '.ras',
                  '.tiff', '.tif']

        if os.path.exists(image_path):
            _, ext = os.path.splitext(image_path)

            if ext.lower() in '.fits':
                # pylance doesn't recognize the return type here
                with pf.open(image_path) as image_file: # type: ignore

                    image = cast(np.ndarray, cast(pf.PrimaryHDU, image_file[0]).data)

                    if len(image.shape) > 2:

                        if image.shape[0] == 3:
                            image = np.swapaxes(image, 0, 1)
                            image = np.swapaxes(image, 1, 2)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    if str(image.dtype) not in ['uint8', 'uint16', 'float32']:
                        image = image.astype('float32')

                    return image

            elif ext.lower() in cv_ext:

                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

                return image

            else:
                raise ValueError('The file you specified ({0:s}) is not a recognizable image.\n'
                                 'Please try again.'.format(image_path))
        else:

            raise ValueError('The file you specified({0:s}) does not exist.\n'
                             'Please try again'.format(image_path))
