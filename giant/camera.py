# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This module defines the Camera object for GIANT, which collects information about a camera and images captured by that
camera in a single place and provides some methods for filtering, sorting, and handling the images.

A camera object is used to collect various pieces of data about a camera in a single place, such as a
:class:`.CameraModel`, a list of :class:`.OpNavImage` objects captured by the camera, a function representing the point
spread function of the camera, and some other pieces of information about the camera and images that are used throughout
the other GIANT routines.  By collecting all this data into a single object, the interface for individual components of
GIANT is unified.

In addition to collecting much of the information GIANT requires in a single location, camera objects provide some
capabilities to make managing image sets easier.  These capabilities include things like the ability to turn an image
(or a set of images) off, so it is no longer considered in the GIANT estimation and measurement routines (and also the
ability to turn images back on), the ability to quickly add new images to be considered by only supplying the path to
the files, the ability to override some of the metadata in the images based off of an external file (like the
attitude of the camera), and also the ability to apply a preprocessor to all images (where you can reorient the image,
remove bad pixels, subtract a flat field, etc..).
"""
import warnings
from datetime import timedelta, datetime
from enum import Enum
from typing import Union, Sequence, Iterable, Callable, Optional, Tuple, List

from pathlib import Path

import numpy as np

from giant.image import OpNavImage, ExposureType
from giant.camera_models import CameraModel
from giant.rotations import slerp, Rotation
from giant._typing import ARRAY_LIKE_2D, PATH
from giant.point_spread_functions import PointSpreadFunction


class AttitudeUpdateMethods(Enum):
    """
    This enumeration provides options for performing quaternion updates on short exposure images using long exposure
    images.

    See :meth:`.update_short_attitude` for more details.
    """

    REPLACE = "replace"
    """
    The replace method where the closest long exposure attitude is used to overwrite the short exposure attitude.
    """

    PROPAGATE = "propagate"
    """
    The delta quaternion method where the closest long exposure attitude is propagated using the attitude function 
    to overwrite the short exposure attitude.
    """

    INTERPOLATE = "interpolate"
    """
    The interpolate method where the 2 closest long exposure attitudes are interpolated using spherical linear 
    interpolation to overwrite the short exposure attitude.
    """


class Camera:
    """
        This class collects images, the :class:`.CameraModel`, and some relevant metadata about the camera into a single
        object for passing to the GIANT estimators and measurements.

        The :class:`Camera` class is primarily a container for collecting and manipulating images and metadata for a
        single physical camera.  This container is passed to the various measurement and estimation routines
        throughout GIANT to provide them with the requisite images and data needed to complete their tasks. The
        :class:`Camera` object is also an iterator over the images that are turned on.  This means you could do
        something like:

            >>> from giant.camera import Camera
            >>> import numpy
            >>> # generate the image data
            >>> image_list = [numpy.random.randn(100, 100) for _ in range(10)]  # type: List[np.ndarray]
            >>> # create an instance with the images included
            >>> cam = Camera(images=image_list, parse_data=False)
            >>> # turn off a few of the images
            >>> cam.image_mask[1], cam.image_mask[5], cam.image_mask[8] = False, False, False
            >>> # iterate over the images that are turned on
            >>> for ind, image in cam:
            >>>    print(ind, numpy.array_equal(image_list[ind], image))
            0 True
            2 True
            3 True
            4 True
            6 True
            7 True
            9 True

        The implementation of the Camera object here is fully functional, however, you may want to subclass this
        object for customization purposes.  For instance you may want to implement the :meth:`preprocessor` method to
        apply corrections to images immediately after loading.  You may also want to update the :meth:`image_check`
        method to use a custom :class:`.OpNavImage` subclass instead of the default (alternatively you could override
        the ``default_image_class`` argument to ``__init__``), or provide more custom functionality.
        """

    def __init__(self, images: Union[Iterable[Union[PATH, ARRAY_LIKE_2D]], PATH, ARRAY_LIKE_2D, None] = None,
                 model: Optional[CameraModel] = None, name: Optional[str] = None, spacecraft_name: Optional[str] = None,
                 frame: Optional[str] = None, parse_data: bool = True, psf: Optional[PointSpreadFunction] = None,
                 attitude_function: Optional[Callable] = None, start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None, metadata_only: bool = False,
                 default_image_class: type = OpNavImage):
        """
        :param images: A single image, or a list of images to store in the camera object.  The image data can either be
                       a string (in which case it is represents the path to the image file), an array of image data
                       (this is generally not recommended), an :class:`.OpNavImage` object already initialized, or a
                       list containing any of these three options.
        :param model: A camera model that represents how 3D points project onto the 2D imaging plane
        :param name:  The name of the camera that is represented by this object.  Not Required
        :param spacecraft_name: The name of the spacecraft that hosts the camera. Not Required
        :param frame: The name of the frame for this camera.  Not Required
        :param parse_data: A flag specifying whether to parse the metadata for each image when it is being loaded.
        :param psf: A callable object that applies a PSF to a 2D image, and provides a
                    :meth`~.PointSpreadFunction.apply_1d` method to apply the PSF to 1D scan lines.  Typically this is a
                    :class:`.PointSpreadFunction` subclass.
        :param attitude_function: A function that returns the attitude of the camera frame with respect to the inertial
                                  frame for an input datetime object.  This is generally a call to a spice routine
                                  generated by the :mod:`.spice_interface` helper functions.
                                  (see the :func:`.create_callable_orientation` and
                                  :func:`.et_callable_to_datetime_callable` functions in particular)
        :param start_date: The time at which images should start being processed
        :param end_date: The time at which images should start being processed
        :param metadata_only: Only load image metadata to an empty OpNavImage instead of loading the full image data.
        :param default_image_class: The class that the images stored in this instance should be an instance of.
        """
        # store the camera model object
        self._model = None
        self.model = model

        # store the image class we want to make sure our images are instances of
        self._default_image_class = default_image_class

        # add the images and create the image mask
        self._image_mask = []
        self._images = []
        if images is not None:
            self.add_images(images, parse_data=parse_data, metadata_only=metadata_only)

        # add the start and end dates:
        self.start_date = start_date
        """
        The initial time to start processing images.
        
        Any images with an :attr:`~.OpNavImage.observation_date` before this epoch are ignored.  This typically should 
        be set to None (no filtering) or a python datetime object. See the :meth:`apply_date_range` method for more 
        details.
        """

        self.end_date = end_date
        """
        The final time to stop processing images.

        Any images with an :attr:`~.OpNavImage.observation_date` after this epoch are ignored.  This typically should 
        be set to None (no filtering) or a python datetime object. See the :meth:`apply_date_range` method for more 
        details.
        """

        self.apply_date_range()

        # store the metadata
        self.name = name
        """
        The name of the camera.  
        
        This attribute is provided for documentation and convenience, but isn't used directly by core GIANT
        functions.  For an example of how one might use this attribute, see the :ref:`getting started <getting-started>`
        page for more details.
        """

        self.frame = frame
        """
        The name of the camera frame corresponding to this camera (typically a spice id).

        This attribute is provided for documentation and convenience, but isn't used directly by core GIANT
        functions.  For an example of how one might use this attribute, see the :ref:`getting started <getting-started>`
        page for more details.
        """

        self.spacecraft_name = spacecraft_name
        """
        The name of the spacecraft hosting this camera (typically set to a spice id).

        This attribute is provided for documentation and convenience, but isn't used directly by core GIANT
        functions.  For an example of how one might use this attribute, see the :ref:`getting started <getting-started>`
        page for more details.
        """

        # store the psf
        self._psf = None
        self.psf = psf

        # store the attitude function
        self._attitude_function = None
        self.attitude_function = attitude_function

    def __iter__(self) -> Iterable[Tuple[int, OpNavImage]]:
        """
        Loop through the images and their indices that are stored in the :attr:`.images` attribute
        and that are turned on according to the :attr:`.image_mask` attribute.

        :returns: A tuple of index, OpNavImage
        """
        for ind, image in enumerate(self._images):
            if self._image_mask[ind]:
                yield ind, image

    def __repr__(self) -> str:
        odict = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                odict[key] = value

        return (self.__module__ + "." + self.__class__.__name__ + "(" +
                ', '.join(['{}={!r}'.format(k, v) for k, v in odict.items()]) + ")")

    def __str__(self) -> str:
        odict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Callable):
                value = value.__module__ + "." + value.__name__

            if not key.startswith("_"):
                odict[key] = value

        return (self.__module__ + "." + self.__class__.__name__ + "(" +
                ', '.join(['{}={!s}'.format(k, v) for k, v in odict.items()]) + ")")

    @property
    def images(self) -> List[OpNavImage]:
        """
        The list of :class:`.OpNavImages` contained in this camera object that should be considered by the GIANT
        routines.

        Note that this attribute is read only.  To add or remove images from the list, use the :meth:`add_images` method
        or :meth:`remove_images` method respectively as these will ensure that the :attr:`.images` and
        :attr:`.image_mask` lists stay in sync.
        """
        return self._images

    @images.setter
    def images(self, data):

        raise AttributeError("The images cannot be set directly.\n"
                             "Use the add_images or remove_images methods instead")

    @property
    def image_mask(self) -> list:
        """
        This property contains a mask for turning images on or off in the GIANT estimation and measurement routines.

        The mask is a list of boolean values, where a ``True`` indicates that the image **should** be considered
        in the GIANT routines and returned when iterating over the images using this class and a ``False`` indicates
        that the image **should not** be considered in the GIANT routines or returned when iterating over the images
        using this class.

        In general, you should not write to, or update this mask directly, and you should never add or remove elements
        from the list directly.  Instead you should use the :meth:`add_images` or :meth:`remove_images` methods to
        add/remove images entirely and use the `*_on` or `*_off` methods to adjust the boolean values within the
        array.

        .. note::
            While it is strongly recommended to not directly modify this attribute, it is not write protected.  Instead
            there is a setter method which ensures that anything set to this attribute is coerced to be the right length
            and type that is expected.
        """
        return self._image_mask

    @image_mask.setter
    def image_mask(self, val):

        if isinstance(val, Sequence):

            if len(val) == len(self._images):
                self._image_mask = list(val)

            elif len(val) == 1:
                self._image_mask = list(val) * len(self._images)

            else:
                raise ValueError("The list you provided us is not the right size")

        elif isinstance(val, bool):
            self._image_mask = [val] * len(self._images)

        elif val is None:
            self._image_mask = [True] * len(self._images)

        else:
            raise ValueError("The image mask must be set as a Sequence or a bool")

    @property
    def psf(self) -> Optional[PointSpreadFunction]:
        """
        An object that applies a point spread function to 1D scan lines and 2D images.

        This object should provide a ``__call__`` method which applies the PSF to a 2D image, and an ``apply_1d`` method
        which applies the PSF to 1D scan lines (as the rows of a 2D array). Typically this is a subclass of
        :class:`.PointSpreadFunction`.

        The PSF object is used when generating templates in GIANT for use with the surface feature and cross
        correlation techniques.

        .. code::

            new_template = camera.psf(template)

        """
        return self._psf

    @psf.setter
    def psf(self, val: Optional[PointSpreadFunction]):

        if isinstance(val, Callable):

            self._psf = val

        elif val is None:

            self._psf = None

        else:

            raise ValueError('The psf object must be callable')

    @property
    def attitude_function(self) -> Callable:
        """
        A function that returns the orientation of the camera frame with respect to the inertial frame as an
        :class:`.Rotation` object for a specific time input as a python :class:`datetime` object.

        This function is used in two different ways.  First, it can be used to replace the attitude metadata
        in all of the images that are turned on directly using the :meth:`update_attitude_from_function` method.
        This is useful when you are parsing the image metadata from an outdated source (such as the header information
        in a fits file) and want to update the attitude to reflect the most recent attitude knowledge.  Second, this
        function can be used to compute delta quaternions in order to propagate a solved-for attitude from one image to
        another image using the :meth:`update_short_attitude` method.  This is useful when you are taking long-short
        exposure sequences and need to update the attitude of the short images based off of the solved-for attitude
        from the long images.

        In both cases the call to this function passes a datetime object representing the UTC
        time we need the attitude for as the only argument and the function will return the attitude
        as an :class:`.Rotation` object.

        In general, this function is a wrapper around spice calls to retrieve the attitude from a ck file.  GIANT
        provides some helper routines in the :mod:`.spice_interface` module to make it easy to generate this function.
        For instance, we can create a valid function for this attribute using the following:

            >>> from giant.utilities.spice_interface import et_callable_to_datetime_callable
            >>> from giant.utilities.spice_interface import create_callable_orientation
            >>> attitude_function = et_callable_to_datetime_callable(create_callable_orientation('J2000', 'MyNavCam'))

        and then we simply need to ensure we furnish a metakernel that provides enough information to compute the
        transformation from the J2000 inertial frame to the `'MyNavCam'` frame.  See the :mod:`.spice_interface`
        documentation for more information about how this works.
        """
        return self._attitude_function

    @attitude_function.setter
    def attitude_function(self, val: Optional[Callable]):
        if isinstance(val, Callable):

            self._attitude_function = val

        elif val is None:

            self._attitude_function = None

        else:

            raise ValueError('The attitude_function object must be callable')

    @property
    def model(self) -> CameraModel:
        """
        The camera model which describes how 3D points in the camera frame are transformed into 2D points in the image.

        The object set to this class will be used by all GIANT routines that make use of a camera model to relate
        3D and 2D points (which is nearly all of them).

        For more information about camera models and their theories, refer to the :mod:`.camera_models` module
        documentation.

        Although not required, it is strongly recommended that the object assigned to this property be a
        subclass of :class:`.CameraModel`.
        """

        return self._model

    @model.setter
    def model(self, val: Optional[CameraModel]):

        if isinstance(val, CameraModel):
            self._model = val

        else:
            self._model = val
            warnings.warn("The camera model you specified is not a subclass of CameraModel.\n"
                          "We'll assume you know what you're doing and have setup the proper methods.\n"
                          "In the future you should subclass CameraModel to avoid this warning.")

    def short_on(self) -> None:
        """
        This method updates the :attr:`.image_mask` attribute so that any image whose :attr:`~.OpNavImage.exposure_type`
        is set to ``SHORT`` or ``DUAL`` is turned on (processed).

        This method checks the :attr:`.exposure_type` attribute of each :class:`.OpNavImage` contained in the
        :attr:`.images` list, and if it is set to ``SHORT`` or ``DUAL``, then the corresponding index of the
        :attr:`.image_mask` list is set to ``True`` so that the image is considered in GIANT measurement and estimation
        routines.

        .. note::
            This method does not turn off images whose :attr:`.exposure_type` attribute is not set to ``SHORT``.  See
            the :meth:`only_short_on` method if you want this functionality.

        .. note::
            The :attr:`.exposure_type` attribute must be set correctly on each image for this method to work correctly.
            The :attr:`.exposure_type` is generally set automatically when an :class:`.OpNavImage` is created according
            to some threshold set by the user.

        """

        for ind, image in enumerate(self._images):
            if image.exposure_type in [ExposureType.SHORT, ExposureType.DUAL]:
                self._image_mask[ind] = True

    def short_off(self) -> None:
        """
        This method updates the :attr:`.image_mask` attribute so that any image whose :attr:`~.OpNavImage.exposure_type`
        is set to ``SHORT`` is turned off (not processed).

        This method checks the :attr:`.exposure_type` attribute of each :class:`.OpNavImage` contained in the
        :attr:`.images` list, and if it is set to ``SHORT`` then the corresponding index of the :attr:`.image_mask` list
        is set to ``False`` so that the image is not considered in GIANT measurement and estimation routines.

        .. note::
            This method does not turn on images whose :attr:`.exposure_type` attribute is not set to ``SHORT``.  See
            the :meth:`only_long_on` method if you want this functionality.

        .. note::
            The :attr:`.exposure_type` attribute must be set correctly on each image for this method to work correctly.
            The :attr:`.exposure_type` is generally set automatically when an :class:`.OpNavImage` is created according
            to some threshold set by the user.

        """

        for ind, image in enumerate(self._images):
            if image.exposure_type == ExposureType.SHORT:
                self._image_mask[ind] = False

    def long_on(self) -> None:
        """
        This method updates the :attr:`.image_mask` attribute so that any image whose :attr:`~.OpNavImage.exposure_type`
        is set to ``LONG`` or ``DUAL`` is turned on (processed).

        This method checks the :attr:`.exposure_type` attribute of each :class:`.OpNavImage` contained in the
        :attr:`.images` list, and if it is set to ``LONG`` or ``DUAL``, then the corresponding index of the
        :attr:`.image_mask` list is set to ``True`` so that the image is considered in GIANT measurement and estimation
        routines.

        .. note::
            This method does not turn off images whose :attr:`.exposure_type` attribute is not set to ``LONG``.  See
            the :meth:`only_long_on` method if you want this functionality.

        .. note::
            The :attr:`.exposure_type` attribute must be set correctly on each image for this method to work correctly.
            The :attr:`.exposure_type` is generally set automatically when an :class:`.OpNavImage` is created according
            to some threshold set by the user.

        """

        for ind, image in enumerate(self._images):
            if image.exposure_type in [ExposureType.LONG, ExposureType.DUAL]:
                self._image_mask[ind] = True

    def long_off(self) -> None:
        """
        This method updates the :attr:`.image_mask` attribute so that any image whose :attr:`~.OpNavImage.exposure_type`
        is set to ``LONG`` is turned off (not processed).

        This method checks the :attr:`.exposure_type` attribute of each :class:`.OpNavImage` contained in the
        :attr:`.images` list, and if it is set to ``LONG``, then the corresponding index of the :attr:`.image_mask` list
        is set to ``False`` so that the image is not considered in GIANT measurement and estimation routines.

        .. note::
            This method does not turn on images whose :attr:`.exposure_type` attribute is not set to ``LONG``.  See
            the :meth:`only_short_on` method if you want this functionality.

        .. note::
            The :attr:`.exposure_type` attribute must be set correctly on each image for this method to work correctly.
            The :attr:`.exposure_type` is generally set automatically when an :class:`.OpNavImage` is created according
            to some threshold set by the user.

        """

        for ind, image in enumerate(self._images):
            if image.exposure_type == ExposureType.LONG:
                self._image_mask[ind] = False

    def all_on(self) -> None:
        """
        This method sets every element of the :attr:`.image_mask` list to ``True`` so that all images are considered in
        the GIANT routines and returned when this class is iterated on.
        """

        self._image_mask[:] = [True] * len(self._images)

    def all_off(self) -> None:
        """
        This method sets every element of the :attr:`.image_mask` list to ``False`` so that no images are considered in
        the GIANT routines or returned when this class is iterated on.
        """

        self._image_mask[:] = [False] * len(self._images)

    def only_short_on(self) -> None:
        """
        This method updated the :attr:`.image_mask` list so that any image whose :attr:`~.OpNavImage.exposure_type`
        attribute is set to ``SHORT`` or ``DUAL`` is turned on (processed) and any image whose
        :attr:`~.OpNavImage.exposure_type` is set to ``LONG`` is turned off (not processed).

        This method checks the :attr:`.exposure_type` attribute of each :class:`.OpNavImage` contained in the
        :attr:`.images` list, and if it is set to ``SHORT`` or ``DUAL``, then the corresponding index of the
        :attr:`.image_mask` list is set to ``True`` so that the image is considered in GIANT measurement and estimation
        routines.  If the :attr:`.exposure_type` attribute is set to ``LONG``, then the corresponding index of the
        :attr:`.image_mask` list is set to ``False`` so that the image is not considered in GIANT measurement and
        estimation routines.

        .. note::
            This method turns off images whose :attr:`.exposure_type` attribute is not set to ``SHORT`` or ``DUAL``.
            See the :meth:`short_on` method if you do not want to change the other :attr:`.image_mask` elements.

        .. note::
            The :attr:`.exposure_type` attribute must be set correctly on each image for this method to work correctly.
            The :attr:`.exposure_type` is generally set automatically when an :class:`.OpNavImage` is created according
            to some threshold set by the user.
        """
        for ind, image in enumerate(self._images):
            self._image_mask[ind] = image.exposure_type in [ExposureType.SHORT, ExposureType.DUAL]

    def only_long_on(self):
        """
        This method updated the :attr:`.image_mask` list so that any image whose :attr:`~.OpNavImage.exposure_type`
        attribute is set to ``LONG`` or ``DUAL`` is turned on (processed) and any image whose
        :attr:`~.OpNavImage.exposure_type` is set to ``SHORT`` is turned off (not processed).

        This method checks the :attr:`.exposure_type` attribute of each :class:`.OpNavImage` contained in the
        :attr:`.images` list, and if it is set to ``LONG`` or ``DUAL``, then the corresponding index of the
        :attr:`.image_mask` list is set to ``True`` so that the image is considered in GIANT measurement and estimation
        routines.  If the :attr:`.exposure_type` attribute is set to ``SHORT``, then the corresponding index of the
        :attr:`.image_mask` list is set to ``False`` so that the image is not considered in GIANT measurement and
        estimation routines.

        .. note::
            This method turns off images whose :attr:`.exposure_type` attribute is not set to ``LONG`` or ``DUAL``.
            See the :meth:`long_on` method if you do not want to change the other :attr:`.image_mask` elements.

        .. note::
            The :attr:`.exposure_type` attribute must be set correctly on each image for this method to work correctly.
            The :attr:`.exposure_type` is generally set automatically when an :class:`.OpNavImage` is created according
            to some threshold set by the user.
        """
        for ind, image in enumerate(self._images):
            self._image_mask[ind] = image.exposure_type in [ExposureType.LONG, ExposureType.DUAL]

    def apply_date_range(self):
        """
        This method filters images by date, setting any whose :attr:`~.OpNavImage.observation_date` is not
        between :attr:`start_date` and :attr:`end_date` to False.

        This method uses the :attr:`.start_date` and :attr:`.end_date` attributes to create a date range.
        It then checks the :attr:`~.OpNavImage.observation_date` attribute of each :class:`.OpNavImage` contained in the
        :attr:`images` to identify which images are in the specified date range.  If an image was not taken
        during the specific date range, then the corresponding index of the :attr:`image_mask` list is set
        to ``False``.

        .. note::
            This method must be called after all other filter methods, such as :meth:`only_long_on`.

        .. note::
            If either the :attr:`.start_date` or :attr:`.end_date` attributes are type None, they will not be considered

        .. note::
            This method does not turn on any images that are turned off, even if they fall within the date
            range.
        """
        if (self.start_date is not None) and (self.end_date is not None):
            for ind, image in self:
                if not (self.start_date <= image.observation_date <= self.end_date):
                    self._image_mask[ind] = False

        elif (self.start_date is None) and (self.end_date is None):
            pass

        elif self.end_date is None:
            for ind, image in self:
                if not (self.start_date <= image.observation_date):
                    self._image_mask[ind] = False

        elif self.start_date is None:
            for ind, image in self:
                if not (image.observation_date <= self.end_date):
                    self._image_mask[ind] = False

    def sort_by_date(self):
        """
        This method is used to sort the images currently loaded to the :attr:`images` attribute by date.

        It also ensures that the :attr:`image_mask` list remains in sync with the images.  The images are sorted
        by the :attr:`~.OpNavImage.observation_date` attribute for each image.

        .. note::
            To ensure that the images are truly sorted by date throughout all image processing steps, this
            method should be called after any :meth:`add_images`, or :meth:`remove_images` method calls needed.
        """
        dates = []
        for image in self._images:
            dates.append(image.observation_date)

        sorted_date_inds = np.argsort(dates)

        sorted_images = []
        sorted_image_mask = []

        for ind in sorted_date_inds:
            sorted_images.append(self._images[ind])
            sorted_image_mask.append(self._image_mask[ind])

        self._images = sorted_images
        self._image_mask = sorted_image_mask

    def add_images(self, data: Union[Iterable[Union[PATH, ARRAY_LIKE_2D]], PATH, ARRAY_LIKE_2D],
                   parse_data: bool = True, preprocessor: bool = True, metadata_only: bool = False):
        """
        This method is used to add images to the :attr:`.images` while also ensuring that the :attr:`.image_mask` list
        remains the same size as the :attr:`.images` list.

        This method is the only way that a user should add images to a :class:`Camera` object after the object has been
        initialized.  It ensures that the :attr:`.images` and :attr:`.image_mask` lists do not get out of sync, ensures
        that the new images are turned on (their corresponding :attr:`.image_mask` values are set to ``True``) and also
        interprets the input in order to create an :class:`.OpNavImage` for each instance.

        There are a few different ways you can specify the images to be added to the camera model.  The first, and most
        effective, is to specify a list of strings representing the paths to the files that contain the image
        information.  Similarly, you can specify a single string representing the path to a single image if you only
        want to add one image.  Inputting the image data in this method allows the :class:`.OpNavImage` class to
        retrieve the required metadata for each image, assuming the user has successfully subclassed the
        :class:`.OpNavImage` class and set up the :meth:`.parse_data` method.

        The next most useful way to enter the image data is by entering either a single, or a list of
        :class:`.OpNavImage` objects.  When using this method, the user should be sure that the appropriate metadata
        has been set for each image.

        The least useful way to enter the image data is by entering the raw image data either as a numpy array, a list
        of numpy arrays, or a list of lists of lists.  In each of these cases, the data contained in the arrays/inner
        lists of lists is interpreted directly as the imaging data and no metadata is attached to the created
        :class:`.OpNavImage` object.  The user must the be sure to go an enter the correct metadata for each image
        to ensure functionality is not broken for other GIANT routines.

        Regardless of how the image data is entered, this method expands the :attr:`.image_mask` list by the number of
        images that are being added and turns each of the new images on.  In addition, if the ``preprocessor`` argument
        is set to ``True`` the :meth:`preprocessor` method is called on each image before it is stored in the
        :attr:`.images` list.

        If you are entering the image data as a string or a list of strings then you can optionally turn off the
        `parse_data` functionality by setting the ``parse_data`` keyword argument to ``False``.  This is not recommended
        however.

        :param data:  The image data to be stored in the :attr:`.images` list
        :param parse_data:  A flag to specify whether to attempt to parse the metadata automatically for the images
        :param preprocessor: A flag to specify whether to run the preprocessor after loading an image.
        :param metadata_only: A flag to specify to only load the metadata for an image, not the image data itself.
        """

        if isinstance(data, (list, tuple)):

            for datum in data:

                image = self.image_check(datum, parse_data=parse_data, metadata_only=metadata_only)

                if preprocessor:
                    self._images.append(self.preprocessor(image))
                else:
                    self._images.append(image)

                if getattr(self.model, 'estimate_multiple_misalignments', False):
                    if hasattr(self.model, 'misalignment'):

                        if isinstance(self.model.misalignment, list):
                            self.model.misalignment.append(np.zeros(3))

                        else:
                            self.model.misalignment = [self.model.misalignment]
                            self.model.misalignment.append(np.zeros(3))

                try:
                    self._image_mask.append(True)
                except AttributeError:
                    pass

        else:

            image = self.image_check(data, parse_data=parse_data, metadata_only=metadata_only)

            self._images.append(self.preprocessor(image))

            if getattr(self.model, 'estimate_multiple_misalignments', False):

                if hasattr(self.model, 'misalignment'):
                    if isinstance(self.model.misalignment, list):
                        self.model.misalignment.append(np.zeros(3))

            try:
                self._image_mask.append(True)
            except AttributeError:
                pass

    def remove_images(self, images: Union[int, slice, Iterable[Union[int, slice]]]):
        """
        This method is used to remove images from the :attr:`.images` list while also ensuring that the
        :attr:`.image_mask` list remains the same size as the :attr:`.images` list.

        This method is the only way that a user should remove images from a :class:`Camera` object after the object has
        been initialized.  It ensures that the :attr:`.images` and :attr:`.image_mask` lists do not get out of sync.

        Images to be removed are specified by index, list of indices, or slice.  The images are removed by using::

            del self.images[ind]
            del self.image_mask[ind]

        If *images* is an iterable, it should be sorted in decreasing order to make sure the proper images
        are removed.

        :param images:  The images to be removed from the camera, as either an index, a slice,
                        or a list of indices and slices
        """

        if isinstance(images, Iterable):

            for image in images:

                del self._images[image]
                del self._image_mask[image]
                if getattr(self.model, 'estimate_multiple_misalignments', False):
                    if hasattr(self.model, 'misalignment'):
                        del self.model.misalignment[image]

        else:

            del self._images[images]
            del self._image_mask[images]
            if getattr(self.model, 'estimate_multiple_misalignments', False):
                if hasattr(self.model, 'misalignment'):
                    del self.model.misalignment[images]

    def image_check(self, data: Union[PATH, ARRAY_LIKE_2D],
                    parse_data: bool = True, metadata_only: bool = False) -> OpNavImage:
        """
        This method is used to interpret the image data that is supplied by the user (either during initialization or
        through the :meth:`add_images` method) and ensure that it is a subclass of :class:`.OpNavImage`

        The input to this method should be a single representation of image data (either an :class:`OpNavImage`,
        Sequence of Sequences, numpy array, or the path to the image file) and the output will be that representation
        converted to an :class:`OpNavImage`.  If you are entering the path to the image file then you can specify
        the optional ``parse_data`` flag which is passed to the OpNavImage initialization function.

        In general it may be desirable to override this method when the :class:`Camera` class is subclassed to customize
        the functionality.  If you simply want to use a subclass of :class:`.OpNavImage` with the same functionality
        then you can simply override the ``default_image_class`` keyword argument to the constructor of this class.

        :param data: The data to be converted into an :class:`.OpNavImage`
        :param parse_data: A flag specifying whether to attempt to automatically parse the metadata for the image
        :param metadata_only: A flag to specify to only load the metadata for an image, not the image data itself.
        :return: The image data converted into an :class:`.OpNavImage` or one of its subclasses
        """

        if isinstance(data, self._default_image_class):
            image = data

        elif isinstance(data, np.ndarray) or isinstance(data, list):
            image = self._default_image_class(data, parse_data=False)

            warnings.warn("The data you gave us is not the appropriate type.\n"
                          "We created one for you but please add in the a priori knowledge to the image class.\n"
                          "See the {} documentation for details.".format(self._default_image_class.__name__))

        elif isinstance(data, (str, Path)):
            if metadata_only:
                image = self._default_image_class([], file=data, parse_data=parse_data)

            else:
                image = self._default_image_class(data, parse_data=parse_data)

        else:
            image = self._default_image_class(data)

            warnings.warn("We're not sure what {0} type is.\n"
                          "We'll assume it's array like for now and hope for the best.\n"
                          "Please be sure that it is array like and \n"
                          "to specify the a priori knowledge to the image class.\n"
                          "See the OpNavImage documentation for details.".format(self._default_image_class.__name__))

        return image

    def preprocessor(self, image: OpNavImage) -> OpNavImage:
        """
        This method is used to globally apply corrections to all images contained in the :class:`Camera` instance.

        The corrections applied by this method generally include things like image flips/transposes to put the image
        into the proper orientation that GIANT expects, dark frame removal to flatten the responsivity of the images
        and other basic image processing steps that apply the same to all images.  The only input to this method is
        the image itself as an :class:`.OpNavImage` subclass and the method should return the corrected
        :class:`OpNavImage` subclass (preserving the metadata).

        This method is applied once, immediately after loading the image.

        :param image: The image to apply the preprocessor corrections to
        :return: The corrected image
        """
        return image

    def _determine_closest_image(self, ind: int, image: OpNavImage) -> OpNavImage:
        """
        This private method determines the closest (in time) long exposure image to a given short exposure image.

        Note that this does not ensure that the returned image is long exposure so you should check yourself.

        :param ind: The index into the :attr:`.images` list of the short exposure image being updated
        :param image: The actual short exposure image object being updated.
        :return: The closest image
        """

        # if we are at the beginning or end of the images list then we only have one option to check
        if ind == 0:

            next_ind = 1

        elif ind == (len(self._images) - 1):

            next_ind = ind - 1

        else:

            # if the previous image is short then only check the following image
            if self.images[ind - 1].exposure_type == ExposureType.SHORT:
                next_ind = ind + 1

            # if the following image is short then only check the previous image
            elif self.images[ind + 1].exposure_type == ExposureType.SHORT:
                next_ind = ind - 1

            # otherwise both are long.  Choose the one with the smallest time difference
            else:
                # check if they both have estimated attitude
                if self.images[ind - 1].pointing_post_fit and (not self.images[ind + 1].pointing_post_fit):
                    next_ind = ind-1
                elif self.images[ind + 1].pointing_post_fit and (not self.images[ind - 1].pointing_post_fit):
                    next_ind = ind + 1
                else:
                    # 2*np.argmin - 1 will either give -1 (previous image) or 1 (next image)
                    delta = 2 * np.argmin([abs(self.images[ind-1].observation_date - image.observation_date),
                                           abs(self.images[ind+1].observation_date - image.observation_date)]) - 1

                    next_ind = ind + delta

        # return the image we are considering
        return self.images[next_ind]

    def _replace(self, ind: int, image: OpNavImage, max_delta: timedelta):
        """
        This private method applies the replace method to update short exposure attitude information from surrounding
        long exposure images.

        This method works on a single image to determine which of the 2 (or 1) surrounding long exposure images
        are closest in time to the supplied short exposure image (within a maximum time difference of *timedelta*.
        It then simply copies the long exposure attitude to the short exposure image.

        If we are successful at updating a short exposure image using this method, then the
        :attr:`.OpNavImage.pointing_post_fit` flag is updated to be ``True``.  Otherwise it is set to ``False``.

        :param ind: The index into the :attr:`.images` list of the short exposure image being updated
        :param image: The actual short exposure image object being updated.
        :param max_delta: The maximum time difference allowed between the short exposure and  long exposure images for
                          an update to be made
        """

        next_image = self._determine_closest_image(ind, image)

        # if this image is a short exposure (only happens if both surrounding images are short exposure)
        # throw a warning and do nothing
        if next_image.exposure_type == ExposureType.SHORT:
            warnings.warn("A short image cannot be preceded or followed by another short image to use "
                          "replace quaternion")
            image.pointing_post_fit = False
            return

        if not next_image.pointing_post_fit:
            warnings.warn("The attitude of the next image has not been estimated.  Unable to replace quaternion.")
            image.pointing_post_fit = False
            return

        # if the difference between the short and long exposure image is too long
        # throw a warning and do nothing
        diff = abs(next_image.observation_date - image.observation_date)
        if diff > max_delta:
            warnings.warn("Two images are separated by too large of a time difference to use replace."
                          "Diff {} between {} and {}".format(diff, image.observation_date, next_image.observation_date))
            image.pointing_post_fit = False
            return

        # copy the long exposure attitude to the short exposure attitude
        image.rotation_inertial_to_camera = next_image.rotation_inertial_to_camera.copy()
        image.pointing_post_fit = True

    # noinspection PyTypeChecker
    def _propagate_attitude(self, ind, image, max_delta):
        """
        This private method applies the delta quaternion method to update short exposure attitude information from
        surrounding long exposure images.

        This method works on a single image to determine which of the 2 (or 1) surrounding long exposure images
        are closest in time to the supplied short exposure image (within a maximum time difference of *timedelta*.
        It then queries the :attr:`.attitude_function` to get the change in the pointing between the two images and
        applies this delta to the long exposure attitude to update the short exposure attitude.

        If we are successful at updating a short exposure image using this method, then the
        :attr:`.OpNavImage.pointing_post_fit` flag is updated to be ``True``.  Otherwise it is set to ``False``.

        :param ind: The index into the :attr:`.images` list of the short exposure image being updated
        :param image: The actual short exposure image object being updated.
        :param max_delta: The maximum time difference allowed between the short exposure and  long exposure images for
                          an update to be made
        """

        next_image = self._determine_closest_image(ind, image)

        # if this image is a short exposure (only happens if both surrounding images are short exposure)
        # throw a warning and do nothing
        if next_image.exposure_type == ExposureType.SHORT:
            warnings.warn("A short image cannot be both preceded and followed by another short image to use "
                          "delta quaternion")
            image.pointing_post_fit = False
            return

        if not next_image.pointing_post_fit:
            warnings.warn("The attitude of the next image has not been estimated.  Unable to use delta quaternion.")
            image.pointing_post_fit = False
            return

        # if the difference between the short and long exposure image is too long
        # throw a warning and do nothing
        diff = abs(next_image.observation_date - image.observation_date)
        if diff > max_delta:
            warnings.warn("Two images are separated by too large of a time difference to use delta quaternion."
                          "Diff {} between {} and {}".format(diff, image.observation_date, next_image.observation_date))
            image.pointing_post_fit = False
            return

        att_prev = self.attitude_function(next_image.observation_date)  # type: Rotation

        att_curr = self.attitude_function(image.observation_date)  # type: Rotation

        # compute the delta quaternion between the long exposure and short exposure image.
        delta_q = att_curr * att_prev.inv()

        # apply the updated delta quaternion
        image.rotation_inertial_to_camera = delta_q * next_image.rotation_inertial_to_camera
        image.pointing_post_fit = True

    def _interp(self, ind, image, max_delta):
        """
        This private method applies the interpolate quaternion method to a given short exposure image.

        If a short exposure image is not surrounded by long exposure images then the quaternion interpolation will not
        work and we fall back to replace.  We also fall back to replace if one of the 2 surrounding images is too far
        away (time) from the short exposure image.

        If all conditions are met then the interpolation method performs spherical linear interpolation between the two
        long exposure images to get the updated attitude for the short exposure image (see :func:`.slerp`).

        If we are successful at updating a short exposure image using this method, then the
        :attr:`.OpNavImage.pointing_post_fit` flag is updated to be ``True``.  Otherwise it is set to ``False``.

        :param ind: The index into the :attr:`.images` list of the short exposure image being updated
        :param image: The actual short exposure image object being updated.
        :param max_delta: The maximum time difference allowed between the short exposure and  long exposure images for
                          an update to be made
        """

        if image.exposure_type == ExposureType.SHORT:

            if ind == 0:
                warnings.warn('A short image is first in the image list, falling back to replace method')
                self._replace(ind, image, max_delta)

            elif ind == (len(self.images) - 1):
                warnings.warn('A short image is last in the image list, falling back to replace method')
                self._replace(ind, image, max_delta)

            else:

                image_prev = self.images[ind - 1]
                image_next = self.images[ind + 1]

                if image_prev.exposure_type == ExposureType.SHORT:
                    warnings.warn("A short image precedes a short image, falling back to replace method")
                    self._replace(ind, image, max_delta)
                    return

                if not image_prev.pointing_post_fit:
                    warnings.warn(
                        "The attitude of the preceding image has not been estimated.  Falling back to replace method.")
                    self._replace(ind, image, max_delta)
                    return

                if image_next.exposure_type == ExposureType.SHORT:
                    warnings.warn("A short image follows a short image, falling back to replace method")
                    self._replace(ind, image, max_delta)
                    return

                if not image_next.pointing_post_fit:
                    warnings.warn(
                        "The attitude of the next image has not been estimated.  Falling back to replace method.")
                    self._replace(ind, image, max_delta)
                    return

                if abs(image.observation_date - image_prev.observation_date) > max_delta:
                    warnings.warn("the time delta between two images is larger than the maximum time delta."
                                  "Falling back to replace method.")
                    self._replace(ind, image, max_delta)

                elif abs(image_next.observation_date - image.observation_date) > max_delta:
                    warnings.warn("the time delta between two images is larger than the maximum time delta."
                                  "Falling back to replace method.")
                    self._replace(ind, image, max_delta)

                else:
                    image.rotation_inertial_to_camera = Rotation(slerp(image_prev.rotation_inertial_to_camera,
                                                                       image_next.rotation_inertial_to_camera,
                                                                       image.observation_date,
                                                                       image_prev.observation_date,
                                                                       image_next.observation_date))
                    image.pointing_post_fit = True

    def update_short_attitude(self,
                              method: Union[str, AttitudeUpdateMethods] = AttitudeUpdateMethods.INTERPOLATE,
                              max_delta: timedelta = timedelta(minutes=5)):
        r"""
        This method updates the attitude metadata for short exposure images based off of the solved for attitudes in
        the long-exposure images.

        There are three different techniques that you can use to update the short exposure attitudes which are selected
        using the `method` key word argument.  The first technique, ``'propagate'``, "propagates" the attitude
        from a long exposure image to the short exposure image using a delta quaternion.  The delta quaternion is
        calculated using the :attr:`.attitude_function` and is computed using

        .. math::
            \delta\mathbf{q}=\mathbf{q}_{sf}\otimes\mathbf{q}_{lf}^{-1}

        where :math:`\delta\mathbf{q}` is the delta quaternion, :math:`\mathbf{q}_{sf}` is the attitude quaternion
        at the short exposure image time according to the :attr:`.attitude_function`, :math:`\mathbf{q}_{lf}^{-1}` is
        the inverse of the attitude quaternion for the long exposure image closest (in time) to the short exposure
        image according to the :attr:`.attitude_function`, and :math:`\otimes` is quaternion multiplication.  The delta
        quaternion is applied according to

        .. math::
            \mathbf{q}_{ss}=\delta\mathbf{q}\otimes\mathbf{q}_{ls}

        where :math:`\mathbf{q}_{ss}` is the solved for attitude for the short exposure image and
        :math:`\mathbf{q}_{ls}` is the solved for attitude for the long exposure image closest (in time) to the short
        exposure image.  This means that to use this method short exposure images must be either preceded or followed by
        a long exposure image in the :attr:`.images` list.

        The next potential method is ``'interpolate'``.  In interpolate, the attitude of a short exposure image that is
        sandwiched between 2 long exposure images is updated by using the SLERP quaternion interpolation method.  The
        SLERP quaternion interpolation method is described in :func:`.slerp` function documentation.  In order to use
        the ``'interpolate'`` method all turned on short exposure images must be immediately preceded and followed by
        long exposure images.

        The final potential method is ``'replace'``.  In the ``'replace'`` method, the attitude for short exposure
        images are replaced with the attitude from the closest (in time) long exposure image to them from the
        :attr:`.images` list.  In order to use the `'replace`' method every turned on short exposure image must be
        preceded or followed by a long exposure image.

        If we are successful at updating a short exposure image using this method, then the
        :attr:`.OpNavImage.pointing_post_fit` flag is updated to be ``True`` for the corresponding image.
        Otherwise it is set to ``False``.

        .. note::
            The attitude is only updated for "short" exposure images that are turned on (it does not matter if the long
            exposure images are turned on or off).

        :param method:  The method to use to update the attitude for the turned on short exposure images
        :param max_delta: The maximum time difference allowed between 2 images for them to be paired as a timedelta
                          object
        """

        if isinstance(method, str):
            method = AttitudeUpdateMethods(method.lower())

        if method == AttitudeUpdateMethods.PROPAGATE:

            if callable(self.attitude_function):
                func = self._propagate_attitude
            else:
                raise ValueError("attitude_function must be callable to use propagate")

        elif method == AttitudeUpdateMethods.INTERPOLATE:

            func = self._interp

        elif method == AttitudeUpdateMethods.REPLACE:

            func = self._replace

        else:
            raise ValueError("Couldn't understand method of {}".format(method))

        for ind, image in self:

            if image.exposure_type == ExposureType.SHORT:

                func(ind, image, max_delta)

    def update_attitude_from_function(self):
        """
        This method is used ot overwrite the attitude information stored in all images that are turned on with
        information from the :attr:`.attitude_function`.

        For each turned on image, the attitude function is queried with the :attr:`~.OpNavImage.observation_date`
        attribute of the image and the resulting Rotation object is set as the new :attr:`.rotation_inertial_to_camera`
        for that image. The image attitude are updated regardless of their exposure type as long as they are turned on.

        When we update the attitude for an image using this method we set the :attr:`.OpNavImage.pointing_post_fit`
        flag to ``False`` for the corresponding image.

        :raises: ValueError if the :attr:`.attitude_function` is not callable.
        """

        if not callable(self.attitude_function):
            raise ValueError("attitude_function must be callable to use update_attitude_from_file")

        for _, image in self:
            image.rotation_inertial_to_camera = self.attitude_function(image.observation_date)
            image.pointing_post_fit = False
