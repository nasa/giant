# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


r"""
This module provides an abstract base class (abc) for implementing GIANT camera models.

This abc provides a design guide for building GIANT compatible camera models.  All user defined camera models should
probably subclass this class to ensure that they implement all of the required properties and methods that GIANT expects
a camera model to have [#]_. In addition, this module provides the functions :func:`save` and :func:`load` which can be
used to write/read camera models from disk in a human and machine readable format.

For a typical use case that doesn't require a custom camera model, see the :class:`.PinholeModel`, :class:`.BrownModel`,
:class:`.OwenModel`, or :class:`.OpenCVModel` classes which provide some of the most common models used in optical
navigation.  These also serve as examples of how to make a concrete implementation of the :class:`CameraModel` abc.

.. rubric:: Footnotes

.. [#] GIANT will not error if you do not subclass :class:`CameraModel`, but it will print warnings to the screen.

Use
___

To implement a fully functional custom camera model for GIANT, you must implement the following methods in addition to
subclassing the :class:`CameraModel` class.

================================================= ======================================================================
Method/Attribute                                  Use
================================================= ======================================================================
:meth:`~CameraModel.project_onto_image`           projects a point from the camera frame onto the image
:meth:`~CameraModel.compute_jacobian`             returns the Jacobian matrix
                                                  :math:`\partial\mathbf{x}_P/\partial\mathbf{c}`
                                                  where :math:`\mathbf{c}` is a vector of camera model parameters (like
                                                  focal length, pixel pitch, distortion coefficients, etc) and
                                                  :math:`\mathbf{x}_P` is a pixel location.
:meth:`~CameraModel.compute_pixel_jacobian`       returns the Jacobian matrix \partial\mathbf{x}_P/\partial\mathbf{x}_C`
                                                  where :math:`\mathbf{x}_C` is a vector in the camera frame that
                                                  projects to :math:`\mathbf{x}_P` which is the pixel location.
:meth:`~CameraModel.compute_unit_vector_jacobian` returns the Jacobian matrix \partial\mathbf{x}_C/\partial\mathbf{x}_P`
                                                  where :math:`\mathbf{x}_C` is a unit vector in the camera frame that
                                                  projects to :math:`\mathbf{x}_P` which is the pixel location.
:meth:`~CameraModel.apply_update`                 updates the camera model based on a vector of delta camera model
                                                  parameters
:meth:`~CameraModel.pixels_to_unit`               transforms pixel coordinates into unit vectors in the camera frame
:meth:`~CameraModel.undistort_pixels`             takes a distorted pixel location and computes the corresponding
                                                  undistorted gnomic location in units of pixels
:meth:`~CameraModel.distort_pixels`               applies the distortion model to gnomic points with units of pixels
:meth:`~CameraModel.project_directions`           converts a direction vector expressed in the camera frame into 
                                                  a unit direction vector expressed in units of pixels in the 
                                                  image frame
:attr:`~CameraModel.estimation_parameters`        a list of strings containing what parameters to estimate when doing
                                                  geometric calibration.
:attr:`~CameraModel.state_vector`                 an NDArray containing the elements of the camera model (corresponding 
                                                  to the :attr:`~CameraModel.estimation_parameters`).
:meth:`~Cameramode.get_state_labels`              converts the list of estimation parameters into human readable
                                                  state label names (used for printing)
:meth:`~Cameramode.check_in_fov`                  computes whether provided vectors in the camera frame fall within the
                                                  field of view of the camera
================================================= ======================================================================

In addition the following methods and attributes are already implemented for most cases but may need to be overridden
for some special cases

================================================= ======================================================================
Method/Attribute                                  Use
================================================= ======================================================================
:meth:`~CameraModel.overwrite`                    overwrites the calling instance with the attributes of another
                                                  instance in place
:meth:`~CameraModel.distortion_map`               generates a set of pixel coordinates+distortion values that can be
                                                  used to create a distortion quiver or contour map.
:meth:`~CameraModel.undistort_image`              undistorts an entire image based on the distortion model (returns a
                                                  warped image)
:meth:`~CameraModel.copy`                         returns a copy of the current model
:meth:`~CameraModel.to_elem`                      a method that stores the model parameters in an element tree element
                                                  for saving the model to file
:meth:`~CameraModel.from_elem`                    a class method that retrieves the model parameters from an element
                                                  tree element for loading a model from a file
:attr:`~CameraModel.n_rows`                       The number of rows in pixels in an image captured by the device
                                                  modeled by this camera model
:attr:`~CameraModel.n_cols`                       The number of columns in pixels in an image captured by the device
                                                  modeled by this camera model
:attr:`~CameraModel.field_of_view`                Half the diagonal field of view of the detector in units of degrees.
================================================= ======================================================================

Finally, if the :meth:`~CameraModel.to_elem` and :meth:`~CameraModel.from_elem` methods are not being overridden, the
:attr:`~CameraModel.important_attributes` attribute should be extended with a list of attributes that must be
saved/loaded to completely reconstruct the camera model.
"""

import copy

from abc import ABCMeta, abstractmethod

import os

from importlib import import_module

import warnings

from enum import Enum

from itertools import repeat

from typing import Tuple, Union, Optional, List, Sequence, Iterable

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate as interp

# apparently lxml has security vulnerabilities but adding warning to documentation to avoid
# loading unverified files
import lxml.etree as etree  # nosec

from giant._typing import ARRAY_LIKE, SCALAR_OR_ARRAY, NONEARRAY, NONENUM, PATH, F_SCALAR_OR_ARRAY, DOUBLE_ARRAY, F_ARRAY_LIKE
from giant.image import OpNavImage


class ReturnShape(Enum):
    """
    This enumeration is used to specify what should be returned from method :meth:`~.CameraModel.undistort_image`.
    """

    FULL = "full"
    """
    Return the full undistorted image in a 2D array large enough to contain all pixels with valid data.  
    Pixels inside of the array which do not have valid data are filled with NaN.
    """

    SAME = "same"
    """
    Return an undistorted image in a 2D array of the same shape as the input image.  If the undistorted image is larger
    than the input image then it will be cropped.  If the undistorted image is smaller than the input image then it will
    be padded.  Pixels which do not have valid data are filled with NaN.
    """


class CameraModel(metaclass=ABCMeta):
    """
    This is the abstract base class for all camera models in GIANT.
    
    A camera model is a mapping from a 3D point expressed in the camera frame to a corresponding 2D point in the image.
    For more description of a camera model refer to the :mod:`.camera_models` documentation.
    
    This class serves as a prototype for implementing a :class:`!CameraModel` in GIANT.  It defines a number of abstract
    methods that need to be implemented for every camera model (:meth:`project_onto_image`, :meth:`compute_jacobian`,
    :meth:`compute_pixel_jacobian`, :meth:`compute_unit_vector_jacobian`, :meth:`apply_update`, :meth:`pixels_to_unit`,
    :meth:`undistort_pixels`, and :meth:`distort_pixels`)
    as well as a few concrete methods that are generally valid for all camera models (:meth:`overwrite`,
    :meth:`distortion_map`, :meth:`undistort_image`, :meth:`copy`, :meth:`to_elem`, :meth:`from_elem`).  This class also
    provides a few attributes (:attr:`field_of_view`, :attr:`n_rows`, :attr:`n_cols`, and :attr:`use_a_priori`) which
    are required for all models.

    Finally, this class provides the beginning of an attribute :attr:`important_attributes` which should be
    updated by each sub-class to ensure some core functionality is not broken (:meth:`__eq__`, :meth:`from_elem`, and
    :meth:`to_elem`).  Essentially, this should be a list of attributes that should (a) be checked when checking for
    equality between two models and (b) be added to/retrieved from elements when writing/reading a model to a file.  The
    values in this list should be valid attributes that return values using ``getattr(self, attr)``.
    
    .. note:: Because this is an ABC, you cannot create an instance of CameraModel (it will raise a ``TypeError``)
    """

    def __init__(self, field_of_view: NONENUM = 0.0, n_rows: int = 1, n_cols: int = 1, use_a_priori: bool = False):
        """
        :param field_of_view: The field of view of the camera in units of degrees.
        :param n_rows: The number of rows in the active pixel array for the camera
        :param n_cols: The number of columns in the active pixel array for the camera
        :param use_a_priori: A flag to specify whether to append the identity matrix to the Jacobian matrix returned
                             by :meth:`compute_jacobian` in order to include the current estimate of the camera model
                             in the calibration process.
        """

        self._field_of_view = 0.0

        self.n_rows = n_rows
        """
        The number of rows in the active pixel array for the camera
        """

        self.n_cols = n_cols
        """
        The number of columns in the active pixel array for the camera
        """

        # set the flag whether to use the current estimate of the model in the calibration
        self.use_a_priori = use_a_priori
        """
        This boolean value is used to determine whether to append the identity matrix to the Jacobian matrix returned 
        by :meth:`compute_jacobian` in order to include the current estimate of the camera model in the calibration 
        process.
        """

        self.important_attributes = ['field_of_view', 'n_rows', 'n_cols', 'use_a_priori']
        """
        A list specifying the important attributes the must be saved/loaded for this camera model to be completely 
        reconstructed. 
        """

        self.field_of_view = field_of_view

    def __eq__(self, other) -> bool:
        """
        Defines the equality check for all :class:`CameraModel` subclasses.

        Camera models are defined as equal if all of the :attr:`important_attributes` attributes are equivalent

        :param other: The other camera model to compare to
        :return: True if the camera models are equivalent, False if otherwise
        """

        # check to see if self and other are the same class
        if not isinstance(other, self.__class__):
            return False

        # check each variable in the important_attributes attribute and see if it is equivalent
        for var in self.important_attributes:

            mine = getattr(self, var)

            theirs = getattr(other, var)

            if not np.array_equal(mine, theirs):
                return False

        return True

    @property
    def field_of_view(self) -> float:
        """
        A radial field of view of the camera specified in degrees.
        
        The field of view should be set to at least the half width diagonal field of view of the camera. The field of
        view is used when querying star catalogs.
        
        The diagonal field of view is defined as
        
        .. code-block:: none
        
            +-----------+
            |          /|
            |         / |
            |        /  |
            |      V/   |
            |     O/    |
            |    F/     |
            |   */      |
            |  2/       |
            |  /        |
            | /         |
            |/          |
            +-----------+

        If you specify this parameter to be ``None``, the field of view will be computed using the camera model if
        possible.
        """

        return self._field_of_view

    @field_of_view.setter
    def field_of_view(self, val):
        if val is not None:
            try:
                self._field_of_view = float(val)
            except ValueError:
                raise ValueError("The field_of_view must be convertible to a float")

        else:
            self.compute_field_of_view()
            
    def compute_field_of_view(self, temperature: float = 0) -> None:
        """
        Computes the half diagonal field of view in degrees and stores it in the field of view argument.
        """
        
        try:
            self._field_of_view = np.arccos(np.prod(self.pixels_to_unit(np.array([[0, self.n_cols],
                                                                                    [0, self.n_rows]]),
                                                                        temperature=temperature),
                                                    axis=-1).sum()) * 90/np.pi   # 90/pi because we want half angle

        except (ValueError, TypeError, AttributeError, IndexError):
            self._field_of_view = 0.0

    @property
    @abstractmethod
    def estimation_parameters(self) -> List[str]:
        """
        A list of strings containing the parameters to estimate when performing calibration with this model.

        This list is used in the methods :meth:`compute_jacobian` and :meth:`apply_update` to determine which parameters
        are being estimated/updated. From the :meth:`compute_jacobian` method, only columns of the Jacobian matrix
        corresponding to the parameters in this list are returned.  In the :meth:`apply_update` method, the update
        vector elements are assumed to correspond to the order expressed in this list.

        Valid values for the elements of this list are dependent on each concrete camera model.  Generally, they
        correspond to attributes of the class, with a few convenient aliases that point to a collection of attributes.
        """
        pass

    @estimation_parameters.setter
    @abstractmethod
    def estimation_parameters(self, val: str | Sequence[str]):  # estimation_parameters should be writeable
        pass

    @property
    @abstractmethod
    def state_vector(self) -> List[float]:
        """
        Returns the fully realized state vector according to :attr:`estimation_parameters` as a length l list.
        """
        
    @property
    def state_vector_length(self) -> int:
        """
        Returns the length of the state vector of camera parameters 
        """
        return len(self.state_vector)

    @abstractmethod
    def get_state_labels(self) -> List[str]:
        """
        Convert a list of estimation parameters into state label names.

        This method interprets the list of estimation parameters (:attr:`estimation_parameters) into human readable
        state labels for pretty printing calibration results and for knowing the order of the state vector.
        In general this returns a list of attributes which can be retrieved from the camera using ``getattr`` with the
        exception of misalignment which must be handled separately.

        :return: The list of state names corresponding to estimation parameters in order
        """
        pass

    @abstractmethod
    def project_onto_image(self, points_in_camera_frame: ARRAY_LIKE, image: int = 0, temperature: float = 0) \
            -> np.ndarray:
        """
        This method transforms 3D points (or directions) expressed in the camera frame into the corresponding 2D image
        locations.
        
        The points input should be either 1 or 2 dimensional, with the first axis being length 3 (each point 
        (direction) in the camera frame is specified as a column).
        
        The optional ``image`` key word argument specifies the index of the image you are projecting onto (this only 
        applies if you have a separate misalignment for each image)

        The optional ``temperature`` key word argument specifies the temperature to use when projecting the points into
        the image.  This only applies when your focal length has a temperature dependence
        
        :param points_in_camera_frame: a shape (3,) or shape (3, n) array of points to project
        :param image: The index of the image being projected onto (only applicable with multiple misalignments)
        :param temperature: The temperature of the camera to use for the projection
        :return: A shape (2,) or shape (2, n) numpy array of image points (with units of pixels)
        """
        return np.zeros(2)

    @abstractmethod
    def project_directions(self, directions_in_camera_frame: ARRAY_LIKE, image: int = 0) -> np.ndarray:
        """
        This method transforms 3D directions expressed in the camera frame into the corresponding 2D image
        directions.

        The direction input should be either 1 or 2 dimensional, with the first axis being length 3 (each direction
        in the camera frame is specified as a column).

        The optional ``image`` key word argument specifies the index of the image you are projecting onto (this only
        applies if you have a separate misalignment for each image)

        This method is different from method :meth:`project_onto_image` in that it only projects the direction component
        perpendicular to the optical axis of the camera (x, y axes of the camera frame) into a unit vector in the image
        plane.  Therefore, you do not get a location in the image out of this, rather a unitless direction in the image.

        :param directions_in_camera_frame: a shape (3,) or shape (3, n) array of points to project
        :param image: The index of the image being projected onto (only applicable with multiple misalignments)
        :return: A shape (2,) or shape (2, n) numpy array of image direction unit vectors
        """
        return np.zeros(2)
 
    @abstractmethod
    def compute_jacobian(self, unit_vectors_in_camera_frame: Sequence[DOUBLE_ARRAY], temperature: F_SCALAR_OR_ARRAY | Sequence[float] = 0) \
            -> np.ndarray:
        r"""
        This method computes the Jacobian matrix :math:`\partial\mathbf{x}_P/\partial\mathbf{c}` where
        :math:`\mathbf{c}` is a vector of camera model parameters.
        
        The vector of camera model parameters contains things like the focal length, the pixel pitch, the distortion
        coefficients, and a misalignment vector.  The ``unit_vectors_in_camera_frame`` should be a shape (m, 3, n) array
        of unit vectors expressed in the camera frame that you wish to calculate the Jacobian for where m is the number
        of images being calibrated. (These unit vectors should correspond to the pixel locations of the measurements
        when projected through the model).
        
        In general this method will not be used by the user and instead is used internally by the calibration estimators
        in :mod:`.calibration`.
        
        :param unit_vectors_in_camera_frame: A (m, 3, n) array of unit vectors expressed in the camera frame
        :param temperature: The temperature of the camera to use for computing the Jacobian matrix.
                            If temperature is an array it must be the same length as the first axis of the
                            ``unit_vectors_in_camera_frame`` input.
        :return: A (n*2, o) (where o is the length of :math:`\mathbf{c}`) array containing the Jacobian matrix
        """
        return np.zeros((2, 1))

    @abstractmethod
    def compute_pixel_jacobian(self, vectors_in_camera_frame: ARRAY_LIKE, image: int = 0, temperature: float = 0) \
            -> np.ndarray:
        r"""
        This method computes the Jacobian matrix :math:`\partial\mathbf{x}_P/\partial\mathbf{x}_C` where
        :math:`\mathbf{x}_C` is a vector in the camera frame that projects to :math:`\mathbf{x}_P` which is the
        pixel location.

        This method is used in the :class:`.LimbScanning` process in order to predict the change in a projected pixel
        location with respect to a change in the projected vector.  The ``vectors_in_camera_frame`` input should
        be a 3xn array of vectors which the Jacobian is to be computed for.

        :param vectors_in_camera_frame: The vectors to compute the Jacobian at
        :param image: The image number to compute the the Jacobian for
        :param temperature: The temperature of the camera at the time the image was taken
        :return: The Jacobian matrix as a nx2x3 array
        """

        return np.zeros((1, 2, 3))

    @abstractmethod
    def compute_unit_vector_jacobian(self, pixel_locations: ARRAY_LIKE, image: int = 0, temperature: float = 0) -> \
            np.ndarray:
        r"""
        This method computes the Jacobian matrix :math:`\partial\mathbf{x}_C/\partial\mathbf{x}_P` where
        :math:`\mathbf{x}_C` is a vector in the camera frame that projects to :math:`\mathbf{x}_P` which is the
        pixel location.

        This method is used in the :class:`.LimbScanning` process in order to predict the change in the unit vector that
        projects to a pixel location with respect to a change in the pixel location.  The
        ``pixel_locations`` input should be a 2xn array of vectors which the Jacobian is to be computed for.

        :param pixel_locations: The pixel locations to compute the Jacobian at
        :param image: The image number to compute the the Jacobian for
        :param temperature: The temperature of the camera at the time the image was taken
        :return: The Jacobian matrix as a nx3x2 array
        """

        return np.zeros((1, 2, 3))

    @abstractmethod
    def apply_update(self, update_vec: F_ARRAY_LIKE):
        r"""
        This method takes in a delta update to camera parameters (:math:`\Delta\mathbf{c}`) and applies the update
        to the current instance in place.
        
        In general the delta update is calculated in the estimators in the :mod:`.calibration` subpackage and this
        method is not used by the user.
        
        The update vector is an array like object where each element corresponds to a specific camera parameter,
        corresponding to the element represented by each column coming from the :meth:`~CameraModel.compute_jacobian`
        method.  For a concrete example of the update vector and how it works, see the concrete camera model
        implementations.
        
        :param update_vec: delta updates to the model parameters
        """
        pass

    @abstractmethod
    def pixels_to_unit(self, pixels: F_ARRAY_LIKE, temperature: float = 0, image: int = 0) -> np.ndarray:
        """
        This method converts pixel image locations to unit vectors expressed in the camera frame.
        
        The pixel locations should be expressed as a shape (2,) or (2, n) array.  They are converted
        to unit vectors by first going through the inverse distortion model (see :meth:`undistort_pixels`) and then 
        being converted to unit vectors in the camera frame according to the definitions of the current model (also 
        including any misalignment terms).
        
        :param pixels: The image points to be converted to unit vectors in the camera frame as a shape (2,) or (2, n) 
                       array
        :param temperature: The temperature to use for the undistortion
        :param image: The image index that the pixels belong to (only important if there are multiple misalignments)
        :return: The unit vectors corresponding to the image locations expressed in the camera frame as a shape (3,) or
                 (3, n) array.
        """
        return np.zeros(3)

    @abstractmethod
    def undistort_pixels(self, pixels: F_ARRAY_LIKE, temperature: float = 0) -> np.ndarray:
        """
        This method computes undistorted pixel locations (gnomic/pinhole locations) for given distorted
        pixel locations according to the current model.

        The ``pixels`` input should be specified as a shape (2,) or (2, n) array of image locations with units of 
        pixels.  The return will be an array of the same shape as ``pixels`` with units of pixels but with distortion
        removed.
        
        :param pixels: The image points to be converted to gnomic (pinhole) locations as a shape (2,) or (2, n) array
        :param temperature: The temperature to use for the undistortion
        :return: The undistorted (gnomic) locations corresponding to the distorted pixel locations as an array of
                 the same shape as ``pixels``
        """
        return np.zeros(2)

    def overwrite(self, model: 'CameraModel'):
        """
        This method replaces self with the properties of ``model`` in place.
        
        This method is primarily used in the calibration classes to maintain the link between the internal and external
        camera models.  Essentially, each instance variable in ``self`` is overwritten by the corresponding instance
        variable in other.

        This method operates by looping through the properties defined in :attr:`important_attributes` and copying the
        value from ``model`` to ``self``.

        :param model: The model to overwrite self with
        :raises ValueError: When ``model`` is not the same type as ``self``
        """

        # check to see if the other model is the same type of self
        if not isinstance(model, self.__class__):
            raise ValueError('Models must be of same type to overwrite')

        # loop through each attribute in important_attributes and copy its value from model to self
        for attribute in self.important_attributes:
            setattr(self, attribute, getattr(model, attribute))

    @abstractmethod
    def distort_pixels(self, pixels: F_ARRAY_LIKE, temperature: float = 0) -> DOUBLE_ARRAY:
        """
        A method that takes gnomic pixel locations in units of pixels and applies the appropriate distortion to them.

        This method is used in the :meth:`distortion_map` method to generate the distortion values for each pixel.

        :param pixels: The pinhole location pixel locations the distortion is to be applied to
        :return: The distorted pixel locations in units of pixels
        """
        return np.zeros(2)
        

    def distortion_map(self, shape: None | Sequence[int] | NDArray[np.integer] = None, step: int = 1) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This method computes the value of the distortion model across an entire image for use in creating distortion 
        maps.
        
        The ``shape`` and ``step`` inputs to this method specify the size of the image (shape) as well as the size of
        the grid steps for computing the distortion values.  The locations the distortion values are computed for are
        generated by::

            rows, cols = np.meshgrid(np.arange(0, shape[0], step), np.arange(0, shape[1], step), indexing='ij')

        If shape is ``None`` then it is set to be ``(self.n_rows, self.n_cols)``.
            
        The value of the distortion is then computed for each row and column location in ``rows`` and ``cols`` and then
        returned, along with the ``rows`` and ``cols`` grids in units of pixels.
        
        In general this method will be used like::
            >>> import matplotlib.pyplot as plt
            >>> from giant.camera_models import CameraModel
            >>> inst = CameraModel(field_of_view=1)  # you can't actually do this
            >>> prows, pcols, dist = inst.distortion_map((1024, 1024), 100)
            >>> plt.figure()
            >>> cs = plt.contour(pcols, prows, np.linalg.norm(dist, axis=0).reshape(prows.shape))
            >>> plt.clabel(cs, inline=True, fontsize=10)
            >>> plt.figure()
            >>> plt.quiver(pcols.flatten(), prows.flatten(), dist[0], dist[1])
            
        to generate distortion maps of the current model.
         
        :param shape: The size of the image or None
        :param step: The size of the step to use in sampling the distortion field
        :return: a tuple containing the ``rows`` grid, ``cols`` grid, and a (2, ``rows.size``) array containing the
                 distortion values in pixels (first row = x distortion values, seconds row=y distortion values)
        """

        if shape is None:
            shape = (self.n_rows, self.n_cols)
        # get the pixels that we are calculating the distortion map for
        row_labels = np.arange(0, shape[0], step)
        col_labels = np.arange(0, shape[1], step)

        rows, cols = np.meshgrid(row_labels, col_labels, indexing='ij')

        pixels = np.array([cols.flatten().tolist(), rows.flatten().tolist()])

        # distort the pixels, calculate the distortion, and return the results
        return rows, cols, self.distort_pixels(pixels) - pixels

    def undistort_image(self, image: np.ndarray | list[np.ndarray | OpNavImage] | OpNavImage, 
                        return_shape: Union[ReturnShape, str, Iterable[ReturnShape | str]] = 'same') -> np.ndarray | list[NDArray]:
        """
        This method takes in an entire image and warps it to remove the distortion specified by the current model.
        
        The image should be input as a (n, m) array of gray-scale illumination values (DN values).
        
        The warping is formed by

        #. generating a grid of subscripts for each cell of the image (these are the distorted pixel locations)
        #. computing the corresponding gnomic location of these points using the :meth:`undistort_pixels` method
        #. re-sampling the undistorted image data to form a new image with distortion removed
        
        In general you should avoid using this function because it is much more computationally expensive than
        working with the nominal distorted image and then undistorting specific points for OpNav measurements.

        If ``return_shape`` is ``'same'`` then the returned image is the same size as the input image (and the
        undistorted image is either cropped or padded to fit this shape).  If ``return_shape`` is ``'full'`` then the
        returned image is the size of what the detector would need to be to capture the image from the camera if it
        was a pinhole model.
                  
        :param image: The image to have the distortion removed from as a (n, m) array of gray-scale illumination values
        :param return_shape: Specify whether to return the full undistorted image or the undistorted image set to the
                             same size as the original
        :return: The undistorted image as an array of shape (n, m) illumination values

        .. note:: The re-sampled image has NaN specified for anywhere that would be considered extrapolation in the
                  re-sampling process.  This means that the undistorted image will generally look somewhat weird around
                  the edges.
        """
        
        temp_image = image if not isinstance(image, list) else image[0]

        row_labels = np.arange(temp_image.shape[0])
        col_labels = np.arange(temp_image.shape[1])

        rows, cols = np.meshgrid(row_labels, col_labels, indexing='ij')

        pixel_subs = np.array([cols.flatten().tolist(), rows.flatten().tolist()])

        undistorted_subs = self.undistort_pixels(pixel_subs, temperature=getattr(temp_image, "temperature", 0.0))

        points = undistorted_subs.T
        
        if isinstance(image, list):
            if isinstance(return_shape, (str, ReturnShape)):
                return_shape = repeat(return_shape)
            res = []
            for im, rs in zip(image, return_shape):

                if ReturnShape(rs) == ReturnShape.SAME:
                    new_subs = pixel_subs[::-1].T
                    shape = im.shape
                else:
                    start = np.ceil(points.min(axis=0)).astype(int)
                    stop = np.floor(points.max(axis=0)).astype(int) + 1
                    new_c = np.arange(start[0], stop[0])
                    new_r = np.arange(start[1], stop[1])
                    gridded_r, gridded_c = np.meshgrid(new_r, new_c, indexing='ij')

                    new_subs = np.vstack([gridded_r.ravel(), gridded_c.ravel()])

                    shape = gridded_r.shape

                res.append(interp.griddata(points, im.ravel(), new_subs, fill_value=np.nan, method='linear').reshape(shape))
            return res
        else:
            if ReturnShape(return_shape) == ReturnShape.SAME:
                new_subs = pixel_subs[::-1].T
                shape = image.shape
            else:
                start = np.ceil(points.min(axis=0)).astype(int)
                stop = np.floor(points.max(axis=0)).astype(int) + 1
                new_c = np.arange(start[0], stop[0])
                new_r = np.arange(start[1], stop[1])
                gridded_r, gridded_c = np.meshgrid(new_r, new_c, indexing='ij')

                new_subs = np.vstack([gridded_r.ravel(), gridded_c.ravel()])

                shape = gridded_r.shape

            return interp.griddata(points, image.ravel(), new_subs, fill_value=np.nan, method='linear').reshape(shape)

    def copy(self) -> 'CameraModel':
        """
        Returns a deep copy of this object, breaking all references with ``self``.
        
        :return: A copy of self that is a separate object
        """
        return copy.deepcopy(self)

    def to_elem(self, elem: etree._Element, misalignment: bool = False) -> etree._Element:
        """
        Stores this camera model in an :class:`lxml.etree.SubElement` object for storing in a GIANT xml file

        This method operates by looping through the attributes in :attr:`important_attributes`, retrieving the value of
        these attributes in self, and then storing them as a sub-element to ``elem``.  If the attribute already exists
        as a sub-element to ``elem`` then it is overwritten.

        The user generally will not use this method and instead will use the module level :func:`save` function.
        
        :param elem: The :class:`lxml.etree.SubElement` class to store this camera model in
        :param misalignment: whether to save the misalignment in the structure (usually you want false).  As is, this 
                             does nothing, subclasses which implement misalignment though should make use of this flag
        :return: The :class:`lxml.etree.SubElement` for this model
        """

        # loop attributes included in this instance's import vars
        for name in self.important_attributes:

            val = getattr(self, name)

            # see if this attribute already exists in the subElement
            node = elem.find(name)

            if node is None:  # if it doesn't, add it
                node = etree.SubElement(elem, name)

            # store the value of this attribute in the subElement
            node.text = ' '.join(repr(val).split())

        return elem

    @classmethod
    def from_elem(cls, elem: etree._Element) -> 'CameraModel':
        """
        This class method is used to construct a new instance of `cls` from an :class:`etree._Element` object

        This method works by first creating an initialized instance of the class.  It then loops through each attribute
        defined in the :attr:`important_attributes` list and searches the element to see if it contains information
        about the current attribute.  If the element contains information for the specified attribute, then this
        information is set in the initialized instance of this class.  If information is not found for the current
        attribute, then a warning is thrown that the element does not contain all the information necessary to define
        the :attr:`important_attributes`.

        .. note:: The user will generally not use this method and instead will use the module level :func:`load`
                  function to retrieve a camera model from a file

        :param elem: The element containing the attribute information for the instance to be created
        :return: An initialized instance of this class with the attributes set according to the `elem` object
        """

        from numpy import array
        from giant.rotations import Rotation

        # create an instance of class.  I'm not sure why copy is needed here but weird things happen if you don't
        inst = cls().copy()

        # loop attributes included in this class's __dict__ attribute and see if they are store in the element
        for prop in inst.important_attributes:

            # try to find this attribute in the subElement
            node = elem.find(prop)

            if node is None:  # if we couldn't find the attribute in the subElement raise a warning and move to the next
                warnings.warn('missing value for {0}'.format(prop))
                continue

            # set the instance attribute with the value from the subElement
            # eval is a security risk, but the warning to not load unverified
            # files is probably sufficient.  Unfortunately I can't see any way
            # around the security threat without going through an extended parser
            setattr(inst, prop, eval(node.text))  # nosec

        return inst

    def instantaneous_field_of_view(self, temperature: float = 0,
                                    center: NONEARRAY = None,
                                    direction: NONEARRAY = None) -> np.ndarray:
        """
        Compute the Instantaneous Field of View (FOV of a single pixel) for the given temperature, location on the focal
        plane, and direction.

        This is computed by determining the line of sight through the center pixel, then the center pixel + the
        direction, and then computing then angle between them (and dividing by the norm of the direction in case it
        isn't 1).  The result will give the IFOV in radians.

        If you do not specify the center or direction, they will be assumed to be the principal point and the x axis
        respectively. Note that this assumes that the principal axis is along the z-axis of the camera frame.  If this
        is not the case for your camera then you must specify the center.

        :param temperature: the temperature at which to compute the IFOV
        :param center: The pixel to compute the IFOV for.  If None then defaults to the principal point
        :param direction: The direction to compute the IFOV in as a length 2 unit vector
        :return: The IFOV of the detector
        """

        if center is None:
            # get the principal point
            center_dir = np.array([[0.], [0.], [1.]])
            center = self.project_onto_image(center_dir, temperature=temperature)
        else:
            center_dir = self.pixels_to_unit(center, temperature=temperature).reshape(3, -1)

        if direction is None:
            direction = np.array([[1], [0.]])

        step_dir = self.pixels_to_unit(center.reshape(2, -1)+direction.reshape(2, -1),
                                       temperature=temperature)

        # compute the IFOV
        return np.arccos((step_dir*center_dir).sum(axis=0))/np.linalg.norm(direction, axis=0, keepdims=True)

    def compute_ground_sample_distance(self, target_position: ARRAY_LIKE,
                                       target_normal: NONEARRAY = None,
                                       camera_step_direction: NONEARRAY = None,
                                       temperature: float = 0) -> SCALAR_OR_ARRAY:
        r"""
        Compute the ground sample distance of the camera at the targets.

        The ground sample distance is computed using

        .. math::

            g = x_1+x_2

        where

        :math:`g` is the ground sample distance,

        .. math::

            x_1 = \frac{r\sin{\theta/2}}{\sin{\delta}}, \\
            x_2 = \frac{r\sin{\theta/2}}{\sin{\gamma}},

        :math:`r=\|\mathbf{r}\|` is the length of the target position vectors :math:`\mathbf{r}`, :math:`\theta` is the
        instantaneous field of views of the detector in the ``camera_step_direction`` towards the target positions,
        :math:`\delta=\frac{\pi}{2}-\theta+\beta`, :math:`\gamma=\frac{\pi}{2}-\theta-\beta`,
        :math:`\beta=\cos^{-1}{\mathbf{n}^T\frac{-\mathbf{r}}{r}}` and :math:`\mathbf{n}` is the unit normal vectors
        ``target_normal_vector``.

        If the ``target_normal_vector`` is ``None``, then it is assumed to be along the line of sight from the camera to
        the targets so that :math:`\delta=\gamma`.  The camera IFOV is computed using
        :meth:`instantaneous_field_of_view`.

        :param target_position: The location of the targets as a 3xn array
        :param target_normal: ``None`` or the unit normal vector of the targets in the camera frame as a 3xn array.
                              If ``None``, the normal vector is assumed to be along the line of sight vector
        :param camera_step_direction: ``None`` or the pixel direction to step when computing the IFOV as a length 2
                                       array.  If ``None``, the x direction is assumed.
        :param temperature: The temperature of the camera when the GSD is to be computed.  This is used in the IFOV
                            calculation.
        :return: The ground sample distances of the camera in the same units as the provided ``target_position`` vector.
        """

        # make sure the position vector is an array with the appropriate shape
        target_position = np.array(target_position).reshape(3, -1)

        # get the distance to the target
        target_distance = np.linalg.norm(target_position, axis=0, keepdims=True)

        line_of_sight_vector = -target_position/target_distance

        # set the target normal vector to be the line of sight, if not provided/ensure its an appropriate shape array
        if target_normal is None:
            target_normal = line_of_sight_vector
        else:
            target_normal = np.array(target_normal).reshape(3, -1)

        # compute the location of the target in the image for computing the IFOV
        target_center = self.project_onto_image(target_position, temperature=temperature)

        # get the IFOV in radians
        ifov = self.instantaneous_field_of_view(temperature=temperature,
                                                center=target_center,
                                                direction=camera_step_direction)
        theta = ifov/2

        # compute the interior angle between the line of sight vector and the normal vector in radians
        gamma = np.arccos(np.clip((target_normal*line_of_sight_vector).sum(axis=0), -1, 1))

        # compute r times half the IFOV
        r_sin_theta = target_distance*np.sin(theta)

        # compute the short side distance
        gsd_short = r_sin_theta/np.sin(np.pi/2-theta+gamma)

        # compute the long side distance
        gsd_long = r_sin_theta/np.sin(np.pi/2-theta-gamma)

        return np.abs(gsd_short+gsd_long)

    @abstractmethod
    def check_in_fov(self, vectors: F_ARRAY_LIKE, image: int = 0, temperature: float = 0) -> NDArray[np.bool]:
        """
        Determines if any points in the array are within the field of view of the camera.

        :param vectors: Vectors to check if they are in the field of view of the camera expressed as a shape (3, n) array in the camera frame.  
        :param image: The index of the image being projected onto (only applicable with multiple misalignments)
        :param temperature: The temperature of the camera to use for the projection
        :return: A boolean array the same length as the number of columns of vectors. False by default, True if the point is in the FOV.
        """
        ...
    


def save(file: PATH, name: str, model: CameraModel, group: Optional[str] = None, misalignment: bool = False):
    """
    This function is used to save a camera model to a GIANT xml file.

    The models are stored as plain text xml trees, where each property is a node of the tree.  The root element for
    the camera models is called `CameraModels`.  You can also optionally specify a `group` in order to be able to
    collect similar camera models together.

    The xml file stores all information necessary for recreating the camera model when it is loaded from a file.  This
    includes the module that defines the camera model, as well as the name of the class that the camera model was an
    instance of.  When saving the camera model to file, this function first looks to see if a camera model of the same
    name and group already exists in the file.  If it does then that camera model is overwritten with the new values.
    If it does not, then the current camera model is added to the file.

    Camera models are converted into xml using the :meth:`~CameraModel.to_elem` method of the class.  This method is
    defined in the :class:`CameraModel` class and thus all models that subclass :class:`CameraModel` (as they should)
    are usable with this function.

    There is an optional keyword argument group which can be used to store the camera model in a sub node of the xml
    tree.  This is mostly just used to organize the save file and allow faster lookup when the file becomes large, but
    it can also be used to distinguish between multiple camera models with the same name, though this is not
    recommended.

    Finally, there is a `misalignment` flag which specifies whether you want to save the misalignment values in the
    file.  This should generally be left as false, which resets the misalignment in the model to be a single
    misalignment of [0, 0, 0] and adjusts the :attr:`~.CameraModel.estimation_parameters` attribute accordingly.  If set
    to true, then the misalignment is stored exactly as it is in the camera model.

    .. warning::
        There is a security risk when loading XML files (exacerbated here by using a eval on some of the field of the
        xml tree).  Do not pass untrusted/unverified files to this function. The files themselves are simple text files
        that can easily be verified for malicious code by inspecting them in a text editor beforehand.

    :param file:  The path of the file to store the camera model in
    :param name: The name to use to store the camera model (i.e. 'Navigation Camera')
    :param model:  The instance of the camera model to store. Should be a subclass of :class:`CameraModel`
    :param group: An optional group to store the camera model into.
    :param misalignment: A flag specifying whether to include the misalignment values in the save file or not.
    """

    if os.path.isfile(file):
        # both etree parse are technically security risks but the user is warned to
        # verify files before loading them since they are easy to inspec
        if isinstance(file, str):
            tree = etree.parse(file)  # nosec
        else:
            tree = etree.parse(str(file))  # nosec

        root = tree.getroot()

    else:

        root = etree.Element('CameraModels')

        tree = etree.ElementTree(root)

    if group is not None:

        group_elem = root.find(group)

        if group_elem is None:
            group_elem = etree.SubElement(root, group)

    else:

        group_elem = root

    model_elem = group_elem.find(name)

    if model_elem is None:
        model_elem = etree.SubElement(group_elem, name, attrib={"module": model.__module__,
                                                                "type": type(model).__name__})

    model.to_elem(model_elem, misalignment=misalignment)

    with open(file, 'wb') as out:

        out.write(etree.tostring(tree, pretty_print=True))


def load(file: PATH, name: str, group: Optional[str] = None) -> CameraModel:
    """
    This function is used to retrieve a camera model from a GIANT xml file.

    This function will return the queried camera model if it exists, otherwise it raises a LookupError.

    If you saved your camera model to a specific group, you can optionally specify this group which may make the search
    faster.  If you have two camera models with the same name but different groups then you must specify group.

    .. warning::
        There is a security risk when loading XML files (exacerbated here by using a eval on some of the field of the
        xml tree and by importing the module the camera model is defined in).  Do not pass untrusted/unverified files to
        this function. The files themselves are simple text files that can easily be verified for malicious code by
        inspecting them in a text editor beforehand.

    :param file: The path to the xml file to retrieve the camera models from.
    :param name: The name of the camera model to retrieve from the file
    :param group: The group that contains the camera model in the file
    :return: The camera model retrieved from the file
    :raises LookupError: when the camera model can't be found in the file
    """

    # both etree parse are technically security risks but the user is warned to
    # verify files before loading them since they are easy to inspec
    if isinstance(file, str):
        tree = etree.parse(file)  # nosec
    else:
        tree = etree.parse(str(file))  # nosec

    root = tree.getroot()

    if group is not None:

        path = group + '/' + name

    else:

        path = './/' + name

    elem = root.find(path)

    if elem is not None:

        mod = import_module(elem.get('module'))

        cls = getattr(mod, elem.get('type'))

        return cls.from_elem(elem)

    else:
        raise LookupError('The specified camera model could not be found in the file') 
