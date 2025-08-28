# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


r"""
This module provides a subclass of :class:`.CameraModel` that implements the distortion free Pinhole camera model.

Theory
______

Recall the experiment that you can use to observe a solar eclipse without damaging your eyes.  You take a piece of
paper, place a small hole in it, and hold it some distance above another sheet of paper.  The sun is then projected onto
the lower piece of paper showing how much is currently obscured by the moon.  The sun on the paper appears much smaller
than the sun in the world because the distance between the two sheets of paper is much smaller than the distance
between the top sheet of paper and the sun.  This is an example of a Pinhole camera, which assumes similar triangles to
relate points in the 3D world.  This is demonstrated in the following diagram.

.. figure:: images/pinholecamera.png
   :alt: The pinhole camera model
   :target: _downloads/cameraModels.pdf

   The pinhole camera model describes a gnomic projection from 3 space to 2 space.

In the above figure, point :math:`\mathbf{x}_B` is rotated and translated to be expressed in the camera frame as point
:math:`\mathbf{x}_C` and this is then projected through the pinhole camera model to point :math:`\mathbf{x}_P` in the
image.  Mathematically this is given as

.. math::
    :nowrap:

    \begin{gather}
    &\mathbf{x}_I = (1+a_1T+a_2T^2+a_3T^3)\frac{f}{z_C}\left[\begin{array}{c}x_C\\y_C\end{array}\right]\\
    &\mathbf{x}_P = \left[\begin{array}{ccc} k_x & 0 & p_x \\ 0 & k_y & p_y\end{array}\right]
    \left[\begin{array}{c} \mathbf{x}_I \\ 1 \end{array}\right]
    \end{gather}

where :math:`f` is the focal length of the camera (the distance between the 2 sheets of paper in our example),
:math:`a_{1-3}` are polynomial coefficients for a temperature dependence on focal length (the camera dimensions may
change due to thermal expansion), :math:`T` is the temperature the projection is occurring at, :math:`k_x` and
:math:`k_y` are one over the pixel pitch values in units of pixels/distance in the :math:`x` and :math:`y` directions
respectively (cameras are not continuous but have discrete receptors for light to enter), and :math:`p_x` and
:math:`p_y` are the location of the principal point of the camera in the image expressed in units of pixels (typically
at the center of the pixel array).

Speeding up the camera model
----------------------------

One of the most common functions of the camera model is to relate pixels in a camera to unit vectors in the 3D camera
frame.  This is done extensively throughout GIANT, particularly when ray tracing.  Unfortunately, this transformation is
iterative (there isn't an analytic solution), which can make things a little slow, particularly when you need to do the
transformation for many pixel locations.

In order to speed up this transformation we can precompute it for each pixel in an detector and for a range of
temperatures specified by a user and then use bilinear interpolation to compute the location of future pixel/temperature
combinations we need.  While this is an approximation, it saves significant time rather than going through the full
iterative transformation, and based on testing, it is accurate to a few thousandths of a pixel, which is more than
sufficient for nearly every use case.  The :class:`.PinholeModel` and its subclasses make precomputing the
transformation, and using the precomputed transformation, as easy as calling :meth:`~PinholeModel.prepare_interp`
once.  Future calls to any method that then needs the transformation from pixels to gnomic locations (on the way to
unit vectors) will then use the precomputed transformation unless specifically requested otherwise.  In addition,
once the :meth:`~PinholeModel.prepare_interp` method has been called, if the resulting camera object is then saved to
a file either using the :mod:`.camera_model`
:func:`~giant.camera_models.camera_model.save`/:func:`~giant.camera_models.camera_model.load` functions  or another
serialization method like pickle/dill, then the precomputed transformation will also be saved and loaded so that it
truly only needs to be computed once.

Since precomputing the transformation can take a somewhat long time, it is not always smart to do so.  Typically if you
have a camera model that you will be using again and again (as is typical in most operations and analysis cases) then
you *should* precompute the transformation and save the resulting camera object to a file that is then used for future
work.  This is usually best done at the end of a calibration script (for a real camera) or in a stand-alone script that
defines the camera, precomputes the transformation, and then saves it off for a synthetic camera for analysis.  If you
are just doing a quick analysis and don't need the camera model repeatedly or for any heavy duty ray tracing then it is
recommended that you *not precompute* the transformation.

Whether you precompute the transformation or not, the use of the camera model should appear unchanged beyond computation
time.

Use
___

This is a concrete implementation of a :class:`.CameraModel`, therefore to use this class you simply need to initialize
it with the proper values.  Typically these values come from either the physical dimensions of the camera, or from
a camera calibration routine performed to refine the values using observed data (see the :mod:`.calibration` sub-package
for details).  For instance, say we have a camera which has an effective focal length of 10 mm, a pix pitch of 2.2 um,
and a detector size of 1024x1024.  We could then create a model for this camera as

    >>> from giant.camera_models import PinholeModel
    >>> model = PinholeModel(focal_length=10, kx=1/2.2e-3, ky=1/2.2e-3,
    ...                      n_rows=1024, n_cols=1024, px=(1024-1)/2, py=(1024-1)/2)

Note that we did not set the field of view, but it is automatically computed for us based off of the prescribed camera
model.

    >>> model.field_of_view
    9.050999753955251

In addition, we can now use our model to project points

    >>> model.project_onto_image([0, 0, 1])
    array([511.5, 511.5])

or to determine the unit vector through a pixel

    >>> model.pixels_to_unit([[0, 500], [0, 100]])
    array([[-0.11113154, -0.00251969],
           [-0.11113154, -0.090161  ],
           [ 0.98757256,  0.99592402]])
"""


from typing import Tuple, Sequence, Iterable, Union, List, cast

# from warnings import warn

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from numpy.typing import NDArray

# the risk of XML is addressed with warnings in the save/load documentation
import lxml.etree as etree  # nosec

from giant.camera_models.camera_model import CameraModel
from giant.rotations import rotvec_to_rotmat, skew, Rotation
from giant._typing import ARRAY_LIKE, NONEARRAY, SCALAR_OR_ARRAY, NONENUM, DOUBLE_ARRAY, F_SCALAR_OR_ARRAY, F_ARRAY_LIKE


class PinholeModel(CameraModel):
    r"""
    This class provides an implementation of the pinhole camera model for projecting 3d points onto images.

    The :class:`PinholeModel` class is a subclass of :class:`CameraModel`.  This means that it includes implementations
    for all of the abstract methods defined in the :class:`CameraModel` class.  This also means that it can be used
    throughout GIANT as the primary camera model, including within the :mod:`calibration` subpackage.  If this class is
    going to be used with the :mod:`calibration` subpackage, the user can set which parameters are estimated and which
    are held fixed by using the ``estimation_parameters`` keyword argument when creating an instance of the class or by
    adjusting the :attr:`estimation_parameters` instance variable on an instance of the class.  The
    ``estimation_parameters`` input/attribute is a string or list of strings specifying which parameters to estimate.
    This means that :attr:`estimation_parameters` could be something like ``'basic'`` which would indicate to estimate
    just the usual parameters, or something like ``['focal_length', 'ky', 'px', 'py']`` to estimate just the terms
    included in the list.

    In addition to the standard set of methods for a :class:`CameraModel` subclass, the :class:`PinholeModel` class
    provides the following additional methods which may or may not be useful to some people:

    =================================  =================================================================================
    Method                             Use
    =================================  =================================================================================
    :meth:`get_projections`            computes the pinhole, image frame, and pixel locations of a 3D point
    :meth:`pixels_to_gnomic`           removes distortion from a point to get the corresponding pinhole location in
                                       units of distance
    =================================  =================================================================================

    The Pinhole model also provides the following additional properties for easy getting/setting:

    ============================  ======================================================================================
    Property                      Description
    ============================  ======================================================================================
    :attr:`field_of_view`         the diagonal field of view of the camera in units of degrees
    :attr:`focal_length`          the diagonal focal length of the camera in units of distance
    :attr:`kx`                    :math:`k_x`, the inverse of the pixel pitch in the x direction in units of
                                  pixels/distance
    :attr:`ky`                    :math:`k_y`, the inverse of the pixel pitch in the y direction in units of
                                  pixels/distance
    :attr:`px`                    :math:`p_{x}`, the x axis pixel location of the principal point of the camera in units
                                  of pixels
    :attr:`py`                    :math:`p_{y}`, the y axis pixel location of the principal point of the camera in units
                                  of pixels
    :attr:`a1`                    :math:`a_1`, the linear coefficient for focal length dependent focal length
    :attr:`a2`                    :math:`a_2`, the quadratic coefficient for focal length dependent focal length
    :attr:`a3`                    :math:`a_3`, the cubic coefficient for focal length dependent focal length
    :attr:`intrinsic_matrix_inv`  The inverse of the intrinsic matrix
    ============================  ======================================================================================
    """

    def __init__(self, intrinsic_matrix: NONEARRAY = None, focal_length: float = 1.,
                 field_of_view: NONENUM = None, use_a_priori: bool = False,
                 misalignment: NONEARRAY = None, estimation_parameters: str | Sequence[str] = 'basic',
                 kx: NONENUM = None, ky: NONENUM = None, px: NONENUM = None, py: NONENUM = None, n_rows: int = 1,
                 n_cols: int = 1, temperature_coefficients: NONEARRAY = None, a1: NONENUM = None, a2: NONENUM = None,
                 a3: NONENUM = None):
        """
        :param intrinsic_matrix: the intrinsic matrix for the camera as a numpy shape (2, 3) array.  Note that this is
                                 overwritten if ``kx``, ``ky``, ``px``, ``py`` are also specified.
        :param focal_length: The focal length of the camera in units of distance.
        :param field_of_view: The field of view of the camera in units of degrees.
        :param use_a_priori: A flag to indicate whether to include the *a priori* state vector in the Jacobian matrix
                             when performing a calibration
        :param misalignment: either a numpy array of shape (3,) or a list of numpy arrays of shape(3,) with each array
                             corresponding to a single image (the list of numpy arrays is only valid when estimating
                             multiple misalignments)
        :param estimation_parameters: A string or list of strings specifying which model parameters to include in the
                                      calibration
        :param kx: The inverse of the pixel pitch along the x axis in units of pixel/distance
        :param ky: The inverse of the pixel pitch along the y axis in units of pixel/distance
        :param px: the x component of the pixel location of the principal point in the image in units of pixels
        :param py: the y component of the pixel location of the principal point in the image in units of pixels
        :param temperature_coefficients: The temperature polynomial coefficients as a length 3 Sequence
        :param a1: the linear coefficient of the focal length temperature dependence
        :param a2: the quadratic coefficient of the focal length temperature dependence
        :param a3: the cubic coefficient of the focal length temperature dependence
        :param n_rows: the number of rows of the active image array
        :param n_cols: the number of columns in the active image array
        """

        self._state_labels = ['focal_length', 'kx', 'ky',
                              'px', 'py', 'a1', 'a2', 'a3', 'misalignment']
        """
        A list of state labels that correspond to the attributes of this class.
        """
        
        # store the element dict for indices into the state vector
        self.element_dict = {
            'basic': [0, 2],
            'intrinsic': np.arange(0, 5),
            'basic intrinsic': [0, 2],
            'temperature dependence': [5, 6, 7],
            'focal_length': [0],
            'kx': [1],
            'ky': [2],
            'px': [3],
            'py': [4],
            'a1': [5],
            'a2': [6],
            'a3': [7],
            'single misalignment': slice(8, None, None),
            'multiple misalignments': slice(8, None, None)
        }

        # set the focal length property
        self._focal_length = 1.0
        self.focal_length = focal_length

        # set the intrinsic matrix
        self.intrinsic_matrix = np.zeros((2, 3))
        r"""
        The 2x3 intrinsic matrix contains the conversion from unitless gnomic locations to a location in an image with 
        units of pixels.
        
        It is defined as 
        
        .. math::
            \mathbf{K} = \left[\begin{array}{ccc} k_x & 0 & p_x \\
            0 & k_y & p_y \end{array}\right] 
        """

        if intrinsic_matrix is not None:
            self.intrinsic_matrix = np.asarray(intrinsic_matrix)

        if kx is not None:
            self.kx = kx
        if ky is not None:
            self.ky = ky
        if px is not None:
            self.px = px
        if py is not None:
            self.py = py

        self.temperature_coefficients = np.zeros(3)
        """
        The coefficients for the polynomial specifying the change in the focal length as a function of temperature.
        """

        # set the temperature dependence
        if temperature_coefficients is not None:
            self.temperature_coefficients = temperature_coefficients

        if a1 is not None:
            self.a1 = a1

        if a2 is not None:
            self.a2 = a2

        if a3 is not None:
            self.a3 = a3

        # set the misalignment attribute
        self.misalignment: DOUBLE_ARRAY | list[DOUBLE_ARRAY] = np.zeros(3)
        """
        Contains either a single rotation vector representing the misalignment between the specified camera frame and
        the actual camera frame, or a list of rotation vectors representing the misalignments between the specified 
        camera frame and the actual camera frame for each image.
        
        Typically you should not interface with this attribute directly and allow other GIANT objects to handle it, 
        because it can get complicated to ensure it is in-sync with the number of images under consideration
        """

        if misalignment is not None:
            self.misalignment = misalignment

        # set a flag for where to use multiple misalignments or not (set by estimation_parameters property)
        self.estimate_multiple_misalignments = False
        """
        This boolean value is used to determine whether multiple misalignments are being estimated/used per image.
        
        If set to ``True`` then one misalignment is estimated for each image and used for each image when projecting
        through the camera model.  When set to ``False`` then a single misalignment is estimated for all images and
        used for all images when projecting through the camera model.  Typically the user shouldn't be setting this 
        attribute directly as it is automatically handled when setting the :attr:`estimation_parameters` attribute
        """

        # set the estimation parameters attribute
        self._estimation_parameters: list[str] = []
        self.estimation_parameters = estimation_parameters

        self._fix_misalignment = []

        self._interp = None
        """
        An instance of SciPy's RegularGridInterpolator for converting pixels to gnomic coordinates.

        This is generated by a call to :meth:`prepare_interp`
        """

        # call the super init
        super().__init__(n_rows=n_rows, n_cols=n_cols,
                         use_a_priori=use_a_priori, field_of_view=field_of_view)

        # store the important attributes for use in the proper functions
        # temporarily duplicate the docstring so that sphinx can pick it up
        self.important_attributes = self.important_attributes + ['kx', 'ky', 'px', 'py', 'focal_length', 'misalignment',
                                                                 'estimate_multiple_misalignments',
                                                                 'estimation_parameters',
                                                                 'a1', 'a2', 'a3', '_interp']
        """
        A list specifying the important attributes the must be saved/loaded for this camera model to be completely 
        reconstructed. 
        """

    def __repr__(self):

        template = "PinholeModel(kx={kx}, ky={ky}, px={px}, py={py}, focal_length={f},\n" \
                   "             a1={a1}, a2={a2}, a3={a3}, n_rows={n_rows}, n_cols={n_cols},\n" \
                   "             field_of_view={fov}, misalignment={mis!r}, \n" \
                   "             estimation_parameters={ep!r}, use_a_priori={ap})\n\n"

        return template.format(
            kx=self.kx, ky=self.ky, px=self.px, py=self.py,
            fov=self.field_of_view, f=self.focal_length,
            mis=self.misalignment, ep=self.estimation_parameters, ap=self.use_a_priori,
            a1=self.a1, a2=self.a2, a3=self.a3, n_rows=self.n_rows, n_cols=self.n_cols
        )

    def __str__(self):
        template = u"Pinhole Camera Model:\n\n" \
                   u" __  __     __       __                              \n" \
                   u"|   x  |   |  f*Xc/Zc  |                   2      3  \n" \
                   u"|      | = |           | * (1 + a1*T + a2*T + a3*T ) \n" \
                   u"|   y  |   |  f*Yc/Zc  |                             \n" \
                   u" --  --     --       --                              \n" \
                   u" __ __     __          __  __ __  \n" \
                   u"|  u  | _ |  kx  0   px  ||  x  | \n" \
                   u"|  v  | - |  0   ky  py  ||  y  | \n" \
                   u" -- --     --          -- |  1  | \n" \
                   u"                           -- --  \n\n" \
                   u"—————————————————————————————————————————\n\n" \
                   u"camera parameters:\n" \
                   u"    f={4}, kx={0}, ky={1}, px={2}, py={3}\n\n" \
                   u"temperature coefficients:\n" \
                   u"    a1={5}, a2={6}, a3={7}\n\n"

        return template.format(self.kx, self.ky, self.px, self.py, self.focal_length,
                               self.a1, self.a2, self.a3)

    @CameraModel.state_vector.getter
    def state_vector(self) -> List[float]:
        """
        Returns the fully realized state vector according to :attr:`estimation_parameters` as a length l list.
        """

        state_vector = []
        for label in self.get_state_labels():
            if 'misalignment' != label:
                state_vector.append(getattr(self, label))
            else:
                if self.estimate_multiple_misalignments:
                    assert isinstance(self.misalignment, list), "something went wrong if this fails"
                    for r in self.misalignment:
                        state_vector.extend(r)
                else:
                    state_vector.extend(self.misalignment)

        return state_vector
    
    def get_state_labels(self) -> List[str]:
        """
        Convert a list of estimation parameters into state label names.

        This method interprets the list of estimation parameters (:attr:`estimation_parameters) into state labels for
        pretty printing calibration results.  In general this returns a list of attributes which can be retrieved from
        the camera using ``getattr`` with the exception of misalignment which must be handled separately.

        :return: The list of state names corresponding to estimation parameters in order
        """

        olist = []

        for param in self.estimation_parameters:
            if 'misalignment' not in param:
                locs = self.element_dict[param]
                for loc in locs:
                    if (param == "basic") and loc >= len(self._state_labels):
                        # this is a misalignment location we should ignore for now
                        continue
                    olist.append(self._state_labels[loc])

            else:
                olist.append(self._state_labels[-1])

        return olist

    @property
    def estimation_parameters(self) -> list[str]:
        r"""
        A list of strings containing the parameters to estimate when performing calibration with this model.

        This list is used in the methods :meth:`compute_jacobian` and :meth:`apply_update` to determine which parameters
        are being estimated/updated. From the :meth:`compute_jacobian` method, only columns of the Jacobian matrix
        corresponding to the parameters in this list are returned.  In the :meth:`apply_update` method, the update
        vector elements are assumed to correspond to the order expressed in this list.

        Valid values for the elements of this list are shown in the following table.  Generally, they correspond to
        attributes of this class, with a few convenient aliases that point to a collection of attributes.

        .. _pinhole-estimation-table:

        ============================  ==================================================================================
        Value                         Description
        ============================  ==================================================================================
        ``'basic'``                   estimate focal length, ky, and a single misalignment
                                      term for all images between the camera attitude and the spacecraft's attitude:
                                      :math:`\left[\begin{array}{ccc} f & k_y & \boldsymbol{\delta\theta}
                                      \end{array}\right]`
        ``'intrinsic'``               estimate focal length, kx, ky, px, and py:
                                      :math:`\left[\begin{array}{ccccc} f & k_x & k_y & p_x & p_y
                                      \end{array}\right]`.  Note that this will likely result in a rank-deficient matrix
                                      without an a priori covariance.  Use ``'basic intrinsic'`` instead.
        ``'basic intrinsic'``         estimate focal length and ky:
                                      :math:`\left[\begin{array}{cc} f & k_y \end{array}\right]`
        ``'focal_length'``            the focal length of the camera:  :math:`f`
        ``'kx'``                      inverse of the pixel pitch along the x axis: :math:`k_x`
        ``'ky'``                      inverse of the pixel pitch along the y axis: :math:`k_y`
        ``'px'``                      x location of the principal point in pixels: :math:`p_x`
        ``'py'``                      y location of the principal point in pixels: :math:`p_y`
        ``'a1'``                      the linear coefficient for a temperature dependent focal length: :math:`a_1`
        ``'a2'``                      the quadratic coefficient for a temperature dependent focal length: :math:`a_2`
        ``'a3'``                      the cubic coefficient for a temperature dependent focal length: :math:`a_3`
        ``'temperature dependence'``  estimate 3 temperature dependence coefficients for the focal length a1, a2, a3:
                                      :math:`\left[\begin{array}{ccc} a_1 & a_2 & a_3 \end{array}\right]`
        ``'single misalignment'``     estimate a single misalignment for all images: :math:`\boldsymbol{\delta\theta}`
        ``'multiple misalignments'``  estimate a misalignment for each image:
                                      :math:`\left[\begin{array}{ccc}\boldsymbol{\delta\theta}_1 & \ldots &
                                      \boldsymbol{\delta\theta}_n \end{array}\right]`
        ============================  ==================================================================================

        Note that it may not be possible to estimate all attributes simultaneously because this may result in a rank
        deficient matrix in the calibration process (for instance, without setting a priori weights, estimating
        ``'focal_length'``, ``'kx'``, and ``'ky'`` together would result in a rank deficient matrix.  Therefore, just
        because you can set something in this list doesn't mean you should.

        For more details about calibrating a camera model, see the :mod:`.calibration` package for details.
        """

        return self._estimation_parameters

    @estimation_parameters.setter
    def estimation_parameters(self, val: Union[str, Sequence[str]]):
        
        self._estimation_parameters = self._validate_parameters(val)
        

        if 'multiple misalignments' in self.estimation_parameters:
            self.estimate_multiple_misalignments = True
        else:
            self.estimate_multiple_misalignments = False
            
    def _validate_parameters(self, val: Union[str, Sequence[str]]) -> list[str]:
        
        if isinstance(val, str):
            
            val = [val.lower()]
        else:
            val = [v.lower() for v in val]

        for elem in val:
            if elem not in self.element_dict:
                raise ValueError('The estimation parameters elements must be one of {}.'.format(
                    self.element_dict.keys()) + ' You specified {}'.format(elem))
                
        return val

    @property
    def kx(self) -> float:
        """
        The inverse of the pixel pitch along the x axis in units of pix/distance.

        This is the conversion factor to convert from gnomic coordinates (in units of distance) to units of pixels.
        It corresponds to the [0, 0] component of the intrinsic matrix
        """
        return self.intrinsic_matrix[0, 0]

    @kx.setter
    def kx(self, val):
        self.intrinsic_matrix[0, 0] = val

    @property
    def kxy(self) -> float:
        """
        The skewness term between xy

        This is the conversion factor to convert from gnomic coordinates (in units of distance) to units of pixels.
        It corresponds to the [0, 1] component of the intrinsic matrix
        """
        return self.intrinsic_matrix[0, 1]

    @kxy.setter
    def kxy(self, val):
        self.intrinsic_matrix[0, 1] = val

    @property
    def ky(self) -> float:
        """
        The inverse of the pixel pitch along the y axis in units of pix/distance.

        This is the conversion factor to convert from pinhole coordinates (in units of distance) to units of pixels.
        It corresponds to the [1, 1] component of the intrinsic matrix
        """
        return self.intrinsic_matrix[1, 1]

    @ky.setter
    def ky(self, val):
        self.intrinsic_matrix[1, 1] = val

    @property
    def kyx(self) -> float:
        """
        The skewness term between yx

        This is the conversion factor to convert from gnomic coordinates (in units of distance) to units of pixels.
        It corresponds to the [1, 0] component of the intrinsic matrix
        """
        return self.intrinsic_matrix[1, 0]

    @kyx.setter
    def kyx(self, val):
        self.intrinsic_matrix[1, 0] = val

    @property
    def px(self) -> float:
        """
        The x pixel location of the principal point of the camera.

        The principal point of the camera is the point in the image where the distortion is zero (the point where the
        optical axis pierces the image).  This corresponds to the [0, 2] component of the intrinsic matrix
        """
        return self.intrinsic_matrix[0, 2]

    @px.setter
    def px(self, val):
        self.intrinsic_matrix[0, 2] = val

    @property
    def py(self) -> float:
        """
        The y pixel location of the principal point of the camera.

        The principal point of the camera is the point in the image where the distortion is zero (the point where the
        optical axis pierces the image).  This corresponds to the [1, 2] component of the intrinsic matrix
        """
        return self.intrinsic_matrix[1, 2]

    @py.setter
    def py(self, val):
        self.intrinsic_matrix[1, 2] = val

    @property
    def a1(self) -> float:
        """
        The linear coefficient for the focal length temperature dependence

        This is the first term in the :attr:`.temperature_coefficients` array and is multiplied by the temperature.
        """

        return self.temperature_coefficients[0]

    @a1.setter
    def a1(self, val):

        self.temperature_coefficients[0] = float(val)

    @property
    def a2(self) -> float:
        """
        The quadratic coefficient for the focal length temperature dependence

        This is the second term in the :attr:`.temperature_coefficients` array and is multiplied by the temperature
        squared.
        """

        return self.temperature_coefficients[1]

    @a2.setter
    def a2(self, val):

        self.temperature_coefficients[1] = float(val)

    @property
    def a3(self) -> float:
        """
        The cubic coefficient for the focal length temperature dependence

        This is the third term in the :attr:`.temperature_coefficients` array and is multiplied by the temperature
        cubed.
        """

        return self.temperature_coefficients[2]

    @a3.setter
    def a3(self, val):

        self.temperature_coefficients[2] = float(val)

    @property
    def focal_length(self) -> float:
        """
        The focal length for the camera expressed in units of distance
        """
        return self._focal_length

    @focal_length.setter
    def focal_length(self, val: float):

        self._focal_length = float(val)

    @property
    def intrinsic_matrix_inv(self) -> np.ndarray:
        r"""
        The inverse of the intrinsic matrix.

        The inverse of the intrinsic matrix is used to convert from units of pixels with an origin at the upper left
        corner of the image to units of distance with an origin at the principal point of the image.

        the intrinsic matrix has an analytic inverse which is given by

        .. math::
            \mathbf{K}^{-1} = \left[\begin{array}{ccc} \frac{1}{k_x} & 0 & \frac{-p_x}{k_x} \\
            0 & \frac{1}{k_y} & \frac{-p_y}{k_y} \end{array}\right]

        To convert from units of pixels to units of distance you would do::
            >>> from giant.camera_models import PinholeModel
            >>> model = PinholeModel(kx=5, ky=10, px=100, py=500)
            >>> ((model.intrinsic_matrix_inv[:, :2]@[[1, 2, 300], [4, 5, 600]]).T + model.intrinsic_matrix_inv[:, 2]).T
            array([[-19.8, -19.6, 40.]
                   [-49.6, -49.5, 10.]])

        .. note:: For the :class:`PinholeModel`, this same functionality is available from :meth:`pixels_to_gnomic`.  In
                  classes with a distortion model (like the rest of the classes in this module) however, the above code
                  will give you distorted gnomic location, while the :meth:`pixels_to_gnomic` will give you
                  undistorted gnomic locations (true pinhole points).

        .. note:: Since the :class:`PinholeModel` class defines the intrinsic matrix as a :math:`2\times 3` matrix this
                  isn't a formal inverse.  To get the true inverse you need to append a row of [0, 0, 1] to both the
                  intrinsic matrix and intrinsic matrix inverse.
        """

        return np.array([[1 / self.kx, 0, -self.px / self.kx],
                         [0, 1 / self.ky, -self.py / self.ky]])

    def adjust_temperature(self, pixel_locations: ARRAY_LIKE, old_temperature: float, new_temperature: float) \
            -> np.ndarray:
        """
        This method adjusts a pixel location to reflect a new image temperature.

        This is done by (a) converting back to the image frame by multiplying by the inverse camera matrix,
        (b) multiplying by the ratio of the new temperature scaling to the old temperature scaling, and
        (c) multiplying by the camera matrix to return to pixel space

        :param pixel_locations: the pixel locations to change the temperature for
        :param old_temperature: the temperature the current pixel locations reflect
        :param new_temperature: the new desired temperature
        :return: the updated pixel locations
        """

        gnomic_distorted = (
            (self.intrinsic_matrix_inv[:, :2] @ pixel_locations).T + self.intrinsic_matrix_inv[:, 2]).T

        temp_change_ratio = self.get_temperature_scale(
            new_temperature) / self.get_temperature_scale(old_temperature)
        new_gnomic_distorted = temp_change_ratio * gnomic_distorted

        return ((self.intrinsic_matrix[:, :2] @ new_gnomic_distorted).T + self.intrinsic_matrix[:, 2]).T

    def get_temperature_scale(self, temperature: SCALAR_OR_ARRAY) -> Union[float, np.ndarray]:
        r"""
        This method computes the scaling to the focal length caused by a shift in temperature.

        The temperature dependence is defined as a third order polynomial in temperature:

        .. math::
            \Delta f = 1 + a_1 T + a_2 T^2 + a_3 T^3

        where :math:`T` is the temperature (usually in units of degrees Celsius).

        You can use this method to get the temperature scaling as either a scalar value for a single parameter, or
        an array of values for an array of temperatures::

            >>> from giant.camera_models import PinholeModel
            >>> model = PinholeModel(a1=1, a2=2, a3=3)
            >>> model.get_temperature_scale(5.5)
            566.125
            >>> model.get_temperature_scale([5.5, -2.5, 0])
            array([ 566.125, -35.875, 1. ])

        :param temperature: The temperature(s) to compute the scaling at
        :return: the temperature scaling either as a float or a numpy ndarray
        """
        # ensure the temperature is a numpy array if it is an array
        temperature = np.asarray(temperature)

        # compute the powers of the temperature
        temp_powers = np.array(
            [temperature, temperature ** 2, temperature ** 3])

        # compute the temperature scaling
        return 1. + self.temperature_coefficients @ temp_powers

    def apply_distortion(self, pinhole_locations: ARRAY_LIKE) -> np.ndarray:
        """
        This method simply returns the pinhole locations since this is the pinhole camera model.  It is included
        for completeness and for assisting with subclassing.

        In general this function is not used by the user and the higher level :meth:`project_onto_image` is used
        which calls this method (along with a few others) instead.  In cases were it is desirable to use this method
        the pinhole locations should be input as a shape (2,) or shape (2, n) array of image plane locations in units of
        distance.  The output from this function is the distorted image plane locations of the points in units of
        distance.

        You can input the pinhole locations as a single length 2 location, or an array of shape (2, n) locations::
            >>> from giant.camera_models import PinholeModel
            >>> model = PinholeModel()
            >>> model.apply_distortion([1, 2])  # the output will be the same as the input since this is the pinhole
            [1, 2]
            >>> model.apply_distortion([[1, 2, 3, 4], [5, 6, 7, 8]])
            [[1, 2, 3, 4], [5, 6, 7, 8]]

        :param pinhole_locations: The image plane location of points to be distorted as a shape (2,) or (2, n) array in
                                  units of distance
        :return: The distorted locations of the points on the image plane as a shape (2,) or (2, n) array in units of
                 distance
        """

        return np.array(pinhole_locations)

    def get_projections(self, points_in_camera_frame: ARRAY_LIKE, image: int = 0, temperature: float = 0) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This method computes and returns the pinhole, and pixel locations for a set of
        3D points expressed in the camera frame.

        In general the user will not use this method and the higher level :meth:`project_onto_image` will be used
        instead.  In cases where it is desirable to use this method, the camera points should be input as a shape (2,)
        or shape (2, n) array of points expressed in the camera frame (units don't matter). This method will then return
        the gnomic locations in units of distance as a shape (2,) or (2, n) numpy array and the pixel locations of the
        points as a shape (2,) or (2, n) numpy array with units of pixels as a length 2 tuple.

        The optional `image` flag specifies which image you are projecting the points onto.  This is only important if
        you have the :attr:`estimate_multiple_misalignments` flag set to true, and have a different alignment set for
        each image.  In general, the optional input `image` should be ignored except during calibration.

        The optional `temperature` input specifies the temperature to perform the projection at.  This is only import
        when your focal length is dependent on temperature and you have entered or calibrated for the temperature
        dependency coefficients.

        You can specify the directions to be input as either a shape (3,) or shape (3, n) array::
            >>> from giant.camera_models import PinholeModel
            >>> model = PinholeModel(kx=300, ky=400, px=500, py=500, focal_length=10, a1=1e-5, a2=1e-6,
            >>>                      misalignment=[[1e-12, -2e-14, 3e-10], [2e-15, 1e-13, 3e-10]],
            >>>                      estimation_parameters=['multiple misalignments'])
            >>> model.get_projections([1, 2, 12000])
            (array([ 0.00083333,  0.00166667]), array([ 500.25      ,  500.66666667]))
            >>> model.get_projections([[1, 2, 3, 4], [2, 5, 6, 7], [12000, 13000, 9000, 5000]], image=1)
            (array([[ 0.00083333,  0.00153846,  0.00333333,  0.008     ],
                    [ 0.00166667,  0.00384615,  0.00666667,  0.014     ]]),
             array([[ 500.25      ,  500.46153846,  501.        ,  502.4       ],
                    [ 500.66666667,  501.53846154,  502.66666667,  505.6       ]]))
            >>> model.get_projections([[1, 2, 3, 4], [2, 5, 6, 7], [12000, 13000, 9000, 5000]], temperature=-1)
            (array([[-0.00166667, -0.00307694, -0.0066667 , -0.01600007],
                    [-0.00333335, -0.00769234, -0.01333339, -0.02800013]]),
             array([[ 499.49999775,  499.07691892,  497.999991  ,  495.1999784 ],
                    [ 498.66666066,  496.92306307,  494.66664266,  488.79994959]]))

        :param points_in_camera_frame: a shape (3,) or shape (3, n) array of points to project
        :param image: The index of the image being projected onto (only applicable with multiple misalignments)
        :param temperature: The temperature to project the points at
        :return: A tuple of the pinhole and pixel locations for a set of 3D points
                 expressed in the camera frame
        """

        # ensure the points are an array
        camera_points = np.asarray(points_in_camera_frame)

        # apply misalignment to the points
        if self.estimate_multiple_misalignments:
            # optimization to avoid matrix multiplication
            if np.any(self.misalignment[image]):
                camera_points = rotvec_to_rotmat(
                    self.misalignment[image]).squeeze() @ camera_points

        else:
            if np.any(self.misalignment):  # optimization to avoid matrix multiplication
                camera_points = rotvec_to_rotmat(
                    self.misalignment).squeeze() @ camera_points

        # get the pinhole locations of the points
        gnomic_locations = self.focal_length * \
            camera_points[:2] / camera_points[2]  # type: np.ndarray

        # apply the temperature scaling
        gnomic_locations *= self.get_temperature_scale(temperature)

        # get the pixel locations of the points, need to mess with transposes due to numpy broadcasting rules
        picture_locations = (
            (self.intrinsic_matrix[:, :2] @ gnomic_locations).T + self.intrinsic_matrix[:, 2]).T

        return gnomic_locations, gnomic_locations, picture_locations

    def project_onto_image(self, points_in_camera_frame: ARRAY_LIKE, image: int = 0, temperature: float = 0) \
            -> np.ndarray:
        """
        This method transforms 3D points or directions expressed in the camera frame into the corresponding 2D image
        locations.

        The points input should be either 1 or 2 dimensional, with the first axis being length 3 (each point
        (direction) in the camera frame is specified as a column).

        The optional ``image`` key word argument specifies the index of the image you are projecting onto (this only
        applies if you have a separate misalignment for each image)

        The optional temperature input specifies the temperature to perform the projection at.  This is only import when
        your focal length is dependent on temperature and you have entered or calibrated for the temperature dependency
        coefficients.

        You can specify the directions to be input as either a shape (3,) or shape (3, n) array::
            >>> from giant.camera_models import PinholeModel
            >>> model = PinholeModel(kx=300, ky=400, px=500, py=500, focal_length=10, a1=1e-5, a2=1e-6,
            >>>                      misalignment=[[1e-12, -2e-14, 3e-10], [2e-15, 1e-13, 3e-10]],
            >>>                      estimation_parameters=['multiple misalignments'])
            >>> model.project_onto_image([1, 2, 12000])
            array([ 500.25      ,  500.66666667])
            >>> model.project_onto_image([[1, 2, 3, 4], [2, 5, 6, 7], [12000, 13000, 9000, 5000]], image=1)
            array([[ 500.25      ,  500.46153846,  501.        ,  502.4       ],
                   [ 500.66666667,  501.53846154,  502.66666667,  505.6       ]])
            >>> model.project_onto_image([[1, 2, 3, 4], [2, 5, 6, 7], [12000, 13000, 9000, 5000]], temperature=-1)
            array([[ 499.49999775,  499.07691892,  497.999991  ,  495.1999784 ],
                   [ 498.66666066,  496.92306307,  494.66664266,  488.79994959]])

        :param points_in_camera_frame: a shape (3,) or shape (3, n) array of points to project
        :param image: The index of the image being projected onto (only applicable with multiple misalignments)
        :param temperature: The temperature to project the points at
        :return: A shape (2,) or shape (2, n) numpy array of image points (with units of pixels)
        """
        _, __, picture_locations = self.get_projections(
            points_in_camera_frame, image=image, temperature=temperature)

        return picture_locations

    def project_directions(self, directions_in_camera_frame: ARRAY_LIKE, image: int = 0) \
            -> np.ndarray:
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
        
        directions_in_camera_frame = np.asanyarray(directions_in_camera_frame)

        if self.estimate_multiple_misalignments:
            # optimization to avoid matrix multiplication
            if np.any(self.misalignment[image]):
                directions_in_camera_frame = (rotvec_to_rotmat(self.misalignment[image]).squeeze() @
                                              directions_in_camera_frame)

        else:
            if np.any(self.misalignment):  # optimization to avoid matrix multiplication
                directions_in_camera_frame = (rotvec_to_rotmat(self.misalignment).squeeze() @
                                              directions_in_camera_frame)

        image_direction = self.intrinsic_matrix[:, :2] @ directions_in_camera_frame[:2]

        return image_direction/np.linalg.norm(image_direction, axis=0, keepdims=True)

    def compute_pixel_jacobian(self, vectors_in_camera_frame: ARRAY_LIKE, image: int = 0, temperature: float = 0) \
            -> np.ndarray:
        r"""
        This method computes the Jacobian matrix :math:`\partial\mathbf{x}_P/\partial\mathbf{x}_C` where
        :math:`\mathbf{x}_C` is a vector in the camera frame that projects to :math:`\mathbf{x}_P` which is the
        pixel location.

        This method is used in the :class:`.LimbScanning` process in order to predict the change in a projected pixel
        location with respect to a change in the projected vector.  The :attr:`vectors_in_camera_frame` input should
        be a 3xn array of vectors which the Jacobian is to be computed for.

        Mathematically the Jacobian matrix is defined as

        .. math::
            \frac{\partial\mathbf{x}_P}{\partial\mathbf{x}_C} =
            \frac{\partial\mathbf{x}_P}{\partial\mathbf{x}_I}
            \frac{\partial\mathbf{x}_I}{\partial\mathbf{x}_C'}
            \frac{\partial\mathbf{x}_C'}{\partial\mathbf{x}_C}

        where

        .. math::
            :nowrap:

            \begin{gather}
            \frac{\partial\mathbf{x}_P}{\partial\mathbf{x}_I} = (1+a_1T+a_2T^2+a_3T^3)
            \mathbf{K}_{2x2} \\
            \frac{\partial\mathbf{x}_I}{\partial\mathbf{x}_C'} = \frac{f}{z_C'}\left[
            \begin{array}{ccc}1 & 0 & \frac{-x_C'}{z_C'} \\ 0 & 1 & \frac{-y_C'}{z_C'} \end{array}\right] \\
            \frac{\partial\mathbf{x}_C'}{\partial\mathbf{x}_C} = \mathbf{T}_{\boldsymbol{\delta\theta}}
            \end{gather}

        :math:`\mathbf{x}_C'` is the camera frame point after applying the misalignment,
        :math:`\boldsymbol{\delta\theta}` is the misalignment vector, :math:`\mathbf{x}_C` is the camera frame point
        before misalignment is applied,
        :math:`\mathbf{x}_P` is the pixel location, :math:`\mathbf{x}_I` is the gnomic location,
        :math:`a_{1-3}` are the temperature coefficients, :math:`T` is the temperature, :math:`k_x` is the inverse of
        the pixel pitch in the x direction,  :math:`k_y` is the inverse of the pixel pitch in the y direction,
        :math:`f` is the focal length, :math:`\mathbf{K}_{2x2}` is the first 2 rows and columns of the
        :attr:`intrinsic_matrix`, and :math:`\mathbf{T}_{\boldsymbol{\delta\theta}}` is the rotation matrix
        corresponding to rotation vector :math:`\boldsymbol{\delta\theta}`.

        :param vectors_in_camera_frame: The vectors to compute the Jacobian at
        :param image: The index of the image being projected onto (only applicable with multiple misalignments)
        :param temperature: The temperature to project the points at
        :return: The Jacobian matrix as a nx2x3 array
        """

        jacobian = []

        for vector in np.asarray(vectors_in_camera_frame).T:

            # get the required projections for the point
            gnomic_location, gnomic_location_distorted, pixel_location = self.get_projections(vector,
                                                                                              image=image,
                                                                                              temperature=temperature)

            # get the camera point after misalignment and shift from principle frame is applied
            if self.estimate_multiple_misalignments:
                # optimization to avoid matrix multiplication
                if np.any(self.misalignment[image]):
                    mis = rotvec_to_rotmat(self.misalignment[image]).squeeze()
                    cam_point = mis @ vector

                else:
                    mis = np.eye(3)
                    cam_point = vector

            else:
                if np.any(self.misalignment):  # optimization to avoid matrix multiplication
                    mis = rotvec_to_rotmat(self.misalignment).squeeze()
                    cam_point = mis @ vector

                else:
                    mis = np.eye(3)
                    cam_point = vector

            # compute the radial distance from the optical axis as well as its powers
            radius = float(np.linalg.norm(gnomic_location))
            radius2 = radius ** 2
            radius3 = radius * radius2
            radius4 = radius2 ** 2

            # --------------------------------------------------------------------------------------------------------------
            # get the partial derivative of the measurement with respect to the input vector
            # --------------------------------------------------------------------------------------------------------------

            # get the partial derivative of the distorted gnomic location with respect to the gnomic location
            ddist_gnom_dgnom = np.eye(2) + self._compute_ddistortion_dgnomic(gnomic_location,
                                                                             radius, radius2, radius3, radius4)

            # get the partial derivative of the pixel location of the point with respect to the dist gnomic location
            dpix_ddist_gnom = self._compute_dpixel_ddistorted_gnomic(
                temperature=temperature)

            # compute the partial derivative of the misaligned vector with respect to a change in the input vector
            dcam_point_dvector = mis

            # compute the partial derivative of the gnomic location with respect to the point in the camera frame
            dgnom_dcam_point = self._compute_dgnomic_dcamera_point(cam_point)

            # compute the partial derivative of the pixel location with respect to the input vector
            dpix_dvector = dpix_ddist_gnom @ ddist_gnom_dgnom @ dgnom_dcam_point @ dcam_point_dvector

            jacobian.append(dpix_dvector)

        return np.array(jacobian)

    def _compute_ddistortion_dgnomic(self, *args) -> np.ndarray:
        """
        This method computes the change in the distortion with respect to a change in the gnomic location.

        For the pinhole model, the distortion doesn't change (since there isn't a distortion) therefore this method
        simply returns the 2x2 0 matrix.  This method is included simply for subclassing purposes.

        :param args: Ignored
        :return: A 2x2 0 matrix
        """
        return np.zeros((2, 2))

    def _compute_dcamera_point_dgnomic(self, gnomic_location: ARRAY_LIKE, vec_length: float) -> np.ndarray:
        r"""
        This method compute the change in the misaligned unit vector with respect to a change in the gnomic location,
        :math:`\partial\mathbf{x}_C'/\partial\mathbf{x}_I`.

        Mathematically this is given by

        .. math::
            \frac{\partial\mathbf{x}_C'}{\partial\mathbf{x}_I} = \left[\begin{array}{cc}
            1/v & 0 \\ 0 & 1/v \\ 0 & 0\end{array}\right] -
            \frac{1}{v^3}\left[\begin{array}{c} \mathbf{x}_I \\ f \end{array}\right]\mathbf{x}_I^T

        :param gnomic_location: The gnomic location we are computing the derivative at
        :param vec_length: The amount the vector is divided by to make it a unit vector
        :return: The change in the misaligned unit vector with respect to a change in the gnomic location
        """
        
        gnomic_location = np.asanyarray(gnomic_location)

        vector_portion = np.vstack([np.eye(2) / vec_length, np.zeros((1, 2))])
        scalar_portion = np.outer(np.concatenate([gnomic_location, [self.focal_length]]),
                                  gnomic_location) / (vec_length ** 3)

        return vector_portion - scalar_portion

    def _compute_dgnomic_ddist_gnomic(self, distorted_gnomic_location: ARRAY_LIKE) -> np.ndarray:
        r"""
        This method computes the change in the gnomic location with respect to a change in the distorted gnomic location
        for the inverse camera model.

        Since this process is iterative this is only an approximate solution.

        Mathematically this is given by

        .. math::
            \frac{\partial\mathbf{x}_I}{\partial\mathbf{x}_I'} = \mathbf{I}_{2\times 2} -
            \left.\frac{\partial\mathbf{x}_I'}{\partial\mathbf{x}_I}\right|_{\mathbf{x}_I=\mathbf{x}_I'}

        :param distorted_gnomic_location: The distorted gnomic location we are calculating the derivative at.
        :return: the partial derivative of the gnomic location with respect to the distorted gnomic location for the
                 inverse camera model
        """

        # compute the radial distance from the principal point and its powers to give to the ddistortion/dgnomic method
        radius = np.linalg.norm(distorted_gnomic_location)
        radius2 = radius * radius
        radius3 = radius2 * radius
        radius4 = radius2 * radius2

        # compute the derivative
        return np.eye(2) - self._compute_ddistortion_dgnomic(distorted_gnomic_location,
                                                             radius, radius2, radius3, radius4)

    def _compute_ddist_gnomic_dpixel(self, temperature: float) -> np.ndarray:
        r"""
        This method computes the change in the distorted gnomic location with respect to a change in the pixel location.

        This change is the same for all pixel locations and is given mathematically by

        .. math::
            \frac{\partial\mathbf{x}_I'}{\partial\mathbf{x}_P} = \frac{\mathbf{K}^{-1}}{1+a_1T+a_2T^2+a_3T^3}

        :param temperature: The temperature of the camera
        :return: The partial derivative of the distorted gnomic location with respect to the pixel location
        """

        return self.intrinsic_matrix_inv[:, :2] / self.get_temperature_scale(temperature)

    def compute_unit_vector_jacobian(self, pixel_locations: ARRAY_LIKE, image: int = 0, temperature: float = 0) \
            -> np.ndarray:
        r"""
        This method computes the Jacobian matrix :math:`\partial\mathbf{x}_C/\partial\mathbf{x}_P` where
        :math:`\mathbf{x}_C` is a vector in the camera frame that projects to :math:`\mathbf{x}_P` which is the
        pixel location.

        This method is used in the :class:`.LimbScanning` process in order to predict the change in the unit vector that
        projects to a pixel location with respect to a change in the pixel location.  The
        ``pixel_locations`` input should be a 2xn array of vectors which the Jacobian is to be computed for.

        :param pixel_locations: The pixel locations to compute the Jacobian at
        :param image: The number of the image we are computing the Jacobian for
        :param temperature: The temperature to compute the Jacobian at
        :return: The Jacobian matrix as a nx3x2 array
        """

        pixel_locations = np.asanyarray(pixel_locations)

        # get the misalignment matrix
        if self.estimate_multiple_misalignments:
            # optimization to avoid matrix multiplication
            if np.any(self.misalignment[image]):
                mis = rotvec_to_rotmat(self.misalignment[image]).squeeze()

            else:
                mis = np.eye(3)

        else:
            if np.any(self.misalignment):  # optimization to avoid matrix multiplication
                mis = rotvec_to_rotmat(self.misalignment).squeeze()

            else:
                mis = np.eye(3)

        # get the distorted gnomic location
        gnomic_distorted = (
            (self.intrinsic_matrix_inv[:, :2] @ pixel_locations).T + self.intrinsic_matrix_inv[:, 2]).T

        # get the gnomic location and the unit vector in the camera frame
        gnomic_locations = self.pixels_to_gnomic(
            pixel_locations, temperature=temperature)

        # append the focal length
        if pixel_locations.ndim == 1:
            cam_points = np.hstack([gnomic_locations, self.focal_length])

        else:
            cam_points = np.vstack(
                [gnomic_locations, self.focal_length * np.ones((1, pixel_locations.shape[1]))])

        unit_vectors = mis.T @ cam_points

        vec_lengths = np.linalg.norm(cam_points, axis=0, keepdims=True)

        unit_vectors /= vec_lengths

        # initialize the Jacobian list
        jacobian = []

        # compute the change in the distorted gnomic location with respect to a change in the pixel location
        # we can do this here once because it is independent of the pixel location
        dgnom_dist_dpixel = self._compute_ddist_gnomic_dpixel(temperature)

        for pixel, gnomic_loc, dist_gnom_loc, cam_point, unit_vector, vec_length in zip(pixel_locations.T,
                                                                                        gnomic_locations.T,
                                                                                        gnomic_distorted.T,
                                                                                        cam_points.T, unit_vectors.T,
                                                                                        vec_lengths.ravel()):
            # compute the change in the camera location with respect to a change in the misaligned camera location
            dunit_dcam_point = mis.T

            # compute the change in the misaligned camera direction with respect to a change in the gnomic location
            dcam_point_dgnomic = self._compute_dcamera_point_dgnomic(
                gnomic_loc, vec_length)

            # compute the change in the gnomic location with respect to a change in the distorted gnomic location
            dgnom_ddist_gnom = self._compute_dgnomic_ddist_gnomic(gnomic_loc)

            # form the Jacobian rows and append to the Jacobian list
            jacobian.append(dunit_dcam_point @ dcam_point_dgnomic @
                            dgnom_ddist_gnom @ dgnom_dist_dpixel)

        return np.array(jacobian)

    @staticmethod
    def _compute_dcamera_point_dmisalignment(unit_vector_camera: ARRAY_LIKE) -> np.ndarray:
        r"""
        Computes the partial derivative of the 3D camera frame location with respect to a change in the misalignment

        Mathematically, this partial is given by

        .. math::
            \frac{\partial\mathbf{x}_C'}{\partial\boldsymbol{\delta\theta}} = \left[\mathbf{x}_C\times\right]

        where :math:`\mathbf{x}_C'` is the camera frame point after applying the misalignment,
        :math:`\boldsymbol{\delta\theta}` is the misalignment vector, :math:`\mathbf{x}_C` is the camera frame point
        before misalignment is applied, and :math:`\left[\bullet\times\right]` is the skew-symmetric cross product
        matrix formed from :math:`\bullet`.

        :param unit_vector_camera: the unit vector through the point in the camera frame
        :return: the partial derivative of the 3D camera frame location with respect to a change in the misalignment
        """

        return -skew(unit_vector_camera)

    def _compute_dpixel_ddistorted_gnomic(self, temperature: float = 0) -> np.ndarray:
        r"""
        Computes the partial derivative of the pixel location with respect to a change in the distorted gnomic location.

        Mathematically, this partial is given by

        .. math::
            \frac{\partial\mathbf{x}_P}{\partial\mathbf{x}_I} = (1+a_1T+a_2T^2+a_3T^3)
            \left[\begin{array}{cc} k_x & 0 \\ 0 & k_y \end{array}\right]

        where :math:`\mathbf{x}_P` is the pixel location, :math:`\mathbf{x}_I` is the gnomic location,
        :math:`a_{1-3}` are the temperature coefficients, :math:`T` is the temperature, :math:`k_x` is the inverse of
        the pixel pitch in the x direction, and :math:`k_y` is the inverse of the pixel pitch in the y direction

        :param temperature: The temperature of the camera
        :return: the partial derivative of the pixel location with respect to a change in the distorted gnomic location
        """

        return self.get_temperature_scale(temperature) * self.intrinsic_matrix[:, :2]

    def _compute_dgnomic_dcamera_point(self, unit_vector_camera: F_ARRAY_LIKE) -> np.ndarray:
        r"""
        Computes the partial derivative of the gnomic location with respect to a change in the 3D camera frame location
        after the misalignment correction is applied.

        Mathematically this is given by

        .. math::
            \frac{\partial\mathbf{x}_I}{\partial\mathbf{x}_C'} = \frac{f}{z_C'}\left[
            \begin{array}{ccc}1 & 0 & \frac{-x_C'}{z_C'} \\ 0 & 1 & \frac{-y_C'}{z_C'} \end{array}\right]

        where all is as defined before.

        :param unit_vector_camera: The 3D camera frame location of the point after misalignment is applied
        :return: The partial derivative of the gnomic location with respect to a change in the 3D camera frame location
        """
        unit_vector_camera = np.asanyarray(unit_vector_camera)
        return self.focal_length / unit_vector_camera[2] * np.hstack([np.eye(2), -unit_vector_camera[:2].reshape(2, 1) /
                                                                      unit_vector_camera[2]])

    @staticmethod
    def _compute_dgnomic_dfocal_length(unit_vector_camera: F_ARRAY_LIKE) -> np.ndarray:
        r"""
        Computes the partial derivative of the gnomic location with respect to a change in the focal length.

        The input is the camera frame point after the misalignment correction has been applied.

        Mathematically this is given by

        .. math::
            \frac{\partial\mathbf{x}_I}{\partial f} = \frac{1}{z_C'}
            \left[\begin{array}{c}x_C'\\y_C'\end{array}\right]

        :param unit_vector_camera: the point being projected onto the image after the misalignment correction is applied
        :return: The partial derivative of the gnomic location with respect to a change in the focal length
        """
        unit_vector_camera = np.asanyarray(unit_vector_camera)
        return unit_vector_camera[:2] / unit_vector_camera[2]

    @staticmethod
    def _compute_dpixel_dintrinsic(gnomic_location_distorted: F_ARRAY_LIKE) -> np.ndarray:
        r"""
        computes the partial derivative of the pixel location with respect to a change in one of the intrinsic matrix
        parameters given the gnomic location of the point we are computing the derivative for.

        Mathematically this is given by

        .. math::
            \frac{\partial\mathbf{x}_P}{\partial\mathbf{k}} = \left[\begin{array}{cccc} x_I & 0 & 1 & 0 \\
            0 & y_I & 0 & 1 \end{array}\right]

        where :math:`\mathbf{k}=[k_x \quad k_y \quad p_x \quad p_y]` is a vector of the intrinsic camera parameters
        and all else is as defined before.

        :param gnomic_location_distorted: the gnomic location of the point to compute the derivative for
        :return: the partial derivative of the pixel location with respect to a change in one of the intrinsic matrix
                 parameters
        """

        # compute the partial derivative of the pixel location with respect to the pixel pitch
        dpix_dkx = [gnomic_location_distorted[0], 0]
        dpix_dky = [0, gnomic_location_distorted[1]]

        # compute the partial derivative of the pixel location with respect to the principal point
        dpix_dpx = [1, 0]
        dpix_dpy = [0, 1]

        # compute the partial derivative of the pixel location with respect to the intrinsic matrix
        return np.array([dpix_dkx, dpix_dky, dpix_dpx, dpix_dpy]).T

    def _compute_dpixel_dtemperature_coeffs(self, gnomic_location_distorted: F_ARRAY_LIKE, temperature: float = 0) \
            -> np.ndarray:
        r"""
        Computes the partial derivative of the pixel coordinates with respect to a change in the temperature
        coefficients for a given gnomic location and temperature.

        Mathematically this is given by

        .. math::
            \frac{\partial\mathbf{x}_P}{\partial\mathbf{a}} = \left[\begin{array}{cc} k_x & 0 \\
            0 & k_y\end{array}\right] \mathbf{x}_I \left[\begin{array}{ccc} T & T^2 & T^3 \end{array}\right]

        where :math:`\mathbf{a}=[a_1 \quad a_2 \quad a_3]` is a vector of the temperature dependence coefficients.

        :param gnomic_location_distorted:  The gnomic location point to compute the derivative for
        :param temperature:  The temperature to compute the derivative at
        :return: The partial derivative of the pixel coordinates with respect to a change in the temperature
                 coefficients
        """

        # compute the powers of the temperature
        temperature_powers = [temperature, temperature *
                              temperature, temperature*temperature*temperature]

        # convert the gnomic location to units of pixels
        gnomic_pixels = self.intrinsic_matrix[:, :2] @ gnomic_location_distorted

        # get the derivative
        return np.outer(gnomic_pixels, temperature_powers)

    def _get_jacobian_row(self, unit_vector_camera: ARRAY_LIKE, image: int, num_images: int, temperature: float = 0) \
            -> np.ndarray:
        r"""
        Calculates the Jacobian matrix for a single point.

        The Jacobian is calculated for every possible parameter that could be included in the state vector in this
        method, and then columns corresponding to the state vectors that the Jacobian is not needed for can be removed
        using the :meth:`_remove_jacobian_columns` method.

        In general you should use the :meth:`compute_jacobian` method in place of this method.

        This method computes the following:

        .. math::
            \frac{\partial\mathbf{x}_P}{\partial\mathbf{c}} = \left[\begin{array}{cccc}
            \frac{\partial\mathbf{x}_P}{\partial f} & \frac{\partial\mathbf{x}_P}{\mathbf{k}} &
            \frac{\partial\mathbf{x}_P}{\partial\mathbf{a}} &
            \frac{\partial\mathbf{x}_P}{\partial\boldsymbol{\delta\theta}}\end{array}\right]

        where, using the chain rule,

        .. math::
            :nowrap:

            \begin{gather}
            \frac{\partial\mathbf{x}_P}{\partial f} =
            \frac{\partial\mathbf{x}_P}{\partial\mathbf{x}_I}
            \frac{\partial\mathbf{x}_I}{\partial f} \\
            \frac{\partial\mathbf{x}_P}{\partial\boldsymbol{\delta\theta}} =
            \frac{\partial\mathbf{x}_P}{\partial\mathbf{x}_I}
            \frac{\partial\mathbf{x}_I}{\partial\mathbf{x}_C'}
            \frac{\partial\mathbf{x}_C'}{\partial\boldsymbol{\delta\theta}}
            \end{gather}

        and all else is as defined before.

        :param unit_vector_camera: The unit vector we are computing the Jacobian for
        :param image: The number of the image we are computing the Jacobian for
        :param num_images:   The total number of images included in our Jacobian matrix
        :param temperature: The temperature to compute the Jacobian at
        :return: The row of the Jacobian matrix corresponding to the input unit vector
        """

        unit_vector_camera = np.asarray(unit_vector_camera).reshape(3)

        unit_vector_camera_mis = unit_vector_camera

        if self.estimate_multiple_misalignments:
            # optimization to avoid matrix multiplication
            if np.any(self.misalignment[image]):
                unit_vector_camera_mis = rotvec_to_rotmat(
                    self.misalignment[image]).squeeze() @ unit_vector_camera

        else:
            if np.any(self.misalignment):  # optimization to avoid matrix multiplication
                unit_vector_camera_mis = rotvec_to_rotmat(
                    self.misalignment).squeeze() @ unit_vector_camera

        # get the required projections for the point
        gnomic_location, _, pixel_location = self.get_projections(unit_vector_camera, image=image,
                                                                  temperature=temperature)

        # --------------------------------------------------------------------------------------------------------------
        # get the partial derivative of the measurement with respect to the misalignment vector
        # --------------------------------------------------------------------------------------------------------------

        # get the partial derivative of the pixel location of the point with respect to the gnomic location
        dpix_dgnom = self._compute_dpixel_ddistorted_gnomic(
            temperature=temperature)

        # compute the partial derivative of the camera location with respect to a change in the misalignment vector
        dcam_point_dmisalignment = self._compute_dcamera_point_dmisalignment(
            unit_vector_camera)

        # compute the partial derivative of the gnomic location with respect to the point in the camera frame
        dgnom_dcam_point = self._compute_dgnomic_dcamera_point(
            unit_vector_camera_mis)

        # compute the partial derivative of the pixel location with respect to the misalignment
        dpix_dmisalignment = dpix_dgnom @ dgnom_dcam_point @ dcam_point_dmisalignment

        # --------------------------------------------------------------------------------------------------------------
        # get the partial derivative of the measurement with respect to the focal length
        # --------------------------------------------------------------------------------------------------------------

        # compute the change in the gnomic location with respect to a change in the focal length
        dgnom_dfocal = self._compute_dgnomic_dfocal_length(
            unit_vector_camera_mis)

        # compute the change in the pixel location with respect to the focal length
        dpix_dfocal = dpix_dgnom @ dgnom_dfocal

        # --------------------------------------------------------------------------------------------------------------
        # get the partial derivative of the measurement with respect to the terms of the intrinsic matrix
        # --------------------------------------------------------------------------------------------------------------

        dpix_dintrinsic = self._compute_dpixel_dintrinsic(gnomic_location)

        # --------------------------------------------------------------------------------------------------------------
        # get the partial derivative of the measurement with respect to the temperature coefficients
        # --------------------------------------------------------------------------------------------------------------

        dpix_dtemperature = self._compute_dpixel_dtemperature_coeffs(
            gnomic_location, temperature=temperature)

        # stack everything together.
        if self.estimate_multiple_misalignments:
            jacobian_row = np.hstack([dpix_dfocal.reshape(2, 1), dpix_dintrinsic, dpix_dtemperature,
                                      np.zeros((2, image * 3)
                                               ), dpix_dmisalignment,
                                      np.zeros((2, (num_images - image - 1) * 3))])

        else:
            jacobian_row = np.hstack([dpix_dfocal.reshape(2, 1), dpix_dintrinsic, dpix_dtemperature,
                                      dpix_dmisalignment])

        return jacobian_row

    def compute_jacobian(self, unit_vectors_in_camera_frame: Sequence[DOUBLE_ARRAY],
                         temperature: F_SCALAR_OR_ARRAY | Sequence[float] = 0) -> np.ndarray:
        r"""
        Calculates the Jacobian matrix for each observation in `unit_vectors_in_camera_frame` for each parameter to be estimated
        as defined in the :attr:`estimation_parameters` attribute.

        This method works by first computing the partial derivatives for all camera parameters for each provided unit
        vector. It then concatenates the rows into the Jacobian matrix and removes any columns of parameters that
        are not specified in the :attr:`estimation_parameters` attribute and sorts the columns according to the order
        of the :attr:`estimation_parameters` attribute. The resulting Jacobian
        will be the appropriate size and in the order specified by :attr:`estimation_parameters`.  There is one
        constraint that the misalignment (if included) must be last in :attr:`estimation_parameters`.

        The `unit_vectors_in_camera_frame` inputs should be formatted as a Sequence of 2d sequences.  Each inner 2D sequence
        should be of shape :math:`3\times x`, where each row corresponds to a component of a unit vector in the camera
        frame. Each inner sequence should contain all observations from a single image, so that if there are :math:`m`
        images being considered, then the outer sequence should be length :math:`m`.  The value of :math:`x` can change
        for each image. If you are estimating multiple misalignments (one for each image) then each misalignment will
        correspond to the order of the image observations in the outer sequence.

        You can also set the :attr:`use_a_priori` to True to have this method append an identity matrix to the bottom of
        this Jacobian if you are solving for an update to your camera model, and not a new one entirely.

        The optional `temperature` input specifies the temperature of the camera for use in estimating temperature
        dependence.  The temperature input should either be a scalar value (float or int), or a list that is the same
        length as `unit_vectors_in_camera_frame`, where each element of the list is the temperature of the camera at the time
        of each image represented by `unit_vectors_in_camera_frame`.  If the `temperature` input is a scalar, then it is assumed
        to be the temperature value for all of the images represented in `unit_vectors_in_camera_frame`.

        :param unit_vectors_in_camera_frame: The points/directions in the camera frame that the jacobian matrix is to be computed
                                    for.  For multiple images, this should be a list of 2D unit vectors where each
                                    element of the list corresponds to a new image.
        :param temperature: A single temperature for all images or a list of temperatures the same length of
                            `unit_vectors_in_camera_frame` containing the temperature of the camera at the time each image was
                            captured
        :return: The Jacobian matrix evaluated for each observation
        """

        # get the number of images being considered
        number_images = len(unit_vectors_in_camera_frame)

        # initialize the Jacobian list
        jacobian = []

        # put the temperature into the correct format
        if not isinstance(temperature, (Sequence, np.ndarray)):
            temperature = [float(temperature)] * len(unit_vectors_in_camera_frame)

        # walk through the observations for each image
        for ind, vecs in enumerate(unit_vectors_in_camera_frame):
            # walk through the observations in the current image
            vecs = np.asarray(vecs)

            for vec in vecs.T:
                # get the full Jacobian row for the current observation
                jac_row = self._get_jacobian_row(
                    vec.T, ind, number_images, temperature=temperature[ind])

                jacobian.append(jac_row)

        # remove un-needed columns from the Jacobian matrix and turn into ndarray
        jacobian = self._remove_unused_misalignment(
            np.concatenate(jacobian, axis=0), unit_vectors_in_camera_frame)
        jacobian = self._remove_jacobian_columns(jacobian)

        # append the identity matrix if we are solving for an update to our model, and not an entirely new independent
        # model
        if self.use_a_priori:
            jacobian = np.pad(
                jacobian, [(0, jacobian.shape[1]), (0, 0)], 'constant', constant_values=0)
            jacobian[-jacobian.shape[1]:, -jacobian.shape[1]:] = np.eye(jacobian.shape[1])

        return jacobian

    def _remove_jacobian_columns(self, jacobian: np.ndarray) -> np.ndarray:
        """
        This method removes columns from the full size Jacobian according to the parameters in
        :attr:`estimation_parameters`.

        The columns are removed and reordered.  In general the user will not use this method, and instead will use the
        :meth:`compute_jacobian` method which calls this method.

        .. note:: This method assumes that Jacobian is organized according to :meth:`_get_jacobian_row`.  If the
                  Jacobian is not organized in this way then the results from this method will be invalid.

        :param jacobian: The Jacobian matrix to remove the columns from
        :return: The reordered and simplified Jacobian
        """

        jac_list = []

        # loop through each element in the estimation parameters attribute and store the appropriate columns
        for element in self.estimation_parameters:
            jac_list.append(jacobian[:, self.element_dict[element]])

        # reform the Jacobian matrix into a ndarray
        return np.concatenate(jac_list, axis=1)

    def _remove_unused_misalignment(self, jacobian: np.ndarray, vecs: Iterable[DOUBLE_ARRAY]) -> np.ndarray:
        """
        This method is used to remove unused misalignment columns from the Jacobian matrix when arbitrary images
        are not included in the calibration

        The unused misalignment columns are determined by matching them with empty lists in the `vecs` list.  When
        an empty index is found, the corresponding columns for that image are removed from the `jacobian` matrix
        and a flag is set in the :attr:`_fix_misalignment` attribute that is used later when applying the update
        vector.

        These methods are only necessary when arbitrary images are excluded from the calibration solution using
        the :attr:`Camera.image_mask` attribute and there are multiple misalignments being estimated.

        :param jacobian: The Jacobian matrix to remove the columns from as a numpy array
        :param vecs: An iterable containing camera points
        :return: the Jacobian matrix with the unneeded misalignment columns removed
        """

        if self.estimate_multiple_misalignments:

            self._fix_misalignment = []

            local_jac_list = [jacobian[:, :getattr(
                self.element_dict['multiple misalignments'], 'start')]]

            misalignment_cols = jacobian[:,
                                         self.element_dict['multiple misalignments']]

            for ind, ivecs in enumerate(vecs):

                if len(ivecs[0]) > 0:

                    local_jac_list.append(
                        misalignment_cols[:, 3 * ind:3 * ind + 3])

                    self._fix_misalignment.append(False)

                else:
                    self._fix_misalignment.append(True)

            return np.concatenate(local_jac_list, axis=1)

        else:
            return jacobian

    def _fix_update_vector(self, update_vec: DOUBLE_ARRAY, parameters: list[int] | NDArray[np.integer]) -> DOUBLE_ARRAY:
        """
        This method is used to fix the update vector when arbitrary images are not included in the calibration and
        multiple misalignments are being estimated.

        The update vector is supplemented with 0 values in the locations that correspond to the misalignment for images
        that were not included in the estimation.  This is necessary to ensure the update is correctly applied to the
        correct alignments.  The location of the 0 values to insert are determined (a) by the beginning of the alignment
        updates in the update vec according the the `parameters` input vector and (b) by a list of boolean values in
        the :attr:`_fix_misalignment` indicating whether 0s need to be added for the image corresponding to the index
        of the :attr:`_fix_misalignment` list.

        :param update_vec: delta updates to the model parameters
        :param parameters: the parameters that correspond to the update vector.
        :return: the fixed update vector
        """

        if self.estimate_multiple_misalignments and np.any(self._fix_misalignment):

            lparameters = list(parameters) # type: ignore

            start = lparameters.index(
                self.element_dict['multiple misalignments'].start)

            misalignment_update = list(update_vec[start:].ravel())

            fixed = []
            fixed.extend(update_vec[:start])

            for fix in self._fix_misalignment:

                if fix:

                    fixed.extend([0, 0, 0])

                else:

                    fixed.extend(misalignment_update[:3])

                    misalignment_update = misalignment_update[3:]

            return np.array(fixed, dtype=np.float64)

        else:
            return update_vec

    def apply_update(self, update_vec: F_ARRAY_LIKE ):
        r"""
        This method takes in a delta update to the camera parameters (:math:`\Delta\mathbf{c}`) and applies the update
        to the current instance in place.

        In general the delta update is calculated in the estimators in the :mod:`.calibration` subpackage and this
        method is not used by the user.

        The order of the update vector is determined by the order of the elements in :attr:`estimation_parameters`.  Any
        misalignment terms must always call last.

        This method operates by walking through the elements in :attr:`estimation_parameters` and retrieving the
        parameter that the element corresponds to.  The value of the ``update_vec`` at the index of the parameter is
        then applied as an additive update to the parameter in self, with the exception of misalignment, which is
        applied as a multiplicative update.

        :param update_vec: An iterable of delta updates to the model parameters
        """

        jacobian_parameters = np.hstack([getattr(self.element_dict[element], 'start', self.element_dict[element])
                                         for element in self.estimation_parameters])

        update_vec = np.asanyarray(update_vec, dtype=np.float64).ravel()
        
        update_vec = self._fix_update_vector(update_vec, jacobian_parameters)

        for ind, parameter in enumerate(jacobian_parameters):

            if parameter == 0:
                self.focal_length += update_vec.item(ind)

            elif parameter == 1:
                self.kx += update_vec.item(ind)

            elif parameter == 2:
                self.ky += update_vec.item(ind)

            elif parameter == 3:
                self.px += update_vec.item(ind)

            elif parameter == 4:
                self.py += update_vec.item(ind)

            elif parameter == 5:
                self.a1 += update_vec.item(ind)

            elif parameter == 6:
                self.a2 += update_vec.item(ind)

            elif parameter == 7:
                self.a3 += update_vec.item(ind)

            elif parameter == 8 or parameter == '8:':

                misalignment_updates = update_vec[ind:].reshape(
                    3, -1, order='F')

                if self.estimate_multiple_misalignments:
                    self.misalignment = [(Rotation(update.T) * Rotation(self.misalignment[ind])).vector
                                         for ind, update in enumerate(misalignment_updates.T)]

                else:
                    self.misalignment = (
                        Rotation(misalignment_updates) *
                        Rotation(self.misalignment)
                    ).vector

                break

    def prepare_interp(self, pixel_bounds: int = 100, temperature_bounds: Tuple[int, int] = (-50, 50),
                       temperature_step: float = 5):
        """
        This method prepares a SciPy RegularGridInterpolator for converting pixels into undistorted gnomic locations.

        This is done by making calls to :meth:`pixels_to_unit` to compute the transformation at every pixel in the
        detector plus/minus the pixel bounds and for each temperature in the temperature bounds using the
        temperature step. (That is: ``cols = np.arange(-pixel_bounds, self.n_cols+pixel_bounds)``,
        ``rows=np.arange(-pixel_bounds, self.n_rows+pixel_bounds)``,
        ``temps = np.arange(temperature_bounds[0], temperature_bounds[1]+temperature_step, temperature_step)``.)

        This method will likely take a little while to run, but only needs to be run once and then the results are saved
        for future use, including if the camera model is dumped to a file.

        :param pixel_bounds: An integer specifying how many pixels to pad when computing the transformation to gnomic
                             locations.
        :param temperature_bounds: A tuple specifying the temperature bounds to compute the transformation over
                                   (inclusive).  If none of :attr:`temperature_coefficients` are non-zero then this is
                                   ignored.
        :param temperature_step: An integer specifying the temperature step size to compute the transformation to gnomic
                                 locations over.  If none of :attr:`temperature_coefficients` are non-zero then this is
                                 ignored.
        """

        col_labels = np.arange(-pixel_bounds, self.n_cols + pixel_bounds)
        row_labels = np.arange(-pixel_bounds, self.n_rows + pixel_bounds)
        cols, rows = np.meshgrid(col_labels, row_labels)
        pix = np.vstack([cols.ravel(), rows.ravel()])
        if np.any(self.temperature_coefficients != 0):
            temperature_labels = np.arange(
                temperature_bounds[0], temperature_bounds[1], temperature_step)

            results = []
            for temp in temperature_labels:
                results.append(self.pixels_to_gnomic(pix, temperature=float(temp)))

            results = np.array(results)

            self._interp = RegularGridInterpolator((rows[:, 0], cols[0], temperature_labels), results,
                                                   bounds_error=False, fill_value=None) # type: ignore
        else:
            self._interp = RegularGridInterpolator(
                (rows[:, 0], cols[0]), self.pixels_to_gnomic(pix))

    def pixels_to_gnomic_interp(self, pixels: F_ARRAY_LIKE, temperature: float = 0) -> np.ndarray:
        r"""
        This method takes an input in pixels and approximates the undistorted gnomic location in units of distance.

        This approximating is done by interpolating values previously computed using :meth:`pixels_to_gnomic` and
        :meth:`prepare_interp` and will in general run much faster than :meth:`pixels_to_gnomic`.  It should usually be
        accurate to better than a few thousandths of a pixel for any pixels within the field of view.  The interpolation
        is done using SciPy's RegularGridInterpolator with a linear interpolation scheme.

        :param pixels: The pixels to be converted as a shape (2,) or (2, n) Sequence
        :param temperature: The temperature for perform the conversion at.
        :return: The undistorted gnomic location of the points
        """
        pixels = np.asanyarray(pixels, dtype=np.float64)

        if self._interp is None:
            raise ValueError(
                'prepare_interp must be called before pixels_to_gnomic_interp')
        else:
            if np.any(self.temperature_coefficients != 0):
                return self._interp(np.hstack([pixels[::-1].T, temperature*np.ones(pixels.shape[-1])])).T[::-1]
            else:
                return self._interp(pixels[::-1].T).T[::-1]

    def pixels_to_gnomic(self, pixels: Sequence[float] | NDArray, temperature: float = 0, _allow_interp: bool = False) -> np.ndarray:
        r"""
        This method takes an input in pixels and computes the undistorted gnomic location in units of distance.

        This conversion is done iteratively (when there is a distortion model involved).  First, the pixel locations
        are converted to units of distance and re-centered at the principal point by multiplying by the inverse
        intrinsic matrix.

        .. math::
            \mathbf{x}_I'=\mathbf{K}^{-1}\left[\begin{array}{c} \mathbf{x}_P \\ 1 \end{array}\right]

        Next, if there is a distortion model, the distortion is removed iteratively using a fixed point algorithm.

        .. math::
           \mathbf{x}_{Ip}' = d(\mathbf{x}_{Ip}) \\
           \mathbf{x}_{In} = \mathbf{x}_{Ip} + (\mathbf{x}_I' - \mathbf{x}_{Ip}')

        where a subscript of :math:`p` indicates the previous iteration's value, a subscript of :math:`n` indicates
        the new value, and :math:`d()` is the distortion model (method :meth:`apply_distortion`).  This iteration is
        repeated until the solution converges, or 20 iterations have been performed.

        The final iteration's value of the undistorted gnomic points are returned.

        :param pixels: The pixels to be converted as a shape (2,) or (2, n) Sequence
        :param temperature: The temperature for perform the conversion at.
        :param _allow_interp: A flag allowing this to dispatch to the interpolation based conversion in
                              :meth:`pixels_to_gnomic_interp`
        :return: The undistorted gnomic location of the points
        """

        if _allow_interp:
            try:
                return self.pixels_to_gnomic_interp(pixels, temperature)
            except ValueError:
                # warn("Attempted a call to pixels_to_gnomic_interp but prepare_interp hasn't been called yet."
                #      "Falling back to the regular method")
                pass

        # get the distorted gnomic location of the points by multiplying by the inverse camera matrix and dividing by
        # the temperature scale
        gnomic_distorted = (
            (self.intrinsic_matrix_inv[:, :2] @ pixels).T + self.intrinsic_matrix_inv[:, 2]).T

        gnomic_distorted /= self.get_temperature_scale(temperature)

        # initialize the fpa guess to be the distorted gnomic location
        gnomic_guess = gnomic_distorted.copy()

        # perform the fpa
        for _ in np.arange(20):

            # get the distorted location assuming the current guess is correct
            gnomic_guess_distorted = self.apply_distortion(gnomic_guess)

            # subtract off the residual distortion from the gnomic guess
            gnomic_guess += gnomic_distorted - gnomic_guess_distorted

            # check for convergence
            if np.all(np.linalg.norm(gnomic_guess_distorted - gnomic_distorted, axis=0) <= 1e-15):
                break

        # return the new gnomic location
        return gnomic_guess

    def undistort_pixels(self, pixels: F_ARRAY_LIKE, temperature: float = 0, allow_interp: bool = True) -> np.ndarray:
        """
        This method computes undistorted pixel locations (gnomic/pinhole locations) for given distorted
        pixel locations according to the current model.

        This method operates by calling :meth:`pixels_to_gnomic` and then re-transforming the undistorted gnomic
        location into an undistorted pixel location using the :attr:`intrinsic_matrix`.

        The ``pixels`` input should be specified as a shape (2,) or (2, n) array of image locations with units of
        pixels.  The return will be an array of the same shape as ``pixels`` with units of pixels but with distortion
        removed.

        :param pixels: The image points to be converted to gnomic (pinhole) locations as a shape (2,) or (2, n) array
        :param temperature: The temperature to use for the undistortion
        :param allow_interp: Allow the approximate conversion using interpolation for speed
        :return: The undistorted (gnomic) locations corresponding to the distorted pixel locations as an array of
                 the same shape as ``pixels``
        """

        # get the undistorted gnomic location
        gnomic = self.pixels_to_gnomic(
            pixels, temperature=temperature, _allow_interp=allow_interp)

        # scale by the temperature
        gnomic *= self.get_temperature_scale(temperature)

        # put back into pixel space using the intrinsic matrix
        return ((self.intrinsic_matrix[:, :2] @ gnomic).T + self.intrinsic_matrix[:, 2]).T

    def pixels_to_unit(self, pixels: F_ARRAY_LIKE, temperature: float = 0, image: int = 0, 
                       allow_interp: bool = True) -> np.ndarray:
        r"""
        This method converts pixel image locations to unit vectors expressed in the camera frame.

        The pixel locations should be expressed as a shape (2,) or (2, n) array.  They are converted
        to unit vectors by first going through the inverse distortion model (see :meth:`pixels_to_gnomic`) and then
        being converted to unit vectors in the camera frame according to the definitions of the current model (also
        including any misalignment terms).  Once the gnomic locations are retrieved using :meth:`pixels_to_gnomic`, the
        unit vectors are formed according to

        .. math::
            \mathbf{x}_C = \mathbf{T}_{\boldsymbol{\delta\theta}}^T\left[\begin{array}{c} \mathbf{x}_I \\ f
            \end{array}\right] \\
            \hat{\mathbf{x}}_C = \frac{\mathbf{x}_C}{\|\mathbf{x}_C\|}

        where :math:`\mathbf{T}_{\boldsymbol{\delta\theta}}^T` is the inverse rotation matrix for the misalignment
        for the image.

        :param pixels: The image points to be converted to unit vectors in the camera frame as a shape (2,) or (2, n)
                       array
        :param temperature: The temperature to use for the undistortion
        :param image: The image index that the pixel belong to (only important if there are multiple misalignments)
        :param allow_interp: Allow the approximate conversion using interpolation for speed
        :return: The unit vectors corresponding to the image locations expressed in the camera frame as a shape (3,) or
                 (3, n) array.
        """

        pixels = np.array(pixels)

        # get the undistorted gnomic locations
        gnomic_locs = self.pixels_to_gnomic(
            pixels, temperature=temperature, _allow_interp=allow_interp)

        # append the focal length
        if pixels.ndim == 1:
            los_vectors = np.hstack([gnomic_locs, self.focal_length])

        else:
            los_vectors = np.vstack(
                [gnomic_locs, self.focal_length * np.ones((1, pixels.shape[1]))])

        # apply misalignment to the unit vectors
        if self.estimate_multiple_misalignments:
            # optimization to avoid matrix multiplication
            if np.any(self.misalignment[image]):
                los_vectors = rotvec_to_rotmat(
                    self.misalignment[image]).squeeze().T @ los_vectors

        else:
            if np.any(self.misalignment):  # optimization to avoid matrix multiplication
                los_vectors = rotvec_to_rotmat(
                    self.misalignment).squeeze().T @ los_vectors

        # convert to unit vector and return
        return los_vectors / np.linalg.norm(los_vectors, axis=0, keepdims=True)

    def distort_pixels(self, pixels: F_ARRAY_LIKE, temperature: float = 0) -> DOUBLE_ARRAY:
        """
        A method that takes gnomic pixel locations in units of pixels and applies the appropriate distortion to them.

        This method is used in the :meth:`distortion_map` method to generate the distortion values for each pixel.

        The distortion is applied by first converting the pixels to gnomic location by multiplying by the inverse camera
        matrix, second applying the distortion to the gnomic locations using :meth:`apply_distortion`, and third
        reconverting to pixel units using the :attr:`intrinsic_matrix` and temperature scale.

        In general this method is not used directly by the user and instead the :meth:`distortion_map` method is used
        to generate a distortion map for the camera model

        :param pixels: The pinhole location pixel locations the distortion is to be applied to
        :param temperature:  The temperature to perform the distortion at
        :return: The distorted pixel locations in units of pixels
        """

        # get the scale for the current temperature
        temp_scale = self.get_temperature_scale(temperature)

        # get the gnomic location using the inverse intrinsic matrix and temperature scale
        gnomic = (
            (self.intrinsic_matrix_inv[:, :2] @ pixels).T + self.intrinsic_matrix_inv[:, 2]).T

        gnomic /= temp_scale

        # apply the distortion
        gnomic_distorted = self.apply_distortion(gnomic)

        # reconvert to pixel units and return
        gnomic_distorted *= temp_scale

        return ((self.intrinsic_matrix[:, :2] @ gnomic_distorted).T + self.intrinsic_matrix[:, 2]).T

    # noinspection PyProtectedMember
    def to_elem(self, elem: etree._Element, misalignment: bool = False) -> etree._Element:
        """
        Stores this camera model in an :class:`etree._Element` object for storing in a GIANT xml file

        :param elem: The :class:`etree._Element` class to store this camera model in
        :param misalignment: A flag about whether to include the misalignment in the :class:`etree._Element`
        :return: The :class:`etree._Element` for this model
        """

        # store a copy of self as it currently is
        copy_of_self = cast(PinholeModel, self.copy())

        # if we don't want to include the misalignment in the save file then set
        # estimation parameters to only include a single misalignment
        # and set the misalignment to be zero
        if not misalignment:

            self.misalignment = np.zeros(3)

            if 'multiple misalignments' in self.estimation_parameters:
                # noinspection PyUnresolvedReferences
                self.estimation_parameters.remove('multiple misalignments')
                # noinspection PyUnresolvedReferences
                self.estimation_parameters.append('single misalignment')
                self.estimate_multiple_misalignments = False

        # store the model into the element
        elem = super().to_elem(elem)

        # reset self to the way it was
        self.overwrite(copy_of_self)

        return elem

    def reset_misalignment(self):
        """
        This method reset the misalignment terms to all be zero (no misalignment).
        """
        if self.estimate_multiple_misalignments:
            new_misalignment = [np.zeros(3, dtype=np.float64)
                                for _ in self.misalignment]
        else:
            new_misalignment = np.zeros(3, dtype=np.float64)

        self.misalignment = new_misalignment

    def get_misalignment(self, image: int = 0) -> Rotation:
        """
        This method returns the Rotation object for the misalignment for the requested image.

        :param image: the image number to get the misalignment for
        :return: The :class:`.Rotation` representing the misalignment between the camera frame and the actual image
                 frame for projection.
        """

        if self.estimate_multiple_misalignments:
            return Rotation(self.misalignment[image])
        else:
            return Rotation(self.misalignment)
            
    def check_in_fov(self, vectors: F_ARRAY_LIKE, image: int = 0, temperature: float = 0) -> NDArray[np.bool]:
        """
        Determines if any points in the array are within the field of view of the camera.

        :param vectors: Vectors to check if they are in the field of view of the camera expressed as a shape (3, n) array in the camera frame.  
        :param image: The index of the image being projected onto (only applicable with multiple misalignments)
        :param temperature: The temperature of the camera to use for the projection
        :return: A boolean array the same length as the number of columns of vectors. False by default, True if the point is in the FOV.
        """

        vectors = np.asanyarray(vectors)
        pixels = self.project_onto_image(vectors, image=image, temperature=temperature)

        in_fov = (pixels[0] >= 0) & (pixels[1] >= 0) & \
                 (pixels[0] <= self.n_cols) & (pixels[1] <= self.n_rows)

        # the camera models break down at extremes so also spot check the angular FOV
        angles = (np.arccos(np.array([[0, 0, 1]]) @ vectors[:, in_fov] / np.linalg.norm(vectors[:, in_fov],
                                                                                         axis=0,
                                                                                         keepdims=True)) *
                  180 / np.pi).ravel()
        if self.field_of_view == 0:
            # get the FOV computed
            self.compute_field_of_view(temperature=temperature)
            
        in_fov[in_fov] = angles < 1.25 * self.field_of_view

        return in_fov
       
        
class CircularPinholeModel(PinholeModel):
    """
    This class overrides the :class:`.PinholeModel` to reimplement the :meth:`.check_in_fov` method to only check the angle between the 
    boresight and the target vectors.
    """
    def check_in_fov(self, vectors: ARRAY_LIKE, image: int = 0, temperature: float = 0) -> np.ndarray:
        """
        Determines if any points in the array are within the field of view of the camera.
        
        :param vectors: Vectors to check if they are in the field of view of the camera expressed as a shape (3, n) array in the camera frame.  
        :param image: ignored
        :param temperature: used to compute the field of view if the field of view is not specified
        :return: A boolean array the same length as the number of columns of vectors. False by default, True if the point is in the FOV.
        """
        # the camera models break down at extremes so also spot check the angular FOV
        angles = (np.arccos(np.array([[0, 0, 1]]) @ vectors / np.linalg.norm(vectors,
                                                                             axis=0,
                                                                             keepdims=True)) *
                  180 / np.pi).ravel()
        if self.field_of_view == 0:
            # get the FOV computed
            self.compute_field_of_view(temperature=temperature)
            
        in_fov = angles < 1.0 * self.field_of_view

        return in_fov
            

