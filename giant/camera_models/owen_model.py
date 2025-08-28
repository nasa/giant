


r"""
This module provides a subclass of :class:`.CameraModel` that implements the distortion modeled Owen (or JPL) camera
model.

Theory
______

The Owen camera model is the pinhole camera model combined with a lens distortion model which projects any point
along a ray emanating from the camera center (origin of the camera frame) to the same 2D point in an image.  Given
some 3D point (or direction) expressed in the camera frame, :math:`\mathbf{x}_C`, the Owen model is defined as

.. math::
    &\mathbf{x}_I = \frac{f}{z_C}\left[\begin{array}{c} x_C \\ y_C \end{array}\right] \\
    &r = \sqrt{x_I^2 + y_I^2} \\
    &\Delta\mathbf{x}_I = \left(\epsilon_2r^2+\epsilon_4r^4+\epsilon_5y_I+\epsilon_6x_I\right)\mathbf{x}_I+
    \left(\epsilon_1r + \epsilon_2r^3\right)\left[\begin{array}{c} -y_I \\ x_I \end{array}\right] \\
    &\mathbf{x}_P = \left[\begin{array}{ccc} k_x & k_{xy} & p_x \\ k_{yx} & k_y & p_y\end{array}\right]
    \left[\begin{array}{c} (1+a_1T+a_2T^2+a_3T^3)(\mathbf{x}_I+\Delta\mathbf{x}_I) \\ 1 \end{array}\right]

where :math:`\mathbf{x}_I` are the image frame coordinates for the point (pinhole location), :math:`f` is the
focal length of the camera in units of distance, :math:`r` is the radial distance from the principal point of the
camera to the gnomic location of the point, :math:`\epsilon_2` and :math:`\epsilon_4` are radial distortion
coefficients, :math:`\epsilon_5` and :math:`\epsilon_6` are tip/tilt/prism distortion coefficients,
:math:`\epsilon_1` and :math:`\epsilon_3` are pinwheel distortion coefficients, :math:`\Delta\mathbf{x}_I` is
the distortion for point :math:`\mathbf{x}_I`, :math:`k_x` and :math:`k_y` are the pixel pitch values in units of
pixels/distance in the :math:`x` and :math:`y` directions respectively, :math:`k_{xy}` and :math:`k_{yx}` are alpha
terms for non-rectangular pixels, :math:`p_x` and :math:`p_y` are the location of the principal point of the
camera in the image expressed in units of pixels, :math:`T` is the temperature of the camera, :math:`a_{1-3}` are
temperature dependence coefficients, and :math:`\mathbf{x}_P` is the pixel location of the point in
the image. For a more thorough description of the Owen camera model checkout
:download:`this memo <docs/cameraModels.pdf>`.

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
sufficient for nearly every use case.  The :class:`.OwenModel` and its subclasses make precomputing the
transformation, and using the precomputed transformation, as easy as calling :meth:`~OwenModel.prepare_interp` once.
Future calls to any method that then needs the transformation from pixels to gnomic locations (on the way to unit
vectors) will then use the precomputed transformation unless specifically requested otherwise.  In addition, once the
:meth:`~OwenModel.prepare_interp` method has been called, if the resulting camera object is then saved to a file
either using the :mod:`.camera_model`
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
and a detector size of 1024x1024 with second order radial distortion.  We could then create a model for this camera as

    >>> from giant.camera_models import OwenModel
    >>> model = OwenModel(focal_length=10, kx=1/2.2e-3, ky=1/2.2e-3, radial2=1e-5,
    ...                   n_rows=1024, n_cols=1024, px=(1024-1)/2, py=(1024-1)/2)

Note that we did not set the field of view, but it is automatically computed for us based off of the prescribed camera
model.

    >>> model.field_of_view
    9.050773898292755

In addition, we can now use our model to project points

    >>> model.project_onto_image([0, 0, 1])
    array([511.5, 511.5])

or to determine the unit vector through a pixel

    >>> model.pixels_to_unit([[0, 500], [0, 100]])
    array([[-0.1111288 , -0.00251967],
           [-0.1111288 , -0.09016027],
           [ 0.98757318,  0.99592408]])

"""

from typing import Union, Tuple, Sequence

import warnings

import numpy as np

from giant._typing import ARRAY_LIKE, NONEARRAY, NONENUM, F_ARRAY_LIKE
from giant.rotations import rotvec_to_rotmat, Rotation
from giant.camera_models.pinhole_model import PinholeModel


class OwenModel(PinholeModel):
    r"""
    This class provides an implementation of the Owen camera model for projecting 3D points onto images and performing
    camera calibration.

    The :class:`OwenModel` class is a subclass of :class:`.CameraModel`.  This means that it includes implementations
    for all of the abstract methods defined in the :class:`.CameraModel` class.  This also means that it can be used
    throughout GIANT as the primary camera model, including within the :mod:`.calibration` subpackage.  If this class is
    going to be used with the :mod:`.calibration` subpackage, the user can set which parameters are estimated and which
    are held fixed by using the ``estimation_parameters`` key word argument when creating an instance of the class or by
    adjusting the :attr:`~OwenModel.estimation_parameters` instance variable on an instance of the class.  The
    ``estimation_parameters`` input/attribute is a string or list of strings specifying which parameters to estimate.
    This means that :attr:`estimation_parameters` could be something like ``'basic'`` which would indicate to estimate
    just eh usual parameters, or something like ``['focal_length', 'ky', 'px', 'py']`` to estimate just the terms
    included in the list.

    In addition to the standard set of methods for a :class:`.CameraModel` subclass, the :class:`OwenModel` class
    provides the following additional methods which may or may not be useful to some people:

    =================================  =================================================================================
    Method                             Use
    =================================  =================================================================================
    :meth:`get_projections`            computes the pinhole, image frame, and pixel locations of a 3D point
    :meth:`pixels_to_gnomic`           removes distortion from a point to get the corresponding pinhole location in
                                       units of distance
    =================================  =================================================================================

    The Owen model also provides the following properties for easy getting/setting:

    ================================ ===================================================================================
    Property                         Description
    ================================ ===================================================================================
    :attr:`field_of_view`            the diagonal field of view of the camera in units of degrees
    :attr:`kx`                       :math:`k_x`, the pixel pitch in the x direction in units of pixels/distance
    :attr:`ky`                       :math:`k_y`, the pixel pitch in the y direction in units of pixels/distance
    :attr:`kxy`                      :math:`k_{xy}`, A alpha term for non-rectangular pixels
    :attr:`kyx`                      :math:`k_{yx}`, A alpha term for non-rectangular pixels
    :attr:`px`                       :math:`p_{x}`, the x axis pixel location of the principal point of the camera in
                                     units of pixels
    :attr:`py`                       :math:`p_{y}`, the y axis pixel location of the principal point of the camera in
                                     units of pixels
    :attr:`radial2`, :attr:`e2`      :math:`\epsilon_2`, the radial distortion coefficient corresponding to :math:`r^2`
    :attr:`radial4`, :attr:`e4`      :math:`\epsilon_4`, the radial distortion coefficient corresponding to :math:`r^4`
    :attr:`tangential_y`, :attr:`e5` :math:`\epsilon_5`, the tip/tilt/prism distortion coefficient corresponding to
                                     :math:`y_I`
    :attr:`tangential_x`, :attr:`e6` :math:`\epsilon_6`, the tip/tilt/prism distortion coefficient corresponding to
                                     :math:`x_I`
    :attr:`pinwheel1`, :attr:`e1`    :math:`\epsilon_1`, the pinwheel distortion coefficient corresponding to :math:`r`
    :attr:`pinwheel2`, :attr:`e3`    :math:`\epsilon_3`, the pinwheel distortion coefficient corresponding to
                                     :math:`r^3`
    :attr:`a1`                       :math:`a_1`, the linear coefficient for focal length dependent focal length
    :attr:`a2`                       :math:`a_2`, the quadratic coefficient for focal length dependent focal length
    :attr:`a3`                       :math:`a_3`, the cubic coefficient for focal length dependent focal length
    :attr:`intrinsic_matrix_inv`     The inverse of the intrinsic matrix
    ================================ ===================================================================================

    .. note:: The distortion attributes are aliases over each other and refer to the same data.  Therefore setting a
              value to :attr:`radial2` would also change the value of :attr:`e2`

    """

    def __init__(self, intrinsic_matrix: NONEARRAY = None, focal_length: float = 1., field_of_view: NONENUM = None,
                 use_a_priori: bool = False, distortion_coefficients: NONEARRAY = None, misalignment: NONEARRAY = None,
                 estimation_parameters: Union[str, Sequence] = 'basic',
                 kx: NONENUM = None, ky: NONENUM = None, kxy: NONENUM = None, kyx: NONENUM = None, px: NONENUM = None,
                 py: NONENUM = None, radial2: NONENUM = None, radial4: NONENUM = None, tangential_x: NONENUM = None,
                 tangential_y: NONENUM = None, pinwheel1: NONENUM = None, pinwheel2: NONENUM = None, e1: NONENUM = None,
                 e2: NONENUM = None, e3: NONENUM = None, e4: NONENUM = None, e5: NONENUM = None, e6: NONENUM = None,
                 a1: NONENUM = None, a2: NONENUM = None, a3: NONENUM = None, n_rows: int = 1, n_cols: int = 1,
                 temperature_coefficients: NONEARRAY = None):
        """
        :param intrinsic_matrix: the intrinsic matrix for the camera as a numpy shape (2, 3) array.  Note that this is
                                 overwritten if ``kx``, ``ky``, ``kxy``, ``kyx``, ``px``, ``py`` are also specified.
        :param focal_length: The focal length of the camera in units of distance
        :param field_of_view: The field of view of the camera in units of degrees.
        :param use_a_priori: A flag to indicate whether to include the a priori state vector in the Jacobian matrix when
                             performing a calibration
        :param distortion_coefficients: A numpy array of shape (6,) containing the six distortion coefficients in order.
                                        Note that this array is overwritten with any distortion coefficients that are
                                        specified independently.
        :param misalignment: either a numpy array of shape (3,) or a list of numpy arrays of shape(3,) with each array
                             corresponding to a single image (the list of numpy arrays is only valid when estimating
                             multiple misalignments)
        :param estimation_parameters: A string or list of strings specifying which model parameters to include in the
                                      calibration
        :param kx: The pixel pitch along the x axis in units of pixel/distance
        :param ky: The pixel pitch along the y axis in units of pixel/distance
        :param kxy: A alpha term for non-rectangular pixels
        :param kyx: A alpha term for non-rectangular pixels
        :param px: the x component of the pixel location of the principal point in the image in units of pixels
        :param py: the y component of the pixel location of the principal point in the image in units of pixels
        :param radial2: the radial distortion coefficient corresponding to the r**2 term
        :param e2: the radial distortion coefficient corresponding to the r**2 term
        :param radial4: the radial distortion coefficient corresponding to the r**4 term
        :param e4: the radial distortion coefficient corresponding to the r**4 term
        :param tangential_x: the tangential distortion coefficient corresponding to the x term
        :param e6: the tangential distortion coefficient corresponding to the x term
        :param tangential_y: the tangential distortion coefficient corresponding to the y term
        :param e5: the tangential distortion coefficient corresponding to the y term
        :param pinwheel1: the pinwheel distortion coefficient corresponding to the r term
        :param e1: the pinwheel distortion coefficient corresponding to the r term
        :param pinwheel2: the pinwheel distortion coefficient corresponding to the r**2 term
        :param e3: the pinwheel distortion coefficient corresponding to the r**2 term
        :param n_rows: the number of rows of the active image array
        :param n_cols: the number of columns in the active image array
        :param a1: the linear coefficient of the focal length temperature dependence
        :param a2: the quadratic coefficient of the focal length temperature dependence
        :param a3: the cubic coefficient of the focal length temperature dependence
        :param temperature_coefficients: The temperature polynomial coefficients as a length 3 Sequence
        """

        # call the super init
        # temporarily set the field of view to 0
        super().__init__(intrinsic_matrix=intrinsic_matrix, kx=kx, ky=ky, px=px, py=py,
                         focal_length=focal_length, misalignment=misalignment, use_a_priori=use_a_priori,
                         a1=a1, a2=a2, a3=a3, n_rows=n_rows, n_cols=n_cols,
                         temperature_coefficients=temperature_coefficients)

        # finish the intrinsic matrix
        if kxy is not None:
            self.kxy = kxy
        if kyx is not None:
            self.kyx = kyx

        # set the distortion coefficients vector
        self.distortion_coefficients = np.zeros(6)
        """
        The distortion coefficients array contains the distortion coefficients for the Brown model 
        [e2, e4, e5, e6, e1, e3].
        """

        if distortion_coefficients is not None:
            self.distortion_coefficients = distortion_coefficients

        if radial2 is not None:
            self.radial2 = radial2
        if radial4 is not None:
            self.radial4 = radial4
        if tangential_x is not None:
            self.tangential_x = tangential_x
        if tangential_y is not None:
            self.tangential_y = tangential_y
        if pinwheel1 is not None:
            self.pinwheel1 = pinwheel1
        if pinwheel2 is not None:
            self.pinwheel2 = pinwheel2

        if e2 is not None:
            self.e2 = e2
        if e4 is not None:
            self.e4 = e4
        if e6 is not None:
            self.e6 = e6
        if e5 is not None:
            self.e5 = e5
        if e1 is not None:
            self.e1 = e1
        if e3 is not None:
            self.e3 = e3

        self._state_labels = ['focal_length', 'kx', 'kxy', 'kyx', 'ky', 'px', 'py', 'e2', 'e4', 'e5', 'e6', 'e1', 'e3',
                              'a1', 'a2', 'a3', 'misalignment']
        """
        A list of state labels that correspond to the attributes of this class.
        """

        # store the element dict for indices into the state vector
        self.element_dict = {
            'basic': [0, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            'intrinsic': np.arange(0, 13),
            'basic intrinsic': [0, 3, 4, 7, 8, 9, 10, 11, 12],
            'temperature dependence': [13, 14, 15],
            'focal_length': [0],
            'kx': [1],
            'kxy': [2],
            'kyx': [3],
            'ky': [4],
            'px': [5],
            'py': [6],
            'radial2': [7],
            'radial4': [8],
            'tangential_y': [9],
            'tangential_x': [10],
            'pinwheel1': [11],
            'pinwheel2': [12],
            'e2': [7],
            'e4': [8],
            'e5': [9],
            'e6': [10],
            'e1': [11],
            'e3': [12],
            'a1': [13],
            'a2': [14],
            'a3': [15],
            'single misalignment': slice(16, None, None),
            'multiple misalignments': slice(16, None, None)
        }

        self.estimation_parameters = estimation_parameters

        # add the distortion parameters to the important variables
        self.important_attributes = self.important_attributes + ['radial2', 'radial4', 'tangential_y', 'tangential_x',
                                                                 'pinwheel1', 'pinwheel2', 'kxy', 'kyx']
        """
        A list specifying the important attributes the must be saved/loaded for this camera model to be completely 
        reconstructed. 
        """

        self.field_of_view = field_of_view

    def __repr__(self):

        template = "OwenModel(kx={kx}, ky={ky}, kxy={kxy}, kyx={kyx}, px={px}, py={py},\n" \
                   "           field_of_view={fov}, focal_length={f}, radial2={r2}, radial4={r4}, \n" \
                   "           tangential_x={tanx}, tangential_y={tany}, pinwheel1={pin1}, pinwheel2={pin2}\n" \
                   "           misalignment={mis!r}, \n" \
                   "           estimation_parameters={ep!r}, use_a_priori={ap}, a1={a1}, a2={a2}, a3={a3}," \
                   "           n_rows={nr}, n_cols={nc})\n\n"

        return template.format(
            kx=self.kx, ky=self.ky, kxy=self.kxy, kyx=self.kyx, px=self.px, py=self.py,
            fov=self.field_of_view, f=self.focal_length, r2=self.radial2, r4=self.radial4,
            tanx=self.tangential_x, tany=self.tangential_y, pin1=self.pinwheel1, pin2=self.pinwheel2,
            mis=self.misalignment, ep=self.estimation_parameters, ap=self.use_a_priori,
            a1=self.a1, a2=self.a2, a3=self.a3, nr=self.n_rows, nc=self.n_cols
        )

    def __str__(self):
        template = "Owen Camera Model:\n\n" \
                   " __  __     __       __    \n" \
                   "|   x  | _ |  f*Xc/Zc  |   \n" \
                   "|   y  | - |  f*Yc/Zc  |   \n" \
                   " --  --     --       --    \n" \
                   "      _________ \n" \
                   "     /  2   2   \n" \
                   "r =\\/  x + y    \n" \
                   " __ __                                      __ __                    __  __   \n" \
                   "|  x' |            2      4                |  x  |               3  |  -y  |  \n" \
                   "|     | = (1 + e2*r + e4*r + e5*y + e6*x ) |     | + (e1*r + e3*r ) |      |  \n" \
                   "|  y' |                                    |  y  |                  |   x  |  \n" \
                   " -- --                                      -- --                    --  --   \n" \
                   " __  __                               __ __   \n" \
                   "|  xt' |                   2      3  |  x' |  \n" \
                   "|      | = (1 + a1*T + a2*T + a3*T ) |     |  \n" \
                   "|  yt' |                             |  y' |  \n" \
                   " --  --                               -- --   \n" \
                   " __ __     __           __  __  __  \n" \
                   "|  u  | _ |  kx  kxy  px  ||  xt' | \n" \
                   "|  v  | - |  kyx ky   py  ||  yt' | \n" \
                   " -- --     --           -- |  1   | \n" \
                   "                            --  --  \n\n" \
                   "————————————————————————————————————————————————————————————————————————————\n\n" \
                   "distortion coefficients:\n" \
                   "    e1={0}, e2={1}, e3={2}, e4={3}, e5={4}, e6={5}\n\n" \
                   "camera parameters:\n" \
                   "    f={12}, kx={6}, ky={7}, kxy={8}, kyx={9}, px={10}, py={11}\n\n" \
                   "temperature coefficients:\n" \
                   "    a1={13}, a2={14}, a3={15}\n\n"

        return template.format(self.e1, self.e2, self.e3, self.e4, self.e5, self.e6,
                               self.kx, self.ky, self.kxy, self.kyx, self.px, self.py,
                               self.focal_length, self.a1, self.a2, self.a3)

    @property
    def kxy(self) -> float:
        """
        A alpha term for non-rectangular pixels.

        This corresponds to the [0, 1] component of the intrinsic matrix
        """
        return self.intrinsic_matrix[0, 1]

    @kxy.setter
    def kxy(self, val):
        self.intrinsic_matrix[0, 1] = val

    @property
    def kyx(self) -> float:
        """
        A alpha term for non-rectangular pixels.

        This corresponds to the [1, 0] component of the intrinsic matrix
        """
        return self.intrinsic_matrix[1, 0]

    @kyx.setter
    def kyx(self, val):
        self.intrinsic_matrix[1, 0] = val

    @property
    def radial2(self) -> float:
        """
        The radial distortion coefficient corresponding to the r**2 term

        This corresponds to the [0] index of the distortion_coefficients array
        """
        return self.distortion_coefficients[0]

    @radial2.setter
    def radial2(self, val):
        self.distortion_coefficients[0] = val

    @property
    def e2(self) -> float:
        """
        The radial distortion coefficient corresponding to the r**2 term

        This corresponds to the [0] index of the distortion_coefficients array
        """
        return self.distortion_coefficients[0]

    @e2.setter
    def e2(self, val):
        self.distortion_coefficients[0] = val

    @property
    def radial4(self) -> float:
        """
        The radial distortion coefficient corresponding to the r**4 term

        This corresponds to the [1] index of the distortion_coefficients array
        """
        return self.distortion_coefficients[1]

    @radial4.setter
    def radial4(self, val):
        self.distortion_coefficients[1] = val

    @property
    def e4(self) -> float:
        """
        The radial distortion coefficient corresponding to the r**4 term

        This corresponds to the [1] index of the distortion_coefficients array
        """
        return self.distortion_coefficients[1]

    @e4.setter
    def e4(self, val):
        self.distortion_coefficients[1] = val

    @property
    def tangential_x(self) -> float:
        """
        The tangential distortion coefficient corresponding to the x term

        this corresponds to the [3] index of the distortion_coefficients array
        """
        return self.distortion_coefficients[3]

    @tangential_x.setter
    def tangential_x(self, val):
        self.distortion_coefficients[3] = val

    @property
    def e6(self) -> float:
        """
        The tangential distortion coefficient corresponding to the x term

        this corresponds to the [3] index of the distortion_coefficients array
        """
        return self.distortion_coefficients[3]

    @e6.setter
    def e6(self, val):
        self.distortion_coefficients[3] = val

    @property
    def tangential_y(self) -> float:
        """
        The tangential distortion coefficient corresponding to the y term

        this corresponds to the [2] index of the distortion_coefficients array
        """
        return self.distortion_coefficients[2]

    @tangential_y.setter
    def tangential_y(self, val):
        self.distortion_coefficients[2] = val

    @property
    def e5(self) -> float:
        """
        The tangential distortion coefficient corresponding to the y term

        this corresponds to the [2] index of the distortion_coefficients array
        """
        return self.distortion_coefficients[2]

    @e5.setter
    def e5(self, val):
        self.distortion_coefficients[2] = val

    @property
    def pinwheel1(self) -> float:
        """
        The pinwheel distortion coefficient corresponding to the r term.

        This corresponds to the [4] index of the distortion_coefficients array.
        """
        return self.distortion_coefficients[4]

    @pinwheel1.setter
    def pinwheel1(self, val):
        self.distortion_coefficients[4] = val

    @property
    def e1(self) -> float:
        """
        The pinwheel distortion coefficient corresponding to the r term.

        This corresponds to the [4] index of the distortion_coefficients array.
        """
        return self.distortion_coefficients[4]

    @e1.setter
    def e1(self, val):
        self.distortion_coefficients[4] = val

    @property
    def pinwheel2(self) -> float:
        """
        The pinwheel distortion coefficient corresponding to the r**3 term.

        This corresponds to the [5] index of the distortion_coefficients array.
        """
        return self.distortion_coefficients[5]

    @pinwheel2.setter
    def pinwheel2(self, val):
        self.distortion_coefficients[5] = val

    @property
    def e3(self) -> float:
        """
        The pinwheel distortion coefficient corresponding to the r**3 term.

        This corresponds to the [5] index of the distortion_coefficients array.
        """
        return self.distortion_coefficients[5]

    @e3.setter
    def e3(self, val):
        self.distortion_coefficients[5] = val

    @property
    def intrinsic_matrix_inv(self) -> np.ndarray:
        r"""
        The inverse of the intrinsic matrix.

        The inverse of the intrinsic matrix is used to convert from units of pixels with an origin at the upper left
        corner of the image to units of distance with an origin at the principal point of the image.

        The intrinsic matrix has an analytic inverse which is given by

        .. math::
            d=k_xk_y-k_{xy}k_{yx} \\
            \mathbf{K}^{-1} = \left[\begin{array}{ccc} \frac{k_y}{d} & -\frac{k_{xy}}{d} &
            \frac{k_{xy}p_y-k_yp_x}{d} \\
            -\frac{k_{yx}}{d} & \frac{k_x}{d} & \frac{k_{yx}p_x-k_xp_y}{d} \end{array}\right]

        To convert from units of pixels to units of distance you would do::
            >>> from giant.camera_models import OwenModel
            >>> model = OwenModel(kx=5, ky=10, px=100, py=500)
            >>> ((model.intrinsic_matrix_inv[:, :2]@[[1, 2, 300], [4, 5, 600]]).T + model.intrinsic_matrix_inv[:, 2]).T
            array([[-19.8, -19.6, 40.]
                   [-49.6, -49.5, 10.]])

        .. note:: The above code will give you distorted gnomic location, while the :meth:`pixels_to_gnomic` will give
                  you undistorted gnomic locations (true pinhole points).

        .. note:: Since the intrinsic matrix is defined as a :math:`2\times 3` matrix this
                  isn't a formal inverse.  To get the true inverse you need to append a row of [0, 0, 1] to both the
                  intrinsic matrix and intrinsic matrix inverse.
        """

        tldet = self.kx * self.ky - self.kxy * self.kyx
        return np.array([[self.ky / tldet, -self.kxy / tldet, (self.py * self.kxy - self.ky * self.px) / tldet],
                         [-self.kyx / tldet, self.kx / tldet, (-self.py * self.kx + self.kyx * self.px) / tldet]])

    def apply_distortion(self, pinhole_locations: ARRAY_LIKE) -> np.ndarray:
        r"""
        This method applies the distortion model to the specified pinhole (gnomic) locations in the image frame.

        In general this function is not used by the user and the higher level :meth:`project_onto_image` is used
        which calls this method (along with a few others) instead.  In cases were it is desirable to use this method
        the pinhole locations should be input as a shape (2,) or shape (2, n) array of image plane locations in units of
        distance.  The output from this function is the distorted image plane locations of the points in units of
        distance.

        For the Owen model, the distortion is defined as

        .. math::
            \Delta\mathbf{x}_I = \left(\epsilon_2r^2+\epsilon_4r^4+\epsilon_5y_I+\epsilon_6x_I\right)\mathbf{x}_I +
            \left(\epsilon_1r + \epsilon_2r^3\right)\left[\begin{array}{c} -y_I \\ x_I \end{array}\right]

        where :math:`\Delta\mathbf{x}_I` is the additive distortion, :math:`\epsilon_2` and :math:`\epsilon_4` are
        radial distortion coefficients, :math:`\epsilon_5` and :math:`\epsilon_6` are tip/tilt/prism distortion
        coefficients, :math:`\epsilon_1` and :math:`\epsilon_3` are pinwheel distortion coefficients,
        :math:`\mathbf{x}_I` is the gnomic location in units of distance, and
        :math:`r = \sqrt{\mathbf{x}_I^T\mathbf{x}_I}` is the radial distance from the optical axis.

        :param pinhole_locations: The image plane location of points to be distorted as a shape (2,) or (2, n) array in
                                  units of distance
        :return: The distorted locations of the points on the image plane as a shape (2,) or (2, n) array in units of
                 distance
        """

        # make sure the input is an array
        pinhole_locations = np.array(pinhole_locations)

        radius = np.linalg.norm(pinhole_locations, axis=0, keepdims=True)
        radius2 = radius ** 2
        radius3 = radius2 * radius
        radius4 = radius2 ** 2

        distortion_values = (self.radial2 * radius2 + self.radial4 * radius4 +
                             self.tangential_y * pinhole_locations[1] +
                             self.tangential_x * pinhole_locations[0]) * pinhole_locations + \
                            (self.pinwheel1 * radius + self.pinwheel2 * radius3) * np.array([-pinhole_locations[1],
                                                                                             pinhole_locations[0]])

        return pinhole_locations + distortion_values

    def get_projections(self, points_in_camera_frame: ARRAY_LIKE, image: int = 0, temperature: float = 0) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This method computes and returns the pinhole, and pixel locations for a set of
        3D points expressed in the camera frame.

        In general the user will not use this method and the higher level :meth:`project_onto_image` will be used
        instead.  In cases where it is desirable to use this method, the camera points should be input as a shape (2,)
        or shape (2, n) array of points expressed in the camera frame (units don't matter)  This method will then return
        the gnomic locations in units of distance as a shape (2,) or (2, n) numpy array and the pixel locations of the
        points as a shape (2,) or (2, n) numpy array with units of pixels as a length 2 tuple.

        The optional image flag specifies which image you are projecting the points onto.  This is only important if you
        have the estimate_multiple_misalignments flag set to true, and have a different alignment set for each image.
        In general, the optional input image should be ignored except during calibration.

        The optional temperature input specifies the temperature to perform the projection at.  This is only import when
        your focal length is dependent on temperature and you have entered or calibrated for the temperature dependency
        coefficients.

        You can specify the directions to be input as either a shape (3,) or shape (3, n) array::
            >>> from giant.camera_models import OwenModel
            >>> model = OwenModel(kx=300, ky=400, px=500, py=500, focal_length=10, a1=1e-5, a2=1e-6, radial2=1e-5,
            >>>                   misalignment=[[1e-12, -2e-14, 3e-10], [2e-15, 1e-13, 3e-10]],
            >>>                   estimation_parameters=['multiple misalignments'])
            >>> model.get_projections([1, 2, 12000])
            (array([ 0.00083333,  0.00166667]), array([0.00112269, 0.00224537]), array([500.33680555, 500.89814814]))
            >>> model.get_projections([[1, 2, 3, 4], [2, 5, 6, 7], [12000, 13000, 9000, 5000]], image=1)
            (array([[0.00083333, 0.00153846, 0.00333333, 0.008     ],
                    [0.00166667, 0.00384615, 0.00666667, 0.014     ]]),
             array([[0.00112269, 0.00417843, 0.02185185, 0.216     ],
                    [0.00224537, 0.01044606, 0.0437037 , 0.378     ]]),
             array([[500.33680556, 501.25352754, 506.55555555, 564.79999998],
                    [500.89814815, 504.17842513, 517.48148149, 651.20000003]]))
            >>> model.get_projections([[1, 2, 3, 4], [2, 5, 6, 7], [12000, 13000, 9000, 5000]], temperature=-1)
            (array([[0.00083333, 0.00153846, 0.00333333, 0.008     ],
                    [0.00166667, 0.00384615, 0.00666667, 0.014     ]]),
             array([[0.00112268, 0.00417839, 0.02185166, 0.21599806],
                    [0.00224535, 0.01044597, 0.04370331, 0.3779966 ]]),
             array([[500.33680252, 501.25351625, 506.55549654, 564.7994167 ],
                    [500.89814006, 504.1783875 , 517.48132409, 651.19863896]]))

        :param points_in_camera_frame: a shape (3,) or shape (3, n) array of points to project
        :param image: The index of the image being projected onto (only applicable with multiple misalignments)
        :param temperature: The temperature to project the points at
        :return: A tuple of the pinhole, distorted pinhole, and pixel locations for a set of 3D points
                 expressed in the camera frame
        """

        # ensure the points are an array
        camera_points = np.asarray(points_in_camera_frame)

        # apply misalignment to the points
        if self.estimate_multiple_misalignments:
            if np.any(self.misalignment[image]):  # optimization to avoid matrix multiplication
                camera_points = rotvec_to_rotmat(self.misalignment[image]).squeeze() @ camera_points

        else:
            if np.any(self.misalignment):  # optimization to avoid matrix multiplication
                camera_points = rotvec_to_rotmat(self.misalignment).squeeze() @ camera_points

        # get the pinhole locations of the points
        pinhole_locations: np.ndarray = self.focal_length * camera_points[:2] / camera_points[2] 

        # get the distorted pinhole locations of the points
        image_locations = self.apply_distortion(pinhole_locations)

        # add the temperature based scaling
        image_locations *= self.get_temperature_scale(temperature)

        # get the pixel locations of the points, need to mess with transposes due to numpy broadcasting rules
        picture_locations = ((self.intrinsic_matrix[:, :2] @ image_locations).T + self.intrinsic_matrix[:, 2]).T

        return pinhole_locations, image_locations, picture_locations

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
            >>> model = OwenModel(kx=300, ky=400, px=500, py=500, focal_length=10, a1=1e-5, a2=1e-6, radial2=1e5,
            >>>                   misalignment=[[1e-12, -2e-14, 3e-10], [2e-15, 1e-13, 3e-10]],
            >>>                   estimation_parameters=['multiple misalignments'])
            >>> model.project_onto_image([1, 2, 12000])
            array([500.33680555, 500.89814814])
            >>> model.project_onto_image([[1, 2, 3, 4], [2, 5, 6, 7], [12000, 13000, 9000, 5000]], image=1)
            array([[500.33680556, 501.25352754, 506.55555555, 564.79999998],
                   [500.89814815, 504.17842513, 517.48148149, 651.20000003]])
            >>> model.project_onto_image([[1, 2, 3, 4], [2, 5, 6, 7], [12000, 13000, 9000, 5000]], temperature=-1)
            array([[500.33680252, 501.25351625, 506.55549654, 564.7994167 ],
                   [500.89814006, 504.1783875 , 517.48132409, 651.19863896]])

        :param points_in_camera_frame: a shape (3,) or shape (3, n) array of points to project
        :param image: The index of the image being projected onto (only applicable with multiple misalignments)
        :param temperature: The temperature to project the points at
        :return: A shape (2,) or shape (2, n) numpy array of image points (with units of pixels)
        """

        _, __, picture_locations = self.get_projections(points_in_camera_frame, image, temperature=temperature)

        return picture_locations

    def _compute_ddistortion_dgnomic(self, gnomic_location: ARRAY_LIKE, radius: float, radius2: float, radius3: float, # type: ignore
                                     radius4: float) -> np.ndarray:
        r"""
        Computes the partial derivative of the distortion with respect to a change in the gnomic location

        Mathematically this is given by:

        .. math::
            \frac{\partial\mathbf{x}_I'}{\partial\mathbf{x}_I} =&
            \left(\epsilon_2r^2+\epsilon_4r^4+\epsilon_5y_I+\epsilon_6x_I\right)\mathbf{I}_{2\times 2}+
            \left(\epsilon_1r+\epsilon_3r^3\right)
            \left[\begin{array}{cc}0 & -1 \\ 1 & 0 \end{array}\right]+\\
            &\left\{\left(2\epsilon_2r+4\epsilon_4r^3\right)
            \\ \mathbf{x}_I+\left(\epsilon_1+3\epsilon_3r^2\right)
            \left[\begin{array}{c} -y_I \\ x_I\end{array}\right]\right\}
            \frac{\mathbf{x}_I^T}{r} +
            \mathbf{x}_I\left[\begin{array}{cc} \epsilon_5 & \epsilon_6\end{array}\right]

        :param gnomic_location: The gnomic location of the point being considered as a shape (2,) numpy array
        :param radius: The radial distance from the optical axis in units of distance
        :param radius2: The radial distance squared from the optical axis in units of distance
        :param radius3: The radial distance cubed from the optical axis in units of distance
        :param radius4: The radial distance to the fourth power from the optical axis in units of distance
        :return: The partial derivative of the distortion with respect to a change in the gnomic location
        """
        if np.abs(radius) >= 1e-8:
            # make sure we have an array
            gnomic_location = np.array(gnomic_location)

            dr_dgnom = gnomic_location / radius

            vec_portion = (self.radial2 * radius2 + self.radial4 * radius4 + self.tangential_y * gnomic_location[1] +
                           self.tangential_x * gnomic_location[0]) * np.eye(2) + \
                          (self.pinwheel1 * radius + self.pinwheel2 * radius3) * np.array([[0, -1], [1, 0]])

            scal_portion = np.outer((2 * self.radial2 * radius + 4 * self.radial4 * radius3) * gnomic_location +
                                    (self.pinwheel1 + 3 * self.pinwheel2 * radius2) *
                                    np.array([-gnomic_location[1], gnomic_location[0]]),
                                    dr_dgnom) + np.outer(gnomic_location, [self.tangential_x, self.tangential_y])

            return vec_portion + scal_portion

        else:
            warnings.warn('small radius, derivative unstable, returning 0')
            return np.zeros((2, 2))

    @staticmethod
    def _compute_dpixel_dintrinsic(gnomic_location_distorted: F_ARRAY_LIKE) -> np.ndarray:
        r"""
        computes the partial derivative of the pixel location with respect to a change in one of the intrinsic matrix
        parameters given the gnomic location of the point we are computing the derivative for.

        Mathematically this is given by

        .. math::
            \frac{\partial\mathbf{x}_p}{\partial\mathbf{k}} = \left[
            \begin{array}{cccccc} x_i & y_i & 0 & 0 & 1 & 0 \\
            0 & 0 & x_i & y_i & 0 & 1 \end{array}\right]

        where :math:`\mathbf{k}=[k_x \quad k_{xy} \quad k_{yx} \quad k_y \quad p_x \quad p_y]` is a vector of the
        intrinsic camera parameters and all else is as defined before.

        :param gnomic_location_distorted: the gnomic location of the point to compute the derivative for
        :return: the partial derivative of the pixel location with respect to a change in one of the intrinsic matrix
                 parameters
        """

        # compute the partial derivative of the pixel location with respect to the pixel pitch
        dpix_dkx = [gnomic_location_distorted[0], 0]
        dpix_dky = [0, gnomic_location_distorted[1]]

        # compute the partial derivative of the pixel location with respect to the non-rectangular pixel terms
        dpix_dkxy = [gnomic_location_distorted[1], 0]
        dpix_dkyx = [0, gnomic_location_distorted[0]]

        # compute the partial derivative of the pixel location with respect to the principal point
        dpix_dpx = [1, 0]
        dpix_dpy = [0, 1]

        # compute the partial derivative of the pixel location with respect to the intrinsic matrix
        return np.array([dpix_dkx, dpix_dkxy, dpix_dkyx, dpix_dky, dpix_dpx, dpix_dpy]).T

    @staticmethod
    def _compute_ddistorted_gnomic_ddistortion(gnomic_location: ARRAY_LIKE, radius: float, radius2: float,
                                               radius3: float, radius4: float) -> np.ndarray:
        r"""
        Computes the partial derivative of the distorted gnomic location with respect to a change in the distortion
        coefficients

        Mathematically this is given by

        .. math::
            \frac{\partial\mathbf{x}_I'}{\partial\mathbf{d}} = \left[
            \begin{array}{cccccc} r^2\mathbf{x}_I & r^4\mathbf{x}_I & y_I\mathbf{x}_I & x_I\mathbf{x}_I &
            r\mathbf{x}_{Ii} & r^3\mathbf{x}_{Ii} \end{array}\right]

        where :math:`\mathbf{d}=[\epsilon_2 \quad \epsilon_4 \quad \epsilon_5 \quad
        \epsilon_6 \quad \epsilon_1 \quad \epsilon_3]` is a vector of the
        intrinsic camera parameters, :math:`\mathbf{x}_{Ii}=[-y_I \quad x_I]^T`, and all else is as defined before.

        :param gnomic_location: The undistorted gnomic location of the point
        :type gnomic_location: np.ndarray
        :param radius: The radial distance from the optical axis in units of distance
        :type radius: float
        :param radius2: The radial distance squared from the optical axis in units of distance
        :type radius2: float
        :param radius3: The radial distance cubed from the optical axis in units of distance
        :type radius3: float
        :param radius4: The radial distance to the fourth power from the optical axis in units of distance
        :type radius4: float
        :return: the partial derivative of the distorted gnomic location with respect to a change in the distortion
                 coefficients
        :rtype: np.ndarray
        """

        gnomic_location = np.array(gnomic_location)

        # the partial derivative of the distorted gnomic location with respect to the radial distortion coefficients
        ddist_gnom_dradial2 = radius2 * gnomic_location
        ddist_gnom_dradial4 = radius4 * gnomic_location

        # the partial derivative of the distorted gnomic location with respect to the tangential distortion coefficients
        ddist_gnom_dtangential_y = gnomic_location[1] * gnomic_location
        ddist_gnom_dtangential_x = gnomic_location[0] * gnomic_location

        # compute the partial derivative of the distorted gnomic location with respect to the pinwheel distortion
        ddist_gnom_dpinwheel1 = radius * np.array([-gnomic_location[1], gnomic_location[0]])
        ddist_gnom_dpinwheel2 = radius3 * np.array([-gnomic_location[1], gnomic_location[0]])

        # compute the partial derivative of the distorted gnomic location with respect to the distortion coefficients
        return np.array([ddist_gnom_dradial2, ddist_gnom_dradial4,
                         ddist_gnom_dtangential_y, ddist_gnom_dtangential_x,
                         ddist_gnom_dpinwheel1, ddist_gnom_dpinwheel2]).T

    def _get_jacobian_row(self, unit_vector_camera: ARRAY_LIKE, image: int, num_images: int,
                          temperature: float = 0) -> np.ndarray:
        r"""
        Calculates the Jacobian matrix for a single point.

        The Jacobian is calculated for every possible parameter that could be included in the state vector in this
        method, and then columns corresponding to the state vectors that the Jacobian is not needed for can be removed
        using the :meth:`_remove_jacobian_columns` method.

        In general you should use the :meth:`compute_jacobian method in place of this method.

        This method computes the following:

        .. math::
            \frac{\partial\mathbf{x}_P}{\partial\mathbf{c}} = \left[\begin{array}{cccc}
            \frac{\partial\mathbf{x}_P}{\partial f} & \frac{\partial\mathbf{x}_P}{\mathbf{k}} &
            \frac{\partial\mathbf{x}_P}{\partial\mathbf{d}} &
            \frac{\partial\mathbf{x}_P}{\partial\mathbf{a}} &
            \frac{\partial\mathbf{x}_P}{\partial\boldsymbol{\delta\theta}}\end{array}\right]

        where all is as defined previously through the use of the chain rule.

        :param unit_vector_camera: The unit vector we are computing the Jacobian for
        :param image: The number of the image we are computing the Jacobian for
        :param num_images:   The total number of images included in our Jacobian matrix
        :param temperature: The temperature to compute the Jacobian at
        :return:
        """

        # ensure the input is an array and the right shape
        unit_vector_camera = np.asanyarray(unit_vector_camera).reshape(3)

        # get the required projections for the point
        gnomic_location, gnomic_location_distorted, pixel_location = self.get_projections(unit_vector_camera,
                                                                                          image=image,
                                                                                          temperature=temperature)

        # get the camera point after misalignment and shift from principle frame is applied
        if self.estimate_multiple_misalignments:
            if np.any(self.misalignment[image]):  # optimization to avoid matrix multiplication
                cam_point = rotvec_to_rotmat(self.misalignment[image]).squeeze() @ unit_vector_camera

            else:
                cam_point = unit_vector_camera

        else:
            if np.any(self.misalignment):  # optimization to avoid matrix multiplication
                cam_point = rotvec_to_rotmat(self.misalignment).squeeze() @ unit_vector_camera

            else:
                cam_point = unit_vector_camera

        # compute the radial distance from the optical axis as well as its powers
        radius = float(np.linalg.norm(gnomic_location))
        radius2 = radius ** 2
        radius3 = radius * radius2
        radius4 = radius2 ** 2

        # --------------------------------------------------------------------------------------------------------------
        # get the partial derivative of the measurement with respect to the misalignment vector
        # --------------------------------------------------------------------------------------------------------------

        # get the partial derivative of the distorted gnomic location with respect to the gnomic location
        ddist_gnom_dgnom = np.eye(2) + self._compute_ddistortion_dgnomic(gnomic_location,
                                                                         radius, radius2, radius3, radius4)

        # get the partial derivative of the pixel location of the point with respect to the distorted gnomic location
        dpix_ddist_gnom = self._compute_dpixel_ddistorted_gnomic(temperature=temperature)

        # compute the partial derivative of the camera location with respect to a change in the misalignment vector
        dcam_point_dmisalignment = self._compute_dcamera_point_dmisalignment(unit_vector_camera)

        # compute the partial derivative of the gnomic location with respect to the point in the camera frame
        dgnom_dcam_point = self._compute_dgnomic_dcamera_point(cam_point)

        # compute the partial derivative of the pixel location with respect to the misalignment
        dpix_dmisalignment = dpix_ddist_gnom @ ddist_gnom_dgnom @ dgnom_dcam_point @ dcam_point_dmisalignment

        # --------------------------------------------------------------------------------------------------------------
        # get the partial derivative of the measurement with respect to the focal length
        # --------------------------------------------------------------------------------------------------------------

        # compute the change in the gnomic location with respect to a change in the focal length
        dgnom_dfocal = self._compute_dgnomic_dfocal_length(cam_point)

        # compute the change in the pixel location with respect to the focal length
        dpix_dfocal = dpix_ddist_gnom @ ddist_gnom_dgnom @ dgnom_dfocal

        # --------------------------------------------------------------------------------------------------------------
        # get the partial derivative of the measurement with respect to the terms of the intrinsic matrix
        # --------------------------------------------------------------------------------------------------------------

        dpix_dintrinsic = self._compute_dpixel_dintrinsic(gnomic_location_distorted)

        # --------------------------------------------------------------------------------------------------------------
        # get the partial derivative of the measurement with respect to the distortion coefficients
        # --------------------------------------------------------------------------------------------------------------

        # compute the partial derivative of the distorted gnomic location with respect to the distortion coefficients
        ddist_gnom_ddist = self._compute_ddistorted_gnomic_ddistortion(gnomic_location,
                                                                       radius, radius2, radius3, radius4)

        # compute the partial derivative of the pixel location with respect to the distortion coefficients
        dpix_ddist = dpix_ddist_gnom @ ddist_gnom_ddist

        # --------------------------------------------------------------------------------------------------------------
        # get the partial derivative of the measurement with respect to the temperature coefficients
        # --------------------------------------------------------------------------------------------------------------

        dpix_dtemperature = self._compute_dpixel_dtemperature_coeffs(gnomic_location_distorted, temperature=temperature)

        # stack everything together.
        if self.estimate_multiple_misalignments:
            jacobian_row = np.hstack([dpix_dfocal.reshape(2, 1), dpix_dintrinsic, dpix_ddist, dpix_dtemperature,
                                      np.zeros((2, image * 3)),
                                      dpix_dmisalignment, np.zeros((2, (num_images - image - 1) * 3))])

        else:
            jacobian_row = np.hstack([dpix_dfocal.reshape(2, 1), dpix_dintrinsic, dpix_ddist, dpix_dtemperature,
                                      dpix_dmisalignment])

        return jacobian_row

    def apply_update(self, update_vec: F_ARRAY_LIKE):
        r"""
        This method takes in a delta update to camera parameters (:math:`\Delta\mathbf{c}`) and applies the update
        to the current instance in place.

        In general the delta update is calculated in the estimators in the :mod:`.calibration` subpackage and this
        method is not used by the user.

        The order of the update vector is determined by the order of the elements in :attr:`estimation_parameters`.  Any
        misalignment terms must always be last.

        This method operates by walking through the elements in :attr:`estimation_parameters` and retrieving the
        parameter that the element corresponds to.  The value of the update_vec at the index of the parameter is then
        applied as an additive update to the parameter in self, with the exception of misalignment, which is applied
        as a multiplicative update.

        :param update_vec: An iterable of delta updates to the model parameters
        :type update_vec: Sequence
        """

        jacobian_parameters = np.hstack([getattr(self.element_dict[element], 'start', self.element_dict[element])
                                         for element in self.estimation_parameters])

        update_vec = self._fix_update_vector(np.asanyarray(update_vec, dtype=np.float64), jacobian_parameters)

        update_vec = np.asarray(update_vec).flatten()

        for ind, parameter in enumerate(jacobian_parameters):

            if parameter == 0:
                self.focal_length += update_vec.item(ind)

            elif parameter == 1:
                self.kx += update_vec.item(ind)

            elif parameter == 2:
                self.kxy += update_vec.item(ind)

            elif parameter == 3:
                self.kyx += update_vec.item(ind)

            elif parameter == 4:
                self.ky += update_vec.item(ind)

            elif parameter == 5:
                self.px += update_vec.item(ind)

            elif parameter == 6:
                self.py += update_vec.item(ind)

            elif parameter == 7:
                self.distortion_coefficients += [update_vec.item(ind), 0, 0, 0, 0, 0]

            elif parameter == 8:
                self.distortion_coefficients += [0, update_vec.item(ind), 0, 0, 0, 0]

            elif parameter == 9:
                self.distortion_coefficients += [0, 0, update_vec.item(ind), 0, 0, 0]

            elif parameter == 10:
                self.distortion_coefficients += [0, 0, 0, update_vec.item(ind), 0, 0]

            elif parameter == 11:
                self.distortion_coefficients += [0, 0, 0, 0, update_vec.item(ind), 0]

            elif parameter == 12:
                self.distortion_coefficients += [0, 0, 0, 0, 0, update_vec.item(ind)]

            elif parameter == 13:
                self.a1 += update_vec.item(ind)

            elif parameter == 14:
                self.a2 += update_vec.item(ind)

            elif parameter == 15:
                self.a3 += update_vec.item(ind)

            elif parameter == 16 or parameter == '16:':

                misalignment_updates = update_vec[ind:].reshape(3, -1, order='F')

                if self.estimate_multiple_misalignments:
                    self.misalignment = [(Rotation(update.T) * Rotation(self.misalignment[ind])).vector
                                         for ind, update in enumerate(misalignment_updates.T)]

                else:
                    self.misalignment = (
                            Rotation(misalignment_updates) * Rotation(self.misalignment)
                    ).vector

                break
