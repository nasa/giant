r"""
This module provides a subclass of :class:`.CameraModel` that implements the Brown camera model, which adds basic
distortion corrections to the Pinhole model.

Theory
------

The Brown camera model is the pinhole camera model combined with a lens distortion model which projects any point
along a ray emanating from the camera center (origin of the camera frame) to the same 2D point in an image.  Given
some 3D point (or direction) expressed in the camera frame, :math:`\mathbf{x}_C`, the Brown model is defined as

.. math::
    &\mathbf{x}_I = \frac{1}{z_C}\left[\begin{array}{c} x_C \\ y_C \end{array}\right] \\
    &r = \sqrt{x_I^2 + y_I^2} \\
    &\Delta\mathbf{x}_I = (k_1r^2+k_2r^4+k_3r^6)\mathbf{x}_I +
    \left[\begin{array}{c} 2p_1x_Iy_I+p_2(r^2+2x_I^2) \\ p_1(r^2+2y_I^2) + 2p_2x_Iy_I \end{array}\right] \\
    &\mathbf{x}_P = \left[\begin{array}{ccc} f_x & \alpha & p_x \\ 0 & f_y & p_y\end{array}\right]
    \left[\begin{array}{c} (1+a_1T+a_2T^2+a_3T^3)(\mathbf{x}_I+\Delta\mathbf{x}_I) \\ 1 \end{array}\right]

where :math:`\mathbf{x}_I` are the image frame coordinates for the point (pinhole location), :math:`r` is the
radial distance from the principal point of the camera to the gnomic location of the point, :math:`k_{1-3}`
are radial distortion coefficients, :math:`p_{1-2}` are tip/tilt/prism distortion coefficients,
:math:`\Delta\mathbf{x}_I` is the distortion for point :math:`\mathbf{x}_I`, :math:`f_x` and :math:`f_y` are the
focal length divided by the pixel pitch in the :math:`x` and :math:`y` directions respectively expressed in units
of pixels, :math:`\alpha` is an alpha term for non-rectangular pixels, :math:`p_x` and :math:`p_y` are the location
of the principal point of the camera in the image expressed in units of pixels, :math:`T` is the temperature of the
camera, :math:`a_{1-3}` are temperature dependence coefficients, and :math:`\mathbf{x}_P` is the pixel location of
the point in the image. For a more thorough description of the Brown camera model checkout
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
sufficient for nearly every use case.  The :class:`.BrownModel` and its subclasses make precomputing the
transformation, and using the precomputed transformation, as easy as calling :meth:`~BrownModel.prepare_interp`
once.  Future calls to any method that then needs the transformation from pixels to gnomic locations (on the way to
unit vectors) will then use the precomputed transformation unless specifically requested otherwise.  In addition,
once the :meth:`~BrownModel.prepare_interp` method has been called, if the resulting camera object is then saved to a
file either using the :mod:`.camera_model`
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
and a detector size of 1024x1024, with radial distortion of 1e-5, -2e-7, 3e-9.  We could then create a model for this
camera as

    >>> from giant.camera_models import BrownModel
    >>> model = BrownModel(fx=10/2.2e-3, fy=10/2.2e-3,
    ...                    n_rows=1024, n_cols=1024, px=(1024-1)/2, py=(1024-1)/2,
    ...                    k1=1e-2, k2=-2e-3, k3=3e-4)

Note that we did not set the field of view, but it is automatically computed for us based off of the prescribed camera
model.

    >>> model.field_of_view
    9.048754127234737

In addition, we can now use our model to project points

    >>> model.project_onto_image([0, 0, 1])
    array([511.5, 511.5])

or to determine the unit vector through a pixel

    >>> model.pixels_to_unit([[0, 500], [0, 100]])
    array([[-0.11110425, -0.00251948],
           [-0.11110425, -0.09015368],
           [ 0.9875787 ,  0.99592468]]
"""

from typing import Sequence, Union, Tuple

import numpy as np
from numpy.typing import NDArray

from giant.camera_models.pinhole_model import PinholeModel, MISALIGNMENT_TYPE
from giant.rotations import rotvec_to_rotmat, Rotation
from giant._typing import NONENUM, NONEARRAY, ARRAY_LIKE


class BrownModel(PinholeModel):
    r"""
    This class provides an implementation of the Brown camera model for projecting 3D points onto images and performing
    camera calibration.

    The :class:`BrownModel` class is a subclass of :class:`.PinholeModel`.  This means that it includes implementations
    for all of the abstract methods defined in the :class:`CameraModel` class.  This also means that it can be used
    throughout GIANT as the primary camera model, including within the :mod:`calibration` subpackage.  If this class is
    going to be used with the :mod:`calibration` subpackage, the user can set which parameters are estimated and which
    are held fixed by using the ``estimation_parameters`` keyword argument when creating an instance of the class or by
    adjusting the :attr:`~BrownModel.estimation_parameters` attribute on an instance of the class.  The
    ``estimation_parameters`` input/attribute is a string or list of strings specifying which parameters to estimate.
    This means that :attr:`~BrownModel.estimation_parameters` could be something like ``'basic'`` which would indicate
    to estimate just the usual parameters, or something like ``['fx', 'fy', 'px', 'py']`` to estimate just the terms
    included in the list.

    In addition to the standard set of methods for a :class:`CameraModel` subclass, the :class:`BrownModel` class
    provides the following additional methods which may or may not be useful to some people:

    =================================  =================================================================================
    Method                             Use
    =================================  =================================================================================
    :meth:`get_projections`            computes the pinhole, image frame, and pixel locations of a 3D point
    :meth:`pixels_to_gnomic`           removes distortion from a point to get the corresponding unitless pinhole
                                       location
    =================================  =================================================================================

    The :class:`BrownModel` class also provides the following properties for easy getting/setting:

    ================================ ===================================================================================
    Property                         Description
    ================================ ===================================================================================
    :attr:`field_of_view`            the diagonal field of view of the camera in units of degrees
    :attr:`kx`, :attr:`fx`           :math:`f_x`, focal length divided by the pixel pitch in the x direction in units of
                                     pixels
    :attr:`ky`, :attr:`fy`           :math:`f_y`, focal length divided by the pixel pitch in the y direction in units of
                                     pixels
    :attr:`kxy`, :attr:`alpha`       :math:`\alpha`, A alpha term for non-rectangular pixels
    :attr:`px`                       :math:`p_{x}`, the x axis pixel location of the principal point of the camera in
                                     units of pixels
    :attr:`py`                       :math:`p_{y}`, the y axis pixel location of the principal point of the camera in
                                     units of pixels
    :attr:`radial2`, :attr:`k1`      :math:`k_1`, the radial distortion coefficient corresponding to :math:`r^2`
    :attr:`radial4`, :attr:`k2`      :math:`k_2`, the radial distortion coefficient corresponding to :math:`r^4`
    :attr:`radial6`, :attr:`k3`      :math:`k_3`, the radial distortion coefficient corresponding to :math:`r^6`
    :attr:`tiptilt_y`, :attr:`p1`    :math:`p_1`, the tip/tilt/prism distortion coefficient corresponding to
                                     :math:`y_I`
    :attr:`tiptilt_x`, :attr:`p2`    :math:`p_2`, the tip/tilt/prism distortion coefficient corresponding to
                                     :math:`x_I`
    :attr:`a1`                       :math:`a_1`, the linear coefficient for focal length dependent focal length
    :attr:`a2`                       :math:`a_2`, the quadratic coefficient for focal length dependent focal length
    :attr:`a3`                       :math:`a_3`, the cubic coefficient for focal length dependent focal length
    :attr:`intrinsic_matrix_inv`     The inverse of the intrinsic matrix
    ================================ ===================================================================================

    .. note:: The distortion attributes are aliases over each other and refer to the same data.  Therefore setting a
              value to :attr:`radial2` would also change the value of :attr:`k1`

    """

    def __init__(self, intrinsic_matrix: NONEARRAY = None, fx: NONENUM = None, fy: NONENUM = None, px: NONENUM = None,
                 py: NONENUM = None, alpha: NONENUM = None, kx: NONENUM = None, ky: NONENUM = None, kxy: NONENUM = None,
                 field_of_view: NONENUM = None, use_a_priori: bool = False,
                 distortion_coefficients: NONEARRAY = None,
                 k1: NONENUM = None, k2: NONENUM = None, k3: NONENUM = None, p1: NONENUM = None, p2: NONENUM = None,
                 radial2: NONENUM = None, radial4: NONENUM = None, radial6: NONENUM = None,
                 tiptilt_y: NONENUM = None, tiptilt_x: NONENUM = None,
                 temperature_coefficients: NONEARRAY = None, a1: NONENUM = None, a2: NONENUM = None, a3: NONENUM = None,
                 misalignment: MISALIGNMENT_TYPE = None,
                 estimation_parameters: Union[str, Sequence] = 'basic', n_rows: int = 1, n_cols: int = 1):
        """
        :param intrinsic_matrix: the intrinsic matrix for the camera as a numpy shape (2, 3) array.  Note that this is
                                 overwritten if ``kx``, ``ky``, ``kxy``, ``kyx``, ``px``, ``py``, ``fx``, ``fy``,
                                 ``alpha`` are also specified.
        :param field_of_view: The field of view of the camera in units of degrees.
        :param use_a_priori: A flag to indicate whether to include the a priori state vector in the Jacobian matrix when
                            performing a calibration
        :param distortion_coefficients: A numpy array of shape (6,) containing the six distortion coefficients in order.
                                        Note that this array is overwritten with any distortion coefficients that are
                                        specified independently.
        :param fx: The pixel pitch along the x axis in units of pixels
        :param fy: The pixel pitch along the y axis in units of pixels
        :param kx: The pixel pitch along the x axis in units of pixels
        :param ky: The pixel pitch along the y axis in units of pixels
        :param kxy: A alpha term for non-rectangular pixels
        :param alpha: An alpha term for non-rectangular pixels
        :param px: the x component of the pixel location of the principal point in the image in units of pixels
        :param py: the y component of the pixel location of the principal point in the image in units of pixels
        :param radial2: the radial distortion coefficient corresponding to the r**2 term
        :param radial4: the radial distortion coefficient corresponding to the r**4 term
        :param radial6: the radial distortion coefficient corresponding to the r**4 term
        :param k1: the radial distortion coefficient corresponding to the r**2 term
        :param k2: the radial distortion coefficient corresponding to the r**4 term
        :param k3: the radial distortion coefficient corresponding to the r**4 term
        :param tiptilt_y: the tip/tilt/decentering distortion coefficient corresponding to the y term
        :param tiptilt_x: the tip/tilt/decentering distortion coefficient corresponding to the x term
        :param p1: the tip/tilt/decentering distortion coefficient corresponding to the y term
        :param p2: the tip/tilt/decentering distortion coefficient corresponding to the x term
        :param temperature_coefficients: The temperature polynomial coefficients as a length 3 Sequence
        :param a1: the linear coefficient of the focal length temperature dependence
        :param a2: the quadratic coefficient of the focal length temperature dependence
        :param a3: the cubic coefficient of the focal length temperature dependence
        :param misalignment: either a numpy array of shape (3,) or a list of numpy arrays of shape(3,) with each array
                             corresponding to a single image (the list of numpy arrays is only valid when estimating
                             multiple misalignments)
        :param estimation_parameters: A string or list of strings specifying which model parameters to include in the
                                      calibration
        :param n_rows: the number of rows of the active image array
        :param n_cols: the number of columns in the active image array
        """

        # call the super init
        super().__init__(intrinsic_matrix=intrinsic_matrix, kx=kx, ky=ky, px=px, py=py,
                         focal_length=1, misalignment=misalignment, use_a_priori=use_a_priori,
                         a1=a1, a2=a2, a3=a3, n_rows=n_rows, n_cols=n_cols,
                         temperature_coefficients=temperature_coefficients)

        # finish the intrinsic matrix
        if kxy is not None:
            self.kxy = kxy

        if fx is not None:
            self.fx = fx
        if fy is not None:
            self.fy = fy
        if alpha is not None:
            self.alpha = alpha

        # set the distortion coefficients vector
        self.distortion_coefficients = np.zeros(5)
        """
        The distortion coefficients array contains the distortion coefficients for the Brown model [k1, k2, k3, p1, p2]
        """

        if distortion_coefficients is not None:
            self.distortion_coefficients = distortion_coefficients

        if k1 is not None:
            self.k1 = k1
        if k2 is not None:
            self.k2 = k2
        if k3 is not None:
            self.k3 = k3
        if p1 is not None:
            self.p1 = p1
        if p2 is not None:
            self.p2 = p2

        if radial2 is not None:
            self.radial2 = radial2
        if radial4 is not None:
            self.radial4 = radial4
        if radial6 is not None:
            self.radial6 = radial6
        if tiptilt_y is not None:
            self.tiptilt_y = tiptilt_y
        if tiptilt_x is not None:
            self.tiptilt_x = tiptilt_x

        self._state_labels = ['fx', 'fy', 'alpha', 'px', 'py', 'k1', 'k2', 'k3', 'p1', 'p2', 'a1', 'a2', 'a3',
                              'misalignment']
        """
        A list of state labels that correspond to the attributes of this class.
        """

        # store the element dict for indexing the update vector and Jacobian matrix
        self.element_dict = {
            'basic': [0, 1, 2, 5, 6, 7, 8, 9, 13, 14, 15],
            'intrinsic': np.arange(0, 10),
            'basic intrinsic': [0, 1, 2, 5, 6, 7, 8, 9],
            'temperature dependence': [10, 11, 12],
            'fx': [0],
            'kx': [0],
            'fy': [1],
            'ky': [1],
            'alpha': [2],
            'kxy': [2],
            'px': [3],
            'py': [4],
            'k1': [5],
            'k2': [6],
            'k3': [7],
            'p1': [8],
            'p2': [9],
            'radial2': [5],
            'radial4': [6],
            'radial6': [7],
            'tiptilt_y': [8],
            'tiptilt_x': [9],
            'a1': [10],
            'a2': [11],
            'a3': [12],
            'single misalignment': slice(13, None, None),
            'multiple misalignments': slice(13, None, None)
        }

        self.estimation_parameters = estimation_parameters

        # add the distortion parameters to the important variables
        self.important_attributes = self.important_attributes + \
            ['k1', 'k2', 'k3', 'p1', 'p2', 'alpha']
        """
        A list specifying the important attributes the must be saved/loaded for this camera model to be completely 
        reconstructed. 
        """

        self.field_of_view = field_of_view

    def __repr__(self) -> str:
        """
        Return a representative string of this object
        """

        template = "BrownModel(fx={fx}, fy={fy}, px={px}, py={py}, alpha={skew},\n" \
                   "           field_of_view={fov}, k1={k1}, k2={k2}, k3={k3}, p1={p1}, p2={p2},\n" \
                   "           misalignment={mis!r}, a1={a1}, a2={a2}, a3={a3}, n_rows={nr}, n_cols={nc},\n" \
                   "           estimation_parameters={ep!r}, use_a_priori={ap})\n\n"

        return template.format(
            fx=self.fx, fy=self.fy, px=self.px, py=self.py, skew=self.alpha,
            fov=self.field_of_view, k1=self.k1, k2=self.k2, k3=self.k3, p1=self.p1, p2=self.p2, mis=self.misalignment,
            ep=self.estimation_parameters, ap=self.use_a_priori, a1=self.a1, a2=self.a2, a3=self.a3, nr=self.n_rows,
            nc=self.n_cols
        )

    def __str__(self):

        template = u"Brown Camera Model:\n\n" \
                   u" __  __     __     __    \n" \
                   u"|   x  | _ |  Xc/Zc  |   \n" \
                   u"|   y  | - |  Yc/Zc  |   \n" \
                   u" --  --     --     --    \n" \
                   u"      _________ \n" \
                   u"     /  2   2   \n" \
                   u"r =\\/  x + y    \n" \
                   u" __ __                               __ __     __               2    2 __   \n" \
                   u"|  x' |            2      4      6  |  x  |   |  2p1*x*y + p2*(r + 2x )  |  \n" \
                   u"|     | = (1 + k1*r + k2*r + k3*r ) |     | + |      2    2              |  \n" \
                   u"|  y' |                             |  y  | + |  p1(r + 2y ) + 2p2*x*y   |  \n" \
                   u" -- --                               -- --     --                      --   \n" \
                   u" __  __                               __ __   \n" \
                   u"|  xt' |                   2      3  |  x' |  \n" \
                   u"|      | = (1 + a1*T + a2*T + a3*T ) |     |  \n" \
                   u"|  yt' |                             |  y' |  \n" \
                   u" --  --                               -- --   \n" \
                   u" __ __     __             __  __  __  \n" \
                   u"|  u  | _ |  fx  alpha  px  ||  xt' | \n" \
                   u"|  v  | - |  0    fy    py  ||  yt' | \n" \
                   u" -- --     --             -- |  1   | \n" \
                   u"                              --  --  \n\n" \
                   u"————————————————————————————————————————————————————————————————————————————\n\n" \
                   u"distortion coefficients:\n" \
                   u"    k1={k1}, k2={k2}, k3={k3}, p1={p1}, p2={p2}\n\n" \
                   u"camera parameters:\n" \
                   u"    fx={fx}, fy={fy}, alpha={skew}, px={px}, py={py}\n\n" \
                   u"temperature coefficients:\n" \
                   u"    a1={a1}, a2={a2}, a3={a3}\n\n"

        return template.format(
            fx=self.fx, fy=self.fy, px=self.px, py=self.py, skew=self.alpha,
            k1=self.k1, k2=self.k2, k3=self.k3, p1=self.p1, p2=self.p2, a1=self.a1, a2=self.a2, a3=self.a3
        )

    @property
    def fx(self) -> float:
        """
        The focal length in units of pixels along the x axis (focal length divided by x axis pixel pitch)

        This is an alias to :attr:`kx` and points to the (0, 0) index of the intrinsic matrix
        """

        return self.intrinsic_matrix[0, 0]

    @fx.setter
    def fx(self, val):
        self.intrinsic_matrix[0, 0] = val

    @property
    def fy(self) -> float:
        """
        The focal length in units of pixels along the y axis (focal length divided by y axis pixel pitch)

        This is an alias to :attr:`ky` and points to the (1, 1) index of the intrinsic matrix
        """
        return self.intrinsic_matrix[1, 1]

    @fy.setter
    def fy(self, val):
        self.intrinsic_matrix[1, 1] = val

    @property
    def kxy(self) -> float:
        """
        An alpha term for non-rectangular pixels.

        This corresponds to the [0, 1] component of the intrinsic matrix
        """
        return self.intrinsic_matrix[0, 1]

    @kxy.setter
    def kxy(self, val):
        self.intrinsic_matrix[0, 1] = val

    @property
    def alpha(self) -> float:
        """
        An alpha term for non-rectangular pixels.

        This is an alias to :attr:`kxy` and points to the to the [0, 1] component of the intrinsic matrix
        """
        return self.intrinsic_matrix[0, 1]

    @alpha.setter
    def alpha(self, val):
        self.intrinsic_matrix[0, 1] = val

    @property
    def k1(self) -> float:
        """
        The radial distortion coefficient corresponding to the r**2 term

        This corresponds to the [0] index of the :attr:`.distortion_coefficients` array
        """
        return self.distortion_coefficients[0]

    @k1.setter
    def k1(self, val):
        self.distortion_coefficients[0] = val

    @property
    def k2(self) -> float:
        """
        The radial distortion coefficient corresponding to the r**4 term

        This corresponds to the [1] index of the :attr:`.distortion_coefficients` array
        """
        return self.distortion_coefficients[1]

    @k2.setter
    def k2(self, val):
        self.distortion_coefficients[1] = val

    @property
    def k3(self) -> float:
        """
        The radial distortion coefficient corresponding to the r**6 term

        This corresponds to the [2] index of the :attr:`.distortion_coefficients` array
        """
        return self.distortion_coefficients[2]

    @k3.setter
    def k3(self, val):
        self.distortion_coefficients[2] = val

    @property
    def p1(self) -> float:
        """
        The tip/tilt/decentering distortion coefficient corresponding to the y term.

        This corresponds to the [3] index of the :attr:`.distortion_coefficients` array.
        """
        return self.distortion_coefficients[3]

    @p1.setter
    def p1(self, val):
        self.distortion_coefficients[3] = val

    @property
    def p2(self) -> float:
        """
        The tip/tilt/decentering distortion coefficient corresponding to the x term.

        This corresponds to the [4] index of the :attr:`.distortion_coefficients` array.
        """
        return self.distortion_coefficients[4]

    @p2.setter
    def p2(self, val):
        self.distortion_coefficients[4] = val

    @property
    def radial2(self) -> float:
        """
        The radial distortion coefficient corresponding to the r**2 term

        This is an alias to the :attr:`k1` attribute and corresponds to the [0] index of the
        :attr:`.distortion_coefficients` array
        """
        return self.distortion_coefficients[0]

    @radial2.setter
    def radial2(self, val):
        self.distortion_coefficients[0] = val

    @property
    def radial4(self) -> float:
        """
        The radial distortion coefficient corresponding to the r**4 term

        This is an alias to the :attr:`k2` attribute and corresponds to the [1] index of the
        :attr:`.distortion_coefficients` array
        """
        return self.distortion_coefficients[1]

    @radial4.setter
    def radial4(self, val):
        self.distortion_coefficients[1] = val

    @property
    def radial6(self) -> float:
        """
        The radial distortion coefficient corresponding to the r**6 term


        This is an alias to the :attr:`k3` attribute and corresponds to the [2] index of the
        :attr:`.distortion_coefficients` array
        """
        return self.distortion_coefficients[2]

    @radial6.setter
    def radial6(self, val):
        self.distortion_coefficients[2] = val

    @property
    def tiptilt_y(self) -> float:
        """
        The tip/tilt/decentering distortion coefficient corresponding to the y term.

        This is an alias to the :attr:`p1` attribute and corresponds to the [3] index of the
        :attr:`.distortion_coefficients` array
        """
        return self.distortion_coefficients[3]

    @tiptilt_y.setter
    def tiptilt_y(self, val):
        self.distortion_coefficients[3] = val

    @property
    def tiptilt_x(self) -> float:
        """
        The tip/tilt/decentering distortion coefficient corresponding to the x term.

        This is an alias to the :attr:`p2` attribute and corresponds to the [3] index of the
        :attr:`.distortion_coefficients` array
        """
        return self.distortion_coefficients[4]

    @tiptilt_x.setter
    def tiptilt_x(self, val):
        self.distortion_coefficients[4] = val

    @PinholeModel.focal_length.setter
    def focal_length(self, val):

        if val != 1:

            raise AttributeError(
                'The focal length is constrained to be length 1 and is read only for this model')

        else:
            self._focal_length = 1

    @property
    def intrinsic_matrix_inv(self) -> np.ndarray:
        r"""
        The inverse of the intrinsic matrix.

        The inverse of the intrinsic matrix is used to convert from units of pixels with an origin at the upper left
        corner of the image to units of distance with an origin at the principal point of the image.

        the intrinsic matrix has an analytic inverse which is given by

        .. math::
            \mathbf{K}^{-1} = \left[\begin{array}{ccc} \frac{1}{f_x} & -\frac{\alpha}{f_xf_y} &
            \frac{\alpha p_y-f_yp_x}{f_xf_y} \\
            0 & \frac{1}{f_y} & \frac{-p_y}{f_y} \end{array}\right]

        To convert from units of pixels to a unitless, distorted gnomic location you would do::
            >>> from giant.camera_models import BrownModel
            >>> model = BrownModel(kx=5, ky=10, px=100, py=500)
            >>> ((model.intrinsic_matrix_inv[:, :2]@[[1, 2, 300], [4, 5, 600]]).T + model.intrinsic_matrix_inv[:, 2]).T
            array([[-19.8, -19.6, 40.]
                   [-49.6, -49.5, 10.]])

        .. note:: The above code will give you distorted gnomic location, while the :meth:`pixels_to_gnomic` will give
                  you undistorted gnomic locations (true pinhole points).

        .. note:: Since the intrinsic matrix is defined as a :math:`2\times 3` matrix this
                  isn't a formal inverse.  To get the true inverse you need to append a row of [0, 0, 1] to both the
                  intrinsic matrix and intrinsic matrix inverse.
        """

        # determinant of top left of intrinsic matrix
        tldet = self.kx * self.ky

        return np.array([[1 / self.kx, -self.kxy / tldet, (self.py * self.kxy - self.ky * self.px) / tldet],
                         [0, 1 / self.ky, -self.py / self.ky]])

    def apply_distortion(self, pinhole_locations: ARRAY_LIKE) -> np.ndarray:
        r"""
        This method applies the distortion model to the specified pinhole (gnomic) locations in the image frame.

        In general this function is not used by the user and the higher level :meth:`project_onto_image` is used
        which calls this method (along with a few others) instead.  In cases were it is desirable to use this method
        the pinhole locations should be input as a shape (2,) or shape (2, n) array of unitless image plane locations.
        The output from this function is the unitless distorted image plane locations.

        For the Brown model, the distortion is defined as

        .. math::
            \Delta\mathbf{x}_I = (k_1r^2+k_2r^4+k_3r^6)\mathbf{x}_I +
            \left[\begin{array}{c} 2p_1x_Iy_I+p_2(r^2+2x_I^2) \\ p_1(r^2+2y_I^2) + 2p_2x_Iy_I \end{array}\right]

        where :math:`\Delta\mathbf{x}_I` is the additive distortion, :math:`k_1` is the second order radial distortion
        coefficient, :math:`k_2` is the fourth order radial distortion coefficient, :math:`k_3` is the sixth order
        radial distortion coefficient, :math:`\mathbf{x}_I` is the unitless gnomic location, :math:`p_1` is the x
        tangential distortion coefficient, :math:`p_2` is the y tangential distortion coefficient, and
        :math:`r = \sqrt{\mathbf{x}_I^T\mathbf{x}_I}` is the radial distance from the optical axis.

        :param pinhole_locations: The unitless image plane location of points to be distorted as a shape (2,) or (2, n)
                                  array.
        :return: The unitless distorted locations of the points on the image plane as a shape (2,) or (2, n) array.
        """

        # ensure we are dealing with an array
        pinhole_locations = np.array(pinhole_locations)

        # compute the powers of the radial distance from the optical axis
        radius2 = (pinhole_locations * pinhole_locations).sum(axis=0)
        radius4 = radius2 * radius2
        radius6 = radius2 * radius4

        # create aliases for easier coding
        rows = pinhole_locations[1]
        cols = pinhole_locations[0]

        # compute the product of the x and y terms
        rows_cols = rows * cols

        # compute the radial distortion
        radial_distortion = (self.k1 * radius2 + self.k2 *
                             radius4 + self.k3 * radius6) * pinhole_locations

        # compute the tip/tilt/decentering distortion
        decentering_distortion = np.vstack([self.p1 * 2 * rows_cols + self.p2 * (radius2 + 2 * cols * cols),
                                            self.p1 * (radius2 + 2 * rows * rows) + self.p2 * 2 * rows_cols])

        if decentering_distortion.size == 2 and pinhole_locations.ndim == 1:
            decentering_distortion = decentering_distortion.ravel()

        # add the distortion to the pinhole locations
        return pinhole_locations + radial_distortion + decentering_distortion

    def get_projections(self, points_in_camera_frame: ARRAY_LIKE,
                        image: int = 0, temperature: float = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

            >>> from giant.camera_models import BrownModel
            >>> model = BrownModel(fx=3000, fy=4000, px=500, py=500, a1=1e-5, a2=1e-6,
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
        :return: A tuple of the pinhole, distorted pinhole, and pixel locations for a set of 3D points
                 expressed in the camera frame
        """

        # ensure the input is an array
        points_in_camera_frame = np.asarray(points_in_camera_frame)

        # apply misalignment to the points
        if self.estimate_multiple_misalignments:
            # optimization to avoid matrix multiplication
            if np.any(self.misalignment[image]):
                points_in_camera_frame = rotvec_to_rotmat(self.misalignment[image]).squeeze() @ \
                    points_in_camera_frame

        else:
            if np.any(self.misalignment):  # optimization to avoid matrix multiplication
                points_in_camera_frame = rotvec_to_rotmat(
                    self.misalignment).squeeze() @ points_in_camera_frame

        # get the unitless image plane location
        pinhole_locations = points_in_camera_frame[:2] / \
            points_in_camera_frame[2]

        # get the distorted image plane location
        image_locations = self.apply_distortion(pinhole_locations) 

        # add the temperature based scaling
        image_locations *= self.get_temperature_scale(temperature)

        # get the pixel locations of the points, need to mess with transposes due to numpy broadcasting rules
        picture_locations = (
            (self.intrinsic_matrix[:, :2] @ image_locations).T + self.intrinsic_matrix[:, 2]).T

        return pinhole_locations, image_locations, picture_locations

    def project_onto_image(self, points_in_camera_frame: ARRAY_LIKE, image: int = 0,
                           temperature: float = 0) -> np.ndarray:
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
            >>> from giant.camera_models import BrownModel
            >>> model = BrownModel(kx=3000, ky=4000, px=500, py=500, a1=1e-5, a2=1e-6,
            >>>                    misalignment=[[1e-12, -2e-14, 3e-10], [2e-15, 1e-13, 3e-10]],
            >>>                    estimation_parameters=['multiple misalignments'])
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
            points_in_camera_frame, image, temperature=temperature)

        return picture_locations

    def compute_pixel_jacobian(self, vectors_in_camera_frame: ARRAY_LIKE, image: int = 0, temperature: float = 0) \
            -> np.ndarray:
        r"""
        This method computes the Jacobian matrix :math:`\partial\mathbf{x}_P/\partial\mathbf{x}_C` where
        :math:`\mathbf{x}_C` is a vector in the camera frame that projects to :math:`\mathbf{x}_P` which is the
        pixel location.

        This method is used in the :class:`.LimbScanning` process in order to predict the change in a projected pixel
        location with respect to a change in the projected vector.  The :attr:`vectors_in_camera_frame` input should
        be a 3xn array of vectors which the Jacobian is to be computed for.

        :math:`\mathbf{x}_C'` is the camera frame point after applying the misalignment,
        :math:`\boldsymbol{\delta\theta}` is the misalignment vector, :math:`\mathbf{x}_C` is the camera frame point
        before misalignment is applied,
        :math:`\mathbf{x}_P` is the pixel location, :math:`\mathbf{x}_I` is the gnomic location,
        :math:`a_{1-3}` are the temperature coefficients, :math:`T` is the temperature, :math:`f_x` is the focal length
        in the x direction,  :math:`f_y` is focal length in the y direction, :math:`\alpha` is the skewness term,
        :math:`k_{1-3}` are the radial distortion terms, :math:`p_{1-2}` are the tangential distortion terms, and
        :math:`\mathbf{T}_{\boldsymbol{\delta\theta}}` is the rotation matrix
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
            radius2 = (gnomic_location * gnomic_location).sum(axis=0)
            radius4 = radius2 ** 2
            radius6 = radius4 * radius2

            # --------------------------------------------------------------------------------------------------------------
            # get the partial derivative of the measurement with respect to the input vector
            # --------------------------------------------------------------------------------------------------------------

            # get the partial derivative of the distorted gnomic location with respect to the gnomic location
            ddist_gnom_dgnom = np.eye(2) + self._compute_ddistortion_dgnomic(gnomic_location,
                                                                             radius2, radius4, radius6)

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
        distorted_gnomic_location = np.array(distorted_gnomic_location)

        # compute the radial distance from the principal point and its powers to give to the ddistortion/dgnomic method
        radius2 = (distorted_gnomic_location ** 2).sum(axis=0)
        radius4 = radius2 * radius2
        radius6 = radius4 * radius2

        # compute the derivative
        return np.eye(2) - self._compute_ddistortion_dgnomic(distorted_gnomic_location,
                                                             radius2, radius4, radius6)

    def _compute_ddistortion_dgnomic(self, gnomic: ARRAY_LIKE, # type: ignore
                                     radius2: float, radius4: float, radius6: float) -> np.ndarray:
        r"""
        Computes the partial derivative of the distorted gnomic location with respect to a change in the gnomic location

        Mathematically this is given by:

        .. math::
            \frac{\partial\mathbf{x}_I'}{\partial\mathbf{x}_I} = &\left(1 + k_1r^2+k_2r^4+k_3r^6\right)
            \mathbf{I}_{2\times 2} + \left[\begin{array}{cc}2p_1y_I+4p_2x_I & 2p_1x_I \\
            2p_2y_I & 4p_1y_I+2p_2x_I \end{array}\right] + \\
            & \left(2k_1+4k_2r^2+6k_3r^4\right)\mathbf{x}_I\mathbf{x}_I^T +
            2 \left[\begin{array}{cc} p_2x_I & p_2y_I \\ p_1x_I & p_1y_I \end{array}\right]

        where all is defined as before

        :param gnomic: The gnomic location of the point being considered as a shape (2,) numpy array
        :param radius2: The radial distance squared from the optical axis
        :param radius4: The radial distance to the fourth power from the optical axis
        :param radius6: The radial distance to the sixth power from the optical axis
        :return: The partial derivative of the distortion with respect to a change in the gnomic location
        """
        
        gnomic = np.asanyarray(gnomic)

        row = gnomic[1]
        col = gnomic[0]

        vector_part = ((self.k1 * radius2 + self.k2 * radius4 + self.k3 * radius6) * np.eye(2) +
                       np.array([[2 * self.p1 * row + 4 * self.p2 * col, 2 * self.p1 * col],
                                 [2 * self.p2 * row, 4 * self.p1 * row + 2 * self.p2 * col]]))

        scalar_part = ((2 * self.k1 + 4 * self.k2 * radius2 + 6 * self.k3 * radius4) * np.outer(gnomic, gnomic) +
                       2 * np.outer([self.p2, self.p1], gnomic))

        return vector_part + scalar_part

    def _compute_ddistorted_gnomic_dgnomic(self, gnomic: ARRAY_LIKE,
                                           radius2: float, radius4: float, radius6: float) -> np.ndarray:
        r"""
        Computes the partial derivative of the distorted gnomic location with respect to a change in the gnomic location

        Mathematically this is given by:

        .. math::
            \frac{\partial\mathbf{x}_I'}{\partial\mathbf{x}_I} = &\left(1 + k_1r^2+k_2r^4+k_3r^6\right)
            \mathbf{I}_{2\times 2} + \left[\begin{array}{cc}2p_1y_I+4p_2x_I & 2p_1x_I \\
            2p_2y_I & 4p_1y_I+2p_2x_I \end{array}\right] + \\
            & \left(2k_1+4k_2r^2+6k_3r^4\right)\mathbf{x}_I\mathbf{x}_I^T +
            2 \left[\begin{array}{cc} p_2x_I & p_2y_I \\ p_1x_I & p_1y_I \end{array}\right]

        where all is defined as before

        :param gnomic: The gnomic location of the point being considered as a shape (2,) numpy array
        :param radius2: The radial distance squared from the optical axis
        :param radius4: The radial distance to the fourth power from the optical axis
        :param radius6: The radial distance to the sixth power from the optical axis
        :return: The partial derivative of the distortion with respect to a change in the gnomic location
        """

        return self._compute_ddistortion_dgnomic(gnomic, radius2, radius4, radius6) + np.eye(2)

    @staticmethod
    def _compute_dpixel_dintrinsic(gnomic_location_distorted: Sequence[float] | NDArray) -> np.ndarray:
        r"""
        computes the partial derivative of the pixel location with respect to a change in one of the intrinsic matrix
        parameters given the gnomic location of the point we are computing the derivative for.

        Mathematically this is given by

        .. math::
            \frac{\partial\mathbf{x}_P}{\partial\mathbf{k}} = \left[\begin{array}{cccc} x_I & 0 & y_I & 1 & 0
            \\ 0 & y_I & 0 & 0 & 1 \end{array}\right]

        where :math:`\mathbf{k}=[f_x \quad f_y \quad \alpha \quad p_x \quad p_y]` is a vector of the intrinsic
        camera parameters and all else is as defined before.

        :param gnomic_location_distorted: the gnomic location of the point to compute the derivative for
        :return: the partial derivative of the pixel location with respect to a change in one of the intrinsic matrix
                 parameters
        """

        # compute the partial derivative of the pixel location with respect to the focal length / pixel pitch
        dpix_dfx = [gnomic_location_distorted[0], 0]
        dpix_dfy = [0, gnomic_location_distorted[1]]

        # compute the partial derivative of the pixel location with respect to the skewness term
        dpix_dskew = [gnomic_location_distorted[1], 0]

        # compute the partial derivative of the pixel location with respect to the principal point
        dpix_dpx = [1, 0]
        dpix_dpy = [0, 1]

        # compute the partial derivative of the pixel location with respect to the intrinsic matrix
        return np.array([dpix_dfx, dpix_dfy, dpix_dskew, dpix_dpx, dpix_dpy]).T

    @staticmethod
    def _compute_ddistorted_gnomic_ddistortion(gnomic_loc: ARRAY_LIKE,
                                               radius2: float, radius4: float, radius6: float) -> np.ndarray:
        r"""
        Computes the partial derivative of the distorted gnomic location with respect to a change in the distortion
        coefficients.

        Mathematically this is given by:

        .. math::
            \frac{\partial\mathbf{x}_I'}{\partial\mathbf{d}} = \left[\begin{array}{ccccc} r^2x_I &
            r^4x_I & r^6x_I & 2x_Iy_I & r^2+2y_I^2 \\
            r^2y_I & r^4y_I & r_6y_I & r^2+2x_I^2 & 2x_Iy_I \end{array}\right]

        where :math:`\mathbf{d}=[k_1 \quad k_2 \quad k_3 \quad p_1 \quad p_2]` is a vector of the distortion
        coefficients and all else is as defined before.

        :param gnomic_loc: The undistorted gnomic location of the point
        :param radius2: The radial distance squared from the optical axis
        :param radius4: The radial distance to the fourth power from the optical axis
        :param radius6: The radial distance to the sixth power from the optical axis
        :return: the partial derivative of the distorted gnomic location with respect to a change in the distortion
                 coefficients
        """

        gnomic_loc = np.array(gnomic_loc)

        # compute the partial derivative of the radial terms
        ddist_gnom_dr2 = radius2 * gnomic_loc
        ddist_gnom_dr4 = radius4 * gnomic_loc
        ddist_gnom_dr6 = radius6 * gnomic_loc

        # compute the partial derivative of the tip/tilt/prism terms
        ddist_gnom_dp1 = [2 * gnomic_loc[0] *
                          gnomic_loc[1], radius2 + 2 * gnomic_loc[1] ** 2]
        ddist_gnom_dp2 = [radius2 + 2 * gnomic_loc[0]
                          ** 2, 2 * gnomic_loc[0] * gnomic_loc[1]]

        # compute the partial derivative of the distorted gnomic location with respect to the distortion coefficients
        return np.array([ddist_gnom_dr2, ddist_gnom_dr4, ddist_gnom_dr6, ddist_gnom_dp1, ddist_gnom_dp2]).T

    def _get_jacobian_row(self, unit_vector_camera: ARRAY_LIKE, image: int, num_images: int,
                          temperature: float = 0) -> np.ndarray:
        r"""
        Calculates the Jacobian matrix for a single point.

        The Jacobian is calculated for every possible parameter that could be included in the state vector in this
        method, and then columns corresponding to the state vectors that the Jacobian is not needed for can be removed
        using the :meth:`_remove_jacobian_columns` method.

        In general you should use the :meth:`.compute_jacobian method in place of this method.

        This method computes the following:

        .. math::
            \frac{\partial\mathbf{x}_P}{\partial\mathbf{c}} = \left[\begin{array}{cccc}
            \frac{\partial\mathbf{x}_P}{\partial\mathbf{k}} &
            \frac{\partial\mathbf{x}_P}{\partial\mathbf{d}} &
            \frac{\partial\mathbf{x}_P}{\partial\mathbf{a}} &
            \frac{\partial\mathbf{x}_P}{\partial\boldsymbol{\delta\theta}}\end{array}\right]

        The partial derivatives above are defined through the use of the chain rule, resulting in:

        .. math::
            :nowrap:

            \begin{gather}
            \frac{\partial\mathbf{x}_P}{\partial\mathbf{d}} =
            \frac{\partial\mathbf{x}_P}{\partial\mathbf{x}_I'}
            \frac{\partial\mathbf{x}_I'}{\partial\mathbf{d}} \\
            \frac{\partial\mathbf{x}_P}{\partial\boldsymbol{\delta\theta}} =
            \frac{\partial\mathbf{x}_p}{\partial\mathbf{x}_I'}
            \frac{\partial\mathbf{x}_I'}{\partial\mathbf{x}_I}
            \frac{\partial\mathbf{x}_I}{\partial\boldsymbol{\delta\theta}}
            \end{gather}

        where all else is as defined previously

        :param unit_vector_camera: The unit vector we are computing the Jacobian for
        :param image: The number of the image we are computing the Jacobian for
        :param num_images:   The total number of images included in our Jacobian matrix
        :param temperature: The temperature to compute the Jacobian at
        :return:
        """

        # ensure the input is an array and the right shape
        unit_vector_camera = np.asarray(unit_vector_camera).reshape(3)

        # get the required projections for the input
        image_loc, image_loc_dist, picture_loc = self.get_projections(unit_vector_camera,
                                                                      image=image,
                                                                      temperature=temperature)

        # get the camera point after misalignment and shift from principle frame is applied
        if self.estimate_multiple_misalignments:
            # optimization to avoid matrix multiplication
            if np.any(self.misalignment[image]):
                cam_point = rotvec_to_rotmat(
                    self.misalignment[image]).squeeze() @ unit_vector_camera

            else:
                cam_point = unit_vector_camera

        else:
            if np.any(self.misalignment):  # optimization to avoid matrix multiplication
                cam_point = rotvec_to_rotmat(
                    self.misalignment).squeeze() @ unit_vector_camera

            else:
                cam_point = unit_vector_camera

        # cam_point = unit_vector_camera

        # compute the radial distance from the optical axis as well as its powers
        # noinspection PyTypeChecker
        radius2: float = np.sum(np.power(image_loc, 2), axis=0) 
        radius4 = radius2 ** 2
        radius6 = radius4 * radius2

        # --------------------------------------------------------------------------------------------------------------
        # get the partial derivative of the measurement with respect to the misalignment vector
        # --------------------------------------------------------------------------------------------------------------

        # get the partial derivative of the distorted gnomic location with respect to the gnomic location
        ddist_gnom_dgnom = self._compute_ddistorted_gnomic_dgnomic(
            image_loc, radius2, radius4, radius6)

        # get the partial derivative of the pixel location of the point with respect to the distorted location
        dpix_ddist_gnom = self._compute_dpixel_ddistorted_gnomic(
            temperature=temperature)

        # get the partial derivative of the camera location with respect to a change in the misalignment vector
        dcam_point_dmisalignment = self._compute_dcamera_point_dmisalignment(
            unit_vector_camera)

        # get the partial derivative of the gnomic location with respect to the point in the camera frame
        dgnom_dcam_point = self._compute_dgnomic_dcamera_point(cam_point)

        # get the partial derivative of the pixel location with respect to the misalignment
        dpix_dmisalignment = dpix_ddist_gnom @ ddist_gnom_dgnom @ dgnom_dcam_point @ dcam_point_dmisalignment

        # --------------------------------------------------------------------------------------------------------------
        # get the partial derivative of the measurement with respect to the intrinsic matrix
        # --------------------------------------------------------------------------------------------------------------

        dpix_dintrinsic = self._compute_dpixel_dintrinsic(image_loc_dist)

        # --------------------------------------------------------------------------------------------------------------
        # get the partial derivative of the measurement with respect to the distortion coefficients
        # --------------------------------------------------------------------------------------------------------------

        # compute the partial derivative of the distorted gnomic location with respect to the distortion coefficients
        dist_gnom_ddist = self._compute_ddistorted_gnomic_ddistortion(
            image_loc, radius2, radius4, radius6)

        # compute the partial derivative of the pixel location with respect to the distortion coefficients
        dpix_ddist = dpix_ddist_gnom @ dist_gnom_ddist
        # --------------------------------------------------------------------------------------------------------------

        # get the partial derivative of the measurement with respect to the temperature coefficients
        # --------------------------------------------------------------------------------------------------------------

        dpix_dtemperature = self._compute_dpixel_dtemperature_coeffs(
            image_loc_dist, temperature=temperature)

        # stack everything together
        if self.estimate_multiple_misalignments:
            jacobian_row = np.hstack([dpix_dintrinsic, dpix_ddist, dpix_dtemperature,
                                      np.zeros((2, image * 3)
                                               ), dpix_dmisalignment,
                                      np.zeros((2, (num_images - image - 1) * 3))])

        else:
            jacobian_row = np.hstack(
                [dpix_dintrinsic, dpix_ddist, dpix_dtemperature, dpix_dmisalignment])

        return jacobian_row

    def apply_update(self, update_vec: ARRAY_LIKE):
        r"""
        This method takes in a delta update to camera parameters (:math:`\Delta\mathbf{c}`) and applies the update
        to the current instance in place.

        In general the delta update is calculated in the estimators in the :mod:`.calibration` subpackage and this
        method is not used by the user.

        The order of the update vector is determined by the order of the elements in
        :attr:`~.BrownModel.estimation_parameters`.  Any misalignment terms must always come last.

        This method operates by walking through the elements in :attr:`~.BrownModel.estimation_parameters` and
        retrieving the parameter that the element corresponds to.  The value of the update_vec at the index of the
        parameter is then applied as an additive update to the parameter in self, with the exception of misalignment,
        which is applied as a multiplicative update.

        :param update_vec: An iterable of delta updates to the model parameters
        """

        jacobian_parameters = np.hstack([getattr(self.element_dict[element], 'start', self.element_dict[element])
                                         for element in self.estimation_parameters])

        update_vec = self._fix_update_vector(np.asanyarray(update_vec, dtype=np.float64), jacobian_parameters)

        update_vec = np.asarray(update_vec).ravel()

        for ind, parameter in enumerate(jacobian_parameters):

            if parameter == 0:
                self.intrinsic_matrix[0, 0] += update_vec.item(ind)

            elif parameter == 1:
                self.intrinsic_matrix[1, 1] += update_vec.item(ind)

            elif parameter == 2:
                self.intrinsic_matrix[0, 1] += update_vec.item(ind)

            elif parameter == 3:
                self.intrinsic_matrix[0, 2] += update_vec.item(ind)

            elif parameter == 4:
                self.intrinsic_matrix[1, 2] += update_vec.item(ind)

            elif parameter == 5:
                self.distortion_coefficients[0] += update_vec.item(ind)

            elif parameter == 6:
                self.distortion_coefficients[1] += update_vec.item(ind)

            elif parameter == 7:
                self.distortion_coefficients[2] += update_vec.item(ind)

            elif parameter == 8:
                self.distortion_coefficients[3] += update_vec.item(ind)

            elif parameter == 9:
                self.distortion_coefficients[4] += update_vec.item(ind)

            elif parameter == 10:
                self.a1 += update_vec.item(ind)

            elif parameter == 11:
                self.a2 += update_vec.item(ind)

            elif parameter == 12:
                self.a3 += update_vec.item(ind)

            elif parameter == 13 or parameter == '13:':

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
