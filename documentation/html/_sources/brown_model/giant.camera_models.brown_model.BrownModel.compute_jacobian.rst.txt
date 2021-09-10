BrownModel.compute\_jacobian
============================

.. currentmodule:: giant.camera_models.brown_model

:mod:`giant.camera_models.brown_model`\:

.. automethod:: BrownModel.compute_jacobian

Example::

    >>> from giant.camera_models import BrownModel
    >>> model = BrownModel(fx=3000, fy=4000, px=500, py=500, a1=1e-5, a2=1e-6,
    >>>                    misalignment = [[1e-12, -2e-14, 3e-10], [2e-15, 1e-13, 3e-10]],
    >>>                    estimation_parameters = ['multiple misalignments'])
    >>> model.compute_jacobian([[[0.5], [0], [1]], [[0.1, 0.2, 0.3], [-0.3, -0.4, -0.5], [4, 5, 6]]],
    >>>                        temperature=[10, -20])
    array([[  0.00000000e+00,  -3.75075000e+03,   0.00000000e+00,
              0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
           [  4.00080000e+03,   2.98059600e-07,  -2.00040000e+03,
              0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
             -5.62612499e+00,  -3.00247537e+03,  -2.25045000e+02],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              4.02330450e+03,   7.50150000e+00,  -1.00020000e+02],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
             -9.60191999e+00,  -3.00540096e+03,  -2.40048000e+02],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              4.02640512e+03,   1.28025600e+01,  -1.60032000e+02],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
             -1.25025000e+01,  -3.00810150e+03,  -2.50050000e+02],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              4.02858333e+03,   1.66700000e+01,  -2.00040000e+02]])

Mathematically the Jacobian matrix is defined to be

.. math::
    \frac{\partial\mathbf{x}_P}{\partial\mathbf{c}} = \left[\begin{array}{cccc}
    \frac{\partial\mathbf{x}_P}{\partial\mathbf{k}} &
    \frac{\partial\mathbf{x}_P}{\partial\mathbf{d}} &
    \frac{\partial\mathbf{x}_P}{\partial\mathbf{a}} &
    \frac{\partial\mathbf{x}_P}{\partial\boldsymbol{\delta\theta}}\end{array}\right]

where, using the chain rule,

.. math::
    :nowrap:

    \begin{gather}
    \frac{\partial\mathbf{x}_P}{\partial\mathbf{d}} =
    \frac{\partial\mathbf{x}_P}{\partial\mathbf{x}_I'}
    \frac{\partial\mathbf{x}_I'}{\partial\mathbf{d}} \\
    \frac{\partial\mathbf{x}_P}{\partial\boldsymbol{\delta\theta}} =
    \frac{\partial\mathbf{x}_p}{\partial\mathbf{x}_I'}
    \frac{\partial\mathbf{x}_I'}{\partial\mathbf{x}_I}
    \frac{\partial\mathbf{x}_I}{\partial\mathbf{x}_C'}
    \frac{\partial\mathbf{x}_C'}{\partial\boldsymbol{\delta\theta}}
    \end{gather}

and

.. math::
    :nowrap:

    \begin{gather}
    \frac{\partial\mathbf{x}_P}{\partial\mathbf{k}} = \left[\begin{array}{cccc} x_I & 0 & y_I & 1 & 0
    \\ 0 & y_I & 0 & 0 & 1 \end{array}\right] \\
    \frac{\partial\mathbf{x}_P}{\partial\mathbf{a}} = \left[\begin{array}{cc} f_x & \alpha \\
    0 & f_y\end{array}\right] \mathbf{x}_I \left[\begin{array}{ccc} T & T^2 & T^3 \end{array}\right] \\
    \frac{\partial\mathbf{x}_P}{\partial\mathbf{x}_I'} = (1+a_1T+a_2T^2+a_3T^3)
    \left[\begin{array}{cc} f_x & \alpha \\ 0 & f_y \end{array}\right] \\
    \frac{\partial\mathbf{x}_I'}{\partial\mathbf{d}} = \left[\begin{array}{ccccc} r^2x_I &
    r^4x_I & r^6x_I & 2x_Iy_I & r^2+2y_I^2 \\
    r^2y_I & r^4y_I & r_6y_I & r^2+2x_I^2 & 2x_Iy_I \end{array}\right] \\
    \begin{split}
    \frac{\partial\mathbf{x}_I'}{\partial\mathbf{x}_I} = &\left(1 + k_1r^2+k_2r^4+k_3r^6\right)
    \mathbf{I}_{2\times 2} + \left[\begin{array}{cc}2p_1y_I+4p_2x_I & 2p_1x_I \\
    2p_2y_I & 4p_1y_I+2p_2x_I \end{array}\right] + \\
    & \left(2k_1+4k_2r^2+6k_3r^4\right)\mathbf{x}_I\mathbf{x}_I^T +
    2 \left[\begin{array}{cc} p_2x_I & p_2y_I \\ p_1x_I & p_1y_I \end{array}\right]
    \end{split}\\
    \frac{\partial\mathbf{x}_I}{\partial\mathbf{x}_C'} = \frac{1}{z_C'}\left[
    \begin{array}{ccc}1 & 0 & \frac{-x_C'}{z_C'} \\ 0 & 1 & \frac{-y_C'}{z_C'} \end{array}\right] \\
    \frac{\partial\mathbf{x}_C'}{\partial\boldsymbol{\delta\theta}} = \left[\mathbf{x}_C\times\right]
    \end{gather}

where :math:`\mathbf{k}=[k_x \quad k_y \quad \alpha \quad p_x \quad p_y]` is a vector of the intrinsic camera parameters,
:math:`\mathbf{a}=[a_1 \quad a_2 \quad a_3]` is a vector of the temperature dependence coefficients,
:math:`\mathbf{x}_C'` is the camera frame point after applying the misalignment,
:math:`\boldsymbol{\delta\theta}` is the misalignment vector, :math:`\mathbf{x}_C` is the camera frame point
before misalignment is applied, :math:`\left[\bullet\times\right]` is the skew-symmetric cross product
matrix formed from :math:`\bullet`,
:math:`\mathbf{x}_P` is the pixel location, :math:`\mathbf{x}_I` is the gnomic location,
:math:`\mathbf{x}_I'` is the distorted gnomic location,
:math:`a_{1-3}` are the temperature coefficients, :math:`T` is the temperature, :math:`f_x` is the effective focal
length in the x direction, :math:`f_y` is the effective focal length in the y direction,
:math:`\mathbf{d}=[k_1 \quad k_2 \quad k_3 \quad p_1 \quad p_2]` is a vector of the distortion coefficients,
:math:`\alpha` is the skewness parameter, :math:`k_1` is the second order radial distortion,
:math:`k_2` is the fourth order radial distortion coefficient,
:math:`k_3` is the sixth order radial distortion coefficient,
:math:`p_1` is the x tangential distortion coefficient, and
:math:`p_2` is the x tangential distortion coefficient.

