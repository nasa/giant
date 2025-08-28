PinholeModel.compute\_unit\_vector\_jacobian
============================================

.. currentmodule:: giant.camera_models.pinhole_model

:mod:`giant.camera_models.pinhole_model`\:

.. automethod:: PinholeModel.compute_unit_vector_jacobian

Mathematically this Jacobian is defined as

.. math::

    \frac{\partial\mathbf{x}_C}{\partial\mathbf{x}_P} =
    \frac{\partial\mathbf{x}_C}{\partial\mathbf{x}_C'}
    \frac{\partial\mathbf{x}_C'}{\partial\mathbf{x}_I}
    \frac{\partial\mathbf{x}_I}{\partial\mathbf{x}_P}

where

.. math::
    :nowrap:

    \begin{gather}
    \frac{\partial\mathbf{x}_C}{\partial\mathbf{x}_C'} = \mathbf{T}_{\boldsymbol{\delta\theta}}^T\\
    \frac{\partial\mathbf{x}_C'}{\partial\mathbf{x}_I} = \left[\begin{array}{cc}
    1/v & 0 \\ 0 & 1/v \\ 0 & 0\end{array}\right] -
    \frac{1}{v^3}\left[\begin{array}{c} \mathbf{x}_I \\ f \end{array}\right]\mathbf{x}_I^T \\
    \frac{\partial\mathbf{x}_I}{\partial\mathbf{x}_P} = \frac{\mathbf{K}^{-1}}{1+a_1T+a_2T^2+a_3T^3}
    \end{gather}

:math:`\mathbf{x}_C'` is the camera frame point after applying the misalignment,
:math:`\boldsymbol{\delta\theta}` is the misalignment vector, :math:`\mathbf{x}_C` is the camera frame point
before misalignment is applied,
:math:`\mathbf{x}_P` is the pixel location, :math:`\mathbf{x}_I` is the gnomic location,
:math:`a_{1-3}` are the temperature coefficients, :math:`T` is the temperature, :math:`k_x` is the inverse of
the pixel pitch in the x direction,  :math:`k_y` is the inverse of the pixel pitch in the y direction,
:math:`f` is the focal length, :math:`\mathbf{T}_{\boldsymbol{\delta\theta}}` is the rotation matrix
corresponding to rotation vector :math:`\boldsymbol{\delta\theta}`,
:math:`v=\sqrt{\mathbf{x}_I^T\mathbf{x}_I + f^2}`, and :math:`\mathbf{K}^{-1}` is the inverse intrinsic matrix.


