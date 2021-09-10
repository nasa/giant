BrownModel.compute\_unit\_vector\_jacobian
==========================================

.. currentmodule:: giant.camera_models.brown_model

:mod:`giant.camera_models.brown_model`\:

.. automethod:: BrownModel.compute_unit_vector_jacobian

Mathematically this Jacobian is defined as

.. math::

    \frac{\partial\mathbf{x}_C}{\partial\mathbf{x}_P} =
    \frac{\partial\mathbf{x}_C}{\partial\mathbf{x}_C'}
    \frac{\partial\mathbf{x}_C'}{\partial\mathbf{x}_I}
    \frac{\partial\mathbf{x}_I}{\partial\mathbf{x}_I'}
    \frac{\partial\mathbf{x}_I'}{\partial\mathbf{x}_P}

where

.. math::
    :nowrap:

    \begin{gather}
    \frac{\partial\mathbf{x}_C}{\partial\mathbf{x}_C'} = \mathbf{T}_{\boldsymbol{\delta\theta}}^T\\
    \frac{\partial\mathbf{x}_C'}{\partial\mathbf{x}_I} = \left[\begin{array}{cc}
    1/v & 0 \\ 0 & 1/v \\ 0 & 0\end{array}\right] -
    \frac{1}{v^3}\left[\begin{array}{c} \mathbf{x}_I \\ 1 \end{array}\right]\mathbf{x}_I^T \\
    \frac{\partial\mathbf{x}_I}{\partial\mathbf{x}_I'} = \mathbf{I}_{2\times 2} -
    \left.\frac{\partial\mathbf{x}_I'}{\partial\mathbf{x}_I}\right|_{\mathbf{x}_I=\mathbf{x}_I'} \\
    \begin{split}
    \frac{\partial\mathbf{x}_I'}{\partial\mathbf{x}_I} = &\left(1 + k_1r^2+k_2r^4+k_3r^6\right)
    \mathbf{I}_{2\times 2} + \left[\begin{array}{cc}2p_1y_I+4p_2x_I & 2p_1x_I \\
    2p_2y_I & 4p_1y_I+2p_2x_I \end{array}\right] + \\
    & \left(2k_1+4k_2r^2+6k_3r^4\right)\mathbf{x}_I\mathbf{x}_I^T +
    2 \left[\begin{array}{cc} p_2x_I & p_2y_I \\ p_1x_I & p_1y_I \end{array}\right]
    \end{split}\\
    \frac{\partial\mathbf{x}_I'}{\partial\mathbf{x}_P} = \frac{\mathbf{K}^{-1}}{1+a_1T+a_2T^2+a_3T^3}
    \end{gather}

:math:`\mathbf{x}_C'` is the camera frame point after applying the misalignment,
:math:`\boldsymbol{\delta\theta}` is the misalignment vector, :math:`\mathbf{x}_C` is the camera frame point
before misalignment is applied,
:math:`\mathbf{x}_P` is the pixel location, :math:`\mathbf{x}_I` is the gnomic location,
:math:`\mathbf{x}_I'` is the distorted gnomic location,
:math:`a_{1-3}` are the temperature coefficients, :math:`T` is the temperature,
:math:`\mathbf{T}_{\boldsymbol{\delta\theta}}` is the rotation matrix
corresponding to rotation vector :math:`\boldsymbol{\delta\theta}`,
:math:`v=\sqrt{\mathbf{x}_I^T\mathbf{x}_I + 1^2}`, and :math:`\mathbf{K}^{-1}` is the inverse intrinsic matrix.


