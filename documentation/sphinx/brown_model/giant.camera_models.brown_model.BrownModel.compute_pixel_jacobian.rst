BrownModel.compute\_pixel\_jacobian
===================================

.. currentmodule:: giant.camera_models.brown_model

:mod:`giant.camera_models.brown_model`\:

.. automethod:: BrownModel.compute_pixel_jacobian


Mathematically the Jacobian matrix is defined as

.. math::
    \frac{\partial\mathbf{x}_P}{\partial\mathbf{x}_C} =
    \frac{\partial\mathbf{x}_P}{\partial\mathbf{x}_I'}
    \frac{\partial\mathbf{x}_I'}{\partial\mathbf{x}_I}
    \frac{\partial\mathbf{x}_I}{\partial\mathbf{x}_C'}
    \frac{\partial\mathbf{x}_C'}{\partial\mathbf{x}_C}

where

.. math::
    :nowrap:

    \begin{gather}
    \frac{\partial\mathbf{x}_P}{\partial\mathbf{x}_I'} = (1+a_1T+a_2T^2+a_3T^3)
    \left[\begin{array}{cc} f_x & \alpha \\ 0 & f_y \end{array}\right] \\
    \begin{split}
    \frac{\partial\mathbf{x}_I'}{\partial\mathbf{x}_I} = &\left(1 + k_1r^2+k_2r^4+k_3r^6\right)
    \mathbf{I}_{2\times 2} + \left[\begin{array}{cc}2p_1y_I+4p_2x_I & 2p_1x_I \\
    2p_2y_I & 4p_1y_I+2p_2x_I \end{array}\right] + \\
    & \left(2k_1+4k_2r^2+6k_3r^4\right)\mathbf{x}_I\mathbf{x}_I^T +
    2 \left[\begin{array}{cc} p_2x_I & p_2y_I \\ p_1x_I & p_1y_I \end{array}\right]
    \end{split}\\
    \frac{\partial\mathbf{x}_I}{\partial\mathbf{x}_C'} = \frac{1}{z_C'}\left[
    \begin{array}{ccc}1 & 0 & \frac{-x_C'}{z_C'} \\ 0 & 1 & \frac{-y_C'}{z_C'} \end{array}\right] \\
    \frac{\partial\mathbf{x}_C'}{\partial\mathbf{x}_C} = \mathbf{T}_{\boldsymbol{\delta\theta}}
    \end{gather}
