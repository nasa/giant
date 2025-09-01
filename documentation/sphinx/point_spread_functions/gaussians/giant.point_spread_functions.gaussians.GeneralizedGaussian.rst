GeneralizedGaussian
===================

.. currentmodule:: giant.point_spread_functions.gaussians

.. autoclass:: GeneralizedGaussian
    :show-inheritance:
    :no-members:
    :members: a_coef, b_coef, c_coef, sigma_x, sigma_y, theta, amplitude, centroid_x, centroid_y, residuals,
              covariance, size, residual_mean, residual_std, residual_rss

    .. attribute:: save_residuals
        :value: False

        This class attribute specifies whether to save the residuals when fitting the specified PSF to data.

        Saving the residuals can be important for in depth analysis but can use a lot of space when many fits are being
        performed and stored so this defaults to off.  To store the residuals simply set this to ``True`` either before or
        after initialization.

.. rubric:: Summary of Methods

.. autosummary::
    :nosignatures:
    :toctree:

    ~GeneralizedGaussian.fit
    ~GeneralizedGaussian.__call__
    ~GeneralizedGaussian.apply_1d
    ~GeneralizedGaussian.apply_1d_sized
    ~GeneralizedGaussian.compute_jacobian
    ~GeneralizedGaussian.determine_size
    ~GeneralizedGaussian.evaluate
    ~GeneralizedGaussian.generate_kernel
    ~GeneralizedGaussian.normalize_amplitude
    ~GeneralizedGaussian.update_state
    ~GeneralizedGaussian.volume
    ~GeneralizedGaussian.shift_centroid
    ~GeneralizedGaussian.compare

|
