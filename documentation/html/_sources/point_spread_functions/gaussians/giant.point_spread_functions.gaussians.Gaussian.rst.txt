Gaussian
========

.. currentmodule:: giant.point_spread_functions.gaussians

.. autoclass:: Gaussian
    :show-inheritance:
    :no-members:
    :members: sigma_x, sigma_y, amplitude, centroid_x, centroid_y, save_residuals, residuals, covariance, size,
              residual_mean, residual_std, residual_rss


.. rubric:: Summary of Methods

.. autosummary::
    :nosignatures:
    :toctree:

    ~Gaussian.fit
    ~Gaussian.__call__
    ~Gaussian.apply_1d
    ~Gaussian.apply_1d_sized
    ~Gaussian.compute_jacobian
    ~Gaussian.determine_size
    ~Gaussian.evaluate
    ~Gaussian.generate_kernel
    ~Gaussian.normalize_amplitude
    ~Gaussian.update_state
    ~Gaussian.volume
    ~Gaussian.shift_centroid
    ~Gaussian.compare

|
