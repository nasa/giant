IterativeGaussian
=================

.. currentmodule:: giant.point_spread_functions.gaussians

.. autoclass:: IterativeGaussian
    :show-inheritance:
    :no-members:
    :members: sigma_x, sigma_y, amplitude, centroid_x, centroid_y, residuals, covariance, size,
              residual_mean, residual_std, residual_rss

    .. attribute:: save_residuals
        :value: False

        This class attribute specifies whether to save the residuals when fitting the specified PSF to data.

        Saving the residuals can be important for in depth analysis but can use a lot of space when many fits are being
        performed and stored so this defaults to off.  To store the residuals simply set this to ``True`` either before or
        after initialization.

    .. attribute:: max_iter
        :value: 20

        An integer defining the maximum number of iterations to attempt in the iterative least squares solution.

    .. attribute:: atol
        :value: 1e-10

        The absolute tolerance cut-off for the iterative least squares. (The iteration will cease when the new estimate is
        within this tolerance for every element from the previous estimate)

    .. attribute:: rtol
        :value: 1e-10

        The relative tolerance cut-off for the iterative least squares. (The iteration will cease when the maximum percent
        change in the state vector from one iteration to the next is less than this value)

.. rubric:: Summary of Methods

.. autosummary::
    :nosignatures:
    :toctree:

    ~IterativeGaussian.fit
    ~IterativeGaussian.__call__
    ~IterativeGaussian.apply_1d
    ~IterativeGaussian.apply_1d_sized
    ~IterativeGaussian.compute_jacobian
    ~IterativeGaussian.converge
    ~IterativeGaussian.fit_lstsq
    ~IterativeGaussian.determine_size
    ~IterativeGaussian.evaluate
    ~IterativeGaussian.generate_kernel
    ~IterativeGaussian.normalize_amplitude
    ~IterativeGaussian.update_state

|
