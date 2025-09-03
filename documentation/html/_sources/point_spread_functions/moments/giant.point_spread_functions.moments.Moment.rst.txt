Moment
======

.. currentmodule:: giant.point_spread_functions.moments

.. autoclass:: Moment
    :show-inheritance:
    :no-members:
    :members: centroid_x, centroid_y, centroid, residual_rss, residual_mean, residual_std, covariance

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

    ~Moment.fit
    ~Moment.__call__
    ~Moment.apply_1d
    ~Moment.evaluate
    ~Moment.generate_kernel
    ~Moment.shift_centroid
    ~Moment.volume
    ~Moment.compare

|
