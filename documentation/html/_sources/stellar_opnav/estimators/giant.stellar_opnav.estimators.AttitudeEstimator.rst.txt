AttitudeEstimator
=================

.. currentmodule:: giant.stellar_opnav.estimators

:mod:`giant.stellar_opnav.estimators`\:

.. autoclass:: AttitudeEstimator
    :no-members:
    :members: post_fit_covariance

    .. attribute:: target_frame_directions

        The unit vectors in the target frame as a 3xn array (:math:`\mathbf{a}_i`).

        Each column should represent the pair of the corresponding column in :attr:`base_frame_directions`.

    .. attribute:: base_frame_directions

        The unit vectors in the base frame as a 3xn array (:math:`\mathbf{b}_i`).

        Each column should represent the pair of the corresponding column in :attr:`target_frame_directions`.

    .. attribute:: weighted_estimation

        A boolean flag specifying whether to apply weights to the camera--database pairs in the estimation

    .. attribute:: weights

        A length n array of the weights to apply if weighted_estimation is True. (:math:`w_i`)

        Each element should represent the pair of the corresponding column in :attr:`target_frame_directions` and
        :attr:`base_frame_directions`.

    .. attribute:: rotation

        The solved for rotation that best aligns the :attr:`base_frame_directions` and :attr:`target_frame_directions` after calling :meth:`estimate`.

        This rotation should go from the base frame to the target frame frame.

.. rubric:: Summary of Methods

.. autosummary::
    :nosignatures:
    :toctree:

    ~AttitudeEstimator.estimate

|
