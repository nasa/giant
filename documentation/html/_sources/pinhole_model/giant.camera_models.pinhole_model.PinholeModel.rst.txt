PinholeModel
============

.. currentmodule:: giant.camera_models.pinhole_model

:mod:`giant.camera_models.pinhole_model`\:

.. autoclass:: PinholeModel
    :show-inheritance:
    :no-members:
    :members: field_of_view, estimation_parameters, focal_length, kx, ky, px, py, temperature_coefficients, a1, a2, a3, intrinsic_matrix, intrinsic_matrix_inv, misalignment, estimate_multiple_misalignments, important_attributes

    .. attribute:: n_rows

        The number of rows in the active pixel array for the camera

    .. attribute:: n_cols

        The number of columns in the active pixel array for the camera

    .. attribute:: use_a_priori

        This boolean value is used to determine whether to append the identity matrix to the Jacobian matrix returned
        by :meth:`~PinholeModel.compute_jacobian` in order to include the current estimate of the camera model in the
        calibration process.

.. rubric:: Summary of Methods

.. autosummary::
    :nosignatures:
    :toctree:

    ~PinholeModel.project_onto_image
    ~PinholeModel.project_directions
    ~PinholeModel.compute_jacobian
    ~PinholeModel.compute_pixel_jacobian
    ~PinholeModel.compute_unit_vector_jacobian
    ~PinholeModel.apply_update
    ~PinholeModel.pixels_to_unit
    ~PinholeModel.undistort_pixels
    ~PinholeModel.distort_pixels
    ~PinholeModel.overwrite
    ~PinholeModel.distortion_map
    ~PinholeModel.undistort_image
    ~PinholeModel.copy
    ~PinholeModel.to_elem
    ~PinholeModel.from_elem
    ~PinholeModel.adjust_temperature
    ~PinholeModel.get_temperature_scale
    ~PinholeModel.get_projections
    ~PinholeModel.pixels_to_gnomic
    ~PinholeModel.prepare_interp
    ~PinholeModel.pixels_to_gnomic_interp
    ~PinholeModel.get_state_labels
    ~PinholeModel.reset_misalignment
    ~PinholeModel.get_misalignment

|
