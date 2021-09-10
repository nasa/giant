giant.calibration.calibration\_class.Calibration
================================================

.. currentmodule:: giant.calibration.calibration_class

.. autoclass:: Calibration
    :show-inheritance:
    :no-members:
    :members: process_stars, model, camera, image_processing, star_id, attitude_estimator, calibration_estimator,
              static_alignment_estimator, temperature_dependent_alignment_estimator, static_alignment,
              temperature_dependent_alignment, alignment_base_frame_func,
              queried_catalogue_star_records, queried_catalogue_image_points, queried_catalogue_unit_vectors,
              queried_weights_inertial, queried_weights_picture,
              ip_extracted_image_points, ip_image_illums, ip_stats, ip_snrs, ip_psfs,
              unmatched_catalogue_star_records, unmatched_catalogue_image_points, unmatched_catalogue_unit_vectors,
              unmatched_extracted_image_points, unmatched_image_illums, unmatched_stats, unmatched_snrs, unmatched_psfs,
              unmatched_weights_inertial, unmatched_weights_picture,
              matched_catalogue_star_records, matched_catalogue_image_points, matched_catalogue_unit_vectors_inertial,
              matched_catalogue_unit_vectors_camera, matched_extracted_image_points, matched_image_illums,
              matched_stats, matched_snrs, matched_psfs, matched_weights_inertial, matched_weights_picture

    .. attribute:: use_weights

        A flag specifying whether to compute weights/use them in the attitude estimation routine

.. rubric:: Summary of Methods

.. autosummary::
    :nosignatures:
    :toctree:

    ~Calibration.id_stars
    ~Calibration.estimate_attitude
    ~Calibration.estimate_calibration
    ~Calibration.estimate_static_alignment
    ~Calibration.estimate_temperature_dependent_alignment
    ~Calibration.calib_summary
    ~Calibration.sid_summary
    ~Calibration.add_images
    ~Calibration.matched_star_residuals
    ~Calibration.remove_matched_stars
    ~Calibration.review_outliers
    ~Calibration.remove_outliers
    ~Calibration.reproject_stars
    ~Calibration.reset_settings
    ~Calibration.reset_image_processing
    ~Calibration.reset_star_id
    ~Calibration.reset_attitude_estimator
    ~Calibration.reset_calibration_estimator
    ~Calibration.reset_static_alignment_estimator
    ~Calibration.reset_temperature_dependent_alignment_estimator
    ~Calibration.update_settings
    ~Calibration.update_image_processing
    ~Calibration.update_star_id
    ~Calibration.update_attitude_estimator
    ~Calibration.update_calibration_estimator
    ~Calibration.update_static_alignment_estimator
    ~Calibration.update_temperature_dependent_alignment_estimator
    ~Calibration.limit_magnitude

|
