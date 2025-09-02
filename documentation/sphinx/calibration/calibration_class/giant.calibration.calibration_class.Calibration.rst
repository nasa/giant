giant.calibration.calibration\_class.Calibration
================================================

.. currentmodule:: giant.calibration.calibration_class

.. autoclass:: Calibration
    :show-inheritance:
    :no-members:
    :members: process_stars, model, camera, scene,
              use_weights, point_of_interest_finder_options, star_id_options, attitude_estimator_options,
              custom_attitude_estimator_class, attitude_estimator_type, denoising,
              geometric_estimator_options, custom_geometric_estimator_class, geometric_estimator_type,
              alignment_base_frame_func, temperature_dependent_alignment_euler_order,
              star_id, attitude_estimator, point_of_interest_finder, geometric_estimator,
              queried_catalog_star_records, queried_catalog_image_points, queried_catalog_unit_vectors,
              queried_weights_inertial, queried_weights_picture,
              extracted_image_points, extracted_image_illums, extracted_psfs, extracted_stats, extracted_snrs,
              unmatched_catalog_star_records, unmatched_catalog_image_points, unmatched_catalog_unit_vectors,
              unmatched_weights_inertial, unmatched_weights_picture,
              unmatched_extracted_image_points, unmatched_image_illums, unmatched_stats, unmatched_snrs, unmatched_psfs,
              matched_catalog_star_records, matched_catalog_image_points, matched_catalog_unit_vectors_inertial,
              matched_catalog_unit_vectors_camera, matched_extracted_image_points, matched_image_illums,
              matched_stats, matched_snrs, matched_psfs, matched_weights_inertial, matched_weights_picture,
              static_alignment, temperature_dependent_alignment

.. rubric:: Summary of Methods

.. autosummary::
    :nosignatures:
    :toctree:

    ~Calibration.add_images
    ~Calibration.id_stars
    ~Calibration.estimate_attitude
    ~Calibration.estimate_geometric_calibration
    ~Calibration.geometric_calibration_summary
    ~Calibration.estimate_static_alignment
    ~Calibration.estimate_temperature_dependent_alignment
    ~Calibration.sid_summary
    ~Calibration.reproject_stars
    ~Calibration.matched_star_residuals
    ~Calibration.remove_matched_stars
    ~Calibration.review_outliers
    ~Calibration.remove_outliers
    ~Calibration.clear_results
    ~Calibration.reset_settings
    ~Calibration.reset_point_of_interest_finder
    ~Calibration.update_point_of_interest_finder
    ~Calibration.reset_star_id
    ~Calibration.update_star_id
    ~Calibration.reset_attitude_estimator
    ~Calibration.update_attitude_estimator
    ~Calibration.reset_geometric_estimator
    ~Calibration.update_geometric_estimator
    ~Calibration.limit_magnitude

|
