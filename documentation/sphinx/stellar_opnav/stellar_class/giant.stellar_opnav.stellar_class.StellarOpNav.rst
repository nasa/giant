StellarOpNav
============

.. currentmodule:: giant.stellar_opnav.stellar_class

:mod:`giant.stellar_opnav.stellar_class`\:

:mod:`giant.stellar_opnav.stellar_class`\:

.. autoclass:: StellarOpNav
    :no-members:
    :members: process_stars, model, camera, scene, 
              use_weights, point_of_interest_finder_options, star_id_options, attitude_estimator_options,
              custom_attitude_estimator_class, attitude_estimator_type, denoising, 
              star_id, attitude_estimator, point_of_interest_finder, 
              queried_catalog_star_records, queried_catalog_image_points, queried_catalog_unit_vectors,
              queried_weights_inertial, queried_weights_picture, use_weights,
              extracted_image_points, extracted_image_illums, extracted_psfs, extracted_stats, extracted_snrs, 
              unmatched_catalog_star_records, unmatched_catalog_image_points, unmatched_catalog_unit_vectors,
              unmatched_weights_inertial, unmatched_weights_picture,
              unmatched_extracted_image_points, unmatched_image_illums, unmatched_stats, unmatched_snrs, unmatched_psfs,
              matched_catalog_star_records, matched_catalog_image_points, matched_catalog_unit_vectors_inertial,
              matched_catalog_unit_vectors_camera, matched_extracted_image_points, matched_image_illums,
              matched_stats, matched_snrs, matched_psfs, matched_weights_inertial, matched_weights_picture

   
.. rubric:: Summary of Methods

.. autosummary::
    :nosignatures:
    :toctree:

    ~StellarOpNav.add_images
    ~StellarOpNav.reset_star_id
    ~StellarOpNav.update_star_id
    ~StellarOpNav.reset_point_of_interest_finder
    ~StellarOpNav.update_point_of_interest_finder
    ~StellarOpNav.reset_attitude_estimator
    ~StellarOpNav.update_attitude_estimator
    ~StellarOpNav.id_stars
    ~StellarOpNav.reproject_stars
    ~StellarOpNav.estimate_attitude
    ~StellarOpNav.sid_summary
    ~StellarOpNav.matched_star_residuals
    ~StellarOpNav.remove_matched_stars
    ~StellarOpNav.review_outliers
    ~StellarOpNav.remove_outliers
    ~StellarOpNav.clear_results

|
