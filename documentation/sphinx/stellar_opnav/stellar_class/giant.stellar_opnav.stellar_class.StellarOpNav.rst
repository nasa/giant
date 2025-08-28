StellarOpNav
============

.. currentmodule:: giant.stellar_opnav.stellar_class

:mod:`giant.stellar_opnav.stellar_class`\:

:mod:`giant.stellar_opnav.stellar_class`\:

.. autoclass:: StellarOpNav
    :no-members:
    :members: process_stars, model, camera, scene, image_processing, star_id, attitude_estimator,
              queried_catalogue_star_records, queried_catalogue_image_points, queried_catalogue_unit_vectors,
              queried_weights_inertial, queried_weights_picture, use_weights,
              ip_extracted_image_points, ip_image_illums, ip_stats, ip_snrs, ip_psfs,
              unmatched_catalogue_star_records, unmatched_catalogue_image_points, unmatched_catalogue_unit_vectors,
              unmatched_extracted_image_points, unmatched_image_illums, unmatched_stats, unmatched_snrs, unmatched_psfs,
              unmatched_weights_inertial, unmatched_weights_picture,
              matched_catalogue_star_records, matched_catalogue_image_points, matched_catalogue_unit_vectors_inertial,
              matched_catalogue_unit_vectors_camera, matched_extracted_image_points, matched_image_illums,
              matched_stats, matched_snrs, matched_psfs, matched_weights_inertial, matched_weights_picture

   
.. rubric:: Summary of Methods

.. autosummary::
    :nosignatures:
    :toctree:

    ~StellarOpNav.id_stars
    ~StellarOpNav.estimate_attitude
    ~StellarOpNav.add_images
    ~StellarOpNav.matched_star_residuals
    ~StellarOpNav.remove_matched_stars
    ~StellarOpNav.reproject_stars
    ~StellarOpNav.reset_attitude_estimator
    ~StellarOpNav.reset_image_processing
    ~StellarOpNav.reset_star_id
    ~StellarOpNav.sid_summary
    ~StellarOpNav.update_attitude_estimator
    ~StellarOpNav.update_image_processing
    ~StellarOpNav.update_star_id
    ~StellarOpNav.review_outliers
    ~StellarOpNav.remove_outliers

|
