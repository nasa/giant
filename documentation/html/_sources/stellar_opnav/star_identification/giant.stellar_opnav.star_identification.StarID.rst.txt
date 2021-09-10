StarID
======

.. currentmodule:: giant.stellar_opnav.star_identification

:mod:`giant.stellar_opnav.star_identification`\:

.. autoclass:: StarID
    :no-members:
    :members: model, camera_velocity, camera_position, extracted_image_points, catalogue, a_priori_rotation_cat2camera,
              max_magnitude, min_magnitude, tolerance, max_combos, ransac_tolerance, second_closest_check, unique_check,
              use_mp, queried_catalogue_image_points, queried_catalogue_star_records, queried_catalogue_unit_vectors,
              queried_weights_inertial, queried_weights_picture,
              unmatched_catalogue_image_points, unmatched_catalogue_star_records, unmatched_catalogue_unit_vectors,
              unmatched_weights_inertial, unmatched_weights_picture,
              matched_catalogue_image_points, matched_extracted_image_points, matched_catalogue_star_records,
              matched_catalogue_unit_vectors, matched_weights_inertial, matched_weights_picture


   
.. rubric:: Summary of Methods

.. autosummary::
    :nosignatures:
    :toctree:

    ~StarID.id_stars
    ~StarID.ransac
    ~StarID.compute_pointing
    ~StarID.project_stars
    ~StarID.query_catalogue
    ~StarID.ransac_iter_test

|
