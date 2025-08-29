StarID
======

.. currentmodule:: giant.stellar_opnav.star_identification

:mod:`giant.stellar_opnav.star_identification`\:

.. autoclass:: StarID
    :no-members:
    :members: model, catalog, max_magnitude, min_magnitude, max_combos, tolerance, ransac_tolerance, 
              second_closest_check, unique_check, use_mp, compute_weights,
              queried_catalog_image_points, queried_catalog_star_records, queried_catalog_unit_vectors,
              queried_weights_inertial, queried_weights_picture,
              unmatched_catalog_image_points, unmatched_catalog_star_records, unmatched_catalog_unit_vectors,
              unmatched_weights_inertial, unmatched_weights_picture,
              matched_catalog_image_points, matched_catalog_star_records, matched_catalog_unit_vectors, 
              matched_extracted_image_points, matched_weights_inertial, matched_weights_picture


   
.. rubric:: Summary of Methods

.. autosummary::
    :nosignatures:
    :toctree:

    ~StarID.id_stars
    ~StarID.ransac
    ~StarID.compute_pointing
    ~StarID.project_stars
    ~StarID.query_catalog
    ~StarID.ransac_iter_test

|
