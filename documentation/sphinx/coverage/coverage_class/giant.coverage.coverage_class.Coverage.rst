giant.coverage.coverage\_class.Coverage
=======================================

.. currentmodule:: giant.coverage.coverage_class

.. autoclass:: Coverage
    :show-inheritance:
    :no-members:
    :members: scene, camera_model, camera_position_function, camera_orientation_function, sun_position_function,
              imaging_times, labels, visibility, gsds, altitudes, facet_visibility, facet_gsds, facet_altitudes,
              albedo_dop, x_terrain_dop, y_terrain_dop, total_dop, dop_jacobians, observation_count,
              ignore_indices, brdf, topography_variations, vec_viewed, targetvecs, targetfacets

.. rubric:: Summary of Methods

.. autosummary::
    :nosignatures:
    :toctree:

    ~Coverage.compute_visibility
    ~Coverage.reduce_visibility_to_facet
    ~Coverage.compute_velocities
    ~Coverage.check_fov
    ~Coverage.determine_footprints
    ~Coverage.compute_dop

|
