Tracker
=======

.. currentmodule:: giant.ufo.ekf_tracker

:mod:`giant.ufo.ekf_tracker`\:

.. autoclass:: Tracker
    :no-members:
    :members: camera, scene, observation_trees, observation_ids, processes, confirmed_particles, dynamics,
              measurement_covariance, state_initializer, maximum_image_timedelta, initial_euclidean_threshold,
              maximum_paths_per_image, maximum_paths_total, maximum_forward_images, maximum_track_length,
              search_distance_function, maximum_mahalanobis_distance_squared, expected_convergence_number,
              reduced_paths_forward_per_image, minimum_number_of_measurements, maximum_residual_standard_deviation,
              maximum_time_outs, maximum_tracking_time_per_image, confirmed_filters, confirmed_standard_deviations,
              kernels_to_load
    

.. rubric:: Summary of Methods

.. autosummary::
   :nosignatures:
   :toctree:

   ~Tracker.filter_ekfs
   ~Tracker.find_initial_pairs
   ~Tracker.save_results
   ~Tracker.track
   


|
