SplitCamera
===========

.. currentmodule:: giant.camera_models.split_camera

:mod:`giant.camera_models.split_camera`\:

.. autoclass:: SplitCamera
    :show-inheritance:
    :no-members:
    :members: field_of_view, n_rows, n_cols, use_a_priori, model1, model2, camera_frame_split_axis, camera_frame_split_threshold, image_plane_split_axis, image_plane_split_threshold, estimation_parameters, state_vector


.. rubric:: Summary of Methods

.. autosummary::
  :nosignatures:
  :toctree:

  ~SplitCamera.project_onto_image
  ~SplitCamera.project_directions
  ~SplitCamera.compute_jacobian
  ~SplitCamera.compute_pixel_jacobian
  ~SplitCamera.compute_unit_vector_jacobian
  ~SplitCamera.apply_update
  ~SplitCamera.pixels_to_unit
  ~SplitCamera.undistort_pixels
  ~SplitCamera.distort_pixels
  ~SplitCamera.overwrite
  ~SplitCamera.to_elem
  ~SplitCamera.from_elem
  ~SplitCamera.check_in_fov

|
