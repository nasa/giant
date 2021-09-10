CameraModel
===========

.. currentmodule:: giant.camera_models.camera_model

:mod:`giant.camera_models.camera_model`\:

.. autoclass:: CameraModel
    :no-members:
    :members: n_rows, n_cols, field_of_view, use_a_priori, estimation_parameters, state_vector, important_attributes

.. rubric:: Summary of Methods

.. autosummary::
   :nosignatures:

   ~CameraModel.project_onto_image
   ~CameraModel.project_directions
   ~CameraModel.compute_jacobian
   ~CameraModel.compute_pixel_jacobian
   ~CameraModel.compute_unit_vector_jacobian
   ~CameraModel.apply_update
   ~CameraModel.pixels_to_unit
   ~CameraModel.undistort_pixels
   ~CameraModel.distort_pixels
   ~CameraModel.overwrite
   ~CameraModel.distortion_map
   ~CameraModel.undistort_image
   ~CameraModel.get_state_labels
   ~CameraModel.copy
   ~CameraModel.to_elem
   ~CameraModel.from_elem

|
