FisheyeModel
============

.. currentmodule:: giant.camera_models.fisheye_model

:mod:`giant.camera_models.fisheye_model`\:

.. autoclass:: FisheyeModel
    :show-inheritance:
    :no-members:
    :members: field_of_view, kx, ky, fx, fy, alpha, kxy, px, py, a1, a2, a3, intrinsic_matrix_inv, k1, k2, k3, k4, distortion_coefficients, state_vector

    .. attribute:: n_rows

        The number of rows in the active pixel array for the camera

    .. attribute:: n_cols

        The number of columns in the active pixel array for the camera

    .. attribute:: use_a_priori

        This boolean value is used to determine whether to append the identity matrix to the Jacobian matrix returned
        by :meth:`~FisheyeModel.compute_jacobian` in order to include the current estimate of the camera model in the
        calibration process.

    .. attribute:: intrinsic_matrix

        The 2x3 intrinsic matrix contains the conversion from unitless gnomic locations to a location in an image with
        units of pixels.

        It is defined as

        .. math::
            \mathbf{K} = \left[\begin{array}{ccc} f_x & \alpha & p_x \\
            0 & f_y & p_y \end{array}\right]

    .. attribute:: temperature_coefficients

        The coefficients for the polynomial specifying the change in the focal length as a function of temperature.

    .. attribute:: estimate_multiple_misalignments


        This boolean value is used to determine whether multiple misalignments are being estimated/used per image.

        If set to ``True`` then one misalignment is estimated for each image and used for each image when projecting
        through the camera model.  When set to ``False`` then a single misalignment is estimated for all images and
        used for all images when projecting through the camera model.  Typically the user shouldn't be setting this
        attribute directly as it is automatically handled when setting the :attr:`~.FisheyeModel.estimation_parameters`
        attribute.

    .. attribute:: estimation_parameters

        A list of strings containing the parameters to estimate when performing calibration with this model.

        This list is used in the methods :meth:`~.FisheyeModel.compute_jacobian` and :meth:`~.FisheyeModel.apply_update` to
        determine which parameters are being estimated/updated. From the :meth:`~.FisheyeModel.compute_jacobian` method,
        only columns of the Jacobian matrix corresponding to the parameters in this list are returned.  In the
        :meth:`~.FisheyeModel.apply_update` method, the update vector elements are assumed to correspond to the order
        expressed in this list.

        Valid values for the elements of this list are shown in the following table.  Generally, they correspond to
        attributes of this class, with a few convenient aliases that point to a collection of attributes.



.. rubric:: Summary of Methods

.. autosummary::
  :nosignatures:
  :toctree:

  ~FisheyeModel.project_onto_image
  ~FisheyeModel.compute_jacobian
  ~FisheyeModel.compute_pixel_jacobian
  ~FisheyeModel.compute_unit_vector_jacobian
  ~FisheyeModel.apply_update
  ~FisheyeModel.pixels_to_unit
  ~FisheyeModel.undistort_pixels
  ~FisheyeModel.distort_pixels
  ~FisheyeModel.overwrite
  ~FisheyeModel.distortion_map
  ~FisheyeModel.undistort_image
  ~FisheyeModel.copy
  ~FisheyeModel.to_elem
  ~FisheyeModel.from_elem
  ~FisheyeModel.adjust_temperature
  ~FisheyeModel.get_temperature_scale
  ~FisheyeModel.pixels_to_gnomic
  ~FisheyeModel.prepare_interp
  ~FisheyeModel.pixels_to_gnomic_interp
  ~FisheyeModel.get_projections
  ~FisheyeModel.apply_distortion
  ~FisheyeModel.get_state_labels
  ~FisheyeModel.reset_misalignment
  ~FisheyeModel.get_misalignment

|
