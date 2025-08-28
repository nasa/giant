BrownModel
==========

.. currentmodule:: giant.camera_models.brown_model

:mod:`giant.camera_models.brown_model`\:

.. autoclass:: BrownModel
    :show-inheritance:
    :no-members:
    :members: field_of_view, kx, ky, fx, fy, alpha, kxy, px, py, a1, a2, a3, intrinsic_matrix_inv, k1, k2, k3, radial2,
              radial4, radial6, p1, p2, tiptilt_y, tiptilt_x, distortion_coefficients, state_vector

    .. attribute:: n_rows

        The number of rows in the active pixel array for the camera

    .. attribute:: n_cols

        The number of columns in the active pixel array for the camera

    .. attribute:: use_a_priori

        This boolean value is used to determine whether to append the identity matrix to the Jacobian matrix returned
        by :meth:`~BrownModel.compute_jacobian` in order to include the current estimate of the camera model in the
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
        attribute directly as it is automatically handled when setting the :attr:`~.BrownModel.estimation_parameters`
        attribute.

    .. attribute:: estimation_parameters

        A list of strings containing the parameters to estimate when performing calibration with this model.

        This list is used in the methods :meth:`~.BrownModel.compute_jacobian` and :meth:`~.BrownModel.apply_update` to
        determine which parameters are being estimated/updated. From the :meth:`~.BrownModel.compute_jacobian` method,
        only columns of the Jacobian matrix corresponding to the parameters in this list are returned.  In the
        :meth:`~.BrownModel.apply_update` method, the update vector elements are assumed to correspond to the order
        expressed in this list.

        Valid values for the elements of this list are shown in the following table.  Generally, they correspond to
        attributes of this class, with a few convenient aliases that point to a collection of attributes.

        .. _brown-estimation-table:

        ============================  ==================================================================================
        Value                         Description
        ============================  ==================================================================================
        ``'basic'``                   estimate fx, fy, alpha all 5 distortion parameters, and a single misalignment
                                      term for all images between the camera attitude and the spacecraft's attitude:
                                      :math:`\left[\begin{array}{cccccccccc} f_x & f_y & \alpha &
                                      k_1 & k_2 & k_3 & p_1 & p_2 & \boldsymbol{\delta\theta} \end{array}\right]`
        ``'intrinsic'``               estimate fx, fy, alpha, px, py, and all 5 distortion parameters:
                                      :math:`\left[\begin{array}{ccccccccccccc} f_x & f_y & \alpha & p_x &
                                      p_y & k_1 & k_2 & k_3 & p_1 & p_2 \end{array}\right]`
        ``'basic intrinsic'``         estimate fx, fy, alpha, and all 5 distortion parameters:
                                      :math:`\left[\begin{array}{ccccccccc} f_x & f_y & \alpha &
                                      k1 & k_2 & k_3 & p_1 & p_2 \end{array}\right]`
        ``'temperature dependence'``  estimate a 3rd order polynomial for temperature dependence:
                                      :math:`\left[\begin{array}{ccc} a_1 & a_2 & a_3 \end{array}\right]`
        ``'fx'``, ``'kx'``            focal length divided by pixel pitch along the x axis:  :math:`f_x`
        ``'alpha'``, ``'kxy'``        alpha term for non-rectangular/rotated pixels: :math:`\alpha`
        ``'fy'``, ``'ky'``            focal length divided by pixel pitch along the y axis: :math:`f_y`
        ``'px'``                      x location of the principal point: :math:`p_x`
        ``'py'``                      y location of the principal point: :math:`p_y`
        ``'radial2'``, ``'k1'``       distortion coefficient corresponding to the :math:`r^2` term: :math:`k_1`
        ``'radial4'``, ``'k2'``       distortion coefficient corresponding to the :math:`r^4` term: :math:`k_2`
        ``'radial6'``, ``'k3'``       distortion coefficient corresponding to the :math:`r^6` term: :math:`k_3`
        ``'tiptilt_y``, ``'p1'``      distortion coefficient corresponding to the y tip/tilt distortion:
                                      :math:`p_1`
        ``'tiptilt_x``, ``'p2'``      distortion coefficient corresponding to the x tip/tilt distortion:
                                      :math:`p_2`
        ``'a1'``                      estimate the linear coefficient for the temperature dependence: :math:`a_1`
        ``'a2'``                      estimate the quadratice coefficient for the temperature dependence: :math:`a_2`
        ``'a3'``                      estimate the cubic coefficient for the temperature dependence: :math:`a_3`
        ``'single misalignment'``     estimate a single misalignment for all images: :math:`\boldsymbol{\delta\theta}`
        ``'multiple misalignments'``  estimate a misalignment for each image:
                                      :math:`\left[\begin{array}{ccc}\boldsymbol{\delta\theta}_1 & \ldots &
                                      \boldsymbol{\delta\theta}_n \end{array}\right]`
        ============================  ==================================================================================

        Note that it may not be possible to estimate all attributes simultaneously because this may result in a rank
        deficient matrix in the calibration process (for instance, without setting a priori weights, estimating
        ``'px'``, ``'py'``, and ``'multiple misalignments'`` together could result in a rank deficient matrix.
        Therefore, just because you can set something in this list doesn't mean you should.

        For more details about calibrating a camera model, see the :mod:`.calibration` package for details.



.. rubric:: Summary of Methods

.. autosummary::
  :nosignatures:
  :toctree:

  ~BrownModel.project_onto_image
  ~BrownModel.project_directions
  ~BrownModel.compute_jacobian
  ~BrownModel.compute_pixel_jacobian
  ~BrownModel.compute_unit_vector_jacobian
  ~BrownModel.apply_update
  ~BrownModel.pixels_to_unit
  ~BrownModel.undistort_pixels
  ~BrownModel.distort_pixels
  ~BrownModel.overwrite
  ~BrownModel.distortion_map
  ~BrownModel.undistort_image
  ~BrownModel.copy
  ~BrownModel.to_elem
  ~BrownModel.from_elem
  ~BrownModel.adjust_temperature
  ~BrownModel.get_temperature_scale
  ~BrownModel.pixels_to_gnomic
  ~BrownModel.prepare_interp
  ~BrownModel.pixels_to_gnomic_interp
  ~BrownModel.get_projections
  ~BrownModel.apply_distortion
  ~BrownModel.get_state_labels
  ~BrownModel.reset_misalignment
  ~BrownModel.get_misalignment

|
