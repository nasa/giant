OwenModel
=========

.. currentmodule:: giant.camera_models.owen_model

:mod:`giant.camera_models.owen_model`\:

.. autoclass:: OwenModel
    :show-inheritance:
    :no-members:
    :members: field_of_view, kx, ky, kxy, kyx, px, py, a1, a2, a3, intrinsic_matrix_inv, radial2, e2, radial4, e4, tangential_y, e5, tangential_x, e6, pinwheel1, e1, pinwheel2, e3, distortion_coefficients

    .. attribute:: n_rows

        The number of rows in the active pixel array for the camera

    .. attribute:: n_cols

        The number of columns in the active pixel array for the camera

    .. attribute:: use_a_priori

        This boolean value is used to determine whether to append the identity matrix to the Jacobian matrix returned
        by :meth:`~OwenModel.compute_jacobian` in order to include the current estimate of the camera model in the
        calibration process.

    .. attribute:: intrinsic_matrix

        The 2x3 intrinsic matrix contains the conversion from unitless gnomic locations to a location in an image with
        units of pixels.

        It is defined as

        .. math::
            \mathbf{K} = \left[\begin{array}{ccc} k_x & k_{xy} & p_x \\
            k_{yx} & k_y & p_y \end{array}\right]

    .. attribute:: temperature_coefficients

        The coefficients for the polynomial specifying the change in the focal length as a function of temperature.

    .. attribute:: estimate_multiple_misalignments


        This boolean value is used to determine whether multiple misalignments are being estimated/used per image.

        If set to ``True`` then one misalignment is estimated for each image and used for each image when projecting
        through the camera model.  When set to ``False`` then a single misalignment is estimated for all images and
        used for all images when projecting through the camera model.  Typically the user shouldn't be setting this
        attribute directly as it is automatically handled when setting the :attr:`~.OwenModel.estimation_parameters`
        attribute.

    .. attribute:: estimation_parameters

        A list of strings containing the parameters to estimate when performing calibration with this model.

        This list is used in the methods :meth:`~.OwenModel.compute_jacobian` and :meth:`~.OwenModel.apply_update` to
        determine which parameters are being estimated/updated. From the :meth:`~.OwenModel.compute_jacobian` method,
        only columns of the Jacobian matrix corresponding to the parameters in this list are returned.  In the
        :meth:`~.OwenModel.apply_update` method, the update vector elements are assumed to correspond to the order
        expressed in this list.

        Valid values for the elements of this list are shown in the following table.  Generally, they correspond to
        attributes of this class, with a few convenient aliases that point to a collection of attributes.

        .. _owen-estimation-table:

        ============================  ======================================================================================
        Value                         Description
        ============================  ======================================================================================
        ``'basic'``                   estimate focal length, kyx, ky, all 6 distortion parameters, and a single misalignment
                                      term for all images between the camera attitude and the spacecraft's attitude:
                                      :math:`\left[\begin{array}{cccccccccc} f & k_y & k_{yx} &
                                      \epsilon_1 & \epsilon_2 & \epsilon_3 & \epsilon_4 & \epsilon_5 &
                                      \epsilon_6 & \boldsymbol{\delta\theta} \end{array}\right]`
        ``'intrinsic'``               estimate focal length, kx, kxy, kyx, ky, px, py, and all 6 distortion parameters:
                                      :math:`\left[\begin{array}{ccccccccccccc} f & k_x & k_{xy} & k_y & k_{yx} & p_x &
                                      p_y & \epsilon_1 & \epsilon_2 & \epsilon_3 & \epsilon_4 & \epsilon_5 &
                                      \epsilon_6\end{array}\right]`
        ``'basic intrinsic'``         estimate focal length, kyx, ky, and all 6 distortion parameters:
                                      :math:`\left[\begin{array}{ccccccccc} f & k_y & k_{yx} &
                                      \epsilon_1 & \epsilon_2 & \epsilon_3 & \epsilon_4 & \epsilon_5 &
                                      \epsilon_6 \end{array}\right]`
        ``'temperature dependence'``  estimate a 3rd order polynomial for temperature dependence:
                                      :math:`\left[\begin{array}{ccc} a_1 & a_2 & a_3 \end{array}\right]`
        ``'focal_length'``            the focal length of the camera:  :math:`f`
        ``'kx'``                      pixel pitch along the x axis:  :math:`k_x`
        ``'kxy'``                     alpha term for non-rectangular/rotated pixels: :math:`k_{xy}`
        ``'kyx'``                     alpha term for non-rectangular/rotated pixels: :math:`k_{yx}`
        ``'ky'``                      pixel pitch along the y axis: :math:`k_y`
        ``'px'``                      x location of the principal point: :math:`p_x`
        ``'py'``                      y location of the principal point: :math:`p_y`
        ``'radial2'``, ``'e2'``       distortion coefficient corresponding to the :math:`r^2` term: :math:`\epsilon_2`
        ``'radial4'``, ``'e4'``       distortion coefficient corresponding to the :math:`r^4` term: :math:`\epsilon_4`
        ``'tangential y'``, ``'e5'``  distortion coefficient corresponding to the y tangential distortion:
                                      :math:`\epsilon_5`
        ``'tangential x'``, ``'e6'``  distortion coefficient corresponding to the x tangential distortion:
                                      :math:`\epsilon_6`
        ``'pinwheel1'``, ``'e1'``     distortion coefficient corresponding to the first order pinwheel distortion:
                                      :math:`\epsilon_1`
        ``'pinwheel2'``, ``'e3'``     distortion coefficient corresponding to the third order pinwheel distortion:
                                      :math:`\epsilon_3`
        ``'a1'``                      estimate the linear coefficient for the temperature dependence: :math:`a_1`
        ``'a2'``                      estimate the quadratice coefficient for the temperature dependence: :math:`a_2`
        ``'a3'``                      estimate the cubic coefficient for the temperature dependence: :math:`a_3`
        ``'single misalignment'``     estimate a single misalignment for all images: :math:`\boldsymbol{\delta\theta}`
        ``'multiple misalignments'``  estimate a misalignment for each image:
                                      :math:`\left[\begin{array}{ccc}\boldsymbol{\delta\theta}_1 & \ldots &
                                      \boldsymbol{\delta\theta}_n \end{array}\right]`
        ============================  ======================================================================================

        Note that it may not be possible to estimate all attributes simultaneously because this may result in a rank
        deficient matrix in the calibration process (for instance, without setting a priori weights, estimating
        ``'px'``, ``'py'``, and ``'multiple misalignments'`` together could result in a rank deficient matrix.
        Therefore, just because you can set something in this list doesn't mean you should.

        For more details about calibrating a camera model, see the :mod:`.calibration` package for details.

.. rubric:: Methods

.. autosummary::
  :nosignatures:
  :toctree:

  ~OwenModel.project_onto_image
  ~OwenModel.project_directions
  ~OwenModel.compute_jacobian
  ~OwenModel.compute_pixel_jacobian
  ~OwenModel.compute_unit_vector_jacobian
  ~OwenModel.apply_update
  ~OwenModel.pixels_to_unit
  ~OwenModel.undistort_pixels
  ~OwenModel.distort_pixels
  ~OwenModel.overwrite
  ~OwenModel.distortion_map
  ~OwenModel.undistort_image
  ~OwenModel.copy
  ~OwenModel.to_elem
  ~OwenModel.from_elem
  ~OwenModel.adjust_temperature
  ~OwenModel.get_temperature_scale
  ~OwenModel.pixels_to_gnomic
  ~OwenModel.prepare_interp
  ~OwenModel.pixels_to_gnomic_interp
  ~OwenModel.get_projections
  ~OwenModel.apply_distortion
  ~OwenModel.get_state_labels
  ~OwenModel.reset_misalignment
  ~OwenModel.get_misalignment

|
