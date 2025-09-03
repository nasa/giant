OpNavImage
==========

.. currentmodule:: giant.image

:mod:`giant.image`\:

.. autoclass:: OpNavImage
    :no-members:
    :members: observation_date, rotation_inertial_to_camera, position, velocity, exposure_type, temperature, saturation

    .. attribute:: exposure
        :type: Optional[Real]

        The exposure length of the image (usually in seconds).

        This attribute is provided for documentation and convenience, but typically isn't used directly by core GIANT
        functions.  Instead, the :attr:`.exposure_type` attribute is used.  For an example of how one might use this
        attribute, see the :ref:`getting started <getting-started>` page for more details.

    .. attribute:: dark_pixels
        :type: Optional[numpy.ndarray]

        A numpy array of "dark" (active but covered) pixels from a detector.

        This array (if set) is used in the :meth:`.find_poi_in_roi` method to determine a rough noise level in the
        entire image.  It typically is set to a region of the detector which contains active pixels but is covered and
        not exposed to any illumination sources, if such a region exists.  If a region like this does not exist in your
        detector then leave set to None.

        Note that if your detector has these pixels, you should probably crop them out of the image data stored in this
        class (though this is not technically necessary).

    .. attribute:: instrument
        :type: Optional[str]

        A string specifying the instrument that this image comes from.

        This attribute is provided for documentation and convenience, but isn't used directly by core GIANT
        functions.  For an example of how one might use this attribute, see the :ref:`getting started <getting-started>`
        page for more details.

    .. attribute:: spacecraft
        :type: Optional[str]

        A string specifying the spacecraft hosting the instrument that this image comes from.

        This attribute is provided for documentation and convenience, but isn't used directly by core GIANT
        functions.  For an example of how one might use this attribute, see the :ref:`getting started <getting-started>`
        page for more details.

    .. attribute:: target
        :type: Optional[str]

        A string specifying the target observed by the instrument in this image.

        This attribute is provided for documentation and convenience, but isn't used directly by core GIANT
        functions.  For an example of how one might use this attribute, see the :ref:`getting started <getting-started>`
        page for more details.

    .. attribute:: pointing_post_fit
        :type: bool

        A flag specifying whether the attitude for this image has been estimated (``True``) or not.

        If this flag is ``True``, then the attitude has been estimated either using observations of stars, or by
        updating the attitude of a short exposure image from the attitude of a long exposure image that has been
        estimated using stars.  This is primarily used for informational purposes, though it is also used as a check
        in the :meth:`.Camera.update_short_attitude` method.

.. rubric:: Summary of Methods

.. autosummary::
  :nosignatures:

  ~OpNavImage.load_image
  ~OpNavImage.parse_data

|
