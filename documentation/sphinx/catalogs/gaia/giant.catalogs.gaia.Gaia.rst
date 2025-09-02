Gaia
====

.. currentmodule:: giant.catalogs.gaia

:mod:`giant.catalogs.gaia`:

.. autoclass:: Gaia
    :no-members:

    .. attribute:: include_proper_motion

        Apply proper motion to queried star locations to get the location at the requested time

    .. attribute:: data_release

        The GAIA data release identifier to use when querying the TAP+ service

    .. attribute:: catalog_file

        The path to the HDF5 file containing a local copy of the catalog


.. rubric:: Summary of Methods

.. autosummary::
   :nosignatures:
   :toctree: gaia

   ~Gaia.query_catalog

|
