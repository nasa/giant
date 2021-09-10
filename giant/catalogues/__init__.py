# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This package provides access to star catalogues for doing stellar OpNav and calibration in GIANT.

Description
-----------

A star catalogue in GIANT is primarily responsible for telling us the location of stars in the inertial frame (at a
specific date), the uncertainty on that position (if available), and the magnitude of the star (or how bright it is).
This data is then packaged into a
`Pandas DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_ with specific
columns from which GIANT can determine this information.

To get this information, we can use an existing star catalogue, like the Tycho 2 and UCAC4 catalogues, or we can use
the provided GIANT catalogue, which is a merging of the Tycho 2 and UCAC4 catalogues into an efficient SQL format.  For
most OpNav scenarios the default GIANT catalogue is sufficient, as it includes stars down to about 18th visual magnitude
and is very efficient for querying data, making stellar OpNav and calibration proceed faster.  In some cases, you may
have other requirements for your catalogue (perhaps dimmer stars or you would like to use a different magnitude for your
stars) in which case you can rebuild the GIANT catalogue using the script :mod:`~.scripts.build_catalogue`.  Just be
aware that rebuilding the catalogue will require you to download the UCAC4 and Tycho 2 catalogues to your computer,
which can take up significant space and can take a long time to download.

Use
---

Star catalogues in GIANT are accessed through a class, which queries the data from wherever it is stored (normally
locally on your machine.)  Typically, you will use the method :meth:`~Catalogue.query_catalogue` with filtering options
for right ascension, declination, and magnitude.  This will then return a pandas dataframe with the requested data with
columns of :attr:`.GIANT_COLUMNS`, which can then be used however you need.  Some catalogues may also provide a method
to return the full dataset for each star (what the full data set it varies from catalogue to catalogue).  You will need
to see the documentation for the particular catalogue you care about if you need this information.

If you need to project the queried stars to get their location on an image, then you can use
:func:`.project_stars_onto_image`, from the :mod:`.catalogues.utilities` package, which will give you the location of
the stars in pixels.

If you want to add a new star catalogue as a source for GIANT, then first reach out to the developers.  We may already
be working on it.  If this doesn't work out, you can also see the :mod:`.meta_catalogue` module documentation for more
details on how to define your own catalogue class.

In addition to the Catalogue classes provided in this package, there is also the :mod:`.catalogues.utilities` module
which provides utilities for unit/epoch/representation conversions and applying proper motion to star tables.  These are
generally useful functions so you may occasionally find yourself using tools from this module as well.
"""

from giant.catalogues.tycho import Tycho2
from giant.catalogues.ucac import UCAC4
from giant.catalogues.giant_catalogue import GIANTCatalogue

__all__ = ['Tycho2', 'UCAC4', 'GIANTCatalogue']
