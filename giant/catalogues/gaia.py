# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


r"""
This module defines the interface to the GAIA star catalogue.

Catalogue Description
=====================

The GAIA is a (nearly) complete sky survey of over 1.8 billion stars with 1.46 billion including proper motion
solutions.  It is generally very accurate for both positions and magnitudes and is mostly complete between magnitudes
3 and 20.

To access the GAIA catalogue, GIANT uses the astroquery interface
(https://astroquery.readthedocs.io/en/latest/gaia/gaia.html) to retrieve the information from the web.  Since the
catalogue is still in development this is the best way to ensure you have the most current solutions.  Alternatively,
if you need more speed or are working in an environment where you cannot access the web, you can use the function
:func:`.download_gaia` to download a subset of the catalogue to a local HDF5 file and then point the class to this file.

Use
===

The GAIA catalogue can be used anywhere that a star catalogue is required in GIANT.
It is stored on the internet in a TAP+ service which allows querying the results and returning them to the machine.
Since the GAIA catalogue is still being revised as more data becomes available, GIANT by default queries to this service
rather than having a local copy of the catalogue (also the catalogue is huge!!!).  As mentioned previously, if this
doesn't work for you for whatever reason, you can use the :func:`.download_gaia` function to download a local copy of
the catalogue and then provide the ``gaia_source_file`` argument to the class constructor to use this rather than live
queries.

Once you have initialized the class, then you can access the catalogue as you would any
GIANT usable catalogue.  Simply call :meth:`~.GAIA.query_catalogue` to get the GIANT records for the stars as a
dataframe with columns according the :attr:`.GIANT_COLUMNS`.

Note that the epoch for the astrometry solutions in the GAIA catalogue changes with each new release.  GIANT tracks
these for you and handles them appropriately, but be aware of this if you are using the catalogue data directly.  Also
note that since in GIANT we primarily care about stars for attitude/calibration purposes, we filter out stars from the
GAIA catalogue which have questionable solutions or for which there is missing data. If you need access to everything in
the catalogue consider the astroquery api discussed previously. For more information about the GAIA catalogue refer to
https://www.cosmos.esa.int/web/gaia-users/archive.
"""

from pathlib import Path
from datetime import datetime

from typing import Optional, List, Dict, Iterable, Union

import pandas as pd

from astroquery.gaia import Gaia as QGaia

from giant.catalogues.meta_catalogue import GIANT_COLUMNS, GIANT_TYPES
from giant.catalogues.meta_catalogue import Catalogue
from giant.catalogues.utilities import (DEG2MAS, PARSEC2KM, STAR_DIST, DEG2RAD,
                                        radec_distance, apply_proper_motion)

from giant._typing import PATH, Real, ARRAY_LIKE

# specify which ucac cols correspond to which GIANT_COLUMNS
# noinspection SpellCheckingInspection
_GAIA_COLS: List[str] = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'phot_g_mean_mag',
                         'ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error', 'ref_epoch',
                         'designation']
"""
This specifies the names of the GAIA columns that are required in converting the a GIANT star record
"""

# specify the mapping of UCAC columns to GIANT columns
_GAIA_TO_GIANT: Dict[str, str] = dict(zip(_GAIA_COLS, GIANT_COLUMNS))
"""
This specifies the mapping of the GAIA column names to the GIANT star record column names
"""

# specify the default GAIA data release to use
GAIA_DR: str = "gaiaedr3"
"""
This specifies the GAIA data release to use when querying the TAP+ service.  Typically this should look like gaiaxxxx
where xxxx is replace with the datarelase string (ie dr2, edr3, etc).
"""


class Gaia(Catalogue):
    """
    This class provides access to the GAIA star catalogue.

    This class is a fully functional catalogue for GIANT and can be used anywhere that GIANT expects a star catalogue.
    As such, it implements the :attr:`include_proper_motion` to turn proper motion on or off as well as the method
    :meth:`query_catalogue` which is how stars are queried into the GIANT format.  In addition, this catalogue provides
    1 additional method :meth:`query_catalogue_raw` to get the raw GAIA records. This method isn't used anywhere by
    GIANT itself, but may be useful if you are doing some advanced analysis.  Note that this method is only really
    useful if you are querying from the TAP+ service, otherwise you will just get a subset of the columns that directly
    correspond to the GIANT columns.

    To use this class simply initialize it, specifying either the data release to use or pointing it to the file of
    the stored catalogue (see :func:`.download_gaia` for details).  Once the class is initialized,
    you can query stars from it using :meth:`query_catalogue` which will return a dataframe of the star records with
    :attr:`.GIANT_COLUMNS` columns.
    """

    def __init__(self, data_release: str = GAIA_DR, catalogue_file: Optional[PATH] = None,
                 include_proper_motion: bool = True):
        """
        :param data_release: The identifier for the data release to use.  Typically this is of the form gaiaxxxx where
                             xxxx is like dr2, edr3, etc.
        :param catalogue_file: A path to the stored catalogue in a HDF5 file.  If this is set to ``None`` then the
                               data will be downloaded through the TAP+ service (requiring an internet connection).
        :param include_proper_motion: A boolean flag specifying whether to apply proper motion when retrieving the stars
        """

        super().__init__(include_proper_motion=include_proper_motion)

        self.data_release: str = data_release
        """
        This specifies which data release of the GAIA catalogue to use when querying the TAP+ service.
        
        Typically this is of the form gaiaxxxx where xxxx is like dr2, edr3, etc. 
       
        If :attr:`.catalogue_file` is not ``None`` then this is ignored.
        """

        if catalogue_file is not None:
            catalogue_file = Path(catalogue_file)

            if not catalogue_file.exists():
                print("We couldn't find the GAIA HDF5 file at the specified location. Falling back to use the TAP+ "
                      "online service.", flush=True)

                catalogue_file = None

        self.catalogue_file: Optional[Path] = catalogue_file
        """
        The path to the HDF5 file containing the subset of the catalogue needed for GIANt.  
        
        If ``None`` then the TAP+ online service will be used instead
        """

        self._catalogue_store: Optional[pd.HDFStore] = None
        """
        The open HDFStore if we are using a local copy of the catalogue
        """

        if self.catalogue_file is not None:
            self._catalogue_store = pd.HDFStore(str(self.catalogue_file), "r")

    def __del__(self):
        if self._catalogue_store is not None:
            self._catalogue_store.close()

    def query_catalogue(self, ids: Optional[ARRAY_LIKE] = None, min_ra: Real = 0, max_ra: Real = 360,
                        min_dec: Real = -90, max_dec: Real = 90, min_mag: Real = -4, max_mag: Real = 20,
                        search_center: Optional[ARRAY_LIKE] = None, search_radius: Optional[Real] = None,
                        new_epoch: Optional[Union[datetime, Real]] = None) -> pd.DataFrame:
        """
        This method queries stars from the catalogue that meet specified constraints and returns them as a DataFrame
        with columns of :attr:`.GIANT_COLUMNS`.

        Stars can either be queried by ID directly or by right ascension/declination/magnitude. You cannot filter using
        both with this method.  If :attr:`apply_proper_motion` is ``True`` then this will shift the stars to the new
        epoch input by the user (``new_epoch``) using proper motion.

        :param ids: A sequence of star ids to retrieve from the catalogue.  The ids are given by the index of the
                    returned data frame (the designation column from the actual catalogue) and should be
                    input as an iterable that yields strings in the appropriate format
        :param min_ra: The minimum ra bound to query stars from in degrees
        :param max_ra: The maximum ra bound to query stars from in degrees
        :param min_dec: The minimum declination to query stars from in degrees
        :param max_dec: The maximum declination to query stars from in degrees
        :param min_mag: The minimum magnitude to query stars from.  Recall that magnitude is inverse (so lower
                        magnitude is a dimmer star)
        :param max_mag: The maximum magnitude to query stars from.  Recall that magnitude is inverse (so higher
                        magnitude is a dimmer star)
        :param search_center: The center of a search cone as a ra/dec pair.
        :param search_radius: The radius about the center of the search cone
        :param new_epoch: The epoch to translate the stars to using proper motion if :attr:`apply_proper_motion` is
                          turned on
        :return: A Pandas dataframe with columns :attr:`GIANT_COLUMNS`.
        """

        cat_recs = self.query_catalogue_raw(ids=ids, min_ra=min_ra, max_ra=max_ra, min_dec=min_dec, max_dec=max_dec,
                                            min_g_mag=min_mag, max_g_mag=max_mag,
                                            search_center=search_center, search_radius=search_radius,
                                            column_subset=_GAIA_COLS)

        giant_record = self.convert_to_giant_catalogue(cat_recs)

        giant_record = giant_record[(giant_record.mag <= max_mag) & (giant_record.mag >= min_mag)]

        if self.include_proper_motion and (new_epoch is not None):
            apply_proper_motion(giant_record, new_epoch, copy=False)

        return giant_record

    def query_catalogue_raw(self, ids: Optional[ARRAY_LIKE] = None, min_ra: Real = 0., max_ra: Real = 360.,
                            min_dec: Real = -90., max_dec: Real = 90.,
                            search_center: Optional[ARRAY_LIKE] = None, search_radius: Optional[Real] = None,
                            max_g_mag: Real = 20., min_g_mag: Real = -1.44,
                            column_subset: Optional[List[str]] = None) -> Union[pd.DataFrame, Iterable[pd.DataFrame]]:
        """
        This method queries stars from the catalogue that meet specified constraints and returns them as a DataFrame
        or as an iterable of dataframes where the columns are the raw catalogue files.

        Stars can either be queried by ID directly or by right ascension/declination/magnitude. You cannot filter using
        both with this method.  This method is not usable by GIANT and it does not apply proper motion.  If you need
        records that are usable by GIANT and with proper motion applied see :meth:`query_catalogue`. For details on what
        the columns are refer to the UCAC4 documentation (can be found online).

        :param ids: A sequence of star ids to retrieve from the catalogue.  The ids are given by zone, rnz and should be
                    input as an iterable that yields tuples (therefore if you have a dataframe you should do
                    ``df.itertuples(false)``
        :param min_ra: The minimum ra bound to query stars from in degrees
        :param max_ra: The maximum ra bound to query stars from in degrees
        :param min_dec: The minimum declination to query stars from in degrees
        :param max_dec: The maximum declination to query stars from in degrees
        :param min_g_mag: The minimum magnitude in the G band to query stars from.  Recall that magnitude is inverse (so
                          lower magnitude is a dimmer star).  The G band is fairly close the visual band
        :param max_g_mag: The maximum magnitude in the G band to query stars from.  Recall that magnitude is inverse (so
                          higher magnitude is a dimmer star).  The G band is fairly close to the visual band.
        :param search_center: The center of a search cone as a ra/dec pair.
        :param search_radius: The radius about the center of the search cone
        :param column_subset: The subset of columns to retrieve from the TAP+ service (not applicable to the local
                              copy).  If ``None`` then all columns are returned which can take a long time
        :return: A Pandas dataframe with columns according to the catalogue columns.
        """

        if ids is not None:
            out = self.get_from_ids(ids, column_subset=column_subset)
        else:
            out = self._get_all_with_criteria(min_ra=min_ra, max_ra=max_ra, min_dec=min_dec, max_dec=max_dec,
                                              search_center=search_center, search_radius=search_radius,
                                              max_g_mag=max_g_mag, min_g_mag=min_g_mag, column_subset=column_subset)

        return out

    def get_from_ids(self, ids: ARRAY_LIKE, column_subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        This returns a dataframe containing the records for each star requested by IDS.

        ``ids`` should be iterable with each element being a string giving the designation for the star (taken from the
        index of the returned dataframes of other methods).

        :param ids: A sequence of star ids to retrieve from the catalogue.  The ids are given by a string.
        :param column_subset: The subset of columns to retrieve from the TAP+ service (not applicable to the local
                              copy).  If ``None`` then all columns are returned which can take a long time
        :return: An Pandas dataframe with columns according to the catalogue columns.
        """
        if self.catalogue_file is None:
            if column_subset is None:
                query = 'SELECT * from {}.gaia_source WHERE designation = "' + '" OR designation = "'.join(ids) + '"'
            else:
                query = ('SELECT ' + ", ".join(column_subset) + ' from {}.gaia_source WHERE designation = "' +
                         '" OR designation = "'.join(ids) + '"')

            QGaia.ROW_LIMIT = len(ids) * 2
            job = QGaia.launch_job_async(query.format(self.data_release))

            return job.get_results().to_pandas().set_index("designation")

        else:
            # noinspection PyTypeChecker
            res: pd.DataFrame = self._catalogue_store.select('stars',
                                                             where='index = "' + '" | index = "'.join(ids) + '"')

            return res

    def _get_all_with_criteria(self, min_ra: Real = 0., max_ra: Real = 360., min_dec: Real = -90., max_dec: Real = 90.,
                               search_center: Optional[ARRAY_LIKE] = None, search_radius: Optional[Real] = None,
                               max_g_mag: Real = 20., min_g_mag: Real = -1.44,
                               column_subset: Optional[List['str']] = None) -> Iterable[pd.DataFrame]:
        """
        This function gets all stars meeting the criteria from the catalogue, yielding the results as DataFrames by
        zone.

        In general, the user should not interact with this method and instead should use :meth:`query_catalogue_raw`.

        :param min_ra: The minimum ra bound to query stars from in degrees
        :param max_ra: The maximum ra bound to query stars from in degrees
        :param min_dec: The minimum declination to query stars from in degrees
        :param max_dec: The maximum declination to query stars from in degrees
        :param min_g_mag: The minimum G band magnitude to query stars from.  Recall that magnitude is inverse (so
                          lower magnitude is a dimmer star)
        :param max_g_mag: The maximum G band magnitude to query stars from.  Recall that magnitude is inverse (so
                          higher magnitude is a dimmer star)
        :param search_center: The center of a search cone as a ra/dec pair.
        :param search_radius: The radius about the center of the search cone
        :param column_subset: The subset of columns to retrieve from the TAP+ service (not applicable to the local
                              copy).  If ``None`` then all columns are returned which can take a long time
        :return: An Iterable of Pandas dataframes with columns according to the catalogue columns.
        """
        # make sure everything is a float (a) to validate input and (b) to protect against sql injection attacks
        min_ra = float(min_ra)
        max_ra = float(max_ra)
        min_dec = float(min_dec)
        max_dec = float(max_dec)
        min_g_mag = float(min_g_mag)
        max_g_mag = float(max_g_mag)

        if self.catalogue_file is None:
            QGaia.ROW_LIMIT = -1
            if column_subset is not None:
                query = 'SELECT ' + ", ".join(column_subset) + ' from {}.gaia_source WHERE '.format(self.data_release)
            else:
                query = 'SELECT * from {}.gaia_source WHERE '.format(self.data_release)

            query = query + f"ra <= {max_ra} AND ra >= {min_ra} AND dec <= {max_dec} AND dec >= {min_dec} AND "
            query = query + f"phot_g_mean_mag <= {max_g_mag} AND phot_g_mean_mag >= {min_g_mag}"

            if search_center is not None:
                query = query + f" AND 1 = CONTAINS(POINT('ICRS', {search_center[0]}, {search_center[1]}), " \
                                f"CIRCLE('ICRS', {search_center[0]}, {search_center[1]}, {search_radius}))"

            return QGaia.launch_job_async(query).get_results().to_pandas().set_index("designation")

        else:
            # determine what the rectangular bounds should look like for the search center/radius
            if search_center is not None:
                min_ra = search_center[0] - search_radius
                max_ra = search_center[0] + search_radius

                min_dec = search_center[1] - search_radius
                max_dec = search_center[1] + search_radius

            # adjust for if we are at a corner case
            if min_dec < -90:
                min_dec = -90
                min_ra = 0
                max_ra = 360

            elif max_dec > 90:
                max_dec = 90
                min_ra = 0
                max_ra = 360

            # in this rare case we just take everything
            if (min_ra < 0) and (max_ra > 360):
                min_ra = 0
                max_ra = 360

            # determine what the query should look like based on the rectangular bounds
            if min_ra < 0:
                query = (f'((ra >= {min_ra + 360} & ra <= {360}) | (ra >= {0} & ra <= {max_ra})) '
                         f'& dec >= {min_dec} & dec <= {max_dec} '
                         f'& phot_g_mean_mag <={max_g_mag} & phot_g_mean_mag >= {min_g_mag}')

            elif max_ra > 360:
                query = (f'((ra >= {min_ra} & ra <= {360}) | (ra >= {0} & ra <= {max_ra - 360})) '
                         f'& dec >= {min_dec} & dec <= {max_dec} '
                         f'& phot_g_mean_mag <={max_g_mag} & phot_g_mean_mag >= {min_g_mag}')

            else:
                query = (f'ra >= {min_ra} & ra <= {max_ra} '
                         f'& dec >= {min_dec} & dec <= {max_dec} '
                         f'& phot_g_mean_mag <={max_g_mag} & phot_g_mean_mag >= {min_g_mag}')

            # noinspection PyTypeChecker
            res: pd.DataFrame = self._catalogue_store.select('stars', where=query)

            # now do the real radial search if it is needed
            if search_center is not None:
                res = res.loc[radec_distance(res.ra * DEG2RAD, res.dec * DEG2RAD,
                                             search_center[0] * DEG2RAD, search_center[1] * DEG2RAD) <=
                              (search_radius * DEG2RAD)]

            return res

    def convert_to_giant_catalogue(self, gaia_records: pd.DataFrame) -> pd.DataFrame:
        """
        This method converts records in the catalogue format into records in the GIANT format.

        This is done by renaming columns and converting units.

        :param gaia_records: The raw records from the catalogue as a pandas DataFrame
        :return: The GIANT records as a Pandas DataFrame
        """

        # prep the gaia data frame (set the full index)
        gaia_records = gaia_records.assign(source=self.data_release)

        # don't want the designation label since thats the index
        records = gaia_records.loc[:, _GAIA_COLS[:-1]]
        records.rename(columns=_GAIA_TO_GIANT, inplace=True)
        records.dtypes.loc[GIANT_COLUMNS] = GIANT_TYPES

        # convert to giant units
        records['distance_sigma'] /= records['distance'] ** 2  # convert parallax std to distance std
        records['distance'] /= 1000  # MAS to arcsecond
        records['distance'] **= -1  # parallax to distance (arcsecond to parsec)
        records['distance'] *= PARSEC2KM  # parsec to kilometers
        records['ra_sigma'] /= DEG2MAS  # to deg
        records['dec_sigma'] /= DEG2MAS  # to deg
        records['ra_proper_motion'] /= DEG2MAS  # MAS/YR to DEG/YR
        records['dec_proper_motion'] /= DEG2MAS  # MAS/YR to DEG/YR
        records['ra_pm_sigma'] /= DEG2MAS  # MAS/YR to DEG/YR
        records['dec_pm_sigma'] /= DEG2MAS  # MAS/YR to DEG/YR
        records['distance_sigma'] *= 1000 * PARSEC2KM  # convert to km

        # fix for stars with no parallax --  The distance standard deviation seems wrong for these
        default_distance_error = 20 / (STAR_DIST / PARSEC2KM / 1000) ** 2 * PARSEC2KM * 1000
        records['distance_sigma'].fillna(value=default_distance_error, inplace=True)
        records['distance'].fillna(STAR_DIST, inplace=True)

        # fix for stars with no proper motion
        records['ra_proper_motion'].fillna(0, inplace=True)
        records['dec_proper_motion'].fillna(0, inplace=True)
        records['ra_pm_sigma'].fillna(0.1, inplace=True)
        records['dec_pm_sigma'].fillna(0.1, inplace=True)

        # fix for stars where the parallax is invalid
        records.loc[records.distance < 0, 'distance'] = STAR_DIST
        records.loc[records.distance < 0, 'distance_sigma'] = default_distance_error

        return records


def download_gaia(save_location: PATH, max_magnitude: float = 16.0, gaia_instance: Optional[Gaia] = None):
    """
    This function downloads a portion of the GAIA catalogue to an HDF5 table file for faster/offline access.

    To use this function requires an active internet connection as the GAIA source information will be downloaded using
    the TAP+ interface through the ``astroquery`` module.  It may also take up a lot of space on your hard drive
    depending on what magnitude you request.

    This function will download all of the stars from the GAIA catalogue up to the specified max_magnitude
    (corresponding to the G band pass) and store them in an HDF5 file at the requested location.  Only columns required
    to generate GIANT star records will be downloaded and stored in the file, so don't use this if you need other
    columns for your analysis.

    This will work in chunks of magnitude (moving 2 magnitude at a time from -6 to the requested max
    magnitude) to ensure that the queries are efficient and don't overwhelm the memory of your computer.  Even then,
    this will likely take a while to run.

    Once you have downloaded the GAIA catalogue, you can create a new instance of the :class:`.Gaia` class, providing
    the path to the file you specified for this function (``save_location``) to the key word argument ``catalogue_file``

    :param save_location: The location to save the file to as a path like object.  Usually this should end in .h5
    :param max_magnitude: The maximum G magnitude to query from the catalogue
    :param gaia_instance: An initialized :class:`.Gaia` object to use to do the querying
    """

    catalogue_store = pd.HDFStore(str(save_location), "w")

    current_min_mag = -6.0 + 1e-10
    current_max_mag = min(current_min_mag + 2, max_magnitude)

    if gaia_instance is None:
        gaia_instance = Gaia()

    first = True
    while current_min_mag < max_magnitude:

        res = gaia_instance.query_catalogue_raw(min_g_mag=current_min_mag, max_g_mag=current_max_mag,
                                                column_subset=_GAIA_COLS)

        if not res.empty:
            res.to_hdf(catalogue_store, "stars", append=~first, format="table",
                       data_columns=["ra", "dec", "phot_g_mean_mag"])
            first = False

        current_min_mag += 2
        current_max_mag = min(current_min_mag+2, max_magnitude)
        print(current_min_mag, flush=True)

    catalogue_store.close()
