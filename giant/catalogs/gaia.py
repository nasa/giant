r"""
This module defines the interface to the GAIA star catalog.

Catalog Description
=====================

The GAIA is a (nearly) complete sky survey of over 1.8 billion stars with 1.46 billion including proper motion
solutions.  It is generally very accurate for both positions and magnitudes and is mostly complete between magnitudes
3 and 20.

To access the GAIA catalog, GIANT uses the astroquery TAP+ interface
(https://astroquery.readthedocs.io/en/latest/gaia/gaia.html) to retrieve the information from the web.  Since the
catalog is still in development this is the best way to ensure you have the most current solutions.  Alternatively,
if you need more speed or are working in an environment where you cannot access the web, you can use the function
:func:`.build_catalog` to download a subset of the catalog to a local HDF5 file and then point the class to this file.

Use
===

The GAIA catalog can be used anywhere that a star catalog is required in GIANT.
It is stored on the internet in a TAP+ service which allows querying the results and returning them to the machine.
Since the GAIA catalog is still being revised as more data becomes available, GIANT by default queries to this service
rather than having a local copy of the catalog (also the catalog is huge!!!).  As mentioned previously, if this
doesn't work for you for whatever reason, you can use the :func:`.build_catalog` function to download a local copy of
the catalog in GIANT format and then provide the ``catalog_file`` argument to the class constructor to use this rather
than live queries.

Once you have initialized the class, then you can access the catalog as you would any
GIANT usable catalog.  Simply call :meth:`~.GAIA.query_catalog` to get the GIANT records for the stars as a
DataFrame with columns according the :attr:`GIANT_COLUMNS`.

Note that the epoch for the astrometry solutions in the GAIA catalog changes with each new release.  GIANT tracks
these for you and handles them appropriately, but be aware of this if you are using the catalog data directly.  Also
note that since in GIANT we primarily care about stars for attitude/calibration purposes, we filter out stars from the
GAIA catalog which have questionable solutions or for which there is missing data. If you need access to everything in
the catalog consider the astroquery api discussed previously. For more information about the GAIA catalog refer to
https://www.cosmos.esa.int/web/gaia-users/archive.
"""

import os
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

from pathlib import Path
from datetime import datetime

from typing import Optional, List, Dict, Iterable, Union, Set, Sequence

from itertools import repeat, starmap
from tqdm import tqdm

import pandas as pd
import numpy as np
import time

from astroquery.gaia import Gaia as QGaia

from scipy.spatial import KDTree

from giant.catalogs.meta_catalog import GIANT_COLUMNS, GIANT_TYPES
from giant.catalogs.meta_catalog import Catalog
from giant.catalogs.utilities import (DEG2MAS, PARSEC2KM, AVG_STAR_DIST, DEG2RAD,
                                        apply_proper_motion)

from giant.utilities.spherical_coordinates import radec_to_unit, radec_distance

from giant._typing import PATH, ARRAY_LIKE, DOUBLE_ARRAY

# specify which gaia cols correspond to which GIANT_COLUMNS
# noinspection SpellCheckingInspection
_GAIA_COLS: List[str] = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'phot_g_mean_mag',
                         'ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error', 'ref_epoch',
                         'designation']
"""
This specifies the names of the GAIA columns that are required in converting the a GIANT star record
"""

# specify the mapping of GAIA columns to GIANT columns
_GAIA_TO_GIANT: Dict[str, str] = dict(zip(_GAIA_COLS, GIANT_COLUMNS))
"""
This specifies the mapping of the GAIA column names to the GIANT star record column names
"""

# specify the default GAIA data release to use
GAIA_DR: str = 'gaiadr3'
"""
This specifies the GAIA data release to use when querying the TAP+ service.  Typically this should look like gaiaxxxx
where xxxx is replaced with the data release string (ie dr2, edr3, etc).
"""

# Gaia DR3 has a maximum number of 3 million rows that can be returned for anonymous users
_GAIA_MAX_ROWS = 3_000_000
"""
This specifies the maximum number of rows that will be returned when querying GAIA DR3 using the TAP+ service.
The query will find all matching rows but truncate the return data to just the first 3 million rows. This limit 
only applies to anonymous users, but we can work around it by splitting out queries into chunks so that no query 
exceeds this limit.
"""

# Default path and file name for local downloaded HDF5 database
DEFAULT_CAT_FILE: Path = (Path(__file__).parent / 'data') / 'giant_cat.hdf5'
"""
This specifies the default location of the HDF5 database file that contains the data for this catalog.

This is stored as Path object.  It defaults to a file called "giant_cat.hdf5" in a "data" directory in the same directory 
containing this file. If you wish to use a different catalog, typically you should simply provide a key
word argument to the class constructor instead of modifying this module attribute.
"""


class Gaia(Catalog):
    """
    This class provides access to the GAIA star catalog.

    This class is a fully functional catalog for GIANT and can be used anywhere that GIANT expects a star catalog.
    As such, it implements the :attr:`.include_proper_motion` to turn proper motion on or off as well as the method
    :meth:`.query_catalog`, which is how stars are queried into the GIANT format.

    To use this class simply initialize it, specifying either the data release to use or pointing it to the file of
    the stored catalog (see :func:`.build_catalog` for details).  Once the class is initialized,
    you can query stars from it using :meth:`.query_catalog` which will return a DataFrame of the star records with
    :attr:`GIANT_COLUMNS` columns.
    """

    def __init__(self, data_release: str = GAIA_DR, catalog_file: Optional[PATH] = None,
                 include_proper_motion: bool = True):
        """
        :param data_release: The identifier for the data release to use.  Typically this is of the form gaiaxxxx where
                             xxxx is like dr2, edr3, etc.
        :param catalog_file: A path to the stored catalog in a HDF5 file.  If this is set to ``None`` then the
                               data will be downloaded through the TAP+ service (requiring an internet connection).
        :param include_proper_motion: A boolean flag specifying whether to apply proper motion when retrieving the stars
        """

        super().__init__(include_proper_motion=include_proper_motion)

        self.data_release: str = data_release
        """
        This specifies which data release of the GAIA catalog to use when querying the TAP+ service.
        
        Typically this is of the form gaiaxxxx where xxxx is like dr2, edr3, etc. 
        
        If :attr:`.catalog_file` is not ``None`` then this is ignored.
        """
        
        self.data_release_formatted: str = f'Gaia {data_release.replace("gaia", "").upper()}'
        """
        This represents the GAIA data release formatted as a string for uniformity in print statements.
        """
        
        if catalog_file is not None:
            catalog_file = Path(catalog_file)

            if not catalog_file.exists():
                print('We could not find the GAIA HDF5 file at the specified location. '
                      'Falling back to use the TAP+ online service.')

                catalog_file = None

        self.catalog_file: Optional[PATH] = catalog_file
        """
        The path to the HDF5 file containing the subset of the catalog needed for GIANT.
        
        If ``None`` then the TAP+ online service will be used instead
        """

        self._catalog_store: Optional[pd.HDFStore] = None
        """
        The open HDFStore if we are using a local copy of the catalog
        """
        
        if self.catalog_file is not None:
            try:
                self._catalog_store = pd.HDFStore(self.catalog_file, 'r')
            except FileNotFoundError:
                print(f"Catalog file not found: {self.catalog_file}.  Will use the TAP+ online service instead.")

    def close(self):
        """
        This closes the HDFStore if we are using a local copy of the catalog if it was open.
        
        If you were using the TAP+ online service then this does nothing and is still safe to call.
        """
        try:
            if self._catalog_store is not None:
                self._catalog_store.close()
        except Exception as e:
            pass
        self._catalog_store = None

    def __del__(self):
        self.close()

    def query_catalog(self, ids: Optional[Iterable[str | int] | Set[str | int]] = None, min_ra: float = 0., max_ra: float = 360.,
                        min_dec: float = -90., max_dec: float = 90., min_mag: float = -4., max_mag: float = 14.,
                        search_center: Optional[Sequence[float] | DOUBLE_ARRAY] = None, search_radius: Optional[float] = None,
                        new_epoch: Optional[Union[datetime, float]] = None) -> pd.DataFrame:
        """
        This method queries stars from the catalog that meet specified constraints and returns them as a DataFrame
        with columns of :attr:`GIANT_COLUMNS`.

        Stars can either be queried by ID directly or by right ascension/declination/magnitude. You CANNOT filter using
        both with this method.  If :attr:`.include_proper_motion` is ``True`` then this will shift the stars to the new
        epoch input by the user (``new_epoch``) using proper motion.

        :param ids: A sequence of star ids to retrieve from the catalog.  The ids are given by the index of the
                    returned data frame (the designation column from the actual catalog) and should be
                    input as an iterable that yields either integers or strings in an appropriate format
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
        :param new_epoch: The epoch to translate the stars to using proper motion if :attr:`.include_proper_motion` is
                          turned on
        :return: A pandas DataFrame with columns :attr:`GIANT_COLUMNS`.
        """

        # we can only query by ids or by other attributes, not both
        if ids is not None:
            giant_records = self._get_from_ids(ids, column_subset=_GAIA_COLS)
        else:
            giant_records = self._get_all_with_criteria(min_ra=min_ra, max_ra=max_ra,
                                                        min_dec=min_dec, max_dec=max_dec,
                                                        min_mag=min_mag, max_mag=max_mag,
                                                        search_center=search_center,
                                                        search_radius=search_radius,
                                                        column_subset=_GAIA_COLS)

        # apply proper motion to the giant catalog query if requested based on the new epoch
        if self.include_proper_motion and (new_epoch is not None):
            apply_proper_motion(giant_records, new_epoch, copy=False)

        return giant_records


    def _get_from_ids(self, ids: Iterable[str | int], column_subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        This returns a DataFrame containing the records for each star requested by ``ids``.

        ``ids`` should be iterable with each element being a string giving the "designation"
        for the star as formatted in the GAIA catalog.  This is typically of the form 
        "Gaia DR3 1234567890123456789".  The ``ids`` can also be provided as integer values.

        In general, the user should not interact with this method and instead should use :meth:`.query_catalog`.
        
        :param ids: A sequence of star ids to retrieve from the catalog.  The ids can be given as strings or integers.
        :param column_subset: The subset of columns to retrieve from the TAP+ service (not applicable to the local
                              copy).  If ``None`` then all columns are returned which can take a long time
        :return: A pandas DataFrame formatted according to the GIANT catalog format
        """
        # remove any duplicate IDs
        ids = list(set(ids))
        
        # No local copy of the catalog, so we need to query the TAP+ web service
        if self.catalog_file is None:
            
            # restrict the number of stars that can be queried at once to avoid
            # truncated results from astroquery
            star_count = len(ids)
            if star_count > _GAIA_MAX_ROWS:
                # determine how many chunks the query will be split into
                num_chunks = 1 + star_count // _GAIA_MAX_ROWS
                
                print(f'More than {_GAIA_MAX_ROWS:,} stars match this query (found {star_count:,} stars).\n'
                      f'Breaking the query up into {num_chunks} chunks '
                      f'of {_GAIA_MAX_ROWS:,} IDs to stay within limit.\n')
                
                dataframes: List[pd.DataFrame] = []
                
                for i in range(0, star_count, _GAIA_MAX_ROWS):
                    # split up the IDs into chunks of up to 3 million IDs
                    ids_partition = ids[i:i+_GAIA_MAX_ROWS]
                    
                    # let the user know how many chunks are left so they can gauge progress
                    print(f'\n{num_chunks} chunks left')
                    num_chunks -= 1
                    
                    # recursive call to _get_from_ids using the appropriate chunk of IDs
                    # recursion depth will never exceed 1
                    dataframes.append(self._get_from_ids(ids_partition, column_subset=column_subset))
                
                # merge together results from all subqueries into a single pandas DataFrame
                # (could crash here if attempting to hold more data than the system memory can support)
                return pd.concat(dataframes)
            
            if column_subset is None:
                # if no column subset is specified, query all columns
                column_subset_query = '*'
            else:
                # format the column subset into a string for the query
                column_subset_query = ', '.join(column_subset)
            
            # format the IDs into a string for the query allowing for the input IDs to be integers or strings
            gaia_ids = [f"'{self.data_release_formatted} {str(star_id).rsplit(' ', 1)[-1]}'" for star_id in ids]
            ids_query = ', '.join(gaia_ids)
            
            # format the query to get the stars with the given IDs
            query = (f'SELECT {column_subset_query} '
                     f'FROM {self.data_release}.gaia_source '
                     f'WHERE designation IN ({ids_query})')

            # query the TAP+ web service for the stars with the given IDs
            print(f'Querying online {self.data_release_formatted} catalog...')
            gaia_records = QGaia.launch_job_async(query).get_results()
            assert gaia_records is not None, "something went wrong with TAP+ query"

            # convert the GAIA results to a pandas DataFrame before converting to the GIANT format
            giant_records: pd.DataFrame = self.convert_to_giant_catalog(gaia_records.to_pandas())
           
            return giant_records
            
        else:
            # format the IDs into a list for the query allowing for the input IDs to be integers or strings
            giant_ids: List[int] = [int(str(star_id).rsplit(' ', 1)[-1]) for star_id in ids]
            
            # query the local copy of the catalog for the stars with the given IDs
            assert self._catalog_store is not None, "Catalog store should not be none at this point"
            giant_records: pd.DataFrame = pd.DataFrame(self._catalog_store.select('stars', where=f'index = {giant_ids}'))

            return giant_records

    def _get_all_with_criteria(self, min_ra: float = 0., max_ra: float = 360.,
                               min_dec: float = -90., max_dec: float = 90.,
                               min_mag: float = -4., max_mag: float = 14.,
                               search_center: Optional[Sequence[float] | DOUBLE_ARRAY] = None,
                               search_radius: Optional[float] = None,
                               column_subset: Optional[List[str]] = None,
                               closed_lower_bound: bool = True,
                               progress_bar: tqdm | None = None) -> pd.DataFrame:
        """
        This function gets all stars meeting the criteria from the catalog, yielding the results as a pandas DataFrame.

        In general, the user should not interact with this method and instead should use :meth:`.query_catalog`.

        :param min_ra: The minimum ra bound to query stars from in degrees
        :param max_ra: The maximum ra bound to query stars from in degrees
        :param min_dec: The minimum declination to query stars from in degrees
        :param max_dec: The maximum declination to query stars from in degrees
        :param min_mag: The minimum G band magnitude to query stars from.  Recall that magnitude is inverse
                        (so lower magnitude is a dimmer star)
        :param max_mag: The maximum G band magnitude to query stars from.  Recall that magnitude is inverse
                        (so higher magnitude is a dimmer star)
        :param search_center: The center of a search cone as a ra/dec pair.
        :param search_radius: The radius about the center of the search cone
        :param column_subset: The subset of columns to retrieve from the TAP+ service (not applicable to the
                              local copy).  If ``None`` then all columns are returned which can take a long time
        :param closed_lower_bound: Flag used when partitioning gaia queries by magnitude ranges so that the 
                                   ranges do not overlap.  ``True`` signals that the lower bound inequality is 
                                   ``<=``, and ``False`` signals that the lower bound inequality is ``<``.
        :param progress_bar: A tqdm progress bar object used to update with the status of the query.
        :return: A pandas DataFrame formatted according to the GIANT catalog format
        """
        # make sure everything is a float (a) to validate input and (b) to protect against sql injection attacks
        min_ra = float(min_ra)
        max_ra = float(max_ra)
        min_dec = float(min_dec)
        max_dec = float(max_dec)
        min_mag = float(min_mag)
        max_mag = float(max_mag)
        
        if self.catalog_file is None:
            # redirect all print statements through this function so that they
            # won't disrupt the progress bar if one is provided
            def print_func(text):
                if progress_bar is None:
                    print(text)
                else:
                    progress_bar.write(text)
                    progress_bar.refresh()
            
            if column_subset is None:
                # if no column subset is specified, query all columns
                column_subset_query = '*'
            else:
                # format the column subset into a string for the query
                column_subset_query = ', '.join(column_subset)
            
            # format the query to count the stars matching the given criteria
            count_query = (f'SELECT COUNT(*) '
                           f'FROM {self.data_release}.gaia_source ')
            
            # format the query to get the stars matching the given criteria
            data_query = (f'SELECT {column_subset_query} '
                          f'FROM {self.data_release}.gaia_source ')
            
            # determine whether the lower bound should be inclusive or exclusive for
            # appropriate queries and print statements
            if closed_lower_bound:
                lower_bound = '='
            else:
                lower_bound = ''
            
            # conditional clause to match the given criteria to be applied to both the count and data queries
            where = (f'WHERE ra >= {min_ra} AND ra <= {max_ra} AND dec >= {min_dec} AND dec <= {max_dec} '
                     f'AND phot_g_mean_mag >{lower_bound} {min_mag} AND phot_g_mean_mag <= {max_mag}')
            
            # add conditional clause to match the given search cone if applicable
            if search_center is not None and search_radius is not None:
                where += (f" AND 1 = CONTAINS(POINT('ICRS', ra, dec), "
                          f"CIRCLE('ICRS', {search_center[0]}, {search_center[1]}, {search_radius}))")
            
            print_func(f'{min_mag} <{lower_bound} G magnitude <= {max_mag}:')
            
            # quick lookup for how many stars will match the given query conditionals
            res = QGaia.launch_job(count_query + where).get_results()
            assert res is not None, "something went wrong with the TAP+ query"
            star_count: int = int(res[0][0]) # type: ignore
            
            # restrict the number of stars that can be queried at once to avoid
            # truncated results from astroquery
            if star_count > _GAIA_MAX_ROWS:
                print_func(f'\tMore than {_GAIA_MAX_ROWS} stars match this query.\n'
                           f'\tBreaking the query up into chunks of G magnitude to stay within the limit.')
                
                dataframes: List[pd.DataFrame] = []
                
                # split up queries into chunks of equal magnitude range so that subsequent queries will
                # return 2 million stars assuming uniform distribution of stars across G magnitudes
                # (it's not uniform, but the simplification buys some time before exceeding the 3 million limit)
                number_steps = int(1 + star_count / ((2./3.) * _GAIA_MAX_ROWS))
                mag_steps = np.linspace(min_mag, max_mag, number_steps + 1)
                for i in range(number_steps):
                    current_min_mag = mag_steps[i]
                    current_max_mag = mag_steps[i+1]
                    
                    # only the first query in a series of chunks can have an inclusive lower bound
                    if i > 0:
                        closed_lower_bound = False
                    
                    # recursive call to _get_all_with_criteria in chunks of G magnitude ranges
                    # (should not hit max recursion depth of 1000 before dataframes variable is
                    # holding more data than the system memory can support)
                    dataframes.append(self._get_all_with_criteria(min_ra=min_ra, max_ra=max_ra,
                                                                  min_dec=min_dec, max_dec=max_dec,
                                                                  min_mag=current_min_mag, max_mag=current_max_mag,
                                                                  search_center=search_center,
                                                                  search_radius=search_radius,
                                                                  column_subset=column_subset,
                                                                  closed_lower_bound=closed_lower_bound,
                                                                  progress_bar=progress_bar))
                
                # merge together results from all subqueries into a single pandas DataFrame
                # (could crash here if attempting to hold more data than the system memory can support)
                return pd.concat(dataframes)
            
            print_func(f'\tQuerying online {self.data_release_formatted} catalog...\n')
            
            # Redirect Gaia TAP+ logging output to custom print_func so progress bar won't be disrupted
            buffer = StringIO()
            with redirect_stdout(buffer), redirect_stderr(buffer):
                # query the TAP+ web service for the stars matching the given criteria
                gaia_records = QGaia.launch_job_async(data_query + where).get_results()
            print_func(f'\033[F\t{buffer.getvalue().strip()}') # \033[F moves cursor up one line
            
            assert gaia_records is not None, "something went wrong with the TAP+ query"
            
            # convert the GAIA results to a pandas DataFrame before converting to the GIANT format
            giant_records: pd.DataFrame = self.convert_to_giant_catalog(gaia_records.to_pandas())
           
            return giant_records

        else:
            # determine what the rectangular bounds should look like for the search center/radius
            if search_center is not None and search_radius is not None:
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
                         f'& mag >= {min_mag} & mag <= {max_mag}')

            elif max_ra > 360:
                query = (f'((ra >= {min_ra} & ra <= {360}) | (ra >= {0} & ra <= {max_ra - 360})) '
                         f'& dec >= {min_dec} & dec <= {max_dec} '
                         f'& mag >= {min_mag} & mag <= {max_mag}')

            else:
                query = (f'ra >= {min_ra} & ra <= {max_ra} '
                         f'& dec >= {min_dec} & dec <= {max_dec} '
                         f'& mag >= {min_mag} & mag <= {max_mag}')

            # query the local copy of the catalog for the stars matching the given criteria
            # noinspection PyTypeChecker
            giant_records: pd.DataFrame = self._catalog_store.select('stars', where=query) # type: ignore

            # now do the float radial search if it is needed
            if search_center is not None and search_radius is not None:
                giant_records = giant_records.loc[radec_distance(giant_records.ra * DEG2RAD,
                                                                 giant_records.dec * DEG2RAD,
                                                                 search_center[0] * DEG2RAD,
                                                                 search_center[1] * DEG2RAD)
                                                  <= (search_radius * DEG2RAD)] # type: ignore
            
            return giant_records

    def convert_to_giant_catalog(self, gaia_records: pd.DataFrame) -> pd.DataFrame:
        """
        This method converts records in the catalog format into records in the GIANT format.

        This is done by renaming columns and converting units.

        :param gaia_records: The GAIA records from the TAP+ service as a pandas DataFrame
        :return: The GIANT records as a pandas DataFrame
        """

        # prep the gaia data frame (set the full index)
        gaia_records = gaia_records.assign(source=self.data_release)
        
        # change index (designation column) from "Gaia DR3 {ID}" to be just
        # the ID as an integer type, allowing for faster indexing
        gaia_records = gaia_records.set_index('designation')
        gaia_records.index = gaia_records.index.map(lambda designation:
                                                    int(designation.rsplit(' ', 1)[-1]))
        gaia_records.index.name = 'IDs'

        # don't want the designation label from _GAIA_COLS since that was the index
        giant_records = gaia_records.loc[:, ['source'] + _GAIA_COLS[:-1]]
        
        # rename the columns to match the GIANT format
        giant_records.rename(columns=_GAIA_TO_GIANT, inplace=True)
        
        # convert the columns to the appropriate date types for the GIANT catalog
        giant_records = giant_records.astype(GIANT_TYPES)

        # convert to giant units
        giant_records['distance_sigma'] /= giant_records['distance'] ** 2  # convert parallax std to distance std
        giant_records['distance'] /= 1000  # MAS to arcsecond
        giant_records['distance'] **= -1  # parallax to distance (arcsecond to parsec)
        giant_records['distance'] *= PARSEC2KM  # parsec to kilometers
        giant_records['ra_sigma'] /= DEG2MAS  # to deg
        giant_records['dec_sigma'] /= DEG2MAS  # to deg
        giant_records['ra_proper_motion'] /= DEG2MAS  # MAS/YR to DEG/YR
        giant_records['dec_proper_motion'] /= DEG2MAS  # MAS/YR to DEG/YR
        giant_records['ra_pm_sigma'] /= DEG2MAS  # MAS/YR to DEG/YR
        giant_records['dec_pm_sigma'] /= DEG2MAS  # MAS/YR to DEG/YR
        giant_records['distance_sigma'] *= 1000 * PARSEC2KM  # convert to km

        # fix for stars with no parallax --  The distance standard deviation seems wrong for these
        default_distance_error = 20 / (AVG_STAR_DIST / PARSEC2KM / 1000) ** 2 * PARSEC2KM * 1000
        giant_records.fillna({'distance_sigma': default_distance_error, 'distance': AVG_STAR_DIST}, inplace=True)

        # fix for stars with no proper motion
        giant_records.fillna({'ra_proper_motion': 0, 'dec_proper_motion': 0,
                              'ra_pm_sigma': 0.1, 'dec_pm_sigma': 0.1}, inplace=True)

        # fix for stars where the parallax is invalid
        giant_records.loc[giant_records.distance < 0, 'distance'] = AVG_STAR_DIST
        giant_records.loc[giant_records.distance < 0, 'distance_sigma'] = default_distance_error

        return giant_records


def _download_gaia(save_location: PATH, max_mag: float = 14., gaia_instance: Optional[Gaia] = None):
    """
    This function downloads a portion of the GAIA catalog to an HDF5 table file for faster/offline access.

    To use this function requires an active internet connection as the GAIA source information will be downloaded using
    the TAP+ interface through the ``astroquery`` module.  It may also take up a lot of storage space on your computer
    depending on what magnitude you request.

    This function will download all of the stars from the GAIA catalog up to the specified ``max_mag``
    (corresponding to the G band pass) and store them in an HDF5 file at the requested ``save_location``.
    Only columns required to generate GIANT star records will be downloaded and stored in the file, so don't use this
    if you need other columns for your analysis.

    This will work in chunks of magnitude (starting with a chunk size of 2 from -4 up to the requested ``max_mag``
    and using smaller chunks as the catalog density increases with larger magnitudes) to ensure that the
    queries are efficient and that the max number of rows for a GAIA TAP+ query is not exceeded
    (indicated by ``_GAIA_MAX_ROWS``) by a query since this will result in a truncated result with missing stars.
    Even then, this will likely take a while to run.

    Once you have downloaded the GAIA catalog, you can create a new instance of the :class:`.Gaia` class, providing
    the path to the file you specified for this function (``save_location``) to the key word argument ``catalog_file``

    :param save_location: The location to save the file to as a path like object.  Usually this should end in .h5 or .hdf5
    :param max_mag: The maximum G magnitude to query from the catalog
    :param gaia_instance: An initialized :class:`.Gaia` object to use to do the querying
    """
    # open the HDF5 store file for writing the catalog to
    catalog_store = pd.HDFStore(save_location, 'w')
    
    if gaia_instance is None:
        gaia_instance = Gaia()

    # if we don't have a local copy of the catalog, we need to build one using astroquery
    if gaia_instance._catalog_store is None:
        # query to count the total number of stars that will be downloaded to track progress
        total_count_query = (f'SELECT COUNT(*) '
                             f'FROM {gaia_instance.data_release}.gaia_source '
                             f'WHERE phot_g_mean_mag <= {max_mag}')
        
        qres = QGaia.launch_job(total_count_query).get_results()
        assert qres is not None, "something went wrong with the TAP+ query"
        total_star_count: int = qres[0][0] # type: ignore
        
        # build a progress bar to track the download progress, updating after each
        # query is completed
        progress_bar = tqdm(total=total_star_count, desc=f'Downloading stars with mag <= {max_mag}',
                            unit=' stars', leave=True, dynamic_ncols=True)
        
        mag_step = 2
        current_min_mag = -4. # Gaia DR3 minimum magnitude is 1.73
        current_max_mag = min(current_min_mag + mag_step, max_mag)
    
        first = True
        closed_lower_bound = True
        lower_bound = '='
        while current_min_mag < max_mag:
            # extract the number of stars that would be returned from the next query
            # (will need to rescale magnitude range if the count exceeds the 3 million limit)
            count_query = (f'SELECT COUNT(*) '
                           f'FROM {gaia_instance.data_release}.gaia_source '
                           f'WHERE phot_g_mean_mag >{lower_bound} {current_min_mag} '
                           f'AND phot_g_mean_mag <= {current_max_mag}')
            lqres = QGaia.launch_job(count_query).get_results()
            assert lqres is not None, "something went wrong with the TAP+ query"
            star_count: int = lqres[0][0] # type: ignore
            
            if star_count > _GAIA_MAX_ROWS:
                # split up queries into chunks of equal magnitude range so that subsequent queries will
                # return 2 million stars assuming uniform distribution of stars across G magnitudes
                # (it's not uniform, but the simplification buys some time before exceeding the 3 million limit)
                mag_step *= (2.0 / 3.0) * (_GAIA_MAX_ROWS / star_count)
                current_max_mag = current_min_mag + mag_step
                continue

            # query the TAP+ web service for all the stars with the given magnitude range
            start = time.time()
            giant_records = gaia_instance._get_all_with_criteria(min_mag=current_min_mag,
                                                                 max_mag=current_max_mag,
                                                                 column_subset=_GAIA_COLS,
                                                                 closed_lower_bound=closed_lower_bound,
                                                                 progress_bar=progress_bar)
            stop = time.time()
            
            progress_bar.write(f'\tQuery took {(stop-start):.1f} seconds to retrieve {len(giant_records):,} stars\n')

            # only the first query will have an inclusive lower bound
            if closed_lower_bound:
                closed_lower_bound = False
                lower_bound = ''
            
            # write the queried stars to the HDF5 store, marking mag, ra, and dec as indexable columns
            if not giant_records.empty:
                giant_records.to_hdf(catalog_store, key='stars', append=not(first),
                                     format='table', data_columns=['mag', 'ra', 'dec'])
                first = False
            
            # increment progress bar by the number of stars retrieved
            progress_bar.update(len(giant_records))
            
            # update the magnitude range for the next query
            current_min_mag = current_max_mag
            current_max_mag = min(current_min_mag + mag_step, max_mag)
        
        progress_bar.close()
    
    else:
        # we already have a gaia instance with a local catalog to pull our stars from
        # this will be much faster than using astroquery to download the stars from scratch

        query = f'mag <= {max_mag}'
        
        # ignore stars with negative indices since that indicates they are not
        # from GAIA and are instead created by star blending
        query_filtered = f'{query} & index >= 0'
        
        # query to count the total number of stars that will be extracted to track progress
        total_star_count = gaia_instance._catalog_store.select('stars', columns=['mag'], where=query_filtered).size
        
        # build a progress bar to track the extraction progress, updating after
        # each chunk is completed
        progress_bar = tqdm(total=total_star_count, desc=f'Extracting stars with {query}', unit=' stars', leave=True, dynamic_ncols=True)
        
        
        first = True
        # get all stars from pre-defined GAIA catalog under max_mag limit
        # and store them in a new GAIA catalog. Limit chunk size to 3 million since
        # that would represent approx. 500 MB in memory as a pandas DataFrame
        for chunk in gaia_instance._catalog_store.select('stars', where=query_filtered, chunksize=3_000_000):
            # write the queried stars to the HDF5 store, marking mag, ra, and dec as indexable columns
            chunk.to_hdf(catalog_store, key='stars', append=not(first),
                         format='table', data_columns=['mag', 'ra', 'dec'])
            first = False
            
            # increment progress bar by the number of stars retrieved
            progress_bar.update(len(chunk))
        
        progress_bar.close()

    catalog_store.close()


def build_catalog(catalog_file: Optional[PATH] = None,
                    giant_catalog_file: Optional[PATH] = None,
                    limiting_magnitude: float = 14., blending_magnitude: float = 8.,
                    limiting_separation: float = 0.04, number_of_stars: int = 0):
    """
    Build a local GIANT catalog:
        a) from the GAIA database for faster query times
        b) from a local GIANT catalog for different star blending options

    This function can be used to build a new GIANT catalog (or overwrite an old one) from the GAIA star
    catalog.  Typically a user will not use this directly and instead will use the command line utility
    :mod:`~.scripts.build_catalog`.

    If you want to use this function, you can adjust how (and where) the catalog is built by adjusting the
    arguments.  Note that this will require downloading the GAIA catalog (if it isn't already available).
    In addition, building the catalog can take a long time, so you will want to have a period
    where you can leave this run for a while without interruption to ensure that the catalog is built successfully and
    not corrupted.

    :param catalog_file: The file to save the giant catalog database to
    :param giant_catalog_file: The filepath containing a pre-built local giant catalog from which
                                 to build the new giant catalog from.
    :param limiting_magnitude: The maximum magnitude to include in the catalog
    :param blending_magnitude: The magnitude of the blended star for it to be included as a blended star in the
                               catalog.
    :param limiting_separation: The maximum separation between stars for them to be considered for blending in degrees.
                                Typically this should be set to around the IFOV on the detector you are considering.
    :param number_of_stars: The maximum number of stars that can be blended together in any group.  To turn off star
                            blending, set this to 0.
    """
    # use the default GIANT catalog file path if none is provided
    if catalog_file is None:
        catalog_file = DEFAULT_CAT_FILE
    else:
        catalog_file = Path(catalog_file)

    # ensure the parent directory for the new database file exists
    catalog_file.parent.mkdir(exist_ok=True, parents=True)

    if catalog_file.exists():
        # if the file already exists, ask the user if they want to rename
        # the old file, otherwise overwrite the old file
        user_response = input(f'\nWARNING: GIANT catalog file already exists at:\n{os.path.realpath(catalog_file)}\n\n'
                              f'Would you like to rename this file so that you will have an old '
                              f'version you can go back to?\n(y/n)? ')
        if user_response[:1].lower() == 'n':
            catalog_file.unlink()
        else:
            base, extension = os.path.splitext(catalog_file.as_posix())
            timestamp = os.path.getmtime(catalog_file)
            timestamp_dt = datetime.fromtimestamp(timestamp)
            old_catalog_file = f'{base}_{timestamp_dt:%d%b%YT%H_%M_%S}{extension}'
            catalog_file.rename(old_catalog_file)
            print(f'\nrenamed: {os.path.basename(catalog_file)}\n     to: {os.path.basename(old_catalog_file)}\n')
    
    # determine the magnitude to query from the catalog
    if number_of_stars == 0:
        query_mag = limiting_magnitude
    else:
        blend_mag = blending_magnitude + (2.5 * np.log10(number_of_stars))
        query_mag = max(limiting_magnitude, blend_mag)

    print(f'\nQuery mag <= {query_mag:.1f}\n')
    
    # set up the Gaia instance to query the catalog
    if giant_catalog_file is None:
        gaia_instance = Gaia()
        print(f'Creating database from {gaia_instance.data_release_formatted} '
              f'using astroquery to download. This might take a while...\n')
    else:
        gaia_instance = Gaia(catalog_file=Path(giant_catalog_file))
        print(f'Creating database from local file:\n{os.path.realpath(giant_catalog_file)}\n')
        
    # download/extract the Gaia stars to the new catalog_file
    _download_gaia(catalog_file, max_mag=query_mag, gaia_instance=gaia_instance)
    
    # check if we are allowed to blend stars
    if number_of_stars > 0:
        # use newly created catalog_file as a basis for star blending
        gaia_instance = Gaia(catalog_file=catalog_file)
        
        # WARNING: This line will attempt to loads entire giant catalog into memory as a pandas DataFrame.
        # This could cause the program to crash or significant slowdowns if using a large enough catalog
        gaia_records = gaia_instance.query_catalog(max_mag=query_mag)
        
        # pair the stars based on distance
        print('\nfinding close star pairs')
        pairs: pd.Series = find_star_pairs(gaia_records, limiting_separation)

        # get rid of the old DataFrame for memory reasons
        del gaia_records

        # blend the stars together
        combined_stars: pd.DataFrame = blend_stars(pairs, gaia_instance, limiting_magnitude)

        # close the read-only file IO for the catalog file so we can append to it
        assert gaia_instance._catalog_store is not None, "should never be None at this point"
        gaia_instance._catalog_store.close()
        
        # reopen the file in read/write mode (allows appending to the file), which
        # is more secure than letting gaia_instance use a read & write-capable file IO
        catalog_store = pd.HDFStore(catalog_file, 'r+')
        
        # append new combined stars to the database, marking mag, ra, and dec as indexable columns
        print('adding blended stars to database')
        combined_stars.to_hdf(catalog_store, key='stars', append=True,
                              format='table', data_columns=['mag', 'ra', 'dec'])
        
        # manually closing catalog file IO is safer than letting it close on its own
        catalog_store.close()


def _repair(star_records: pd.DataFrame, pairs: pd.DataFrame) -> pd.Series:
    """
    This helper function combines multiple pairs of stars and gets rid of duplicates

    Don't use this yourself.

    :param star_records: The DataFrame of star records
    :param pairs: The DataFrame specifying groups of stars
    :return: The pandas Series specifying groups of stars after correcting the groupings
    """
    # get the unique right hand sides
    unique_others = pairs.b.unique()

    paired_dict: pd.Series = pairs.groupby('a')['b'].apply(set)
    removes = []
    
    # use a tqdm progress bar to let the user know how many star pairs are being
    # condensed and to give them an idea of how long it will take
    for primary, others in tqdm(paired_dict.items(),
                                desc='Condensing star pairs',
                                total=len(paired_dict)):
        sets = []
        starts = []
        # look for where primary is also paired to another star
        if (primary in unique_others) and (primary not in removes):
            for local_primary, local_others in paired_dict.items():
                if local_primary == primary:
                    continue
                elif primary in local_others:
                    # if the others is a subset of the first group we don't need to do anything
                    if others.issubset(local_others):
                        # should we also remove local_primary here?
                        continue
                    # otherwise store them for use
                    sets.append(local_others)
                    starts.append(local_primary)
            # if anything needs modified
            if starts:
                for local_primary, local_others in zip(starts, sets):
                    if len(starts) == 1:
                        # keep which ever has the higher magnitude on the left hand side and discard the other
                        if star_records.loc[local_primary, 'mag'] < star_records.loc[primary, 'mag']:  # type: ignore
                            local_others.update(others)
                            removes.append(primary)
                        else:
                            others.update(local_others)
                            removes.append(local_primary)
                    else:
                        # keep the brightest magnitude
                        best = primary
                        best_mag = star_records.loc[primary, 'mag']  # type: ignore
                        best_set = others
                        for o, ls in zip(starts, sets):
                            if best_mag > star_records.loc[o, 'mag']:  # type: ignore
                                best_mag = star_records.loc[o, 'mag']
                                best = o
                                best_set = ls
                        # get rid of whichever aren't the brightest and feed it
                        if best != primary:
                            removes.append(primary)
                            best_set.update(others)
                        for local_local_primary, local_local_others in zip(starts, sets):
                            if local_local_primary == best:
                                continue
                            best_set.update(local_local_others)
                            removes.append(local_local_primary)

    # get rid of the bad ones
    unique_pairs: pd.Series = paired_dict.drop(removes)
    
    # let the user know how many unique star pairs were found
    print(f'\nfound {len(unique_pairs)} unique star pairs')
    
    return unique_pairs

def find_star_pairs(star_records: pd.DataFrame, max_separation: float) -> pd.Series:
    """
    This identifies possible star pairs based on separation angle.

    Stars are paired if their max separation is less that the input ``max_separation`` in degrees. This is done by
    creating unit vectors for all of the stars and then doing a pair query using a KDTree.  The pairs are sorted based
    on magnitude so that the first star in each pair is brighter.

    The result of this function will be a DataFrame where the first column "a" is the primary star and the second column
    "b" is a set of stars that should be combined with "a".

    Generally this is not used directly by the user.  Instead see :func:`build_catalog` or script
    :mod:`~.scripts.build_catalog`.

    :param star_records: The DataFrame containing the stars that are to be paired
    :param max_separation: The maximum separation in degrees between stars for them to be paired
    :return: A pandas Series specifying stars to pair together.
    """

    # get the unit vectors
    units = radec_to_unit(star_records.ra.to_numpy() * DEG2RAD, star_records.dec.to_numpy() * DEG2RAD).T

    # build the kdtree
    # noinspection PyArgumentList
    tree = KDTree(units, compact_nodes=False, balanced_tree=False)

    # find the pairs.  Tell pycharm to stop complaining because numpy/scipy don't document right
    # noinspection PyArgumentList,PyUnresolvedReferences
    pairs = tree.query_pairs(np.sin(max_separation * np.pi / 360) * 2, output_type='ndarray')

    # get the pairs
    pairs = pd.DataFrame(star_records.index.to_numpy()[pairs], columns=['a', 'b'])

    # sort the pairs on magnitude
    for pair in pairs.itertuples():

        if star_records.loc[pair.a, 'mag'] > star_records.loc[pair.b, 'mag']:  # type: ignore
            pairs.loc[pair.Index, 'a'] = pair.b
            pairs.loc[pair.Index, 'b'] = pair.a

    # let the user know how many star pairs were found
    print(f'found {len(pairs)} total star pairs\n')
    
    # condense pairs so that stars aren't in multiple pairs
    return _repair(star_records, pairs)

def _blend_stars(star_group: tuple, gaia_instance: Gaia, limiting_mag: float,
                 reference_mag: float) -> Optional[pd.Series]:
    """
    This helper function computes a blended star from the input star indices.

    This function queries the star records from the database for memory reasons (so that we can use multiprocessing).

    The stars are blended into a single record with a combined magnitude, right ascension, declination, and proper
    motion (distance is not considered).  This is based off of an internal note on blending stars that Sean Semper sent.

    :param star_group: The group of stars to be blended
    :param gaia_instance: An initialized :class:`.Gaia` object to use to query the stars in a group by ID
    :param limiting_mag: The magnitude that the blended stars must reach for them to be included
    :param reference_mag: The reference magnitude to use when blending the stars.
    :return: A series with the blended star, or ``None`` if the limiting magnitude wasn't met
    """

    # extract the primary star and the other stars' IDs from the input star_group
    primary_id: int = star_group[0]
    other_ids: Set[int] = star_group[1]

    # get all the IDs that we will need to query
    star_ids: Set[int] = other_ids.union({primary_id})
    star_records: pd.DataFrame = gaia_instance.query_catalog(ids=star_ids)

    # get the primary star from the DataFrame
    primary_star: pd.Series = star_records.loc[primary_id] # type: ignore
    # get the rest of the stars from the DataFrame
    other_stars: pd.DataFrame = star_records[star_records.index.isin(other_ids)]

    # compute the weights for each star
    primary_weight = 1. / (10 ** (0.4 * (primary_star.mag - reference_mag)))
    other_weights = 1. / (10 ** (0.4 * (other_stars.mag.to_numpy() - reference_mag)))

    # compute the combined magnitude
    combined_mag = -2.5 * np.log10((10 ** (-0.4 * primary_star.mag) + 10 ** (-0.4 * other_stars.mag.to_numpy())).sum())

    # if the blended star is too dim stop here
    if combined_mag > limiting_mag:
        return None
    
    # get the brightest star among the other_stars
    brightest_other = other_stars.loc[other_stars.mag.idxmin()]
    
    # determine the reference declination from the brightest star
    if primary_star.mag <= brightest_other.mag:
        ref_dec = np.cos(primary_star.dec*DEG2RAD)
    else:
        # I think this is dead code since _repair() already ensures that
        # the primary star is the brightest star from its group
        ref_dec = np.cos(brightest_other.dec*DEG2RAD)

    # use the primary star Series as a basis for the blended star
    combined_star: pd.Series = primary_star.copy()
    
    # set the blended magnitude
    combined_star.mag = combined_mag
    
    # update the star ID to be negative
    combined_star.name *= -1 # type: ignore

    # compute the combined position
    denominator = primary_weight + other_weights.sum()
    combined_star.ra = (primary_weight * primary_star.ra +
                        (other_weights * other_stars.ra.to_numpy()).sum()) / denominator
    combined_star.dec = (primary_weight * primary_star.dec +
                         (other_weights * other_stars.dec.to_numpy()).sum()) / denominator
    combined_star.ra_proper_motion = (primary_weight * primary_star.ra_proper_motion +
                                      (other_weights * other_stars.ra_proper_motion.to_numpy()).sum()) / denominator
    combined_star.dec_proper_motion = (primary_weight * primary_star.dec_proper_motion +
                                       (other_weights * other_stars.dec_proper_motion.to_numpy()).sum()) / denominator

    return combined_star

def blend_stars(star_groups: pd.Series, gaia_instance: Gaia, limiting_mag: float,
                reference_mag: float = 4.) -> pd.DataFrame:
    """
    Blends groups of stars together into a single "apparent" star as viewed by a camera.

    Star magnitude, right ascension, declination, and proper motion are all blended in the final product.  The blending
    is based off of an internal memo by Sean Semper.

    The star_groups input should provide 2 columns, the first column "a" should provide the primary (brightest) star in each
    group.  The second column "b" should provide a set of all of the stars that are to be blended to each other and "a".
    This is what is returned by :func:`find_star_pairs`.  This function uses the database to retrieve the individual
    star records for memory purposes.

    The blended star is given an ID that is the negative of the brightest star in the group.  The blended stars are
    returned as a pandas DataFrame.

    Typically this is not used directly by the user.  Instead see :func:`build_catalog` or script
    :mod:`.scripts.build_catalog`.

    :param star_groups: The pandas Series specifying the groups of stars to blend
    :param gaia_instance: An initialized :class:`.Gaia` object to use to query the stars in a group by ID
    :param limiting_mag: The limiting magnitude that blended stars must achieve for them to be included
    :param reference_mag: The reference magnitude to use when blending the stars
    :return: The DataFrame of the blended apparent stars
    """

    # get the number of groups we need to blend for reporting and progress tracking
    number_of_groups = len(star_groups)

    print(f'{number_of_groups} stars to blend\n')

    # combine the stars.  Perhaps can use multiprocessing for this
    combined_stars = list(tqdm(starmap(_blend_stars,
                                  zip(star_groups.items(),
                                      repeat(gaia_instance),
                                      repeat(limiting_mag),
                                      repeat(reference_mag))),
                               total=number_of_groups, desc='Blending Stars'))

    # return the DataFrame by concatenating the individual blended Series as
    # columns, and then transposing and converting to the GIANT data types.
    # This will ignore Nones returned when the combined brightness is below the limiting_mag
    return pd.concat(combined_stars, axis=1).T.astype(GIANT_TYPES)
