


r"""
This module defines the interface to the Tycho 2 star catalog.

Catalog Description
=====================

The Tycho2 is a bright star catalog containing positions, proper motions, and photometry for the 2.5 million brightest
stars in the sky based solely on observations from the Hipparcos satellite.  This corresponds to nearly complete
coverage down to a visual magnitude magnitude of about 11.0.

The Tycho 2 catalog uses a csv text file to store the stars.  It is not very efficient for querying either large or
small numbers of stars.  It also does not include blended stars (stars that are close enough together to appear as a
single source in an image).  If you need faster retrieval and/or blended stars then you should use the
:mod:`.giant_catalog` instead.

For a more thorough description of the Tycho2 star catalog see https://www.cosmos.esa.int/web/hipparcos/tycho-2.

Use
===

The Tycho 2 catalog can be used anywhere that a star catalog is required in GIANT.
It is stored in 3 csv files plus an index csv that takes about 500 MB of disk space.  If you attempt to
initialize the class and point it to a directory that does not contain the Tycho 2 data it will ask you if you want to
download the catalog (note that the Tycho 2 data is not included by default so if you have not downloaded it yourself
you will definitely need to).  If you answer yes, be aware that it may take a very long time to download.

Once you have initialized the class (and downloaded the data files), then you can access the catalog as you would any
GIANT usable catalog.  Simply call :meth:`~.Tycho2.query_catalog` to get the GIANT records for the stars as a
dataframe with columns according the :attr:`.GIANT_COLUMNS`.  This class also provides a helper method,
:meth:`~.Tycho2.query_catalog_raw`, which can be used to retrieve the raw catalog entries (instead of the GIANT
entries).
"""

import os
from pathlib import Path

from io import StringIO

import time

from datetime import datetime

import warnings

from typing import Optional, TextIO, List, Union, Sequence, Iterable

import numpy as np
import pandas as pd

from giant.catalogs.meta_catalog import Catalog, GIANT_TYPES, GIANT_COLUMNS
from giant.catalogs.utilities import DEG2RAD, AVG_STAR_DIST, DEG2MAS, apply_proper_motion, AVG_STAR_DIST_SIGMA
from giant.utilities.spherical_coordinates import radec_distance

from giant._typing import PATH, ARRAY_LIKE, DOUBLE_ARRAY


TYCHO_DIR: Path = Path(__file__).resolve().parent / "data" / "TYCHO2" 
"""
This gives the default location of the Tycho 2 catalog files.

The default location is a directory called "data" in the directory containing this source file.
"""


class Tycho2(Catalog):
    """
    This class provides access to the Tycho 2 star catalog.

    This class is a fully functional catalog for GIANT and can be used anywhere that GIANT expects a star catalog.
    As such, it implements the :attr:`include_proper_motion` to turn proper motion on or off as well as the method
    :meth:`query_catalog` which is how stars are queried into the GIANT format.  In addition, this catalog provides
    1 additional method :meth:`query_catalog_raw` which returns the raw Tycho 2 records for stars instead of the GIANT
    records. This method isn't used anywhere by GIANT itself, but may be useful if you are doing some advanced analysis.

    To use this class simply initialize it, pointing to the directory where the Tycho 2 catalog files index.dat,
    suppl_1.dat, and tyc2.dat are contained.  If the catalog files do not exist it will ask you if you want to
    download them, and if you answer yes, it will download the Tycho 2 catalog (which takes a long time in most
    instances).  Once the class is initialized, you can query stars from it using :meth:`query_catalog` which will
    return a dataframe of the star records with :attr:`.GIANT_COLUMNS` columns.
    """

    def __init__(self, directory: PATH = TYCHO_DIR, include_proper_motion: bool = True):
        """
        :param directory: The directory containing the Tycho 2 catalog files.  This should contain index.dat,
                          suppl_1.dat, and tyc2.dat as csv files.
        :param include_proper_motion: A boolean flag specifying whether to apply proper motion when retrieving the stars
        """

        # call the subclass
        super().__init__(include_proper_motion=include_proper_motion)

        directory = Path(directory)

        self._root: Path = directory
        """
        The root directory where the catalog files are stored
        """

        if not directory.exists():
            print("Tycho data not found at {}".format(directory), flush=True)
            user_response = input("Would you like to download the Tycho data to this directory (y/n)?\n"
                                  "    WARNING: THIS REQUIRES AN INTERNET CONNECTION, WILL TAKE A LONG TIME, AND WILL"
                                  " USE UP 500 MB OF SPACE!\n    ")

            if user_response[:1].lower() == 'y':
                # make sure the directory exists
                directory.mkdir(exist_ok=True, parents=True)
                download_tycho(directory)

            else:
                raise FileNotFoundError('The Tycho data is not available in the specified directory.  Cannot initialize'
                                        'The Tycho2 class.')

        # store the index file for the catalog
        self._index_file: Path = self._root / 'index.dat'
        """
        The index file for the catalog
        """

        if not self._index_file.exists():
            print("Tycho index data missing at {}".format(directory), flush=True)
            user_response = input("Would you like to download the Tycho data to this directory (y/n)?\n"
                                  "    WARNING: THIS REQUIRES AN INTERNET CONNECTION, WILL TAKE A LONG TIME, AND WILL"
                                  " USE UP 500 MB OF SPACE!\n    ")

            if user_response[:1].lower() == 'y':
                download_tycho(directory)
            else:
                raise FileNotFoundError('The Tycho2 2 index file could not be located.\n'
                                        'Please ensure that "index.dat" is in your tycho directory to proceed')

        # build the index in memory
        self._index = pd.read_csv(self._index_file, sep='|', header=None, index_col=False,
                                  names=['mstars', 'sstars', 'minra', 'maxra', 'mindec', 'maxdec'],
                                  dtype={'mstars': np.uint32, 'sstars': np.uint16,
                                         'minra': np.float64, 'maxra': np.float64,
                                         'mindec': np.float64, 'maxdec': np.float64})
        """
        The index specifying where stars are in the catalog
        """

        # store the main data file for the catalog
        try:

            self._main: TextIO = (self._root / 'tyc2.dat').open('r')
            """
            The main file object for the catalog
            """

            # figure out the length of a line
            self._main.readline()
            self._main_line_length = self._main.tell()
            """
            The length of a line in the main table
            """

            self._main.seek(0)

        except FileNotFoundError:
            print("Tycho data missing at {}".format(directory), flush=True)
            user_response = input("Would you like to download the Tycho data to this directory (y/n)?\n"
                                  "    WARNING: THIS REQUIRES AN INTERNET CONNECTION, WILL TAKE A LONG TIME, AND WILL"
                                  " USE UP 500 MB OF SPACE!\n    ")

            if user_response[:1].lower() == 'y':
                download_tycho(directory)
                self._main = (self._root / 'tyc2.dat').open('r')
                self._main.readline()
                self._main_line_length = self._main.tell()
                self._main.seek(0)

            else:

                raise FileNotFoundError('The Tycho2 2 main file could not be located.\n'
                                        'Please ensure that "tyc2.dat" is in your tycho directory to proceed')

        # store supplement 1 for the catalog
        try:

            self._sup1: TextIO = (self._root / 'suppl_1.dat').open('r')
            """
            The first supplement file object 
            """

            self._sup1.readline()
            self._sup1_line_length: int = self._sup1.tell()
            """
            The length of a line in the first supplement table
            """

            self._sup1.seek(0)

        except FileNotFoundError:

            print("Tycho supplement data missing at {}".format(directory), flush=True)
            user_response = input("Would you like to download the Tycho data to this directory (y/n)?\n"
                                  "    WARNING: THIS REQUIRES AN INTERNET CONNECTION, WILL TAKE A LONG TIME, AND WILL"
                                  " USE UP 500 MB OF SPACE!\n    ")

            if user_response[:1].lower() == 'y':
                download_tycho(directory)
                self._sup1 = (self._root / 'suppl_1.dat').open('r')
                self._sup1.readline()
                self._sup1_line_length = self._sup1.tell()
                self._sup1.seek(0)
            else:
                raise FileNotFoundError('The first supplement file could not be located.\n'
                                        'Please ensure that "suppl_1.dat" is in your tycho directory to proceed')

        # store the column names and dtypes for the catalog
        # noinspection SpellCheckingInspection
        self._names: List[str] = ['TYCID',
                                  'pflag',
                                  'RAmdeg', 'DEmdeg', 'pmRA', 'pmDE',
                                  'e_RAmdeg', 'e_DEmdeg', 'e_pmRA', 'e_pmDE',
                                  'EpRAm', 'EpDEm',
                                  'Num',
                                  'q_RAmdeg', 'q_DEmdeg', 'q_pmRA', 'q_pmDE',
                                  'BTmag', 'e_BTmag', 'VTmag', 'e_VTmag',
                                  'prox',
                                  'TYC', 'HIPpCCDM',
                                  'RAdeg', 'DEdeg',
                                  'EpRAm1990', 'EpDEm1990',
                                  'e_RAdeg', 'e_DEdeg',
                                  'posflg',
                                  'corr']
        """
        The names of the columns from the main file
        """

        # noinspection SpellCheckingInspection
        self._sup1_names: List[str] = ['TYCID',
                                       'flag',
                                       'RAdeg', 'DEdeg', 'pmRA', 'pmDE',
                                       'e_RAdeg', 'e_DEdeg', 'e_pmRA', 'e_pmDE',
                                       'mflag',
                                       'BTmag', 'e_BTmag', 'VTmag', 'e_VTmag',
                                       'prox',
                                       'TYC', 'HIPpCCDM']
        """
        The names of the columns from the first supplement file
        """

        # noinspection SpellCheckingInspection
        self._sup1_rename: List[str] = ['TYCID',
                                        'flag',
                                        'RAmdeg', 'DEmdeg', 'pmRA', 'pmDE',
                                        'e_RAmdeg', 'e_DEmdeg', 'e_pmRA', 'e_pmDE',
                                        'mflag',
                                        'BTmag', 'e_BTmag', 'VTmag', 'e_VTmag',
                                        'prox', 'TYC', 'HIPpCCDM']
        """
        The names of the columns that the first supplement records are renamed to so they can be merged with the main 
        catalog files.
        """

        # store the data types for the columns
        # noinspection SpellCheckingInspection
        self._dtypes: List[type] = [np.str_,  # TYC1, TYC2, TYC3
                                    np.str_,  # pflag
                                    np.float64, np.float64,  # RAmdeg, DEmdeg
                                    np.float64, np.float64,  # pmRA, pmDE
                                    np.str_, np.float64,  # e_RAmdeg, e_DEmdeg
                                    np.float64, np.float64,  # e_pmRA, e_pmDE
                                    np.float64, np.float64,  # EpRAm, EpDEm
                                    np.float64,  # Num
                                    np.float64, np.float64,  # q_RAmdeg, q_DEmdeg
                                    np.float64, np.float64,  # q_pmRA, q_pmDE
                                    np.float64, np.float64,  # BTmag, e_BTmag
                                    np.float64, np.float64,  # VTmag, e_VTmag
                                    np.float64,  # prox
                                    np.str_,  # TYC
                                    np.str_,  # HIP, CCDM
                                    np.float64, np.float64,  # RAdeg, DEdeg
                                    np.float64, np.float64,  # EpRA-1990, EpDE-1990
                                    np.float64, np.float64,  # e_RAdeg, e_DEdeg
                                    np.str_,  # posflg
                                    np.float64]  # corr
        """
        This list specifies the types of each column of the main catalog (as raw types)
        """

        # noinspection SpellCheckingInspection
        self._sup1_dtypes = [np.str_,  # TYC1, TYC2, TYC3
                             np.str_,  # flag
                             np.float64, np.float64,  # RAdeg, DEdeg
                             np.float64, np.float64,  # pmRA, pmDE
                             np.float64, np.float64,  # e_RAdeg, e_DEdeg
                             np.float64, np.float64,  # e_pmRA, e_pmDE
                             np.str_,  # mflag
                             np.float64, np.float64,  # BTmag, e_BTmag
                             np.float64, np.float64,  # VTmag, e_VTmag
                             np.float64,  # prox
                             np.str_,  # TYC
                             np.str_]  # HIP, CCDM
        """
        This list specifies the types of each column of the secondary file (as raw types)
        """

    def empty_frame(self) -> pd.DataFrame:
        """
        This simple helper function returns an empty dataframe with the appropriate columns.

        :return: The empty dataframe
        """

        return self._process_results([])

    def nan_frame(self, index: Optional[str] = None) -> pd.DataFrame:
        """
        This simple helper function returns a dataframe with a single NaN filled row.

        The index of the row will either be all 0 or will be the input value

        :return: The nan filled dataframe
        """
        if index is not None:
            return self._process_results([pd.DataFrame([[index] +
                                                        [np.nan if x != np.str_ else '' for x in self._dtypes[1:]]],
                                                       columns=self._names)])
        else:
            return self._process_results([pd.DataFrame([['0 0 0'] +
                                                        [np.nan if x != np.str_ else '' for x in self._dtypes[1:]]],
                                                       columns=self._names)])

    def retrieve_record(self, tycho_id: str) -> pd.DataFrame:
        """
        This method can be used to retrieve a single star by ID from the tycho 2 main catalog or first supplement
        file.

        The star is returned in the raw Tycho catalog format, not in the GIANT format.

        :param tycho_id: The tycho id as a string with each component separated by a space
        :return: The found star record, or a record filled with NaN
        """

        zone = int(tycho_id.split()[0]) - 1

        start = self._index.iloc[zone]
        stop = self._index.iloc[zone + 1]

        self._main.seek((start.mstars - 1) * self._main_line_length, os.SEEK_SET)

        ind = 0

        for line in self._main:

            if tycho_id in line[:12]:
                self._main.seek(0, os.SEEK_SET)

                stream = StringIO(line)

                record = pd.read_csv(stream, sep='|', header=None, index_col=False,
                                     names=self._names, dtype=dict(zip(self._names, self._dtypes)),
                                     na_values=[' ' * length for length in range(20)])

                return self._process_results([record])

            if ind == stop.mstars + 1:
                break

            ind += 1

        self._sup1.seek((start.sstars - 1) * self._sup1_line_length, os.SEEK_SET)

        ind = 0

        for line in self._sup1:

            if tycho_id in line[:12]:
                self._sup1.seek(0, os.SEEK_SET)

                stream = StringIO(line)

                record = pd.read_csv(stream, sep='|', header=None, index_col=False,
                                     names=self._sup1_names, dtype=dict(zip(self._sup1_names, self._sup1_dtypes)),
                                     na_values=[' ' * length for length in range(20)])

                return self._process_results([record], rtype='supp')

            if ind == stop.sstars + 1:
                break

            ind += 1

        warnings.warn('Tycho2 record for star {} not found'.format(tycho_id))

        return self.nan_frame(index=tycho_id)

    def query_catalog(self, ids: Optional[Iterable[str | int]] = None, min_ra: float = 0, max_ra: float = 360,
                        min_dec: float = -90, max_dec: float = 90, min_mag: float = -4, max_mag: float = 20,
                        search_center: Optional[Sequence[float] | DOUBLE_ARRAY] = None, search_radius: Optional[float] = None,
                        new_epoch: Optional[Union[datetime, float]] = None) -> pd.DataFrame:

        """
        This method queries stars from the catalog that meet specified constraints and returns them as a DataFrame
        with columns of :attr:`.GIANT_COLUMNS`.

        Stars can either be queried by ID directly or by right ascension/declination/magnitude. You cannot filter using
        both with this method.  If :attr:`apply_proper_motion` is ``True`` then this will shift the stars to the new
        epoch input by the user (``new_epoch``) using proper motion.

        :param ids: A sequence of star ids to retrieve from the catalog.  The ids are given by zone, rnz and should be
                    input as an iterable that yields tuples (therefore if you have a dataframe you should do
                    ``df.itertuples(false)``
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

        if ids is not None:
            out = []
            for star_id in ids:

                out.append(self.retrieve_record(str(star_id)))

            cat_recs = pd.concat(out)
        else:
            # query the catalog to get the full records
            cat_recs = self._get_all_with_criteria(min_ra=min_ra, max_ra=max_ra, min_dec=min_dec, max_dec=max_dec,
                                                   min_visual_mag=min_mag, max_visual_mag=max_mag,
                                                   search_radius=search_radius, search_center=search_center)

        # drop anything that isn't well known
        cat_recs = cat_recs[~cat_recs.loc[:, 'pmRA':'pmDE'].isnull().any(axis=1)]

        # convert each to the format GIANT expects
        giant_records = self.convert_to_giant_format(cat_recs)

        # apply the proper motion if requested
        if self.include_proper_motion and (new_epoch is not None):
            apply_proper_motion(giant_records, new_epoch, copy=False)

        return giant_records

    def query_catalog_raw(self, ids: Optional[Iterable[str | int]] = None, min_ra: float = 0, max_ra: float = 360,
                            min_dec: float = -90, max_dec: float = 90,
                            min_visual_mag: float = -4, max_visual_mag: float = 20,
                            search_center: Optional[Sequence[float] | DOUBLE_ARRAY] = None,
                            search_radius: Optional[float] = None) -> pd.DataFrame:

        """
        This method queries stars from the catalog that meet specified constraints and returns them as a DataFrame
        where the columns are the raw catalog columns.

        Stars can either be queried by ID directly or by right ascension/declination/magnitude. You cannot filter using
        both with this method.  This method is not usable by GIANT and it does not apply proper motion.  If you need
        records that are usable by GIANT and with proper motion applied see :meth:`query_catalog`. For details on what
        the columns are refer to the Tycho 2 documentation (can be found online).

        :param ids: A sequence of star ids to retrieve from the catalog.  The ids are given by string and should be
                    ``'{TYC1} {TYC2} {TYC3}'`` where ``TYC*`` are the 3 components of the Tycho ID of the star.
        :param min_ra: The minimum ra bound to query stars from in degrees
        :param max_ra: The maximum ra bound to query stars from in degrees
        :param min_dec: The minimum declination to query stars from in degrees
        :param max_dec: The maximum declination to query stars from in degrees
        :param min_visual_mag: The minimum visual magnitude to query stars from.  Recall that magnitude is inverse (so
                               lower magnitude is a dimmer star)
        :param max_visual_mag: The maximum visual magnitude to query stars from.  Recall that magnitude is inverse (so
                               higher magnitude is a dimmer star)
        :param search_center: The center of a search cone as a ra/dec pair.
        :param search_radius: The radius about the center of the search cone
        :return: A Pandas dataframe with the original columns form the star catalog.
        """

        if ids is not None:
            out = []
            for star_id in ids:

                out.append(self.retrieve_record(str(star_id)))

            return pd.concat(out)
        else:
            # query the catalog to get the full records
            return self._get_all_with_criteria(min_ra=min_ra, max_ra=max_ra, min_dec=min_dec, max_dec=max_dec,
                                               min_visual_mag=min_visual_mag, max_visual_mag=max_visual_mag,
                                               search_radius=search_radius, search_center=search_center)

    def _get_all_with_criteria(self, min_ra: float = 0., max_ra: float = 360., min_dec: float = -90., max_dec: float = 90.,
                               search_center: Optional[Sequence[float] | DOUBLE_ARRAY] = None, search_radius: Optional[float] = None,
                               max_visual_mag: float = 20., min_visual_mag: float = -1.44,
                               max_b_mag: float = 20., min_b_mag: float = -1.44,
                               ) -> pd.DataFrame:
        """
        This function gets all stars meeting the criteria from the catalog, yielding the results as DataFrames.

        In general, the user should not interact with this method and instead should use :meth:`query_catalog_raw`.

        :param min_ra: The minimum ra bound to query stars from in degrees
        :param max_ra: The maximum ra bound to query stars from in degrees
        :param min_dec: The minimum declination to query stars from in degrees
        :param max_dec: The maximum declination to query stars from in degrees
        :param min_visual_mag: The minimum visual magnitude to query stars from.  Recall that magnitude is inverse (so
                               lower magnitude is a dimmer star)
        :param max_visual_mag: The maximum visual magnitude to query stars from.  Recall that magnitude is inverse (so
                               higher magnitude is a dimmer star)
        :param min_b_mag: The minimum b magnitude to query stars from.  Recall that magnitude is inverse (so
                          lower magnitude is a dimmer star)
        :param max_b_mag: The maximum b magnitude to query stars from.  Recall that magnitude is inverse (so
                          higher magnitude is a dimmer star)
        :param search_center: The center of a search cone as a ra/dec pair.
        :param search_radius: The radius about the center of the search cone
        :return: An Iterable of Pandas dataframes with columns according to the catalog columns.
        """

        # retrieve the required columns from the index for ease of use
        ind_min_ra = self._index.minra
        ind_max_ra = self._index.maxra
        ind_min_dec = self._index.mindec
        ind_max_dec = self._index.maxdec

        # determine which GSC zones we need to check
        if search_center is not None:
            assert search_radius is not None, "the search radius needs to be specified if the search center is specified"
            min_ra2 = search_center[0] - search_radius
            max_ra2 = search_center[0] + search_radius

            min_dec2 = search_center[1] - search_radius
            max_dec2 = search_center[1] + search_radius

            if min_dec2 < -90:
                # if the search region includes the south pole take all RA
                min_dec2 = -90
                min_ra2 = 0
                max_ra2 = 360

            elif max_dec2 > 90:
                # if the search region includes the north pole take all RA
                max_dec2 = 90
                min_ra2 = 0
                max_ra2 = 360

            # if the first point of ares is included then we need to query both the high and low RA
            if (min_ra2 < 0) and (max_ra2 < 360):
                index_check = ((ind_min_ra <= max_ra) & (ind_max_ra >= min_ra) &
                               (ind_min_dec <= max_dec) & (ind_max_dec >= min_dec)) & \
                              ((((ind_min_ra <= max_ra2) & (ind_max_ra >= 0)) |
                                ((ind_min_ra <= 360) & (ind_max_ra >= min_ra2 + 360))) &
                               (ind_min_dec <= max_dec2) & (ind_max_dec >= min_dec2))

            elif (min_ra2 < 0) and (max_ra2 >= 360):  # if we need the whole ra band
                min_ra2 = 0
                max_ra2 = 360
                index_check = (((ind_min_ra <= max_ra) & (ind_max_ra >= min_ra) &
                               (ind_min_dec <= max_dec) & (ind_max_dec >= min_dec)) & 
                              ((ind_min_ra <= max_ra2) & (ind_max_ra >= min_ra2) &
                               (ind_min_dec <= max_dec2) & (ind_max_dec >= min_dec2)))

            elif max_ra2 >= 360:
                # if the first point of ares is included then we need to query both the high and low RA
                index_check = ((ind_min_ra <= max_ra) & (ind_max_ra >= min_ra) &
                               (ind_min_dec <= max_dec) & (ind_max_dec >= min_dec)) & \
                              ((((ind_min_ra <= 360) & (ind_max_ra >= min_ra2)) |
                                ((ind_min_ra <= max_ra2-360) & (ind_max_ra >= 0))) &
                               (ind_min_dec <= max_dec2) & (ind_max_dec >= min_dec2))
            else:
                # other wise nothing special
                index_check = ((ind_min_ra <= max_ra) & (ind_max_ra >= min_ra) &
                               (ind_min_dec <= max_dec) & (ind_max_dec >= min_dec)) & \
                              ((ind_min_ra <= max_ra2) & (ind_max_ra >= min_ra2) &
                               (ind_min_dec <= max_dec2) & (ind_max_dec >= min_dec2))

        else:

            index_check = ((ind_min_ra < max_ra) & (ind_max_ra > min_ra) &
                           (ind_min_dec < max_dec) & (ind_max_dec > min_dec))

        # ################################################# MAIN FILE ##################################################
        # get the start and stop lines for each zone we need to consider
        start_lines: pd.Series = self._index.loc[index_check].mstars - 1 
        end_lines: pd.Series = self._index.loc[start_lines.index + 1].mstars 

        results = []

        for start, stop in zip(start_lines, end_lines):

            # seek to the proper point in the file
            self._main.seek(start * self._main_line_length, os.SEEK_SET)

            # read the file for the current GSC chunk
            df = pd.read_csv(self._main, sep='|', header=None, index_col=False,
                             names=self._names, dtype=dict(zip(self._names, self._dtypes)),
                             nrows=stop - start + 1, na_values=[' ' * length for length in range(20)])

            # perform comparisons
            visual_check = (df.VTmag >= min_visual_mag) & (df.VTmag <= max_visual_mag)
            b_check = (df.BTmag >= min_b_mag) & (df.BTmag <= max_b_mag)

            test = ((df.RAmdeg >= min_ra) & (df.RAmdeg <= max_ra) & (df.DEmdeg >= min_dec) & (df.DEmdeg <= max_dec) &
                    ((visual_check & b_check) | (df.VTmag.isnull() & b_check) | (df.BTmag.isnull() & visual_check)))

            if search_center is not None:
                assert search_radius is not None, "the search radius must be specified if the search center is specified"
                # check the radial distance if we're doing a cone search
                test &= radec_distance(df.RAmdeg * DEG2RAD, df.DEmdeg * DEG2RAD,
                                       search_center[0] * DEG2RAD, search_center[1] * DEG2RAD) <= (
                                    search_radius * DEG2RAD)

            # check to see if anything met the criteria
            if test.any():
                results.append(df.loc[test])

        # ############################################### SUPPLEMENT FILE ##############################################
        # get the start and stop lines for each zone we need to consider
        start_lines: pd.Series = self._index.loc[index_check].sstars - 1 
        end_lines: pd.Series = self._index.loc[start_lines.index + 1].sstars 

        sup1results = []

        for start, stop in zip(start_lines, end_lines):

            # seek to the proper point in the file
            self._sup1.seek(start * self._sup1_line_length, os.SEEK_SET)

            # if there are no supplement stars to consider
            if (stop - start + 1) == 0:
                continue

            # read the file for the current GSC chunk
            df = pd.read_csv(self._sup1, sep='|', header=None, index_col=False,
                             names=self._sup1_names, dtype=dict(zip(self._sup1_names, self._sup1_dtypes)),
                             nrows=stop - start + 1, na_values=[' ' * length for length in range(20)])

            # perform comparisons
            visual_check = (df.VTmag >= min_visual_mag) & (df.VTmag <= max_visual_mag)
            b_check = (df.BTmag >= min_b_mag) & (df.BTmag <= max_b_mag)

            test = ((df.RAdeg >= min_ra) & (df.RAdeg <= max_ra) & (df.DEdeg >= min_dec) & (df.DEdeg <= max_dec) &
                    ((visual_check & b_check) | (df.VTmag.isnull() & b_check) | (df.BTmag.isnull() & visual_check)))

            if search_center is not None:
                # check the radial distance if we're doing a cone search
                assert search_radius is not None, "the search radius must be specified if the search center is specified"
                test &= radec_distance(df.RAdeg * DEG2RAD, df.DEdeg * DEG2RAD,
                                       search_center[0] * DEG2RAD, search_center[1] * DEG2RAD) <= (
                                    search_radius * DEG2RAD)

            # check to see if anything met the criteria
            if test.any():
                sup1results.append(df.loc[test])

        return pd.concat([self._process_results(results), self._process_results(sup1results, rtype='supp')])

    def _process_results(self, res: List[pd.DataFrame], rtype: str = 'main') -> pd.DataFrame:
        """
        This modifies the star records to use the same format, have the right index, and label whether they are
        main or supplemental stars.

        :param res: The frames to modify and join together
        :param rtype: the type of frames, either main or supp
        :return: The concatenated and modified dataframe
        """

        if res:

            # if we found anything then concatenate all of the results together
            big_df: pd.DataFrame = pd.concat(res) 

            # split the TYC ID column into its components
            # noinspection SpellCheckingInspection
            tycid = big_df['TYCID'].str.split(expand=True)
            big_df['TYC1'] = tycid[0].astype(np.uint16)
            big_df['TYC2'] = tycid[1].astype(np.uint16)
            big_df['TYC3'] = tycid[2].astype(np.uint8)

            if 'supp' in rtype:
                out = big_df.loc[:, 'flag':].set_index(['TYC1', 'TYC2', 'TYC3']).assign(tycho_source='supp')
                return out.rename(columns=dict(zip(self._sup1_names, self._sup1_rename)))
            else:
                # noinspection SpellCheckingInspection
                out = big_df.loc[:, 'pflag':].set_index(['TYC1', 'TYC2', 'TYC3']).assign(tycho_source='main')

                return out

        else:
            return pd.DataFrame(columns=self._names[1:] +
                                ['TYC1', 'TYC2', 'TYC3'] + ['tycho_source']).set_index(['TYC1', 'TYC2', 'TYC3'])

    @staticmethod
    def convert_to_giant_format(tycho_recs):
        """
        This static method converts raw tycho2 records into GIANT records so they can be used in star id.
        
        It mostly consists of updating names of columns, changing units, and creating a couple new columns.
        
        Generally, a user probably won't interact with this directly
        """

        tycho_recs['distance'] = AVG_STAR_DIST
        tycho_recs['distance_sigma'] = AVG_STAR_DIST_SIGMA

        tycho_cols = ['RAmdeg', 'DEmdeg', 'distance', 'pmRA', 'pmDE', 'VTmag', 'e_RAmdeg', 'e_DEmdeg', 'distance_sigma',
                      'e_pmRA', 'e_pmDE']

        records = tycho_recs.loc[:, tycho_cols].rename(columns=dict(zip(tycho_cols, GIANT_COLUMNS)))
        records['epoch'] = 2000

        records = records.astype(GIANT_TYPES)

        # convert to giant units
        records['ra_sigma'] /= DEG2MAS
        records['dec_sigma'] /= DEG2MAS
        records['ra_proper_motion'] /= DEG2MAS
        records['dec_proper_motion'] /= DEG2MAS
        records['ra_pm_sigma'] /= DEG2MAS
        records['dec_pm_sigma'] /= DEG2MAS

        main_stars = tycho_recs['tycho_source'] == 'main'

        # update ra_sigma and dec_sigma to J2000 for the main records
        ra_shift_time = 2000.0 - tycho_recs.loc[main_stars, 'EpRAm']
        dec_shift_time = 2000.0 - tycho_recs.loc[main_stars, 'EpDEm']

        records = records.assign(epoch=2000.0)

        # technically this miscalculates the error for subsequent computations
        records.loc[main_stars, 'ra_sigma'] = np.sqrt(records.loc[main_stars, 'ra_sigma'] ** 2 +
                                                      ra_shift_time ** 2 * records.loc[main_stars, 'ra_pm_sigma'] ** 2)
        records.loc[main_stars, 'dec_sigma'] = np.sqrt(records.loc[main_stars, 'dec_sigma'] ** 2 +
                                                       dec_shift_time ** 2 * records.loc[main_stars, 'dec_pm_sigma'] ** 2)

        # set the epoch for the supplement stars
        records.loc[~main_stars, "epoch"] = 1991.25

        return records


def download_tycho(target_directory: PATH):
    """
    This function downloads the Tycho2 catalog from vizier to the target directory.

    This is done over ftp.  It requires an active internet connection that can connect to cdsarc.u-strasbg.fr

    .. warning::

        This download will take a long time and use up approximately 500 MB of space.

    .. warning::

        This download has no way to verify the integrity of the files because no hash is provided.  While the vizier
        service is trusted, use this function at your own risk

    :param target_directory: the directory to save the Tycho catalog to
    """

    # we minimize the security risk here by using FTPS
    import ftplib  # nosec

    from gzip import decompress
    
    target_directory = Path(target_directory)

    target_directory.mkdir(exist_ok=True, parents=True)

    # FTPS is secure
    ftp = ftplib.FTP_TLS('cdsarc.u-strasbg.fr')  # nosec

    # anonymous login since we're just grabbing data
    ftp.connect()
    ftp.sendcmd('USER anonymous')
    ftp.sendcmd('PASS anonymous@a.com')

    ftp.cwd('pub/cats/I/259/')

    lines = []

    ftp.retrlines('LIST', callback=lines.append)

    tyc_file = target_directory / "tyc2.dat"
    if tyc_file.exists():
        # need to delete this file since we will append to it
        tyc_file.unlink()

    for line in lines:

        # file
        name = line.split()[-1]
        if ".dat" in name:
            start = time.time()

            if 'tyc2' in name:
                local = target_directory / "tyc2.dat"
                mode = "ab"
            else:
                local = target_directory / name.replace('.gz', '')
                mode = 'wb'

            with local.open(mode) as download_file:

                writer = download_file.write

                # noinspection PyTypeChecker,SpellCheckingInspection
                ftp.retrbinary('RETR {}'.format(name), writer)

            print('{} done in {:.3f}'.format(name, time.time()-start), flush=True)

    # there is some risk here because no hash is provided with the files but what can you do?
    print('decompressing the data')
    for file in target_directory.glob('*'):
        with file.open('rb') as out_file:
            decompressed = decompress(out_file.read())
        with file.open('wb') as out_file:
            out_file.write(decompressed)
