# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


r"""
This module defines the interface to the UCAC4 star catalogue.

Catalogue Description
=====================

The UCAC4 is a catalogue of 113 million stars with 105 million including proper motion solutions.  It is complete to
about a magnitude of 16.  It is generally very accurate for positions, but less so for magnitudes.

The UCAC 4 uses a file based database system that is fast for small queries, but can be slow for queries over a large
area.  It also does not include blended stars (stars that are close enough that they appear as a single star in an
image).  If you need faster retrieval and/or blended stars then you should use the :mod:`.giant_catalogue` instead,
which is built primarily using the UCAC4 catalogue.

Use
===

The UCAC4 catalogue can be used anywhere that a star catalogue is required in GIANT.
It is stored in a number of index and binary data files on disk and takes about 9 GB of space.  If you attempt to
initialize the class and point it to a directory that does not contain the UCAC4 data it will ask you if you want to
download the catalogue (note that the UCAC4 data is not included by default so if you have not downloaded it yourself
you will definitely need to).  If you answer yes, be aware that it may take a very long time to download.

Once you have initialized the class (and downloaded the data files), then you can access the catalogue as you would any
GIANT usable catalogue.  Simply call :meth:`~.UCAC4.query_catalogue` to get the GIANT records for the stars as a
dataframe with columns according the :attr:`.GIANT_COLUMNS`.  This class also provides 2 helper methods,
:meth:`~.UCAC4.query_catalogue_raw` which can be used to retrieve the raw catalogue entries (instead of the GIANT
entries) and :meth:`~.UCAC4.cross_ref_tycho` which can be used to get the raw Tycho 2 catalogue records for a UCAC4
star (if available).
"""

import os
import warnings
import copy
import time
from pathlib import Path
from datetime import datetime

from enum import Enum

from operator import lt, gt

from sqlite3 import Connection


from typing import Optional, List, Dict, TextIO, Iterable, Union, Tuple, Callable, BinaryIO, Any

import numpy as np

import pandas as pd

from giant.catalogues.meta_catalogue import GIANT_COLUMNS, GIANT_TYPES
from giant.catalogues.meta_catalogue import Catalogue
from giant.catalogues.utilities import (DEG2MAS, MAS2RAD, PARSEC2KM, STAR_DIST, DEG2RAD,
                                        radec_distance, apply_proper_motion)
from giant.catalogues.tycho import Tycho2

from giant._typing import PATH, Real, ARRAY_LIKE


# specify which ucac cols correspond to which GIANT_COLUMNS
# noinspection SpellCheckingInspection
_UCAC_COLS: List[str] = ['ra', 'spd', 'parallax', 'pmra', 'pmdc', 'apasm_v',
                         'sigra', 'sigdc', 'sigpara', 'sigpmr', 'sigpmd']
"""
This specifies the names of the UCAC columns that are required in converting the a GIANT star record
"""

# specify the mapping of UCAC columns to GIANT columns
_UCAC_TO_GIANT: Dict[str, str] = dict(zip(_UCAC_COLS, GIANT_COLUMNS))
"""
This specifies the mapping of the UCAC column names to the GIANT star record column names
"""

# specify the default UCAC4 directory location
UCAC_DIR: Path = Path(__file__).resolve().parent / "data" / "UCAC4"
"""
This specifies the default location of the UCAC4 directory, which is in the data directory contained in the same 
directory containing this source file.
"""


class UCAC4(Catalogue):
    """
    This class provides access to the UCAC4 star catalogue.

    This class is a fully functional catalogue for GIANT and can be used anywhere that GIANT expects a star catalogue.
    As such, it implements the :attr:`include_proper_motion` to turn proper motion on or off as well as the method
    :meth:`query_catalogue` which is how stars are queried into the GIANT format.  In addition, this catalogue provides
    2 additional methods :meth:`cross_ref_tycho` and :meth:`query_catalogue_raw` to get the corresponding Tycho 2 record
    for the input stars or the raw UCAC4 records. These methods aren't used anywhere by GIANT itself, but may be useful
    if you are doing some advanced analysis.

    To use this class simply initialize it, pointing to the directory where the u4b and u4i catalogue directories are
    contained.  If the catalogue files do not exist it will ask you if you want to download them, and if you answer yes,
    it will download the UCAC4 catalogue (which takes a long time in most instances).  Once the class is initialized,
    you can query stars from it using :meth:`query_catalogue` which will return a dataframe of the star records with
    :attr:`.GIANT_COLUMNS` columns.
    """

    def __init__(self, directory: PATH = UCAC_DIR, include_proper_motion: bool = True):
        """
        :param directory: The directory containing the UCAC4 data.  This should contain 2 sub directories u4i and u4b.
        :param include_proper_motion: A boolean flag specifying whether to apply proper motion when retrieving the stars
        """

        super().__init__(include_proper_motion=include_proper_motion)

        directory = Path(directory)

        if not directory.exists():
            print("UCAC data not found at {}".format(directory), flush=True)
            user_response = input("Would you like to download the UCAC data to this directory (y/n)?\n"
                                  "    WARNING: THIS REQUIRES AN INTERNET CONNECTION, WILL TAKE A LONG TIME, AND WILL"
                                  " USE UP 9 GB OF SPACE!\n    ")

            if user_response[:1].lower() == 'y':
                download_ucac(directory)

            else:
                raise FileNotFoundError('The UCAC data is not available in the specified directory.  Cannot initialize'
                                        'The UCAC4 class.')

        self.root_directory: Path = directory
        """
        The root directory containing the u4i and u4b catalogue file
        """

        self.bytes_per_rec = 78
        """
        The number of bytes per record in the binary files according to the UCAC4 documentation
        """

        self._index_file: Path = self.root_directory / 'u4i' / 'u4index.asc'
        """
        The index file to use to index into the binary files
        """

        if not self._index_file.is_file():
            print("UCAC index file not found at {}".format(directory), flush=True)
            user_response = input("Would you like to download the UCAC data to this directory (y/n)?\n"
                                  "    WARNING: THIS REQUIRES AN INTERNET CONNECTION, WILL TAKE A LONG TIME, AND WILL"
                                  " USE UP 9 GB OF SPACE!\n    ")

            if user_response[:1].lower() == 'y':
                download_ucac(directory)

            else:
                raise FileNotFoundError("We can't find the index file for the database.  This is a required file\n"
                                        "that should be in the u4i directory in the UCAC directory. The file should\n"
                                        "be called u4index.asc.  Please locate this file and place it in the proper \n"
                                        "directory.  Please also verify the rest of your UCAC directory.")

        self._zone_files: List[Path] = [(self.root_directory / 'u4b' / 'z{0:0>3d}'.format(zone))
                                        for zone in np.arange(1, 901)]
        """
        The list of zone file objects that allow reading the data
        """

        for file in self._zone_files:
            if not file.exists():

                print("UCAC zone files not found at {}".format(directory), flush=True)
                user_response = input("Would you like to download the UCAC data to this directory (y/n)?\n"
                                      "\tWARNING: THIS REQUIRES AN INTERNET CONNECTION, WILL TAKE A LONG TIME, AND WILL"
                                      " USE UP 9 GB OF SPACE!\n    ")

                if user_response[:1].lower() == 'y':
                    download_ucac(directory)

                else:
                    raise FileNotFoundError("We can't find  the zone files for the database. These are required files"
                                            "that should be in the u4b directory in the UCAC directory. The files "
                                            "should be called Z### from 001 to 900. Please locate these files and place"
                                            " them in the proper directory.  Please also verify the rest of your UCAC "
                                            "directory.")
                break

        try:
            # noinspection SpellCheckingInspection
            self._supplement_file: TextIO = (self.root_directory / 'u4i' / 'u4supl.dat').open('r')
            """
            The supplement file object
            """

        except FileNotFoundError:
            print("UCAC supplement file not found at {}".format(directory), flush=True)
            user_response = input("Would you like to download the UCAC data to this directory (y/n)?\n"
                                  "    WARNING: THIS REQUIRES AN INTERNET CONNECTION, WILL TAKE A LONG TIME, AND WILL"
                                  " USE UP 9 GB OF SPACE!\n    ")

            if user_response[:1].lower() == 'y':
                download_ucac(directory)
                # noinspection SpellCheckingInspection
                self._supplement_file = (self.root_directory / 'u4i' / 'u4supl.dat').open('r')

            else:
                # noinspection SpellCheckingInspection
                raise FileNotFoundError("We can't find the supplement file for the database. This is a required file\n"
                                        "that should be in the u4i directory in the UCAC4 directory. The file should\n"
                                        "be called u4supl.dat.  Please locate this file and place it in \n"
                                        "the proper directory. Please also verify the rest of your directory directory."
                                        )

        try:
            self._hpm_file: BinaryIO = (self.root_directory / 'u4i' / 'u4hpm.dat').open('rb')
            """
            The high proper motion supplement file.
            
            This is a text file which contains the proper motions for high proper motion stars
            """

        except FileNotFoundError:
            print("UCAC high proper motion file not found at {}".format(directory), flush=True)
            user_response = input("Would you like to download the UCAC data to this directory (y/n)?\n"
                                  "    WARNING: THIS REQUIRES AN INTERNET CONNECTION, WILL TAKE A LONG TIME, AND WILL"
                                  " USE UP 9 GB OF SPACE!\n    ")

            if user_response[:1].lower() == 'y':
                download_ucac(directory)
                self._hpm_file = (self.root_directory / 'u4i' / 'u4hpm.dat').open('rb')

            else:
                raise FileNotFoundError("We can't find the high proper motion file.  This is \n"
                                        "a required file that should be in the u4i directory in the UCAC4 directory.\n"
                                        "The file should be called u4hpm.dat. Please locate this file and place it in\n"
                                        "the proper directory. Please also verify the rest of your directory directory."
                                        )

        try:
            # noinspection SpellCheckingInspection
            self._hippo_file: BinaryIO = (self.root_directory / 'u4i' / 'hipsupl.dat').open('rb')
            """
            The Hipparcos cross reference supplement file.
            
            This contains bright stars included in the hipparcos catalogue, including parallax
            """

        except FileNotFoundError:
            print("UCAC hipparcos supplement file not found at {}".format(directory), flush=True)
            user_response = input("Would you like to download the UCAC data to this directory (y/n)?\n"
                                  "    WARNING: THIS REQUIRES AN INTERNET CONNECTION, WILL TAKE A LONG TIME, AND WILL"
                                  " USE UP 9 GB OF SPACE!\n    ")

            if user_response[:1].lower() == 'y':
                download_ucac(directory)
                # noinspection SpellCheckingInspection
                self._hippo_file = (self.root_directory / 'u4i' / 'hipsupl.dat').open('rb')

            else:
                # noinspection SpellCheckingInspection
                raise FileNotFoundError("We can't find the supplement file containing the hipparcos data.  This is \n"
                                        "a required file that should be in the u4i directory in the UCAC4 directory.\n"
                                        "The file should be hipsupl.dat. Please locate this file and place it in\n"
                                        "the proper directory. Please also verify the rest of your directory directory."
                                        )

        try:
            self._tycho_cross_file: BinaryIO = (self.root_directory / 'u4i' / 'u4xtycho').open('rb')
            """
            This is the Tycho 2 cross reference file, which maps UCAC4 ids to Tycho 2 ids.
            """

        except FileNotFoundError:
            print("UCAC tycho supplement file not found at {}".format(directory), flush=True)
            user_response = input("Would you like to download the UCAC data to this directory (y/n)?\n"
                                  "    WARNING: THIS REQUIRES AN INTERNET CONNECTION, WILL TAKE A LONG TIME, AND WILL"
                                  " USE UP 9 GB OF SPACE!\n    ")

            if user_response[:1].lower() == 'y':
                download_ucac(directory)
                self._tycho_cross_file = (self.root_directory / 'u4i' / 'u4xtycho').open('rb')

            else:
                warnings.warn('Unable to find the Tycho2 2 cross reference file. '
                              'This is required to get a true complete \n'
                              'catalogue. Proceeding without the tycho catalogue '
                              'but be aware that your results will be \n'
                              'incomplete for bright stars. To use the Tycho2 catalogue please locate the u4xtycho file'
                              'and place it in the u4i directory.')

        # noinspection SpellCheckingInspection
        self.cat_names: List[str] = ['ra', 'spd', 'magm', 'maga', 'sigmag', 'objt', 'cdf',  # 15 bytes
                                     'sigra', 'sigdc', 'na1', 'nu1', 'cu1',  # 5 bytes
                                     'cepra', 'cepdc', 'pmra', 'pmdc', 'sigpmr', 'sigpmd',  # 10 bytes
                                     'pts_key', 'j_m', 'h_m', 'k_m', 'icqflg_j', 'icqflg_h', 'icqflg_k', 'e2mpho_j',
                                     'e2mpho_h', 'e2mpho_k',  # 16 bytes
                                     'apasm_b', 'apasm_v', 'apasm_g', 'apasm_r', 'apasm_i', 'apase_b', 'apase_v',
                                     'apase_g', 'apase_r', 'apase_i', 'gcflg',  # 16 bytes
                                     'icf',  # 4 bytes
                                     'leda', 'x2m', 'rnm', 'zn2', 'rn2']  # 12 bytes
        """
        This is a list of the names of the catalogue star record columns according to the UCAC4 documentation.
        """

        self.cat_formats: List[str] = ['i4', 'i4', 'i2', 'i2', 'i1', 'i1', 'i1',  # 15 bytes
                                       'i1', 'i1', 'i1', 'i1', 'i1',  # 5 bytes
                                       'i2', 'i2', 'i2', 'i2', 'i1', 'i1',  # 10 bytes
                                       'i4', 'i2', 'i2', 'i2', 'i1', 'i1', 'i1', 'i1', 'i1', 'i1',  # 16 bytes
                                       'i2', 'i2', 'i2', 'i2', 'i2', 'i1', 'i1', 'i1', 'i1', 'i1', 'i1',  # 16 bytes
                                       'i4',  # 4 bytes
                                       'i1', 'i1', 'i4', 'i2', 'i4']  # 12 bytes
        """
        This is a list of the types for the raw UCAC4 star record columns according to the UCAC4 documentation
        
        The types are numpy dtype strings
        """

        self.cat_dtype: np.dtype = np.dtype({'names': self.cat_names, 'formats': self.cat_formats})
        """
        This is the numpy datatype that represents a record in the catalogue binary files.
        
        This is used when accessing the binary files as memory maps.
        """

        self.hpm_formats = copy.copy(self.cat_formats)
        """
        This is a list of the format strings for the high proper stars.  
        
        It is the mostly same as the regular table since the stars are included in the main catalogue but need data 
        from an extra table.
        """

        # these are the 2 columns that need a slightly different format
        new_hpm_format_inds = [14, 15]

        for ind in new_hpm_format_inds:
            self.hpm_formats[ind] = 'i8'

        self.hpm_dtype: np.dtype = np.dtype({'names': self.cat_names, 'formats': self.hpm_formats})
        """
        This specifies the numpy datatype used to represent the high proper motion stars.
        """


        self.index: Optional[np.ndarray] = None
        """
        This contains the index into the catalogue that is used to efficiently retrieve stars.
        
        The index is stored as a numpy structured array.  It gives the number of stars for each block in each zone file.
        """

        # make the index
        self.build_index()

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

        :param ids: A sequence of star ids to retrieve from the catalogue.  The ids are given by zone, rnz and should be
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

        cat_recs = self.query_catalogue_raw(ids=ids, min_ra=min_ra, max_ra=max_ra, min_dec=min_dec, max_dec=max_dec,
                                            min_visual_mag=min_mag, max_visual_mag=max_mag,
                                            search_center=search_center, search_radius=search_radius,
                                            generator=False)

        giant_record = self.convert_to_giant_catalogue(cat_recs)

        giant_record = giant_record[(giant_record.mag <= max_mag) & (giant_record.mag >= min_mag)]

        if self.include_proper_motion and (new_epoch is not None):
            apply_proper_motion(giant_record, new_epoch, copy=False)

        return giant_record

    def query_catalogue_raw(self, ids: Optional[ARRAY_LIKE] = None, min_ra: Real = 0., max_ra: Real = 360.,
                            min_dec: Real = -90., max_dec: Real = 90.,
                            search_center: Optional[ARRAY_LIKE] = None, search_radius: Optional[Real] = None,
                            max_visual_mag: Real = 20., min_visual_mag: Real = -1.44,
                            generator: bool = False) -> Union[pd.DataFrame, Iterable[pd.DataFrame]]:
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
        :param min_visual_mag: The minimum visual magnitude to query stars from.  Recall that magnitude is inverse (so
                               lower magnitude is a dimmer star)
        :param max_visual_mag: The maximum visual magnitude to query stars from.  Recall that magnitude is inverse (so
                               higher magnitude is a dimmer star)
        :param search_center: The center of a search cone as a ra/dec pair.
        :param search_radius: The radius about the center of the search cone
        :param generator: A boolean flag specifying whether to return the stars as a generator (preserving memory) or to
                          to collect all of the stars into a single dataframe.  If ``True`` then the function will yield
                          dataframes (which may themselves contain single or multiple stars).
        :return: A (Iterable of) Pandas dataframe(s) with columns according to the catalogue columns.
        """

        if ids is not None:
            out = self.get_from_ids(ids)
        else:
            out = self._get_all_with_criteria(min_ra=min_ra, max_ra=max_ra, min_dec=min_dec, max_dec=max_dec,
                                              search_center=search_center, search_radius=search_radius,
                                              max_visual_mag=max_visual_mag, min_visual_mag=min_visual_mag)

        if generator is False:
            results = []
            for res in out:
                results.append(res)

            if results:
                return pd.concat(results)

            else:
                # noinspection SpellCheckingInspection
                return pd.DataFrame(columns=self.cat_names + ['zone', 'rnz', 'parallax', 'sigpara'])

        else:
            return out

    def get_from_ids(self, ids: ARRAY_LIKE) -> Iterable[pd.DataFrame]:
        """
        This is a generator which returns single records for each requested ID.

        ``ids`` should be iterable with each element being length 2 and returning zone, rnz.

        The yields will be the raw catalogue data.

        :param ids: A sequence of star ids to retrieve from the catalogue.  The ids are given by zone, rnz and should be
                    input as an iterable that yields tuples (therefore if you have a dataframe you should do
                    ``df.itertuples(false)``
        :return: An Iterable of Pandas dataframes with columns according to the catalogue columns.
        """
        for zone, rnz in ids:
            offset = (rnz - 1) * self.bytes_per_rec

            records = pd.DataFrame.from_records(np.fromfile(self._zone_files[zone - 1], dtype=self.cat_dtype,
                                                            offset=offset, count=1).astype(self.hpm_dtype),
                                                index='rnm')
            # append new columns
            records = records.assign(zone=zone, rnz=rnz, parallax=0., sigpara=0.)
            yield records

    def _get_all_with_criteria(self, min_ra: Real = 0., max_ra: Real = 360., min_dec: Real = -90., max_dec: Real = 90.,
                               search_center: Optional[ARRAY_LIKE] = None, search_radius: Optional[Real] = None,
                               max_visual_mag: Real = 20., min_visual_mag: Real = -1.44) -> Iterable[pd.DataFrame]:
        """
        This function gets all stars meeting the criteria from the catalogue, yielding the results as DataFrames by
        zone.

        In general, the user should not interact with this method and instead should use :meth:`query_catalogue_raw`.

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
        :return: An Iterable of Pandas dataframes with columns according to the catalogue columns.
        """

        if search_center is not None:
            # determine which zone files we need to check

            # find the bounds based on the specified min ra/dec and the search center stuff
            min_ra = float(max(search_center[0] - search_radius, min_ra))
            max_ra = float(min(search_center[0] + search_radius, max_ra))

            min_dec = float(max(search_center[1] - search_radius, min_dec))
            max_dec = float(min(search_center[1] + search_radius, max_dec))

            if min_dec < -90.0:
                # if the search region includes the south pole take all RA
                min_dec = -90
                min_ra = 0
                max_ra = 360

            elif max_dec > 90.0:
                # if the search region includes the north pole take all RA
                max_dec = 90
                min_ra = 0
                max_ra = 360

            if min_ra < 0:
                # if the first point of ares is included then we need to query both the high and low RA
                min_ra += 360
                # check the upper bounds
                zone_start_a, block_start_a = self.get_zone_block(min_ra, min_dec)
                # check the lower bounds
                zone_start_b, block_start_b = self.get_zone_block(0, min_dec)
                # determine the limiting case
                zone_start = min(zone_start_a, zone_start_b)
                block_start = min(block_start_a, block_start_b)

                # check the upper bounds
                zone_end_a, block_end_a = self.get_zone_block(360, max_dec)
                # check the lower bounds
                zone_end_b, block_end_b = self.get_zone_block(max_ra, max_dec)
                # determine the limiting case
                zone_end = max(zone_end_a, zone_end_b)
                block_end = max(block_end_a, block_end_b)

            elif max_ra > 360:
                # if the first point of ares is included then we need to query both the high and low RA
                max_ra -= 360
                # check the upper bounds
                zone_start_a, block_start_a = self.get_zone_block(min_ra, min_dec)
                # check the lower bounds
                zone_start_b, block_start_b = self.get_zone_block(0, min_dec)
                # determine the limiting case
                zone_start = min(zone_start_a, zone_start_b)
                block_start = min(block_start_a, block_start_b)

                # check the upper bounds
                zone_end_a, block_end_a = self.get_zone_block(360, max_dec)
                # check the lower bounds
                zone_end_b, block_end_b = self.get_zone_block(max_ra, max_dec)
                # determine the limiting case
                zone_end = max(zone_end_a, zone_end_b)
                block_end = max(block_end_a, block_end_b)

            else:
                # Nothing special needs to happen because we are in the ra domain
                zone_start, block_start = self.get_zone_block(min_ra, min_dec)
                zone_end, block_end = self.get_zone_block(max_ra, max_dec)

        else:
            # if we don't have to deal with the search radius stuff then just use what the user gave us
            zone_start, block_start = self.get_zone_block(float(min_ra), float(min_dec))
            zone_end, block_end = self.get_zone_block(float(max_ra), float(max_dec))

        # walk through each zone that we need to search
        for zone in np.arange(zone_start, zone_end + 1):

            # Get the beginning and ending record number in the current file
            start, _ = self.index[self.get_index_ind(zone, block_start)]
            end_start, end_num = self.index[self.get_index_ind(zone, block_end)]

            number_of_stars = end_start + end_num - start

            # Get the starting byte for the first record we need from this file and seek to it
            start_byte = self.bytes_per_rec * start

            # read in the correct rows.  First access as a memmap to conserve memory while we further limit the
            # results
            # noinspection PyTypeChecker
            data = np.memmap(self._zone_files[zone - 1], dtype=self.cat_dtype, mode='r',
                             offset=start_byte, shape=(number_of_stars,))

            # begin to further limit the results
            if search_center is not None:
                bearing_check = radec_distance(data['ra'] * MAS2RAD, data['spd'] * MAS2RAD - np.pi / 2,
                                               search_center[0] * DEG2RAD,
                                               search_center[1] * DEG2RAD) <= (search_radius * DEG2RAD)
            else:
                bearing_check = (data['ra'] >= min_ra * DEG2MAS) & (data['ra'] <= max_ra * DEG2MAS) & \
                                (data['spd'] >= (min_dec + 90) * DEG2MAS) & (data['spd'] <= (max_dec + 90) * DEG2MAS)

            # limit the magnitude based on the APASS V magnitude if available.
            # If it is not then default to the ucac magnitude but that is crappy..
            mag_check = (
                                (data['apasm_v'] <= max_visual_mag * 1000) &
                                (data['apasm_v'] >= min_visual_mag * 1000)
                        ) | (
                                (data['magm'] <= max_visual_mag * 1000) &
                                (data['magm'] >= min_visual_mag * 1000)
                        )

            valid_data = bearing_check & mag_check

            if not valid_data.any():
                continue

            records = pd.DataFrame.from_records(np.array(data[valid_data]).astype(self.hpm_dtype),
                                                index='rnm')

            # append new columns
            records = records.assign(zone=zone, rnz=np.where(valid_data)[0] + 1 + start,
                                     parallax=0., sigpara=0.)

            # check to see if hpm stars were encountered
            # noinspection SpellCheckingInspection
            hpm_check = (records["pmra"] == 32767)

            if hpm_check.any():

                hpm_records = records[hpm_check].copy()

                for rnm in hpm_records.index:
                    # should probably just store this in memory instead of doing a binary
                    hpm_line = binary_search(self._hpm_file, rnm).decode().split()

                    # noinspection SpellCheckingInspection
                    hpm_records.loc[rnm, 'pmra'] = int(hpm_line[3])
                    # noinspection SpellCheckingInspection
                    hpm_records.loc[rnm, 'pmdc'] = int(hpm_line[4])

                records[hpm_check] = hpm_records

            # check to see if any hipparcos stars are included
            hip_flag = records['icf'].values // (10 ** 8)
            hip_check = (hip_flag == 7) | (hip_flag == 9)

            if hip_check.any():
                running_numbers = records[hip_check].index

                # get the information about each hipparcos star and add it to the data
                for rnm in running_numbers:
                    hip_rec = binary_search(self._hippo_file, rnm).decode().split()

                    records.loc[rnm, 'parallax'] = float(hip_rec[8])
                    # noinspection SpellCheckingInspection
                    records.loc[rnm, 'sigpara'] = float(hip_rec[13])

            yield records

    def build_index(self):
        """
        This method builds the in memory index into the catalogue files.

        The index is stored in the :attr:`index` attribute as a numpy array.  It specifies the number of stars in each
        block in each zone to make it easy to seek to the right place.
        """

        # noinspection SpellCheckingInspection
        self.index = np.genfromtxt(self._index_file, dtype=[('start', np.uint32), ('nstars', np.uint16)],
                                   usecols=[0, 1])

    @staticmethod
    def get_zone_block(ra: Real, dec: Real) -> Tuple[int, int]:
        """
        This tells you the zone and block number that correspond to a specific ra dec pair

        :param ra: the right ascension under consideration in units of degrees
        :param dec: The declination under consideration in units of degrees
        :return: a tuple containing the zone and block location for the ra/dec
        """

        # zones are based on dec
        zone = int(np.floor((dec + 90) / 0.2) + 1)
        # blocks are based on ra
        ra_ind = int(np.floor(ra / 0.25) + 1)

        # need this check because ra is < 360, not <=
        if ra_ind > 1440:
            ra_ind = 1440

        # need this check for when the max dec is 90, because this is included in the last file
        if zone > 900:
            zone = 900

        return zone, ra_ind

    @staticmethod
    def get_index_ind(zone: int, ra_ind: int) -> int:
        """
        This determines the location in the index that corresponds to the requested zone and block

        :param zone: The zone number
        :param ra_ind: The right ascension index
        :return:
        """

        return (zone - 1) * 1440 + ra_ind - 1

    @staticmethod
    def convert_to_giant_catalogue(ucac_records: pd.DataFrame) -> pd.DataFrame:
        """
        This method converts records in the catalogue format into records in the GIANT format.

        This is done by renaming columns and converting units.

        :param ucac_records: The raw records from the catalogue as a pandas DataFrame
        :return: The GIANT records as a Pandas DataFrame
        """

        # prep the ucac data frame (set the full index)
        ucac_records = ucac_records.assign(source='UCAC4').reset_index().rename(columns={'index': 'rnm'})
        ucac_records = ucac_records.set_index(['source', 'zone', 'rnz', 'rnm'])

        records = ucac_records.loc[:, _UCAC_COLS]
        records.rename(columns=_UCAC_TO_GIANT, inplace=True)
        records = records.assign(epoch=2000.0)
        records.dtypes.loc[GIANT_COLUMNS] = GIANT_TYPES

        # replace invalid magnitudes with the ucac model magnitude
        invalid_mag = records['mag'] == 20000

        records.loc[invalid_mag, 'mag'] = ucac_records.loc[invalid_mag, 'magm'].astype(np.float64)

        # convert to giant units
        records['ra'] /= DEG2MAS  # MAS to DEG
        records['dec'] /= DEG2MAS  # MAS to DEG
        records['dec'] -= 90.  # SPD to DEC
        records['distance_sigma'] /= records['distance'] ** 2  # convert parallax std to distance std
        records['distance'] /= 1000  # MAS to arcsecond
        records['distance'] **= -1  # parallax to distance (arcsecond to parsec)
        records['distance'] *= PARSEC2KM  # parsec to kilometers
        records['mag'] /= 1000.  # mMAG to MAG
        records['ra_sigma'] += 128  # to uint
        records['ra_sigma'] /= DEG2MAS  # to deg
        records['dec_sigma'] += 128  # to uint
        records['dec_sigma'] /= DEG2MAS  # to deg
        records['ra_proper_motion'] /= 10 * DEG2MAS  # 0.1 MAS/YR to DEG/YR
        records['dec_proper_motion'] /= 10 * DEG2MAS  # 0.1 MAS/YR to DEG/YR
        records['ra_pm_sigma'] /= 10 * DEG2MAS  # 0.1 MAS/YR to DEG/YR
        records['dec_pm_sigma'] /= 10 * DEG2MAS  # 0.1 MAS/YR to DEG/YR
        records['distance_sigma'] *= 1000  # 1/MAS to parsec
        records['distance_sigma'] *= PARSEC2KM  # parsec to km

        # convert the sigmas to J2000
        # noinspection SpellCheckingInspection
        ra_shift_time = 2000 - (ucac_records['cepra'] / 100 + 1900)
        # noinspection SpellCheckingInspection
        dec_shift_time = 2000 - (ucac_records['cepdc'] / 100 + 1900)

        records['ra_sigma'] = np.sqrt(records['ra_sigma'] ** 2 + ra_shift_time ** 2 * records['ra_pm_sigma'] ** 2)
        records['dec_sigma'] = np.sqrt(records['dec_sigma'] ** 2 + dec_shift_time ** 2 * records['dec_pm_sigma'] ** 2)

        # fix for stars with no parallax --  The distance standard deviation seems wrong for these
        default_distance_error = 20 / (STAR_DIST / PARSEC2KM / 1000) ** 2 * PARSEC2KM * 1000
        records['distance_sigma'].fillna(value=default_distance_error, inplace=True)
        records['distance'].replace([np.inf, -np.inf], STAR_DIST, inplace=True)

        # fix for stars where the parallax is invalid
        records.loc[records.distance < 0, 'distance'] = STAR_DIST
        records.loc[records.distance < 0, 'distance_sigma'] = default_distance_error

        # specify that the epoch of the stars is J2000
        return records

    def cross_ref_tycho(self, ucac_labels: ARRAY_LIKE, tycho_cat: Optional[Tycho2] = None) -> pd.DataFrame:
        """
        This retrieves the Tycho 2 catalogue records for the requested UCAC4 star Zone, RNZ values.

        :param tycho_cat: The tycho catalogue instance to use.  If None, a default instance will be created
        :param ucac_labels:  The UCAC 4 labels as an iterable of Zone, RNZ pairs
        :return: The raw Tycho 2 star records.
        """

        if tycho_cat is None:
            tycho_cat = Tycho2()

        out = []
        # loop through all of the UCAC stars
        for label in ucac_labels:

            # compose the UCAC id as included in the u4xtycho file
            query = label[0] * 1e6 + label[1]

            # search the u4xtycho file to see if this corresponds to a tycho star
            line = binary_search(self._tycho_cross_file, query, column=1)

            if line is not None:
                line = line.decode()

                # determine the tycho2 id from the u4xtycho file
                t2id = ' '.join([line[:4], line[5:10], line[11]])

                # get the tycho2 star record
                rec = tycho_cat.retrieve_record(t2id)

                out.append(rec)

            else:
                warnings.warn('The requested UCAC4 star does not have a tycho reference associated with it: {}'.format(
                    label
                ))
                # todo: may need to figure out how to get a unique id here so it doesn't get overwritten
                out.append(tycho_cat.nan_frame())

        return pd.concat(out)

    def dump_to_sqlite(self, database_connection: Connection, limiting_mag: Real = 20, use_tycho_mag: bool = False,
                       return_locations: bool = False, return_mag: Optional[Real] = None) -> Optional[pd.DataFrame]:
        """
        Use this to write the catalogue to a sqlite3 database in the GIANT format.

        You can control what stars/data are included using the key word argument inputs.  You can also have this return
        the star magnitude/locations for doing blending stars.

        In general you should not use this directly.  Instead you should use :func:`~.giant_catalogue.build_catalogue`
        or script :mod:`~.scripts.build_catalogue`.

        :param database_connection: The connection to the database the data is to be dumped to
        :param limiting_mag: The maximum magnitude to include in the catalogue.  This is based off of the APASM_V
                             magnitude or the UCAC4 magm magnitude, depending on which is available
        :param use_tycho_mag: This flag stores the magnitude from the Tycho2 catalogue for each star that is in the
                              Tycho2 catalogue.  Note that this will be very slow.
        :param return_locations: This flag specifies to return locations for stars for doing blending
        :param return_mag: This flag specifies to only return locations for stars that are brighter than this magnitude.
                           If ``None`` then all stars are returned.
        :return: A dataframe of the dumped stars that meet the ``return_mag`` condition or ``None`` if
                 ``return_locations`` is ``False``
        """
        from .tycho import Tycho2

        # if we want to use the tycho information in place of UCAC4 then create the tycho catalogue interface
        if use_tycho_mag:
            tycho = Tycho2()
        else:
            tycho = None

        # list for returning the results if we are doing that
        out = []

        start = time.time()

        print('dumping zone {}'.format(1), flush=True)

        # loop through each zone file and dump it
        for ind, records in enumerate(self.query_catalogue_raw(max_visual_mag=limiting_mag, generator=True)):

            # convert into the GIANT format
            giant_records = self.convert_to_giant_catalogue(records)

            # if we are cross referencing the tycho catalogue do it
            if use_tycho_mag:
                tycho_recs = self.cross_ref_tycho(giant_records.index.droplevel(['source', 'rnm']).tolist(),
                                                  tycho_cat=tycho)

                # add a column so we can see where the tycho information came from
                giant_records.loc[:, 'tycho id'] = ''

                for star_index, magnitude_index in enumerate(giant_records.index):
                    if not np.isnan(tycho_recs.iloc[star_index].VTmag):
                        # update the magnitude if we found a tycho record
                        giant_records.loc[magnitude_index, 'tycho id'] = '{}-{}-{}'.format(
                            *tycho_recs.iloc[star_index].name
                        )
                        giant_records.loc[magnitude_index, 'mag'] = tycho_recs.iloc[star_index].VTmag

            # set the index to be the rnm
            giant_records = giant_records.reset_index().set_index('rnm')
            # dump it out to the GIANT catalogue in the stars table
            giant_records.to_sql('stars', database_connection, if_exists='append')
            print('zone dumped in {:.3f} secs'.format(time.time() - start), flush=True)
            if return_locations:
                if return_mag is not None:
                    out.append(giant_records.loc[giant_records.mag <= return_mag, ["ra", "dec", "mag"]])
                else:
                    out.append(giant_records.loc[:, ["ra", "dec", "mag"]])

            zone = ind + 2
            start = time.time()

            print('dumping zone {}'.format(zone), flush=True)

        if return_locations:
            return pd.concat(out)


class ColumnOrder(Enum):
    """
    This enumeration specifies whether a column is sorted in ascending or descending order.

    This is intended to be used as an input to :func:`binary_search`.
    """

    ASCENDING = "ASCENDING"
    """
    The column is sorted in ascending order (smallest first)
    """

    DESCENDING = "DESCENDING"
    """
    The column is sorted in DESCENDING order (smallest last)
    """


def binary_search(file: BinaryIO, label: Any, column: int = 0,
                  separator: Optional[str] = None, column_conversion: Callable = float,
                  order: Union[ColumnOrder, str] = ColumnOrder.ASCENDING,
                  start: int = 0, stop: Optional[int] = None, line_length: Optional[int] = None) -> Optional[bytes]:
    """
    This helper function does a binary search on a sorted file with fixed width lines.

    The binary search is performed by successively checking the midpoint between the current block of the file under
    consideration and using it to determine whether to search to the left or right of the midpoint for the next
    iteration.  As such, this requires the lines to be sorted on the column that is being searched.  This also requires
    that the column being searched is orderable (implements comparison operators) after conversion from a string.

    The conversion into an orderable type is controlled using the ``column_conversion`` keyword argument.  This is
    applied to the specified column (controlled by keyword arguments ``column`` and ``separator``) to create an
    orderable object.  This can be any callable, so long as it returns an orderable object, but typically is a python
    type like ``int`` or ``float``.  Note that strings are orderable as well, therefore you can make
    ``column_conversion`` ``str``, however be aware that the ordering of strings can be confusing when white space is
    involved (for instance ``'10'`` is less than ``'2'`` according to string comparisons).  Therefore, unless your
    numbers are 0 padded (ie ``'02'``), we recommend using a numeric type for the ``column_conversion``.

    If the searched for label is found in the column then the line in which it is found is returned (as a bytes object).
    If it is not found then ``None`` is returned.

    :param file: The file object to search.  This should be opened in binary read mode so that we can seek
    :param label: the label we are searching for in the file object.  This must support equality comparison (==) with
                  the type that is returned by ``column_conversion``.
    :param column: the column index that is to be searched
    :param separator: The separator spec for splitting the file.  If ``None`` then defaults to white space.  This is
                      passed directly to ``str.split``
    :param column_conversion: The callable to convert the column into an orderable object.  Typically this should be one
                              of the python builtin types (like ``float`` or ``int``) but it can be ay callable so long
                              as the return supports less than/greater than operators.  This is applied as
                              ``column_conversion(line.split(sep=separator))`` where ``line`` is the current line under
                              consideration.
    :param order: How the column being searched is sorted.  This should be either ``ASCENDING`` or ``DESCENDING`` (one
                  of the :class:`ColumnOrder` enum values)
    :param start: Where to start in the file in bytes. Typically this is unused unless you know you can skip part of
                  the file
    :param stop: Where to stop the search in bytes.  If this is ``None`` then  it will be set to the length of the file.
                 Typically this is unused unless you know you can skip part of the file
    :param line_length: The number of bytes in each line.  If ``None`` then this will be computed from the file.
    :return:
    """

    # if the line length for the file wasn't specified determine it
    if line_length is None:
        file.seek(0, os.SEEK_SET)
        file.readline()
        line_length = file.tell()

    # if the end of our search isn't specified, use the end of the file
    if stop is None:
        file.seek(0, os.SEEK_END)
        stop = file.tell()

    if isinstance(order, str):
        order = ColumnOrder(order.upper())

    # determine which comparison operator to use based on the sort order
    if order is ColumnOrder.ASCENDING:
        left_comparison = lt
    else:
        left_comparison = gt

    while True:
        # determine the number of lines in the file
        stop_line = stop // line_length
        # determine the line we are starting on
        start_line = start // line_length

        # determine the line we are searching next
        mid_line = (stop_line - start_line) // 2 + start_line

        mid = mid_line * line_length

        # seek to the line to search
        file.seek(mid, os.SEEK_SET)

        line = file.readline()

        # reached the end of the file, return None to indicate failure
        if not line:
            return None

        # get the label from this line
        line_lab = column_conversion(line.split(sep=separator)[column])

        if line_lab == label:
            # if this is the line we want return it
            return line
        elif mid_line == start_line or mid_line == stop_line:
            # if we have run out of lines to search stop
            return None
        elif left_comparison(line_lab, label):
            # if the search line is below the label we want then search from it to the stop
            start = mid
        else:
            # if the search line is above the label we want then search from the start to the search line
            stop = mid


def check_file(file: Path, md5s: Optional[dict] = None) -> Tuple[bool, Optional[dict]]:
    """
    This checks the md5sum of a file to ensure it wasn't corrupted.

    .. warning::

        This function can only verify the integrity of the files through md5 sums which are insecure.  While the vizier
        service is trusted, use this function at your own risk

    :param file: The file to check the md5 of
    :param md5s: A dictionary containing the md5 sums for each file
    :return: a flag specifying if the file was correct and the dictionary of md5 sums used to check the file
    """

    import hashlib

    if md5s is None:
        md5_file = file.parent / "md5sum.txt"  # type: Path
        if md5_file.exists():
            md5s = {}
            with md5_file.open('r') as m_file_obj:
                for line in m_file_obj:
                    if line.startswith('#'):
                        continue
                    else:
                        fmd5, m_file_obj = line.split()
                        md5s[m_file_obj.replace('*', '')] = fmd5

        else:
            return False, None

    if file.exists():
        base = file.name
        if base != "md5sum.txt":
            # MD5 has a security risk but its better than nothing...
            return md5s.get(base, None) == hashlib.md5(file.open('rb').read()).hexdigest(), md5s  # nosec
        else:
            return True, md5s
    else:
        return False, md5s


def download_ucac(target_directory: Path):
    """
    This function downloads the UCAC4 catalogue from vizier to the target directory.

    This is done over ftp.  It requires an active internet connection that can connect to cdsarc.u-strasbg.fr

    .. warning::

        This download will take a long time and use up approximately 9 GB of space.

    .. warning::

        This download can only verify the integrity of the files through md5 sums which are insecure.  While the vizier
        service is trusted, use this function at your own risk

    :param target_directory: the directory to save the UCAC catalogue to
    """

    # we minimize the security risk here by using FTPS
    import ftplib  # nosec

    # FTPS is secure
    ftp = ftplib.FTP_TLS('cdsarc.u-strasbg.fr')  # nosec

    # anonymous login since we're just grabbing data
    ftp.connect()
    ftp.sendcmd('USER anonymous')
    ftp.sendcmd('PASS anonymous@a.com')

    ftp.cwd('pub/cats/I/322A/UCAC4/')

    lines = []

    ftp.retrlines('LIST', callback=lines.append)

    for line in lines:

        if line.startswith('d'):
            # directory
            name = line.split()[-1]

            if name in ["u4b", "u4i"]:

                local_directory = target_directory / name
                local_directory.mkdir(parents=True, exist_ok=True)

                download_lines = []

                ftp.retrlines('LIST {}'.format(name), callback=download_lines.append)

                md5s = None

                for dl in download_lines:
                    start = time.time()
                    file = dl.split()[-1]

                    local = local_directory / file

                    downloaded, md5s = check_file(local, md5s)

                    if not downloaded:
                        with local.open('wb') as download_file:

                            # noinspection SpellCheckingInspection
                            ftp.retrbinary('RETR {}/{}'.format(name, file), download_file.write)

                        if name == 'u4b':
                            print('{} of z900 done in {:.3f}'.format(file, time.time()-start), flush=True)
                        else:
                            print('{} done in {:.3f}'.format(file, time.time()-start), flush=True)
                    else:
                        print('{} already downloaded'.format(file))
