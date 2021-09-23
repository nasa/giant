# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This module defines the interface to the default GIANT star catalogue.

Catalogue Description
=====================

The default GIANT star catalogue is a blending of the UCAC4 and Tycho2 catalogues into a stripped down sqlite3 database.
This makes retrieval of stars very fast instead of using the UCAC4 or Tycho2 catalogue files themselves and also
includes blended stars (stars that are so close together they will appear as a single star in most cameras).  This
catalogue includes stars down to about 16th magnitude and should be sufficient for nearly all stellar OpNav needs.

The current implementation of the database storing this catalogue is a single table with a primary index column of rnm,
which contains the unique ID of the provided star from the UCAC4 catalogue.  For blended stars, the index number becomes
the negative of the unique ID of the brightest star in the group.  In addition to the index, the following columns are
provided

================= ====== ======== ======================================================================================
Column            Units  Type     Description
================= ====== ======== ======================================================================================
source            N/A    string   The original catalogue source of the star (UCAC4 or Tycho2)
zone              N/A    integer  The UCAC4 zone number for the star, if applicable
rnz               N/A    integer  The star number in the UCAC4 zone if applicable
ra                deg    double   The right ascension of the star in degrees
dec               deg    double   The declination of the star in degrees
distance          km     double   The distance to the star in km.  If not known then this is replaced with the average
                                  distance to a star
ra_proper_motion  deg/yr double   The proper motion of the right ascension of the star in degrees per SI year
dec_proper_motion deg/yr double   The proper motion of the declination of the star in degrees per SI year
mag               mag    double   The visual magnitude of the star.  This is the APASM_V magnitude if available,
                                  otherwise the MAGM (model magnitude) from the UCAC4 catalogue.
ra_sigma          deg    double   The right ascension uncertainty in units of degrees
dec_sigma         deg    double   The declination uncertainty in units of degrees
distance_sigma    km     double   The distance uncertainty in units of kilometers.  For stars for which this is not
                                  known it is set to a large number
ra_pm_sigma       deg/yr double   The right ascension proper motion uncertainty in degrees per SI year.
dec_pm_sigma      deg/yr double   The declination proper motion uncertainty in degrees per SI year.
================= ====== ======== ======================================================================================

While this implementation mirrors the :attr:`.GIANT_COLUMNS` currently, and probably will in the future, it isn't
guaranteed to stay that way.  In addition, while this is currently a blend of the UCAC4 and Tycho2 catalogues, in the
future it will likely be built from the GAIA DR2 catalogue since this provides much more accurate stars
positions/magnitudes for many more stars.

Use
===

The GIANT catalogue can be used anywhere that a star catalogue is required in GIANT and is generally the default
catalogue that is used if you do not override it.  It is stored in a sqlite file in a directory called data in the same
directory hosting this file, though it is possible to override this if desired (which you may want to do if you need
different versions of the catalogue for different cameras). To access the default catalogue simply initialize this class
with no arguments and then call :meth:`~.GIANTCatalogue.query_catalogue` to retrieve the star records that you want.

This implementation also provides 2 helper methods that can retrieve the original star record from either the Tycho 2 or
UCAC4 catalogue (if the star exists in them and is not a blended star).  These are
:meth:`~.GIANTCatalogue.get_ucac4_record` and :meth:`~.GIANTCatalogue.get_tycho2_record`.  They take in a pandas
DataFrame of stars retrieved from this catalogue and return a dataframe with the original records for those stars.
This can be useful if you need more information about a star than what GIANT typically considers.  Just note that these
methods will require that the entire UCAC4/Tycho2 star catalogues be downloaded if they aren't already.

This module also provides a few functions that can be used to build a new version of this catalogue,
:func:`build_catalogue`, :func:`find_star_pairs`, and :func:`blend_stars`.  Typically you won't interact with these
directly and instead will use the script :mod:`~.scripts.build_catalogue` which provides a command line interface,
however they're provided for those who are interested in what they do or in doing some more advanced things.
"""

from pathlib import Path
import time

from itertools import repeat, starmap

from warnings import filterwarnings, catch_warnings

import sqlite3

from datetime import datetime

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from giant.catalogues.utilities import radec_to_unit, apply_proper_motion, radec_distance, DEG2RAD
from giant.catalogues.meta_catalogue import Catalogue
from giant._typing import PATH, Real, ARRAY_LIKE


DEFAULT_CAT_FILE: Path = (Path(__file__).parent / "data") / 'giant_cat.db'
"""
This specifies the default location of the sqlite3 database file that contains the data for this catalogue.

This is stored as Path object.  It defaults to a file called "giant_cat.db" in a "data" directory in the same directory 
containing this file. If you wish to use a different catalogue, typically you should simply provide a key
word argument to the class constructor instead of modifying this module attribute.
"""


_STARS_TABLE_SQL: str = '''CREATE TABLE IF NOT EXISTS "stars" (
  "rnm" INTEGER UNIQUE PRIMARY KEY NOT NULL ON CONFLICT REPLACE,
  "source" TEXT,
  "zone" INTEGER,
  "rnz" INTEGER,
  "ra" REAL,
  "dec" REAL,
  "distance" REAL,
  "ra_proper_motion" REAL,
  "dec_proper_motion" REAL,
  "mag" REAL,
  "ra_sigma" REAL,
  "dec_sigma" REAL,
  "distance_sigma" REAL,
  "ra_pm_sigma" REAL,
  "dec_pm_sigma" REAL,
  "epoch" REAL
)'''
"""
This SQL command creates a new table in the sqlite3 database file called stars for storing the GIANT catalogue

This shouldn't be used by the user and may change without notice.
"""


def build_catalogue(database_file: Optional[PATH] = None, limiting_magnitude: Real = 12, number_of_stars: int = 0,
                    use_tycho_mag: bool = False, limiting_separation: float = 0.04, blending_magnitude: Real = 8,
                    ucac_dir: Optional[PATH] = None):
    """
    Build a sqlite3 catalogue from the UCAC catalogue for faster query times.

    This function can be used to build a new GIANT catalogue (or overwrite an old one) from the UCAC4/Tycho2 star
    catalogues.  Typically a user will not use this directly and instead will use the command line utility
    :mod:`~.scripts.build_catalogue`.

    If you want to use this function, you can adjust how (and where) the catalogue is built by adjusting the key word
    arguments.  Note that this will require downloading the UCAC4 catalogue (if it isn't already available) and possibly
    the Tycho2 Catalogue.  In addition, building the catalogue can take a long time, so you will want to have a period
    where you can leave this run for a while without interruption to ensure that the catalogue is built successfully and
    not corrupted.

    :param database_file: The file to save the catalogue database to
    :param limiting_magnitude: The maximum magnitude to include in the catalogue
    :param number_of_stars: The maximum number of stars that can be blended together in any group.  To turn off star
                            blending, set this to 0.
    :param use_tycho_mag: A flag specifying whether to replace the UCAC4/APASM_V magnitudes with the Tycho VT magnitude.
                          The Tycho VTMag is more accurate, but this can make things take even longer to compile so by
                          default it is not used.
    :param limiting_separation: The maximum separation between stars for them to be considered for blending in degrees.
                                Typically this should be set to around the IFOV on the detector you are considering.
    :param blending_magnitude: The magnitude of the blended star for it to be included as a blended star in the
                               catalogue.
    :param ucac_dir: The directory containing the UCAC4 data files.  This is passed to the :class:`.UCAC4` class.
    """

    with catch_warnings():
        # ignore dumb warnings
        filterwarnings('ignore', message='The requested UCAC4')

        # import the UCAC4 interface and default directory
        from giant.catalogues.ucac import UCAC4, UCAC_DIR

        # use the default if the user didn't specify
        if database_file is None:
            database_file = DEFAULT_CAT_FILE
        else:
             database_file = Path(database_file)

        # make sure the directory for the database file exists
        database_file.parent.mkdir(exist_ok=True, parents=True)

        if database_file.exists():
            database_file.unlink()
            
        # connect to the database
        db_con = sqlite3.connect(str(database_file))

        # get the UCAC instance
        if ucac_dir is None:
            ucac = UCAC4(UCAC_DIR)
        else:
            ucac = UCAC4(ucac_dir)

        # determine the magnitude to query from the catalogue
        if number_of_stars == 0:
            blend_mag = -4.0
            query_mag = limiting_magnitude
        else:
            blend_mag = blending_magnitude + (2.5 * np.log10(number_of_stars))
            query_mag = max(limiting_magnitude, blend_mag)

        print('Query mag {}'.format(query_mag), flush=True)

        # Create the table/indices in the database
        db_con.execute("DROP TABLE IF EXISTS stars")
        db_con.execute(_STARS_TABLE_SQL)
        db_con.execute("CREATE UNIQUE INDEX idx_rnm on stars(rnm)")
        db_con.execute("CREATE INDEX idx_ra on stars(ra)")
        db_con.execute("CREATE INDEX idx_dec on stars(dec)")
        db_con.execute("CREATE INDEX idx_mag on stars(mag)")
        db_con.commit()

        print('creating database', flush=True)
        if number_of_stars == 0:
            # dump the UCAC4 stars to the database
            ucac.dump_to_sqlite(db_con, limiting_mag=query_mag, use_tycho_mag=use_tycho_mag,
                                return_locations=False)
        else:
            # dump the UCAC4 stars to the database
            records = ucac.dump_to_sqlite(db_con, limiting_mag=query_mag, use_tycho_mag=use_tycho_mag,
                                          return_locations=True, return_mag=blend_mag)

            # pair the stars based on distance
            print('finding close star pairs', flush=True)
            pairs = find_star_pairs(records, limiting_separation)

            # get rid of the old dataframe for memory reasons
            del records

            # blend the stars together
            print('blending stars', flush=True)
            combined_stars = blend_stars(pairs, db_con, limiting_magnitude)

            # rename the index column
            combined_stars.index.name = 'rnm'

            # dump to the stars table in the database
            print('adding blended stars to db', flush=True)
            combined_stars.to_sql('stars', db_con, if_exists='append')


def _repair(star_records: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """
    This helper function combines multiple pairs and gets rid of duplicates

    Don't use this yourself.

    :param star_records: The dataframe of star records
    :param pairs: The dataframe specifying groups of stars
    :return: The dataframe specifying groups of stars after correcting the groupings
    """
    # get the unique right hand sides
    unique_others = pairs.b.unique()

    paired_dict: pd.DataFrame = pairs.groupby('a')['b'].apply(set)
    removes = []
    for initial, others in paired_dict.iteritems():
        sets = []
        starts = []
        # look for where initial is also paired to another star
        if (initial in unique_others) and (initial not in removes):
            for local_initial, local_others in paired_dict.iteritems():
                if local_initial == initial:
                    continue
                elif initial in local_others:
                    # if the others is a subset of the first group we don't need to do anything
                    if others.issubset(local_others):
                        # should we also remove local_initial here?
                        continue
                    # otherwise store them for use
                    sets.append(local_others)
                    starts.append(local_initial)
            # if anything needs modified
            if starts:
                for local_initial, local_others in zip(starts, sets):
                    if len(starts) == 1:
                        # keep which ever has the higher magnitude on the left hand side and discard the other
                        if star_records.loc[local_initial, "mag"] < star_records.loc[initial, "mag"]:
                            local_others.update(others)
                            removes.append(initial)

                        else:
                            others.update(local_others)
                            removes.append(local_initial)

                    else:
                        # keep the brightest magnitude
                        best = initial
                        best_mag = star_records.loc[initial, "mag"]
                        best_set = others
                        for o, ls in zip(starts, sets):
                            if best_mag > star_records.loc[o, "mag"]:
                                best_mag = star_records.loc[o, "mag"]
                                best = o
                                best_set = ls

                        # get rid of whichever aren't the brightest and feed it
                        if best != initial:
                            removes.append(initial)
                            best_set.update(others)
                        for local_local_initial, local_local_others in zip(starts, sets):
                            if local_local_initial == best:
                                continue
                            best_set.update(local_local_others)
                            removes.append(local_local_initial)

    # get rid of the bad ones
    return paired_dict.drop(removes)


def find_star_pairs(star_records: pd.DataFrame, max_separation: float) -> pd.DataFrame:
    """
    This identifies possible star pairs based on separation.

    Stars are paired if their max separation is less that the input ``max_separation`` in degrees. This is done by
    creating unit vectors for all of the stars and then doing a pair query using a KDTree.  The pairs are sorted based
    on magnitude so that the first star in each pair is brighter.

    The result of this function will be a dataframe where the first column "a" is the primary star and the second column
    "b" is a set of stars that should be combined with "a".

    Generally this is not used directly by the user.  Instead see :func:`build_catalogue` or script
    :mod:`~.scripts.build_catalogue`.

    :param star_records: The dataframe containing the stars that are to be paired
    :param max_separation: The maximum separation in degrees between stars for them to be paired
    :return: A dataframe specifying stars to pair together.
    """

    # get the unit vectors
    units = radec_to_unit(star_records.ra.values * DEG2RAD, star_records.dec.values * DEG2RAD).T

    # build the kdtree
    # noinspection PyArgumentList
    tree = cKDTree(units, compact_nodes=False, balanced_tree=False)

    # find the pairs.  Tell pycharm to stop complaining because numpy/scipy don't document right
    # noinspection PyArgumentList,PyUnresolvedReferences
    pairs = tree.query_pairs(np.sin(max_separation * np.pi / 360) * 2, output_type='ndarray')

    # get the pairs
    pairs = pd.DataFrame(star_records.index.values[pairs], columns=['a', 'b'])

    # sort the pairs on magnitude
    for pair in pairs.itertuples():

        if float(star_records.loc[pair.a, "mag"]) > float(star_records.loc[pair.b, "mag"]):
            pairs.loc[pair.Index, "a"] = pair.b
            pairs.loc[pair.Index, "b"] = pair.a

    # condense pairs so that stars aren't in multiple pairs
    return _repair(star_records, pairs)


def _blend_stars(input_group: tuple, database_connection: sqlite3.Connection,
                 index: int, limiting_mag: Real, reference_mag: Real) -> Optional[pd.Series]:
    """
    This helper function computes a blended star from the input star indices.

    This function queries the star records from the database for memory reasons (so that we can use multiprocessing).

    The stars are blended into a single record with a combined magnitude, right ascension, declination, and proper
    motion (distance is not considered).  This is based off of an internal note on blending stars that Sean Semper sent.

    :param input_group: The group of stars to be blended
    :param database_connection: The database connection to retrieve the star records from
    :param index: The index of the group of stars that we're working on.  Purely for printing purposed
    :param limiting_mag: The magnitude that the blended stars must reach for them to be included
    :param reference_mag: The reference magnitude to use when blending the stars.
    :return: A series with the blended star, or ``None`` if the limiting magnitude wasn't met
    """

    # interpret the group
    initial = input_group[0]
    group = list(input_group[1])

    start = time.time()

    # all the ids we need to query
    star_ids = [initial] + group
    # the security risk is minimized here by calling int on all of the elements in the star_ids list
    # its already minimal because a user should never be directly interacting with this function anyway
    star_records = pd.read_sql('select * from stars where rnm in {}'.format(tuple(map(int, star_ids))),  # nosec
                               database_connection, index_col='rnm')

    # get the initial star from the dataframe
    initial_star = star_records.loc[initial]
    # get the rest of the stars from the dataframe
    other_stars = star_records.loc[group]

    # compute the weights for each star
    initial_weight = 1 / (10 ** (0.4 * (initial_star.mag - reference_mag)))
    other_weights = 1 / (10 ** (0.4 * (other_stars.mag.values - reference_mag)))

    # determine the reference declination from the brightest star
    if (initial_star.mag <= other_stars.mag).all():
        ref_dec = np.cos(initial_star.dec*DEG2RAD)
    else:
        ref_dec = np.cos(other_stars.loc[other_stars.mag == other_stars.mag.min(), "dec"].values[0]*DEG2RAD)

    # compute the combined magnitude
    combined_mag = -2.5 * np.log10((10 ** (-0.4 * initial_star.mag) + 10 ** (-0.4 * other_stars.mag.values)).sum())

    # if the blended star is too dim stop here
    if combined_mag > limiting_mag:
        return None

    # make the Series to return
    combined_star = initial_star.copy()
    # set the blended magnitude
    combined_star.mag = combined_mag
    # update the RNM to be negative
    combined_star.name *= -1

    # compute the combined position
    denominator = initial_weight + other_weights.sum()
    combined_star.ra = (initial_weight * initial_star.ra * ref_dec +
                        (other_weights * other_stars.ra.values * ref_dec).sum()) / denominator / ref_dec
    combined_star.dec = (initial_weight * initial_star.dec +
                         (other_weights * other_stars.dec.values).sum()) / denominator
    combined_star.ra_proper_motion = ((initial_weight * initial_star.ra_proper_motion * ref_dec +
                                      (other_weights * other_stars.ra_proper_motion.values * ref_dec).sum()) /
                                      denominator / ref_dec)
    combined_star.dec_proper_motion = ((initial_weight * initial_star.dec_proper_motion +
                                        (other_weights * other_stars.dec_proper_motion.values).sum()) / denominator)

    # give a status
    print('Pair {} blended in {:.3f}'.format(index, time.time() - start))

    return combined_star


def blend_stars(groups: pd.DataFrame, database_connection: sqlite3.Connection, limiting_mag: Real,
                ref_mag: Real = 4) -> pd.DataFrame:
    """
    Blends groups of stars together into a single "apparent" star as viewed by a camera.

    Star magnitude, right ascension, declination, and proper motion are all blended in the final product.  The blending
    is based off of an internal memo by Sean Semper.

    The groups input should provide 2 columns, the first column "a" should provide the primary (brightest) star in each
    group.  The second column "b" should provide a set of all of the stars that are to be blended to each other an "a".
    This is what is returned by :func:`find_star_pairs`.  This function uses the database to retrieve the individual
    star records for memory purposes.

    The blended star is given an id that is the negative of the brightest star in the group.  The blended stars are
    returned as a pandas dataframe.

    Typically this is not used directly by the user.  Instead se :func:`build_catalogue` or script
    :mod:`.scripts.build_catalogue`.

    :param groups: The dataframe specifying the groups to blend
    :param database_connection: The connection to the sqlite3 database to retrieve the stars from
    :param limiting_mag: The limiting magnitude that blended stars must achieve for them to be included
    :param ref_mag: The reference magnitude to use when blending the stars
    :return: The dataframe of the blended apparent stars
    """

    # get the number of groups we need to blend for reporting purposes
    number_of_groups = len(groups)

    # notify the user
    print('{} stars to blend'.format(number_of_groups), flush=True)

    # combine the stars.  Perhaps can use multiprocessing for this
    combined_stars = list(starmap(_blend_stars,
                                  zip(groups.iteritems(),
                                      repeat(database_connection),
                                      range(1, number_of_groups + 1),
                                      repeat(limiting_mag),
                                      repeat(ref_mag))))

    # return the dataframe by concatenating the individual blended series.  This will ignore Nones
    return pd.concat(combined_stars, axis=1).T


class GIANTCatalogue(Catalogue):
    """
    This class provides access to the default GIANT star catalogue built from the UCAC4 and Tycho2 Catalogues.

    This class is a fully functional catalogue for GIANT and can be used anywhere that GIANT expects a star catalogue.
    As such, it implements the :attr:`include_proper_motion` to turn proper motion on or off as well as the method
    :meth:`query_catalogue` which is how stars are queried into the GIANT format.  In addition, this catalogue provides
    2 additional methods :meth:`get_ucac4_record` and :meth:`get_tycho2_record` to get the original record that was
    used to create the GIANT catalogue record.  These methods aren't used anywhere by GIANT itself, but may be useful if
    you are doing some advanced analysis.

    To use this class simply initialize it, pointing to the file where the database is stored (if you are using one that
    is different from the default).  If the catalogue file does not exist it will ask you if you want to build it, and
    if you answer yes, it will dispatch to :func:`build_catalogue` (which takes a long time in most instances).  Once
    the class is initialized, you can query stars from it using :meth:`query_catalogue` which will return a dataframe of
    the star records with :attr:`.GIANT_COLUMNS` columns.
    """

    def __init__(self, db_file: PATH = DEFAULT_CAT_FILE, include_proper_motion: bool = True):
        """
        :param db_file: The file containing the sqlite3 database that the stars are stored in
        :param include_proper_motion: A boolean flag specifying whether to apply proper motion when retrieving the stars
        """

        super().__init__(include_proper_motion=include_proper_motion)

        self.catalogue_path: Path = Path(db_file)
        """
        The path to the catalogue file containing the database
        """

        self._catalogue: Optional[sqlite3.Connection] = None
        """
        The sqlite3 catalogue connection 
        """

        if db_file.exists():
            try: 
                self._catalogue = sqlite3.connect(str(self.catalogue_path))
                self._catalogue.execute("SELECT * FROM stars LIMIT 1")
            except (sqlite3.OperationalError, sqlite3.DatabaseError):
                print("GIANT catalogue corrupted at {}".format(db_file), flush=True)
                user_response = input("Would you like to build the GIANT catalogue from the UCAC catalogue (y/n)?\n"
                                      "    WARNING: THIS MAY TAKE A LONG TIME AND WILL USE UP 600 MB OF SPACE!\n    ")

                if user_response[:1].lower() == 'y':
                    build_catalogue(db_file)

                    self._catalogue = sqlite3.connect(str(self.catalogue_path))

                else:
                    raise sqlite3.DatabaseError('The GIANT catalogue database file is corrupted')

        else:
            print("GIANT catalogue not found at {}".format(db_file), flush=True)
            user_response = input("Would you like to build the GIANT catalogue from the UCAC catalogue (y/n)?\n"
                                  "    WARNING: THIS MAY TAKE A LONG TIME AND WILL USE UP 600 MB OF SPACE!\n    ")

            if user_response[:1].lower() == 'y':
                build_catalogue(db_file)

                self._catalogue = sqlite3.connect(str(self.catalogue_path))

            else:
                raise FileNotFoundError('The GIANT catalogue database file cannot be found')

    def __reduce__(self):
        return self.__class__, (self.catalogue_path, self.include_proper_motion)

    @property
    def catalogue(self) -> sqlite3.Connection:
        """
        This is a sqlite3 connection object which is used to read from the catalogue.

        It should not be used externally unless you really know what you're doing...
        """

        return self._catalogue

    @catalogue.setter
    def catalogue(self, err):
        raise AttributeError('You cannot set the catalogue directly.  It is purely for internal use.\n' +
                             'If you really know what you are doing, you can set the catalogue file by accessing\n' +
                             'the _catalogue attribute, however this is highly warned against.')

    @catalogue.deleter
    def catalogue(self):
        self._catalogue.close()

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

        :param ids: A sequence of star ids to retrieve from the catalogue.  What these ids are vary from catalogue to
                    catalogue so see the catalogue documentation for details.
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
            records = self.get_from_ids(ids)
        else:

            records = self.get_all_with_criteria(min_ra=min_ra, max_ra=max_ra, min_dec=min_dec, max_dec=max_dec,
                                                 min_mag=min_mag, max_mag=max_mag,
                                                 search_center=search_center, search_radius=search_radius)

        if self.include_proper_motion and (new_epoch is not None):
            apply_proper_motion(records, new_epoch, copy=False)

        return records

    def get_from_ids(self, ids: ARRAY_LIKE) -> pd.DataFrame:
        """
        This method queries star records from the database based off of ID (``rnm`` in the database).

        This can be used if you are interested in a particular set of stars.  The stars are returned in the GIANT
        DataFrame format according the :attr:`.GIANT_COLUMNS`.

        Note that this does not apply proper motion.  If you need to apply proper motion see :meth:`query_catalogue`

        :param ids: The ids of the stars to retrieve as an iterable
        :return: The dataframe of stars according to :attr:`.GIANT_COLUMNS`
        """

        # map(int, ids) protects against sql injection
        return pd.read_sql(f'select * from stars where rnm in {tuple(map(int, ids))}', self._catalogue,  # nosec
                           index_col='rnm')

    def get_all_with_criteria(self, min_ra: Real = 0, max_ra: Real = 360,
                              min_dec: Real = -90, max_dec: Real = 90, min_mag: Real = -4, max_mag: Real = 20,
                              search_center: Optional[ARRAY_LIKE] = None,
                              search_radius: Optional[Real] = None) -> pd.DataFrame:
        """
        This method queries star records from the database based off of location and magnitude requirements.

        Note that this does not apply proper motion.  If you need to apply proper motion see :meth:`query_catalogue`.

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
        :return: A Pandas dataframe with columns :attr:`GIANT_COLUMNS`.
        """

        # make sure everything is a float (a) to validate input and (b) to protect against sql injection attacks
        min_ra = float(min_ra)
        max_ra = float(max_ra)
        min_dec = float(min_dec)
        max_dec = float(max_dec)
        min_mag = float(min_mag)
        max_mag = float(max_mag)

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

        # determine what the query should look like based on the rectangular bounds
        # the security risk is non-existent here due to calling float on all of the values before they are used as
        # parameters
        if min_ra < 0:
            query = ('SELECT * FROM stars WHERE ((ra >= {} AND ra <= {}) OR (ra >= {} AND ra <= {})) '   # nosec
                     'AND dec >= {} AND dec <= {} AND mag <={} ' 
                     'AND mag >= {}'.format(min_ra + 360, 360, 0, max_ra, min_dec, max_dec, max_mag, min_mag))

        elif max_ra > 360:
            query = ('SELECT * FROM stars WHERE ((ra >= {} AND ra <= {}) OR (ra >= {} AND ra <= {})) '   # nosec
                     'AND dec >= {} AND dec <= {} AND mag <={} ' 
                     'AND mag >= {}'.format(min_ra, 360, 0, max_ra - 360, min_dec, max_dec, max_mag, min_mag))

        else:
            query = ('SELECT * FROM stars WHERE ra >= {} AND ra <= {} '  # nosec
                     'AND dec >= {} AND dec <= {} AND mag <={} '
                     'AND mag >= {}'.format(min_ra, max_ra, min_dec, max_dec, max_mag, min_mag))

        # query the results from the database
        records = pd.read_sql(query, self.catalogue, index_col='rnm')

        # now do the real radial search if it is needed
        if search_center is not None:
            records = records.loc[radec_distance(records.ra * DEG2RAD, records.dec * DEG2RAD,
                                                 search_center[0] * DEG2RAD, search_center[1] * DEG2RAD) <=
                                  (search_radius * DEG2RAD)]
        if "epoch" not in records.columns:
            records = records.assign(epoch=2000.0)

        return records

    @staticmethod
    def get_tycho2_record(stars: pd.DataFrame, ucac_directory: Optional[PATH] = None,
                          tycho_directory: Optional[PATH] = None) -> pd.DataFrame:
        """
        This method can be used to retrieve the corresponding full (not GIANT) Tycho 2 records for a set of GIANT
        catalogue stars.

        This method requires that the UCAC4 and Tycho 2 catalogues be available, and will request to download them if
        they are not available (which takes a long time).  For a description of the columns refer to the Tycho 2
        documentation.

        Note that these records are not directly usable for GIANT, therefore only use this if you need the raw records
        yourself.

        Any stars that are not available in the Tycho 2 catalogue (blended stars or stars that are too dim) will be
        included in the output dataframe but with NANs for all columns

        :param stars: The stars to retrieve the Tycho records for
        :param ucac_directory: The directory containing the UCAC4 star catalogue files (or None to use the default)
        :param tycho_directory: The directory containing the Tycho 2 star catalogue files (or None to use the default)
        :return: The raw Tycho 2 records
        """

        from giant.catalogues.ucac import UCAC4
        from giant.catalogues.tycho import Tycho2

        ucac = UCAC4(directory=ucac_directory)

        tycho = Tycho2(directory=tycho_directory)

        return ucac.cross_ref_tycho(stars.loc[:, ['zone', 'rnz']], tycho_cat=tycho)

    @staticmethod
    def get_ucac4_record(stars: pd.DataFrame, ucac_directory: Optional[PATH] = None) -> pd.DataFrame:
        """
        This method can be used to retrieve the corresponding full (not GIANT) UCAC4 records for a set of GIANT
        catalogue stars.

        This method requires that the UCAC4 catalogue be available, and will request to download it if it is not (which
        takes a long time).  For a description of the columns refer to the UCAC4 documentation.

        Note that these records are not directly usable for GIANT, therefore only use this if you need the raw records
        yourself.

        Any stars that are not available in the UCAC4 catalogue (blended stars) will be included in the output dataframe
        but with NANs for all columns

        :param stars: The UCAC4 records for the stars
        :param ucac_directory: The directory containing the UCAC4 star catalogue files (or None to use the default)
        :return: The raw UCAC4 records
        """

        from giant.catalogues.ucac import UCAC4

        ucac = UCAC4(directory=ucac_directory)

        return ucac.query_catalogue_raw(ids=stars.loc[:, ['zone', 'rnz']].itertuples(False), generator=False)
