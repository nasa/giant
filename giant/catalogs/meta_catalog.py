"""
This module defines the abstract base class (abc) for defining GIANT star catalogs that will work for Stellar OpNav
and camera calibration as well as the column definitions for the dataframe used to contain stars in GIANT.

The abc documents the required interface that must be implemented for each star catalog for it to be fully
functional in GIANT.  As such, when you define a new catalog in GIANT you should subclass this class and be sure to
implement all of its abstract methods. You should only worry about this abc when you are defining a new catalog.  If
you are using an existing catalog then you can ignore this documentation and read the documentation for the catalog
you are using for more specific details.

The column definitions are stored as 2 module attributes :attr:`GIANT_COLUMNS` and :attr:`GIANT_TYPES` which specify the
column names and the column types respectively of the dataframe used to store star records for use in GIANT.

Use
---

To implement a full function GIANT catalog, you must implement the following instance attribute

============================================= ==========================================================================
Instance Attribute                            Description
============================================= ==========================================================================
:attr:`~.Catalog.include_proper_motion`       boolean flag specifying whether to apply proper motion to the queried
                                              locations to translate them to the specified time before returning.
                                              Technically this isn't actually required, but it is strongly recommended.
============================================= ==========================================================================

In addition, you need to implement the following method

============================================= ==========================================================================
Method                                        Description
============================================= ==========================================================================
:attr:`~.Catalog.query_catalog`           A method which queries the catalog for requested Ra/Dec/Mag of stars
                                              and, if requested, applies proper motion to translate those stars to the
                                              requested time.
============================================= ==========================================================================

So long as your class implements these things it will be fully functional in GIANT and can be used for star
identification purposes.
"""

from abc import ABCMeta, abstractmethod

from datetime import datetime

from typing import Union, Optional, List, Type, Dict, Iterable, Sequence, Any

import numpy as np
import pandas as pd

from giant.utilities.spherical_coordinates import radec_to_unit, unit_to_radec
from giant.ray_tracer.utilities import correct_stellar_aberration
from giant.camera_models import CameraModel
from giant.image import OpNavImage

from giant._typing import DOUBLE_ARRAY


GIANT_COLUMNS: List[str] = ['ra', 'dec', 'distance', 'ra_proper_motion', 'dec_proper_motion', 'mag',
                            'ra_sigma', 'dec_sigma', 'distance_sigma', 'ra_pm_sigma', 'dec_pm_sigma', 'epoch']
"""
This specifies the name of the DataFrame columns used to store star observations in GIANT.

These columns represent the minimum set of data that GIANT needs to know about a star to use it for star identification.
A description of each column follows.

===================== ======== =================================================================================
column                units    description
===================== ======== =================================================================================
`'ra'`                deg      The right ascension of the star after correcting for proper motion
`'dec'`               deg      The declination of the star after correcting for proper motion
`'distance'`          km       The distance to the star from the Solar system barycenter (converted from
                               parallax). This column has a default value of 5.428047027e15 if no parallax
                               information is provided by the catalog.
`'ra_proper_motion'`  deg/year The proper motion for the right ascension
`'dec_proper_motion'` deg/year The proper motion for the declination
`'mag'`               N/A      The apparent magnitude of the star according to the star catalog
`'ra_sigma'`          deg      The formal uncertainty in the right ascension according to the catalog
`'dec_sigma'`         deg      The formal uncertainty in the declination according to the catalog
`'distance_sigma'`    km       The formal uncertainty in the distance according to the catalog 
                               (converted from parallax).  This has a default value of 
                               1.9949433041226756e+19 km for stars with no parallax information.
`'ra_pm_sigma'`       deg/year The formal uncertainty in the right ascension proper motion according to 
                               the catalog
`'dec_pm_sigma'`      deg/year The formal uncertainty in the declination proper motion according to the
                               catalog
`'epoch'`             MJD year The current epoch of the ra/dec/proper motion as a float
===================== ======== =================================================================================

"""


GIANT_TYPES: Dict[str, Type] = dict(zip(GIANT_COLUMNS, [np.float64] * len(GIANT_COLUMNS)))
"""
This specifies the data type for each column of the GIANT star dataframe.

This is generally all double precision float values, though that could change in a future release.
"""


class Catalog(metaclass=ABCMeta):
    """
    This is the abstract base class for star catalogs that GIANT can use for star identification.

    This class defines the minimum required attributes and methods that GIANT expects when interfacing with a star
    catalog.  It is also set up to implement duck typing, so if you don't want to you don't need to subclass this
    class directly when defining a new catalog, though it is still strongly encouraged.

    To define a new Catalog you pretty much only need to implement a :meth:`query_catalog` method with the call
    signature specified in the abstract method documentation.  This is what GIANT will use when retrieving stars from
    the catalog.  Beyond that, you should probably also implement and use an instance attribute
    :attr:`include_proper_motion` (although this is not required) as a flag that the user could use to turn proper
    motion applying on or off on a call to :meth:`query_catalog`.

    If you are just trying to use a GIANT catalog, you don't need to worry about this class, instead see one of the
    concrete implementations in :mod:`.ucac`, :mod:`.tycho`, or :mod:`.giant_catalog`

    .. note:: Because this is an ABC, you cannot create an instance of this class.
    """

    def __init__(self, include_proper_motion: bool = True):
        """
        :param include_proper_motion: A flag indicating that proper motion should be applied to query results from this
                                      catalog
        """

        self.include_proper_motion: bool = include_proper_motion
        """
        Apply proper motion to queried star locations to get the location at the requested time
        """

    @abstractmethod
    def query_catalog(self, ids: Optional[Iterable[Any]] = None, min_ra: float = 0, max_ra: float = 360,
                        min_dec: float = -90, max_dec: float = 90, min_mag: float = -4, max_mag: float = 20,
                        search_center: Optional[Sequence[float] | DOUBLE_ARRAY] = None, search_radius: Optional[float] = None,
                        new_epoch: Optional[Union[datetime, float]] = None) -> pd.DataFrame:
        """
        This method queries stars from the catalog that meet specified constraints and returns them as a DataFrame
        with columns of :attr:`.GIANT_COLUMNS`.

        Stars can either be queried by ID directly or by right ascension/declination/magnitude. You cannot filter using
        both with this method.  If :attr:`apply_proper_motion` is ``True`` then this will shift the stars to the new
        epoch input by the user (``new_epoch``) using proper motion.

        :param ids: A sequence of star ids to retrieve from the catalog.  What these ids are vary from catalog to
                    catalog so see the catalog documentation for details.
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

        pass
    
    
    def get_stars_directions_and_pixels(self: 'Catalog', image: OpNavImage, model: CameraModel, max_mag: float, min_mag: float = -4, image_number: int = 0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        This function produces the visible stars in an image, including their records, their inertial unit vectors, and their pixel locations.
        
        The function queries the catalog using information contained in the OpNavImage input and the min/max mag inputs. 
        Important attributes of the OpNavImage are `observation_date`, `rotation_inertial_to_camera`, `temperature`, `position`, and `velocity`.
        
        The function corrects the catalog unit vectors for parallax and for stellar aberration.
        
        :param image: The OpNavImage used to specify metadata about the camera
        :param model: The camera model used to project unit vectors onto the image
        :param max_mag: the maximum magnitude star to query from the catalog
        :param min_mag: the minimum magnitude star to query from the catalog
        :param image_number: The number of the image being processed
        :returns: A tuple containing the star records as a pandas DataFrame, the star inertial unit vectors (corrected for aberration and parallax), and the pixels the stars project to
        """
        
        
        # get the ra/dec of the z axis of the camera in the inertial frame
        ra, dec = unit_to_radec(image.rotation_inertial_to_camera.matrix[-1])

        # query the star catalog for stars in the field of view
        stars = self.query_catalog(search_center=(float(np.rad2deg(ra)), float(np.rad2deg(dec))), 
                                     search_radius=model.field_of_view, 
                                     new_epoch=image.observation_date,
                                     max_mag=max_mag,
                                     min_mag=min_mag)

        
        # convert the star locations into unit vectors in the inertial frame
        ra_rad = np.deg2rad(stars['ra'].to_numpy())
        dec_rad =np.deg2rad(stars['dec'].to_numpy())
        catalog_unit_vectors = radec_to_unit(ra_rad, dec_rad)

        # correct the unit vectors for parallax using the distance attribute of the star records and the camera inertial
        # location
        catalog_points = catalog_unit_vectors * stars['distance'].to_numpy()

        camera2stars_inertial = catalog_points - image.position.reshape(3, 1)

        # correct the stellar aberration
        camera2stars_inertial = correct_stellar_aberration(camera2stars_inertial, image.velocity)

        # form the corrected unit vectors
        camera2stars_inertial /= np.linalg.norm(camera2stars_inertial, axis=0, keepdims=True)

        # rotate the unit vectors into the camera frame
        rot2camera = image.rotation_inertial_to_camera.matrix
        catalog_unit_vectors_camera = rot2camera @ camera2stars_inertial

        # store the inertial corrected unit vectors and the projected image locations
        return (stars, 
                camera2stars_inertial, 
                catalog_unit_vectors_camera, 
                model.project_onto_image(catalog_unit_vectors_camera, 
                                         temperature=image.temperature, 
                                         image=image_number))

    @classmethod
    def __subclasshook__(cls, other: type) -> bool:

        if cls is Catalog:
            return hasattr(other, 'query_catalog')

        return NotImplemented
