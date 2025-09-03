"""
This module provides functions for ingesting WCS header data into a format GIANT can understand.
"""


from typing import cast

import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, CartesianRepresentation
from astropy.units import Quantity

from giant.rotations import Rotation
from giant._typing import ARRAY_LIKE

from giant.calibration.estimators import IterativeNonlinearLSTSQ

from giant.camera_models import CameraModel


def get_wcs_orientation(input: WCS, boresight_pixel: ARRAY_LIKE) -> Rotation:
    r"""
    This function converts an astropy WCS object into a GIANT Rotation object.

    The conversion happens using astropy's pixel_to_world method of the wcs object to 
    define a 2 vector frame to account for various projections and coordinate systems 
    used in astronomical imaging.

    If you have a working wcs defined in a fits file you could do something like:


        >>> from giant.utilities.wcs import get_wcs_orientation
        >>> from astropy.io import fits
        >>> from astropy.wcs
        >>> with fits.open("my_file.fits") as fits_file:
        ...        wcs_l = WCS(fits_file[0])
        >>> rotation = get_wcs_orientation(wcs_l)
        >>> print(rotation)
        Rotation([...])


    :param input: The astropy WCS object to be converted
    :return: A Rotation object representing the rotation from the inertial to the camera 
             frame defined by the WCS
    """

    # boresight
    boresight_pixel = np.array(boresight_pixel).ravel()
    z_sc = cast(SkyCoord, input.pixel_to_world(*boresight_pixel))
    z_dir: np.ndarray = cast(Quantity, cast(CartesianRepresentation, z_sc.cartesian).get_xyz()).value

    x_const_sc = cast(SkyCoord, input.pixel_to_world(boresight_pixel[0]+1, boresight_pixel[1]))
    x_const: np.ndarray = cast(Quantity, cast(CartesianRepresentation, x_const_sc.cartesian).get_xyz()).value

    y_dir = np.cross(z_dir, x_const)
    y_dir /= np.linalg.norm(y_dir)

    x_dir = np.cross(y_dir, z_dir)
    x_dir /= np.linalg.norm(x_dir)

    return Rotation([x_dir, y_dir, z_dir])


def get_wcs_model(input: WCS, camera_model_guess: CameraModel, sample_step: int = 100, rotation_world_to_camera: Rotation | None = None) -> CameraModel:
    """
    This function creates a giant :ref:`.CameraModel` that mimics the input WCS object.

    We do this by doing a recalibration using the output of the WCS model to feed as an input into the calibration process.
    Essentially, we sample across the detector, use the WCS to convert each sample into a unit vector in the world frame,
    rotate the unit vectors in the world frame into the camera frame, and then do the calibration process using these vectors 
    and the original pixels.  The sampling is done in a grid between (0, 0) and (camera_model_guess.n_cols, camera_model_guess.n_rows)
    with a step size of sample_step.
    
    The camera_model_guess should be a subclass instance of the :ref:`.CameraModel` class for the camera model you would like to use
    which is seeded with an initial guess (that at least can do a projection to a meaningful value, even if its wrong).  Additionally,
    this instance should have the :ref:`.CameraModel.estimation_parameters` set correctly for what parameters you would like estimated.
    The same rules apply as for a normal calibration.
    
    If you have already extracted the rotation between the world and camera frame from the wcs object you can provide it in the
    optional rotation_world_to_camera frame, otherwise it will be extracted from the WCS using the :ref:`get_wcs_orientation` function 
    from this module assuming the boresight pixel is the center of the image
    
    :param input: The WCS to fit the model to
    :param camera_model_guess: the initial guess for the camera model/type of camera model to be extracted
    :param sample_step: the number of pixels between each sample
    """
    
    # get the rotation from the wcs if not provided
    if rotation_world_to_camera is None:
        rotation_world_to_camera = get_wcs_orientation(input, [(camera_model_guess.n_cols-1)/2, (camera_model_guess.n_rows-1)/2])
        
    
    # create a grid of pixels to sample
    cols, rows = np.meshgrid(np.arange(0, camera_model_guess.n_cols, sample_step),
                             np.arange(0, camera_model_guess.n_rows, sample_step))
    
    # use the wcs to get the corresponding unit vectors for each pixel in the camera frame
    world_coordinates = input.pixel_to_world(cols.ravel(), rows.ravel())
    
    world_units = np.array([w.cartesian.get_xyz().value for w in world_coordinates]).T
    
    camera_units = rotation_world_to_camera.matrix @ world_units
    
    # set up the calibration estimator
    estimator = IterativeNonlinearLSTSQ(camera_model_guess.copy(), measurements=np.array([cols.ravel(), rows.ravel()]), 
                                        camera_frame_directions=[camera_units], temperatures=[0],)
    
    # do the estimation
    estimator.estimate()
    
    # reocompute the field of view
    estimator.model.field_of_view = None
    
    # return the result
    return estimator.model
    
    
    
