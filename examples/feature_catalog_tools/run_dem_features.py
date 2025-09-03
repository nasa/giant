"""This file builds a csv file of optimal landmarks and uses DEM data to create Maplet
    objects that represent the csv features on the DEM map projection.

    Given a body, spacecraft, and trajectory, this script builds the csv file of
    optimized landmark's on the body within the camera field of view based on the SC trajectory.

    To run this script, you will also need to download:
     
    DEM Data: LDEM_128.LBL from https://imbrium.mit.edu/BROWSE/LOLA_GDR/CYLINDRICAL/ELEVATION/

    Albedo Data: LDAM_10_FLOAT.LBL from https://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/FLOAT_IMG/

    Note that the directory location of the DEM data, albedo data, feature catalog csv file,
    and Maplets are defined in dem_constants.py. These values can be viewed and/or changed
    within the constants file.
"""

from datetime import datetime, timedelta
import pickle
import argparse

import numpy as np
import spiceypy as spice
from pathlib import Path
from osgeo import gdal

import dem_constants as constants

from giant.camera_models import PinholeModel
from giant.ray_tracer.shapes import Ellipsoid, Point
from giant.utilities.spice_interface import SpiceOrientation, SpicePosition, datetime_to_et
from giant.ray_tracer.scene import SceneObject, Scene

from determine_dem_landmark_needs import get_spk_coverage, body_position, body_orientation
from determine_dem_landmark_needs import sun_position, sun_orientation, prepare_data, output_features
from dem_to_landmarks import load_feature_csv, dem_interp, albedo_interp, feature_to_maplet

"""ARG HELPER FUNCTION"""

def build_arg_parser() -> argparse.Namespace:
    """
    Build an argument parser.

    :return: An argparse.ArgumentParser containing parameter values.
    """
    # Set up arg parser
    description = """Identify features/landmarks for TGS"""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-mk', '--metakernel', type=str, default='./kernels/meta_kernel.tm',
                        help="Metakernel filepath (default = ./kernels/meta_kernel.tm)", metavar='FILEPATH')
    parser.add_argument('-s', '--start', type=str, default='', nargs = '+',
                        help="Starting epoch of imaging ('DD-MMM-YYYY HH:MM') - defaults to start of SPK coverage",
                        metavar=('DATE','TIME'))
    parser.add_argument('-e', '--end', type=str, default='', nargs = '+',
                        help="Ending epoch of imaging ('DD-MMM-YYYY HH:MM') - defaults to end of SPK coverage",
                        metavar=('DATE','TIME'))
    parser.add_argument('-c', '--cadence', type=float, default=0.5,
                         help="Cadence/time step of imaging in minutes (default = 0.5 min)")
    parser.add_argument('-n', '--n_features_per_img_min', type=int, default=5,
                        help="Minimum number of features per image to try to reach (default = 5)")
    parser.add_argument('-in', '--incidence', type=float, default=[0.0, 70.0], nargs=2,
                        help="Acceptable range for angle between sun direction and normal vector in deg (default = 0–70)",
                        metavar=('INCIDENCE_MIN','INCIDENCE_MAX'))
    parser.add_argument('-ex', '--exidence', type=float, default=[0.0, 65.0], nargs=2,
                        help="Acceptable range for angle between camera direction and normal vector in deg (default = 0–65)",
                        metavar=('EXIDENCE_MIN','EXIDENCE_MAX'))
    parser.add_argument('-ph', '--phase', type=float, default=[0.0, 120.0], nargs=2,
                        help="Acceptable range for angle between sun direction and camera direction in deg (default = 0–120)",
                        metavar=('PHASE_MIN','PHASE_MAX'))
    parser.add_argument('-gsd', '--gsd_tolerance', type=float, default=3.0,
                        help="Use existing features only if they have a GSD between (1/gsd_tolerance)*cam_gsd and"+\
                             " gsd_tolerance*cam_gsd (default = 3.0)")
    parser.add_argument('-csvf', '--csv_filepath_feat', type=str, default='./results/dem_feature_list.csv',
                        help="Filepath for csv to output feature list to (default = ./results/dem_feature_list.csv)", metavar='FILEPATH')
    parser.add_argument('-csvi', '--csv_filepath_img', type=str, default='./results/dem_image_list.csv',
                        help="Filepath for csv to output feature list to (default = ./results/dem_image_list.csv)", metavar='FILEPATH')
    # Parse args
    return parser.parse_args()

"""PARAMETER SET UP"""

if __name__ == "__main__":
    # Parse args
    args = build_arg_parser()

    metakernel = args.metakernel
    start_time = datetime.strptime(' '.join(args.start),'%d-%b-%Y %H:%M') if args.start else None
    stop_time = datetime.strptime(' '.join(args.end),'%d-%b-%Y %H:%M') if args.end else None
    time_step = timedelta(minutes=args.cadence)
    n_features_per_img_min = args.n_features_per_img_min
    gsd_tolerance = args.gsd_tolerance  # Use existing features only if they are between (1/gsd_tolerance)*cam_gsd and gsd_tolerance*cam_gsd
    incidence_angle_range = args.incidence # deg, angle between sun direction and normal vector
    exidence_angle_range = args.exidence  # deg, angle between camera direction and normal vector
    phase_angle_range = args.phase # deg, angle between sun direction and camera direction
else:
    metakernel = './kernels/meta_kernel.tm'
    start_time = None # start of spk coverage will be used
    stop_time = None # end of spk coverage will be used
    time_step = timedelta(minutes=0.5)
    gsd_tolerance = 3  # Use existing features only if they are between (1/gsd_tolerance)*cam_gsd and gsd_tolerance*cam_gsd
    incidence_angle_range = [0, 70]  # deg, angle between sun direction and normal vector
    exidence_angle_range = [0, 65]  # deg, angle between camera direction and normal vector
    phase_angle_range = [0, 120]  # deg, angle between sun direction and camera direction

    n_features_per_img_min = 5  # Try to add features up to this number if there are fewer existing features than this number

if __name__ == "__main__":

    """CONSTANT/GENERAL SETTINGS"""

    # Other settings
    apparent_diameter_threshold = 100  # pixels, only find features if body is larger than this
    feature_spacing_limit = 50  # pixels, smallest distance allowed between features most of the time
    same_feature_limit = 1e-6  # km, threshold for considering two ray trace results to be the same feature

    # Spice IDs (Note: these 2 variables are not used for the moon case so did not redefine).
    sc_ID = constants.SC_ID
    camera_frame = constants.CAM_FRAME

    # Spice moon info.
    body_ID = constants.BODY_ID
    body_frame = constants.BODY_FRAME
    # Defined false to use Ellipsoid shape model instead of loading kernel.
    use_shape_model = False
    if use_shape_model:
        shape_model_file = './kernels/67P_SHAPEX_160404_64.pickle'
    body_mean_radius = constants.BODY_MEAN_RADIUS
    # Use the body mean radius for all 3 radii.
    body_radius = [body_mean_radius, body_mean_radius, body_mean_radius]

    # Define the camera model parameters.
    pitch = constants.PITCH
    n_cols = constants.NCOLS
    n_rows = constants.NROWS
    horizontal_fov = constants.HORZ_FOV
    focal_length = (n_cols/2 * pitch) / np.tan(np.deg2rad(horizontal_fov/2))

    camera_model = PinholeModel(focal_length=focal_length, kx=1/pitch, ky=1/pitch,
                                n_cols=n_cols, n_rows=n_rows, px=(n_cols-1)/2, py=(n_rows-1)/2)

    spice.furnsh(metakernel)

    # Set up imaging epochs.
    if not start_time or not stop_time:
        spk_coverage = get_spk_coverage(sc_ID)
    if not start_time:
        start_time = spk_coverage[0] + timedelta(seconds=1, microseconds= -spk_coverage[0].microsecond) # pyright: ignore[reportPossiblyUnboundVariable]
    if not stop_time:
        stop_time = spk_coverage[1] - timedelta(microseconds=spk_coverage[0].microsecond) # pyright: ignore[reportPossiblyUnboundVariable]

    times = np.arange(start_time, stop_time, time_step).tolist()

    start_et = datetime_to_et(start_time)
    stop_et = datetime_to_et(stop_time)
    step_et = time_step.total_seconds() / 60
    ets = np.arange(start_et, stop_et, step_et)

    # Spacecraft position in body fixed frame for plotting trajectory.
    # Replaced with fake orbit W/ RP = 10,000 km, epoch (ets),
    # MU (const) = 4.9048695 × 10^12 m^3/s^2, everything else will be 0.
    elt = np.array([body_mean_radius + constants.ORBIT_ALT, 0, 0, 0, 0, 0, 0, constants.BODY_MU]) #km^3/s^2

    # Create an array of sc states. Expecting an n x 3 np.array.
    sc_positions = np.array([spice.conics(elt, float(et)) for et in ets])

    # Set up spice functions.
    body_orientation_inertial = SpiceOrientation(body_frame, 'J2000')
    """The body fixed frame to the inertial frame orientation."""

    sun_position_inertial = SpicePosition('Sun', 'J2000', 'None', sc_ID)
    """The sun inertial position with respect to the spacecraft in the inertial frame."""

    # Set up the shape model. Note: using ellipsoid option (i.e. set use_shape_model to False).
    if use_shape_model:
        # Load the shape model.
        print("\n Loading shape model...")
        with open(shape_model_file, 'rb') as tree_file: # pyright: ignore[reportPossiblyUnboundVariable]
            body_shape = pickle.load(tree_file)
        print(" Shape model loaded")
    else:
        # Create an ellipsoid object for the shape model.
        body_shape = Ellipsoid(np.zeros(3, dtype=np.float64),
                            principal_axes=np.array(body_radius))

    # Set up the scene objects.
    body_object = SceneObject(
        body_shape, position_function=body_position, orientation_function=body_orientation, name='Body')

    sun_object = SceneObject(
        Point([0, 0, 0]), position_function=sun_position, orientation_function=sun_orientation, name='Sun')

    scene = Scene(target_objs=[body_object], light_obj=sun_object)

    """BUILD THE FEATURE CSV FILE"""

    # Generate all of the data
    DATA_HISTORY = []
    prepare_data(len(times)-1, np.asanyarray(times), apparent_diameter_threshold, feature_spacing_limit,
                 same_feature_limit, body_object, sun_object, camera_model, scene)

    # Output features to a csv
    output_features(args.csv_filepath_feat) # pyright: ignore[reportPossiblyUnboundVariable]

    """BUILD MAPLET OBJECTS FROM DEM FEATURES"""

    # Define the directory to save feature maplets to.
    maplet_out_dir = constants.MAPLET_DIR
    maplet_out_dir.mkdir(exist_ok=True, parents=True)

    # Load in the list of features from a feature csv file.
    feature_df = load_feature_csv(constants.FEATURE_CATALOG_CSV)
    feature_df.drop_duplicates(inplace=True)

    # Load in the DEM shape data.
    lbl = constants.DEM_DATA
    dem_data: gdal.Dataset = gdal.Open(lbl)
    band: gdal.Band = dem_data.GetRasterBand(1)

    # Retrieve the image data and convert to km.
    scale_image = 1000 # km
    raw_data = band.ReadAsArray()
    assert isinstance(raw_data, np.ndarray), "something is wrong with the DEM data"
    img_data = (raw_data.astype(np.float32) * band.GetScale() + band.GetOffset())
    img_data /= scale_image
    del raw_data

    # Interpolate the DEM shape lat, lon coordinates into a grid.
    interp = dem_interp(dem_data, img_data)

    # Load in the albedo data.
    alb_img = constants.ALBEDO_DATA
    alb_data: gdal.Dataset = gdal.Open(alb_img)
    alb_band: gdal.Band = alb_data.GetRasterBand(1)
    raw_alb_data = alb_band.ReadAsArray()
    assert isinstance(raw_alb_data, np.ndarray), "something is wrong with the albedo data"
    albedo_map = (raw_alb_data.astype(np.float32) * alb_band.GetScale() + alb_band.GetOffset())

    # Interpolate the albedo lat/lon coordinates into a grid.
    a_interp = albedo_interp(alb_data, albedo_map)

    # Define the underlying maplet grid.
    msize = constants.MAPLET_SIZE # 99 x 99 grid.

    mgx, mgy = np.meshgrid(np.arange(-(msize//2), msize//2+1),
                            np.arange(-(msize//2), msize//2+1))

    # Loop through each feature and create a maplet for each feature in the DEM map.
    for ind, feature in feature_df.iterrows():

        # Convert the feature to a maplet object.
        maplet = feature_to_maplet(feature, msize, mgx, mgy, interp, a_interp)

        f_ind = int(ind) + 1 # pyright: ignore[reportArgumentType]
        # Save the maplet to a MAP file.
        name = 'AA{:04d}.MAP'.format(f_ind)
        maplet.write(maplet_out_dir / (name))        
        print(f'Feature {f_ind} of {feature_df.shape[0]} finished', flush=True)

        # Include to create GIANT OBJ files for each maplet as well.
        # maplet_to_obj(maplet, name)
