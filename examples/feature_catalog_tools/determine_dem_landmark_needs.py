#!/usr/bin/env python

"""Given a body, spacecraft, and trajectory, this script builds a csv file of
   optimized landmark's on the body within the camera field of view based on the
   SC trajectory. This script also displays the features found on the body to a web app.
"""

from datetime import datetime, timedelta
from typing import Tuple, Union, Any
from copy import deepcopy
import argparse
import pickle
import csv
import webbrowser

import os
import numpy as np
import spiceypy as spice

from numpy.typing import NDArray

import dem_constants as constants

from giant.camera_models import CameraModel, PinholeModel
from giant.ray_tracer.shapes import Ellipsoid, Point
from giant.rotations import Rotation
from giant.utilities.spice_interface import SpiceOrientation, SpicePosition, datetime_to_et
from giant.ray_tracer.scene import SceneObject, Scene
from giant.ray_tracer.rays import Rays
from giant.ray_tracer.kdtree import KDTree
from giant.ray_tracer.illumination import ILLUM_DTYPE
from giant._typing import PATH

import matplotlib.path as mpath

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

# turn off Dash logging every callback in terminal
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

""" HELPER FUNCTIONS """
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

def get_spk_coverage(sc_ID: float | str=-123) -> list[datetime]:
    """Get the spk timespan coverage of the images.
    
    :param sc_ID: The NAIF Spice spracecraft ID.

    :return: A list of datetimes representing the spk coverage of the images.
    """
    # Get coverage of last furnished bsp that contains sc_ID
    sc_ID = int(sc_ID)
    coverage = []
    n_spks = spice.ktotal('SPK')
    for i in range(n_spks):
        kernel_path = spice.kdata(i, 'SPK')[0]
        ids = spice.spkobj(kernel_path)
        if '.bsp' in kernel_path and sc_ID in ids:
            # Get spk coverage
            coverage = spice.stypes.SPICEDOUBLE_CELL(2000)
            spice.scard(0, coverage)
            spice.spkcov(kernel_path, sc_ID, coverage)
            coverage = [et_to_datetime(et) for et in coverage]

    return coverage

def et_to_datetime(et: float) -> datetime:
    """Convert an ET to a datetime object.

    :param et: A float representing an ET.

    :return: A datetime object created from the ET input.
    """
    return datetime.strptime(spice.et2utc(et, 'ISOD', 6), '%Y-%jT%H:%M:%S.%f') # type: ignore

def unit_to_lat_lon(unit_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """A helper function to convert a unit vector to lat, lon.
    
    :param unit_vector: The unit vector to convert to lat, lon.

    :return: The resulting latitude and longitude converted from the unit vector.
    """
    # Compute the ra, dec of the unit vector.
    dec = np.arcsin(unit_vector[2])
    ra = np.arctan2(unit_vector[1], unit_vector[0])

    # Convert the ra, dec to lat, lon coordinates.
    longitude = ra
    longitude[longitude < 0] += 2*np.pi
    latitude = dec

    return latitude, longitude

def plot_shape(body_shape: Union[Ellipsoid, KDTree]) -> go.Mesh3d:
    """Create a plot object given a body shape.
    
    :param body_shape: The body shape object to create.

    :return: A splotly.graph_objects.Mesh3d representing a body shape.
    """

    if isinstance(body_shape, Ellipsoid):
        phi = np.linspace(0, 2*np.pi)
        theta = np.linspace(-np.pi/2, np.pi/2)
        phi, theta=np.meshgrid(phi, theta)

        x = np.cos(theta) * np.sin(phi) * body_shape.principal_axes[0]
        y = np.cos(theta) * np.cos(phi) * body_shape.principal_axes[1]
        z = np.sin(theta) * body_shape.principal_axes[2]

        surf = go.Mesh3d({'x': x.flatten(), 
                        'y': y.flatten(), 
                        'z': z.flatten(), 
                        'alphahull': 0},
                        color='rgb(127, 127, 127)')
    else:
        x = body_shape.surface.vertices[:,0]
        y = body_shape.surface.vertices[:,1]
        z = body_shape.surface.vertices[:,2]

        i = body_shape.surface.facets[:,0]
        j = body_shape.surface.facets[:,1]
        k = body_shape.surface.facets[:,2]

        # Set up trace
        surf = go.Mesh3d(x=x, y=y, z=z,
                        i=i, j=j, k=k,
                        color='rgb(127, 127, 127)')

    return surf

def great_circle(start_point, end_point):
    start_dir = start_point/np.linalg.norm(start_point)
    end_dir = end_point/np.linalg.norm(end_point)
    plane_normal = np.cross(start_dir, end_dir)
    plane_normal /= np.linalg.norm(plane_normal)

    local_x = start_dir
    local_y = np.cross(plane_normal, local_x)
    local_z = plane_normal

    rot = np.array([local_x, local_y, local_z])

    start_local = rot @ start_point
    end_local = rot @ end_point

    x_local = np.linspace(start_local[0], end_local[0], 100)
    y_local = np.linspace(start_local[1], end_local[1], 100)
    z_local = np.zeros(100)

    out = rot.T @ np.array([x_local, y_local, z_local])
    out /= np.linalg.norm(out, axis=0, keepdims=True)
    out *= np.linalg.norm(start_point)

    return out

def is_visible(illum: np.ndarray) -> NDArray[np.bool]:
    """Determine whether the illum point is visible at the point in time on the body.
    
    :param illum: The illumination data for a single point on the body at a point in time.

    :return: A list of booleans that represent whether the illumination features are visible.
    """

    n_features = len(illum)
    incidence = -illum['incidence']
    exidence  = illum['exidence']
    normal    = illum['normal']
    visible   = illum['visible']

    incidence_angle = [np.rad2deg(np.arccos(np.dot(i, n))) for i,n in zip(incidence,normal)]
    exidence_angle  = [np.rad2deg(np.arccos(np.dot(e, n))) for e,n in zip(exidence,normal)]
    phase_angle     = [np.rad2deg(np.arccos(np.dot(i, e))) for i,e in zip(incidence,exidence)]

    for i in range(n_features):
        visible[i] = visible[i] and \
                  (incidence_angle_range[0] <= incidence_angle[i] <= incidence_angle_range[1] ) and \
                  (exidence_angle_range[0]  <= exidence_angle[i]  <= exidence_angle_range[1]) and \
                  (phase_angle_range[0]     <= phase_angle[i]     <= phase_angle_range[1])
        
    return visible

def output_features(csv_filepath: PATH):
    """Output the list of features to a csv file.
    
    :param csv_filepath: The csv filepath to output the feature list to.
    """

    print('\n Feature list:')
    with open(csv_filepath, 'w') as f:
        csvwriter = csv.writer(f, delimiter='\t')

        # header line
        fields = ['# feature ID', 'gsd [m/px]', 'body_fixed_x [km]', 'body_fixed_y [km]', 'body_fixed_z [km]']
        csvwriter.writerow(fields)
        print('\n'+ '  '.join(fields))

        # features
        final_features = DATA_HISTORY[-1]['features'] + DATA_HISTORY[-1]['new']
        for feature in final_features:
            row = [f"{feature[11]:10.0f}", f"{1000*feature[2]:14.3f}", f"{feature[3]:17.8f}", f"{feature[4]:18.8f}", f"{feature[5]:18.8f}"]
            csvwriter.writerow(row)
            print('  '.join(row))

    print(f'\nFeatures output to {csv_filepath}')

    return

def output_image_list(csv_filepath: PATH):
    """Output the list of features per image to a csv file.
    
    :param csv_filepath: The csv filepath to output the list of features per image to.
    """
    print('\n Image list:')
    with open(csv_filepath, 'w') as f:
        csvwriter = csv.writer(f)

        # header line
        fields = ['# Image Number', ' Image Epoch [UTC]', 'Feature ID', ' Sample Location (px)' , ' Line Location (px)' , ' Feature GSD [m/px]',  ' Body-Fixed X [km]', ' Body-Fixed Y [km]', ' Body-Fixed Z [km]']
        csvwriter.writerow(fields)
        print('\n'+ '     '.join(fields))

        # features
        for i in range(len(times)):
            image_epoch = times[i]
            image_features = DATA_HISTORY[i]['old'] + DATA_HISTORY[i]['new']
            for f, feature in enumerate(image_features):
                row = [f"{i+1:10d}", f"{image_epoch}", f"{feature[11]:10.0f}", f"{feature[0]:14.2f}", f"{feature[1]:14.2f}", f"{1000*feature[2]:14.3f}", f"{feature[3]:17.8f}", f"{feature[4]:17.8f}", f"{feature[5]:17.8f}"]
                csvwriter.writerow(row)
                print('  '.join(row))

    print(f'\nFeatures output to {csv_filepath}')

    return

""" POSITION AND ORIENTATION FUNCTIONS """

def body_position(epoch: datetime) -> np.ndarray:
    """The body position with respect to the camera frame.
    
    :param epoch: The datetime to evaluate the position at.

    :return: The body position with respect to the spacecraft in the camera frame.
    """
    return -camera_orientation(epoch).matrix @ camera_position(epoch)

def body_orientation(epoch: datetime) -> Rotation:
    """The body fixed to camera orientation.
    
    :param epoch: The datetime to evaluate the orientation at.

    :return: The rotation from the body frame to the camera frame.
    """
    return camera_orientation(epoch) * body_orientation_inertial(epoch)

def camera_position(epoch: datetime) -> np.ndarray:
    """The spacecraft position with respect to the body in the inertial frame.
    
    :param epoch: The datetime to evaluate the position at.

    :return: The camera position at the input epoch.
    """
    return body_orientation_inertial(epoch).matrix @ spice.conics(elt, spice.datetime2et(epoch))[:3] # type: ignore

def camera_orientation(epoch: datetime) -> Rotation:
    """The camera orientation.
    
    :param epoch: The datetime to evaluate the orientation at.

    :return: The inertial to camera frame orientation at the input epoch.
    """
    # Define camera frame as +Z pointed at body
    z_dir = -camera_position(epoch)
    z_dir /= np.linalg.norm(z_dir)

    # +X pointed towards sun
    x_const = sun_position_inertial(epoch)

    x_const /= np.linalg.norm(x_const)

    # +Y completes right hand rule
    y_dir = np.cross(z_dir, x_const)
    y_dir /= np.linalg.norm(y_dir)

    x_dir = np.cross(y_dir, z_dir)
    x_dir /= np.linalg.norm(x_dir)

    return Rotation(np.array([x_dir, y_dir, z_dir]))

def sun_position(epoch: datetime) -> np.ndarray:
    """The sun position with respect to the camera frame.
    
    :param epoch: The datetime to evaluate the position at.

    :return: The sun position with respect to the spacecraft in the camera frame.
    """
    return camera_orientation(epoch).matrix @ sun_position_inertial(epoch)

def sun_orientation(*args) -> Rotation:
    """The sun orientation.
    
    :return: An identity rotation representing the inertial to body frame.
    """
    # Always set the sun orientation to be the identity rotation (J2000)
    # because its orientation is not applicable.
    return Rotation([0, 0, 0])

""" MAIN FEATURE FINDER FUNCTION HANDLING """

# Define a feature data history array to store feature data throughout the script.
DATA_HISTORY = []
def prepare_data(n: int, times: np.ndarray, apparent_diameter_threshold: int,
                 feature_spacing_limit: int, same_feature_limit: float, body_object: SceneObject,
                 sun_object: SceneObject, camera_model: CameraModel, scene: Scene):
    """Function to prepare the image and feature data and store the results in DATA_HISTORY.
    
    :param n: The number of features to process.
    :param times: An array of imaging epochs.
    :param apparent_diameter_threshold: The threshold to compare against to only find features
                                        larger than this apparent diameter [pixels].
    :param feature_spacing_limit: The smallest distance allowed between features [pixels].
    :param same_feature_limit: The threshold for considering two ray trace results to be the same
                               feature [km].
    :param body_object: The scene object representing the target body.
    :param sun_object: The scene object representing the sun.
    :param camera_model: The camera model object.
    :param scene: The scene object containing the body and sun objects.
    """

    # Find features for each imaging epoch.
    print("\n Finding features...\n")
    while len(DATA_HISTORY) < n+1:
        i = len(DATA_HISTORY)
        try:
            working: dict[str, Any] = deepcopy(DATA_HISTORY[-1])
            working["features"].extend(working["new"])
            working["new"] = []
            working["old"] = []
            working["camera bounds 3d"] = []
        except IndexError:
            working: dict[str, Any] = {"features": [], "old": [], "new": [],
                       "camera bounds 3d": [], "feature count history": [],
                       "feature count per image": []}
        DATA_HISTORY.append(working)

        image_time = times[i]

        body_object.place(image_time)
        sun_object.place(image_time)

        # Determine the extent of the body.
        apparent_diameter = body_object.get_apparent_diameter(camera_model)
        existing_features = []
        if apparent_diameter > apparent_diameter_threshold:

            # print image info
            print(f"{image_time}: ad={apparent_diameter:.1f} pixels, type=TRN")

            # determine the bounds of the camera FOV at the body surface (for 3D plot)
            fov_center_ray = Rays([0, 0, 0.0], [0, 0, 1.0])
            fov_center_trace = body_object.shape.trace(fov_center_ray) # type: ignore
            fov_center_distance = np.linalg.norm(fov_center_trace['intersect'])

            fov_corners_pxln = np.array([[0, camera_model.n_cols, camera_model.n_cols,              0],
                                         [0,             0,       camera_model.n_rows, camera_model.n_rows]])
            camera_bounds_cf = fov_center_distance * camera_model.pixels_to_unit(fov_corners_pxln)
            camera_bounds_bf = body_object.orientation.matrix.T @ (camera_bounds_cf.T - body_object.position).T

            working['camera bounds 3d'] = np.hstack([camera_bounds_bf, camera_bounds_bf[:, 0, np.newaxis]]).T
            
            # check if any existing features are in the FOV
            camera_fov = mpath.Path(fov_corners_pxln.T, closed=False)

            features = working['features']
            for feature in features:
                feature_bf = feature[3:6]
                feature_cf = body_object.position + body_object.orientation.matrix @ (feature_bf).T
                feature_pxln = camera_model.project_onto_image(feature_cf)
                feature[0:2] = feature_pxln

                if existing_features:
                    distance_from_existing = np.linalg.norm(np.array(existing_features)[:,0:2] - feature_pxln, axis=1)
                else:
                    distance_from_existing = np.inf

                if camera_fov.contains_point(feature_pxln.tolist()) and ~np.any(distance_from_existing < feature_spacing_limit):
                    # figure out the GSD of the camera at this point
                    position_normal_camera = body_object.orientation.matrix @ feature[3:9].reshape(
                        -1, 3).T
                    position_normal_camera[:, 0] += body_object.position

                    cam_gsd = camera_model.compute_ground_sample_distance(
                        position_normal_camera[:, 0], position_normal_camera[:, 1])

                    feature_gsd = feature[2]

                    if feature_gsd/gsd_tolerance <= cam_gsd <= gsd_tolerance*feature_gsd:
                        # figure out if feature is still visible and illuminated
                        feature_ray = Rays(np.zeros(3, dtype=np.float64), camera_model.pixels_to_unit(feature_pxln))
                        feature_trace = scene.get_illumination_inputs(feature_ray, return_intersects=True)

                        same_body_location = np.linalg.norm(feature_trace[1]['intersect'] - feature_cf) < same_feature_limit

                        if same_body_location and is_visible(feature_trace[0]):
                            existing_features.append(feature)

            working['old'] = existing_features

            # If there aren't already n_features_per_img_min number of features, add new features.
            if len(existing_features) < n_features_per_img_min:
                # create new features
                fov_center_pxln = np.array([[camera_model.n_cols / 2],
                                            [camera_model.n_rows / 2]])

                # Creates an array of ideal feature locations in the shape of a five-side of a die.
                new_feature_pxln = np.array([[camera_model.n_cols / 4, 3 * camera_model.n_cols / 4,
                                              3 * camera_model.n_cols / 4, camera_model.n_cols / 4,
                                              camera_model.n_cols / 2],
                                             [camera_model.n_rows / 4, camera_model.n_rows / 4,
                                              3 * camera_model.n_rows / 4, 3 * camera_model.n_rows / 4,
                                              camera_model.n_rows / 2]])

                # Use ray tracing to check whether target is located at ideal feature locations and
                # whether the feature is visible.
                new_feature_rays = Rays(np.zeros(3, dtype=np.float64),
                                        camera_model.pixels_to_unit(new_feature_pxln))
                new_feature_trace = scene.get_illumination_inputs(new_feature_rays, return_intersects=True)

                # For rays not hitting body or not visible, take midpoint between ray and center
                # of image until it hits body and is visible or is too close to middle.
                for k in np.where(~new_feature_trace[0]['visible'])[0]:
                    test_pxln_wrt_center = (new_feature_pxln[:,k] - fov_center_pxln.T).T * 3/4
                    while np.linalg.norm(test_pxln_wrt_center) > feature_spacing_limit:
                        test_pxln = fov_center_pxln + test_pxln_wrt_center
                        test_ray = Rays(np.zeros(3, dtype=np.float64),camera_model.pixels_to_unit(test_pxln))
                        test_trace = scene.get_illumination_inputs(test_ray, return_intersects=True)
                        if is_visible(test_trace[0]):
                            new_feature_trace[0][k] = test_trace[0]
                            new_feature_trace[1][k] = test_trace[1]
                            new_feature_pxln[:,k] = test_pxln.T
                            break
                        test_pxln_wrt_center *= 3/4 # move ray 1/4 of the way to center on next loop

                new_feature_locs_camera = new_feature_trace[1]

                # figure out the gsd at the new feature locs
                new_feature_gsds = camera_model.compute_ground_sample_distance(
                    new_feature_locs_camera['intersect'].T, new_feature_locs_camera['normal'].T).ravel() # type: ignore

                new_feature_bf = body_object.orientation.matrix.T @ (
                    new_feature_locs_camera['intersect'] - body_object.position).T
                new_feature_bf_unit = new_feature_bf / np.linalg.norm(new_feature_bf, axis=0, keepdims=True)

                new_feature_latlon = np.rad2deg(np.array(unit_to_lat_lon(new_feature_bf_unit)))

                new_feature_normal_bf = body_object.orientation.matrix.T @ new_feature_locs_camera['normal'].T

                # only take visible features
                keep_features = np.where(is_visible(new_feature_trace[0]))[0]

                # only take the furthest from the existing features in image coordinates
                if existing_features:
                    existing_features = np.array(existing_features)
                    distance_from_existing = np.array([np.min(np.linalg.norm(existing_features[:,0:2] - pxln, axis=1))
                                                            for pxln in new_feature_pxln[:].T])

                    # set distance to inf for non-visible features so they are sorted last and not included in this cut
                    distance_from_existing[~is_visible(new_feature_trace[0])] = np.inf
                    n_to_throw_out = len(keep_features) - n_features_per_img_min + len(existing_features)
                    throw_out = np.argsort(distance_from_existing)[0:n_to_throw_out]
                    keep_features = keep_features[~np.isin(keep_features,throw_out)]
                else:
                    keep_features = keep_features[:n_features_per_img_min]


                for ind, k in enumerate(keep_features, len(working['features'])):
                    feature_id = ind + 1; 
                    working['new'].append(np.array([new_feature_pxln[0][k], new_feature_pxln[1][k], new_feature_gsds[k],
                                                    new_feature_bf[0][k], new_feature_bf[1][k], new_feature_bf[2][k],
                                                    new_feature_normal_bf[0][k], new_feature_normal_bf[1][k], new_feature_normal_bf[2][k],
                                                    new_feature_latlon[0][k], new_feature_latlon[1][k], feature_id]))

        working['feature count history'].append(
            len(working['features']) + len(working['new']))
        working['feature count per image'].append(
            len(existing_features) + len(working['new']))

""" SETTING DEFINITION HANDLING """

if __name__ == "__main__":
    # Set up arg parser
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

# Other settings
apparent_diameter_threshold = 100  # pixels, only find features if body is larger than this
feature_spacing_limit = 50  # pixels, smallest distance allowed between features most of the time
same_feature_limit = 1e-6  # km, threshold for considering two ray trace results to be the same feature

# Spice (Note: these 2 variables are not used for the moon case so did not redefine).
sc_ID = constants.SC_ID
camera_frame = constants.CAM_FRAME

# Spice moon info.
body_ID = constants.BODY_ID
body_frame = constants.BODY_FRAME
# Defined false to use Ellipsoid shape model instead of loading kernel.
use_shape_model = False
if use_shape_model:
    shape_model_file = './kernels/67P_SHAPEX_160404_64.pickle'
body_mean_radius = constants.BODY_MEAN_RADIUS # km
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
    start_time = spk_coverage[0] + timedelta(seconds=1, microseconds= -spk_coverage[0].microsecond) # type: ignore
if not stop_time:
    stop_time = spk_coverage[1] - timedelta(microseconds=spk_coverage[0].microsecond) # type: ignore

times = np.arange(start_time, stop_time, time_step).tolist()

start_et = datetime_to_et(start_time)
stop_et = datetime_to_et(stop_time)
step_et = time_step.total_seconds() / 60
ets = np.arange(start_et, stop_et, step_et)

# Spacecraft position in body fixed frame for plotting trajectory.
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

""" WEB APP HANDLING """

# Set up web app
app = dash.Dash("Feature Finder", meta_tags=[{
                "name": "viewport", "content": "width=device-width, initial-scale=1"}])
app.title = "Feature Finder"

server = app.server

app_color = {"graph_bg": "#000000",
             "graph_line": "#FFFFFF", "graph_text": "#FFFFFF",
             "2d_bg": "#D3D3D3", "2d_line": "#000000", "2d_text": "#000000"}

app.layout = html.Div(
    [
        # slider
        html.Div(
            [
                dcc.Slider(0, len(times)-1, 1, value=0,
                           id="frame-slider",
                           marks={i: str(times[i])
                                  for i in range(0, len(times), 250)},
                           updatemode="drag",
                           tooltip={"always_visible": True,
                                    "placement": "bottom"},
                           ),
            ],
            style={"width": "100%"},
        ),
        # running total
        html.Div(
            children=[
            # 3d
                html.Div(
                    [
                        dcc.Graph(
                            id="3d-view",
                            figure=dict(
                                layout=dict(
                                    plot_bgcolor=app_color["graph_bg"],
                                    paper_bgcolor=app_color["2d_bg"],
                                    responsive=True
                                ),
                            ),
                            config=dict(autosizable=True) # pyright: ignore[reportArgumentType]
                        ),
                    ],

                    style={"display": "inline-block",
                           "width": "50%"},
                ),
                # running total
                html.Div(
                    [
                        dcc.Graph(
                            id="running-total",
                            figure=dict(
                                layout=dict(
                                    plot_bgcolor=app_color["graph_bg"],
                                    paper_bgcolor=app_color["graph_bg"],
                                    responsive=True
                                ),
                            ),
                            config=dict(autosizable=True) # pyright: ignore[reportArgumentType]
                        ),

                        dcc.Graph(
                            id="lat-lon-view",
                            figure=dict(
                                layout=dict(
                                    plot_bgcolor=app_color["2d_bg"],
                                    paper_bgcolor=app_color["2d_bg"],
                                    responsive=True
                                ),
                            ),
                            config=dict(autosizable=True) # pyright: ignore[reportArgumentType]
                        ),
                    ],
                    style={"display": "inline-block", "width": "50%"},
                ),
            ],
            style={"width": "100%"},

        ),
    ],
    className="app__container",
)

# Slider callback function
@app.callback(Output("3d-view", "figure"),
              Output("lat-lon-view", "figure"),
              Output("running-total", "figure"),
              Input("frame-slider", "value"))
# Input("3d-view-update", "n_intervals"))
def update_3d_view(n):
    image_time = times[n]

    current = DATA_HISTORY[n]

    current_et = datetime_to_et(image_time)
    # Update to use spice.conics
    # current_position = spice.spkpos(
    #     sc_ID, current_et, body_frame, 'NONE', body_ID)[0]
    current_position = spice.conics(elt, current_et)

    sun_position = spice.spkpos(
        'SUN', current_et, body_frame, 'LT+S', body_ID)[0]

    traces = [plot_shape(body_shape),
              go.Scatter3d(x=sc_positions[:, 0], y=sc_positions[:, 1], z=sc_positions[:, 2], mode='lines',
                           line={'color': 'gray'}, name='orbit'),
              go.Scatter3d(x=[current_position[0]], y=[current_position[1]], z=[current_position[2]], mode='markers',
                           marker={'color': 'green', 'size': 10},
                           name='S/C'), ]

    lat_lon_traces = []
    px_ln_traces = []
    if current['features']:
        existing_features = np.array(current["features"])

        traces.append(go.Scatter3d(x=existing_features[:, 3],
                                   y=existing_features[:, 4],
                                   z=existing_features[:, 5],
                                   mode="markers",
                                   name="existing feature set",
                                   hovertext="existing feature",
                                   marker={'color': 'black', 'size': 4}))

        lat_lon_traces.append(go.Scatter(x=existing_features[:, 10],
                                         y=existing_features[:, 9],
                                         mode="markers",
                                         name="existing feature (unused)",
                                         hovertext="existing feature",
                                         marker={'color': 'black'}))

    if current['new']:
        new_features = np.array(current["new"])

        traces.append(go.Scatter3d(x=new_features[:, 3],
                                   y=new_features[:, 4],
                                   z=new_features[:, 5],
                                   mode="markers",
                                   name="added feature",
                                   hovertext="added feature",
                                   marker={'color': 'yellow', 'size': 4}))
        px_ln_traces.append(go.Scatter(x=new_features[:, 0],
                                         y=new_features[:, 1],
                                         mode="markers",
                                         name="added feature",
                                         hovertext="added feature",
                                         marker={'color': 'yellow'},
                                         showlegend = False))
        lat_lon_traces.append(go.Scatter(x=new_features[:, 10],
                                         y=new_features[:, 9],
                                         mode="markers",
                                         name="added feature",
                                         hovertext="added feature",
                                         marker={'color': 'yellow'}))

    if current['old']:
        old_features = np.array(current["old"])

        traces.append(go.Scatter3d(x=old_features[:, 3],
                                   y=old_features[:, 4],
                                   z=old_features[:, 5],
                                   mode="markers",
                                   name="reused feature",
                                   hovertext="reused feature",
                                   marker={'color': 'blue', 'size': 4}))
        px_ln_traces.append(go.Scatter(x=old_features[:, 0],
                                         y=old_features[:, 1],
                                         mode="markers",
                                         name="reused feature",
                                         hovertext="reused feature",
                                         marker={'color': 'blue'},
                                         showlegend = False))
        lat_lon_traces.append(go.Scatter(x=old_features[:, 10],
                                         y=old_features[:, 9],
                                         mode="markers",
                                         name="reused feature",
                                         hovertext="reused feature",
                                         marker={'color': 'blue'}))

    layout = dict(
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        font={"color": "#fff"},
        # height=700,
    )

    if len(current["camera bounds 3d"]) > 0:
        traces.append(go.Scatter3d(x=current["camera bounds 3d"][:,0],
                                   y=current["camera bounds 3d"][:,1],
                                   z=current["camera bounds 3d"][:,2],
                                   mode="lines",
                                   name="camera bounds",
                                   line={'color': 'red'}))

    fig = go.Figure(data=traces, layout=layout)
    current_position /= np.linalg.norm(current_position)/1.1
    fig.update_layout(dict(
        scene_camera=dict(up=dict(x=0, y=0, z=1),
                          center=dict(x=0, y=0, z=0),
                          eye=dict(x=current_position[0], y=current_position[1], z=current_position[2]))),
                      scene=dict(
        xaxis={
            "title": "Body X, km",
            "showbackground": False,
        },
        yaxis={
            "title": "Body Y, km",
            "showbackground": False
        },
        zaxis={
            "title": "Body Z, km",
            "showbackground": False
        },
        aspectmode='data',
        bgcolor=app_color["graph_bg"],),
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["2d_bg"],
        showlegend=False,
        height=800,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    sun_position /= np.linalg.norm(sun_position) / 1e5
    lightposition = {'x': float(sun_position[0]),
                     'y': float(sun_position[1]),
                     'z': float(sun_position[2])}
    fig.update_traces(lightposition=lightposition, selector=dict(type='mesh3d'))


    fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Current FOV", ""),horizontal_spacing=0.15)
    fig2.update_layout(dict( height=400,
                       plot_bgcolor=app_color["2d_bg"], paper_bgcolor=app_color["2d_bg"]),
                       yaxis=dict(color=app_color["2d_line"]), xaxis=dict(color=app_color["2d_line"]),
                       font_color=app_color["2d_line"], showlegend=True,
                       legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
                       margin=dict(t=60,r=40),)
    fig2.add_traces(px_ln_traces,rows=1,cols=1)
    fig2.update_xaxes(row=1,col=1,
        range=[0.0, camera_model.n_cols], tickmode='array', tickvals=[0, camera_model.n_cols/2.0, camera_model.n_cols],
        linecolor=app_color["2d_line"], gridcolor=app_color["2d_line"], showline=True,
        zerolinecolor=app_color["2d_line"], mirror=True, title="Sample, px",
        constrain="domain")
    fig2.update_yaxes(row=1,col=1, tickmode='linear', tick0=0, dtick=camera_model.n_rows/2,
        range=[0.0, camera_model.n_rows], linecolor=app_color["2d_line"], gridcolor=app_color["2d_line"], showline=True,
        zerolinecolor=app_color["2d_line"], mirror=True, title="Line, px",
        scaleanchor="x",
        scaleratio=1,
    )
    fig2.add_traces(lat_lon_traces,rows=1,cols=2)
    fig2.update_xaxes(row=1,col=2,
        range=[-180.0, 180.0], tickmode='linear', tick0=-180, dtick=60,
                      linecolor=app_color["2d_line"], gridcolor=app_color["2d_line"], showline=True,
                      zerolinecolor=app_color["2d_line"], mirror=True, title="Longitude, deg")
    fig2.update_yaxes(row=1,col=2,
        range=[-90.0, 90.0], tickmode='linear', tick0=-90, dtick=30,
                       linecolor=app_color["2d_line"], gridcolor=app_color["2d_line"], showline=True,
    zerolinecolor=app_color["2d_line"], mirror=True, title="Latitude, deg")

    fig3 = go.Figure(data=go.Scatter(x=times, y=current["feature count history"], mode="lines", line={
                     "color": "black"}, name="total"), layout=layout)
    fig3.add_trace(go.Scatter(x=times, y=current["feature count per image"], mode="lines", line={
                     "color": "red"}, name='per image'))

    if n < 3/4 * len(DATA_HISTORY):
        fig3_legend = dict(yanchor="top", y=0.98, xanchor="right", x=0.99,
                            bgcolor="rgb(225,225,225)", bordercolor="Black", borderwidth=1)
    else:
        fig3_legend = dict(yanchor="top", y=0.98, xanchor="left", x=0.01,
                            bgcolor="rgb(225,225,225)", bordercolor="Black", borderwidth=1)

    fig3.update_layout(dict(plot_bgcolor=app_color["2d_bg"], paper_bgcolor=app_color["2d_bg"]),
                       yaxis=dict(color=app_color["2d_line"]), xaxis=dict(color=app_color["2d_line"]),
                       font_color=app_color["2d_line"], showlegend=True,
                       title=str(image_time),
                       height=400,
                       legend=fig3_legend,
                       margin=dict(r=40),)

    fig3.update_xaxes(
        range=[times[0], times[-1]], linecolor=app_color["2d_line"], gridcolor=app_color["2d_line"], showline=True, zerolinecolor=app_color["2d_line"], mirror=True, title="Time")
    fig3.update_yaxes(autorange=True,
                      linecolor=app_color["2d_line"], gridcolor=app_color["2d_line"], showline=True, zerolinecolor=app_color["2d_line"], mirror=True, title="Count of features")

    return fig, fig2, fig3

if __name__ == "__main__":

    # Generate all of the data
    prepare_data(len(times)-1, times, apparent_diameter_threshold, feature_spacing_limit, # pyright: ignore[reportArgumentType]
                 same_feature_limit, body_object, sun_object, camera_model, scene)

    # Output image list to a csv
    output_image_list(args.csv_filepath_img) # pyright: ignore[reportPossiblyUnboundVariable]

    # Output features to a csv
    output_features(args.csv_filepath_feat) # pyright: ignore[reportPossiblyUnboundVariable]

    # Run and open the web app
    host = 'localhost'
    os.environ['HOST'] = host
    port = 9990
    print(f'\n\nIf a browser window does not open, go to to http://{host}:{port}/ in your browser to see visualization\n')
    webbrowser.open(f'http://{host}:{port}', new=2)
    app.run_server(debug=False, host=host, port=port)

