#!/usr/bin/env python

"""This file takes a csv file of landmarks and uses DEM data to create Maplet
    objects that represent the csv features on the DEM map projection.

    To run this script, you first must run the determine_dem_landmark_needs.py
    script to create the csv file of the landmarks to create DEM map projections of.

    To run this script, you will also need to download:
     
    DEM Data: LDEM_128.LBL from https://imbrium.mit.edu/BROWSE/LOLA_GDR/CYLINDRICAL/ELEVATION/

    Albedo Data: LDAM_10_FLOAT.LBL from https://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/FLOAT_IMG/

    Note that the directory location of the DEM data, albedo data, feature catalog csv file,
    and Maplets are defined in dem_constants.py. These values can be viewed and/or changed
    within the constants file.
"""

import numpy as np
import pandas as pd
from osgeo import gdal, osr

import dem_constants as constants

from scipy.interpolate import RegularGridInterpolator
from giant.utilities.stereophotoclinometry import Maplet
from giant._typing import PATH

"""HELPER FUNCTIONS"""

def unit_to_lat_lon(unit_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

def bodyfixed_to_latlon(bf_x: np.ndarray, bf_y: np.ndarray, bf_z: np.ndarray) -> list[tuple[float, float]]:
    """A helper function to convert body fixed position vector to lat, lon.
    
    :param bf_x: The x-component of the body fixed position vector.
    :param bf_y: The y-component of the body fixed position vector.
    :param bf_z: The z-component of the body fixed position vector.

    :return: The resulting latitude and longitude converted from the body fixed position.
    """
    lat = np.arctan2(bf_y, bf_x)
    lon = np.arcsin(bf_z / np.linalg.norm(np.hstack([bf_x, bf_y, bf_z])))

    return list(zip(lat, lon))

def bodyfixed_to_direction(bf_x: np.ndarray, bf_y: np.ndarray,
                           bf_z: np.ndarray) -> list[tuple[float, float, float]]:
    """A helper function to convert body fixed position vector to direction vector.
    
    :param bf_x: The x-component of the body fixed position vector.
    :param bf_y: The y-component of the body fixed position vector.
    :param bf_z: The z-component of the body fixed position vector.

    :return: The resulting direction vector components converted from the body fixed position.
    """
    r = np.linalg.norm(np.hstack([bf_x, bf_y, bf_z]))
    nx, ny, nz = bf_x / r, bf_y / r, bf_z / r

    return list(zip(nx, ny, nz))

def load_feature_csv(filepath: PATH) -> pd.DataFrame:
    """A helper function to load in the feature csv file, fill in the lat, lon and
    normal (direction) components, and create a dataFrame of the feature data.

    The csv file must have the columns:
    feature ID, gsd [m/px], body_fixed_x [km], body_fixed_y [km], body_fixed_z [km]

    The following columns will be added to the csv file by this function:
    latitude, longitude, normal_x [km], normal_y [km], normal_z [km]

    All the above columns will also be the columns of the returned dataFrame.

    :param filepath: The csv file containing feature data.
    
    :return: A pandas.dataFrame containing the feature data.
    """

    print(f'\nFeature updates output to {filepath}')

    # Read the csv file into a pandas dataframe.
    df = pd.read_csv(filepath, delimiter='\t')

    # Fill the feature csv file with lat/lon and normal (direction) values for each feature.
    bf_x, bf_y, bf_z = df['body_fixed_x [km]'].to_numpy(), df['body_fixed_y [km]'].to_numpy(), df['body_fixed_z [km]'].to_numpy()
    
    df[['latitude', 'longitude']] = bodyfixed_to_latlon(bf_x, bf_y, bf_z)
    df[['normal_x [km]', 'normal_y [km]', 'normal_z [km]']] = bodyfixed_to_direction(bf_x, bf_y, bf_z)

    # Add the new lat, lon and normal variable columns to the csv file.
    df.to_csv(filepath, index=False, sep='\t', float_format='%15.8f')

    return df

def dem_interp(dem_data: gdal.Dataset, img_data: np.ndarray) -> RegularGridInterpolator:
    """Interpolate the DEM map into a rectilinear grid in arbitrary dimensions.
    
    :param dem_data: The DEM map projection.
    :param img_data: An array representing the band covered by the DEM image data.

    :return: A 3D grid of the DEM map.
    """

    # Determine the lat, lon from the dem map projection.
    proj_crs: osr.SpatialReference = dem_data.GetSpatialRef()
    latlon_crs = osr.SpatialReference()
    latlon_crs.CopyGeogCSFrom(proj_crs)
    transform_proj_to_latlon: osr.CoordinateTransformation = osr.CreateCoordinateTransformation(proj_crs, latlon_crs)
    gt_from_map = dem_data.GetGeoTransform()
    
    longitude = []
    for x in range(dem_data.RasterXSize):
        lon = np.deg2rad(transform_proj_to_latlon.TransformPoint(*gdal.ApplyGeoTransform(gt_from_map, x, dem_data.RasterYSize))[1])
        if lon < 0:
            lon += 2*np.pi
        longitude.append(lon)

    latitude = [] 
    for y in range(dem_data.RasterYSize-1, -1, -1):
        latitude.append(np.deg2rad(transform_proj_to_latlon.TransformPoint(*gdal.ApplyGeoTransform(gt_from_map, 0, y))[0]))

    latitude = np.array(latitude)
    longitude = np.array(longitude)

    # Interpolate the lat, lon coordinates into a grid.
    interp = RegularGridInterpolator((latitude, longitude), img_data, bounds_error=False,
                                     fill_value=img_data.mean())
    
    return interp

def albedo_interp(alb_data: gdal.Dataset, albedo_map: np.ndarray) -> RegularGridInterpolator:
    """Interpolate the albedo map into a rectilinear grid in arbitrary dimensions.
    
    :param alb_data: The albedo map projection.
    :param albedo_map: An array reprsenting the band covered by the albedo map. 

    :return: A 3D grid of the albedo map.
    """

    # Determine the lat, lon from the albedo map projection.
    alb_proj_crs: osr.SpatialReference = alb_data.GetSpatialRef()
    alb_latlon_crs = osr.SpatialReference()
    alb_latlon_crs.CopyGeogCSFrom(alb_proj_crs)
    alb_transform_proj_to_latlon: osr.CoordinateTransformation = osr.CreateCoordinateTransformation(alb_proj_crs, alb_latlon_crs)
    alb_gt_from_map = alb_data.GetGeoTransform()
    
    alb_longitude = []
    for x in range(alb_data.RasterXSize):
        lon = np.deg2rad(alb_transform_proj_to_latlon.TransformPoint(*gdal.ApplyGeoTransform(alb_gt_from_map, x, alb_data.RasterYSize))[1])
        if lon < 0 and x > 0:
            lon += 2*np.pi
        alb_longitude.append(lon)

    alb_latitude = [] 
    for y in range(alb_data.RasterYSize-1, -1, -1):
        alb_latitude.append(np.deg2rad(alb_transform_proj_to_latlon.TransformPoint(*gdal.ApplyGeoTransform(alb_gt_from_map, 0, y))[0]))
        
    alb_latitude = np.array(alb_latitude)
    alb_longitude = np.array(alb_longitude)

    # Interpolate the lat, lon coordinates into a grid.
    a_interp = RegularGridInterpolator((alb_latitude, alb_longitude), albedo_map, bounds_error=False,
                                       fill_value=albedo_map.mean())
    
    return a_interp

def feature_in_bodyfixed(feature: pd.Series, mgx: np.ndarray,
                         mgy: np.ndarray, interp: RegularGridInterpolator) -> tuple:
    """Convert feature attributes into the body fixed frame.

    :param feature: The feature to convert to a maplet object.
    :param mgx: The maplet grid array of x-coordinates.
    :param mgy: The maplet grid array of y-coordinates.
    :param interp: The DEM interpolated grid.

    :return: A tuple of maplet feature attributes in the body fixed frame.
             In order, this includes: the rotation from the maplet to body fixed frame,
             the shift from the maplet frame to the body fixed frame in the body fixed
             frame, an array containing the lat, lon corrdinates of the maplet in the
             body fixed frame, and the vectors for the maplet in the body fixed frame.
    """
    # Define the rotation from the maplet frame to the body fixed frame.
    maplet_z = feature["normal_x [km]":"normal_z [km]"].to_numpy()

    # Constrain x to be positive in the direction of +x body fixed.
    maplet_x_const = np.array([1, 0, 0])

    maplet_y = np.cross(maplet_z, maplet_x_const)
    maplet_y /= np.linalg.norm(maplet_y)

    maplet_x = np.cross(maplet_y, maplet_z)
    maplet_x /= np.linalg.norm(maplet_x)

    # Rotate the maplet to the body fixed frame.
    rot_maplet_to_bf = np.array([maplet_x, maplet_y, maplet_z]).T

    # Define the shift from the maplet frame to the body fixed frame in the body fixed frame.
    if feature.longitude < 0:
        feature.longitude += 2*np.pi
    shift_maplet_to_bf_in_bf = (
        feature["normal_x [km]":"normal_z [km]"].to_numpy() * interp(feature["latitude":"longitude"])).reshape(3, 1)

    # Get the true locations of the grid points in the maplet frame.
    mx = mgx * feature['gsd [m/px]']
    my = mgy * feature['gsd [m/px]']

    # Convert the grid points to locations in the body fixed frame.
    bf_grid_cells = rot_maplet_to_bf @ [
        mx.ravel(), my.ravel(), np.zeros(mx.size)] + shift_maplet_to_bf_in_bf
    bf_grid_cell_units = bf_grid_cells / \
        np.linalg.norm(bf_grid_cells, axis=0, keepdims=True)

    # Get the lat, lon for the maplet grid.
    m_latlon = np.vstack(unit_to_lat_lon(bf_grid_cell_units)).T

    # Get the radii for the maplet.
    m_radii = interp(m_latlon)

    # Get the vectors for the maplet in the body fixed frame.
    maplet_bf_locs = bf_grid_cell_units * m_radii

    return rot_maplet_to_bf, shift_maplet_to_bf_in_bf, m_latlon, maplet_bf_locs

def feature_to_maplet(feature: pd.Series, msize: int, mgx: np.ndarray, mgy: np.ndarray,
                      interp: RegularGridInterpolator, a_interp: RegularGridInterpolator) -> Maplet:
    """Create a maplet for each feature on the DEM map projection.

    :param feature: The feature to convert to a maplet object.
    :param msize: The 1D maplet grid size which defines the grid as msize x msize.
    :param mgx: The maplet grid array of x-coordinates.
    :param mgy: The maplet grid array of y-coordinates.
    :param interp: The DEM interpolated grid.
    :param a_interp: The albedo interpolated grid.

    :return: A maplet representing the feature on the DEM map.
    """
    # Get various feature attributes in the body fixed frame including
    # conversion variables from the maplet frame to the body fixed frame.
    rot_maplet_to_bf, shift_maplet_to_bf_in_bf, m_latlon, maplet_bf_locs = feature_in_bodyfixed(feature, mgx, mgy, interp)

    # Rotate and translate the maplet back into the maplet frame.
    maplet_locs_mframe = rot_maplet_to_bf.T @ (maplet_bf_locs - shift_maplet_to_bf_in_bf)

    # The heights for the maplet are now just the z values.
    m_heights = maplet_locs_mframe[2].reshape(mgx.shape)

    # Get the albedos for the maplet.
    m_albedos = a_interp(m_latlon).reshape(mgx.shape)

    # Create the maplet object for the landmark feature.
    maplet = Maplet()

    # Defines the xy grid for the maplet.
    maplet.scale = feature['gsd [m/px]']
    maplet.size = msize // 2
    maplet.position_objmap = shift_maplet_to_bf_in_bf.ravel()
    maplet.rotation_maplet2body = rot_maplet_to_bf

    # Defines the height (z) for the maplet.
    m_heights /= maplet.scale

    maplet.hscale = np.abs(m_heights).max()/np.iinfo('i2').max
    maplet.heights = m_heights.T

    # Defines the albedo for the maplet (not shape data).
    maplet.albedos = m_albedos.T
    maplet.albedos /= maplet.albedos.mean()
    maplet.albedos = np.clip(maplet.albedos, 0, 2.55)

    return maplet

def maplet_to_obj(maplet: Maplet, name: str):
    """Convert a maplet object to GIANT's OBJ maplet format and write it to an output file.

    The output file will be named: temp_obj_files/ + name + .obj

    :param maplet: The Maplet object to output to a GIANT obj file.
    :param name: The name of the maplet to define the OBJ file.
    """

    # Convert the maplet grid to GIANT's triangular grid.
    tris = maplet.get_triangles()

    # Write the GIANT grid to an OBJ output file.
    with open('temp_obj_files/' + name + '.obj', "w") as ofile:
        for v in tris.vertices:
            ofile.write('v {} {} {}\n'.format(*v))

        for f in tris.facets:
            ofile.write('f {} {} {}\n'.format(*f))

if __name__ == "__main__":

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
    raw_data = band.ReadAsArray()
    
    assert isinstance(raw_data, np.ndarray), "Something is wrong with the data in the DEM file"

    # Retrieve the image data and convert to km.
    scale_image = 1000 # km
    img_data = (raw_data.astype(np.float32) * band.GetScale() + band.GetOffset())
    img_data /= scale_image
    del raw_data

    # Interpolate the DEM shape lat, lon coordinates into a grid.
    interp = dem_interp(dem_data, img_data)

    # Load in the albedo data.
    alb_img = constants.ALBEDO_DATA
    alb_data: gdal.Dataset = gdal.Open(alb_img)
    alb_band: gdal.Band = alb_data.GetRasterBand(1)
    alb_raw_data = alb_band.ReadAsArray()
    assert isinstance(alb_raw_data, np.ndarray), "Something is wrong with the albedo data"
    albedo_map = (alb_raw_data.astype(np.float32) * alb_band.GetScale() + alb_band.GetOffset())
    del alb_raw_data

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
        
        f_num = int(ind)+1 # type: ignore

        # Save the maplet to a MAP file.
        name = 'AA{:04d}.MAP'.format(f_num)
        maplet.write(maplet_out_dir / (name))        
        print(f'Feature {f_num} of {feature_df.shape[0]} finished', flush=True)

        # Include to create GIANT OBJ files for each maplet as well.
        # maplet_to_obj(maplet, name)
