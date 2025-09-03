"""This file tests the results of the DEM feature catalog tool.

    To run this test, you first must run the script: run_dem_features.py
    This script will produce the necessary feature catalog csv file and
    the resulting Maplet objects from the DEM projection of the features.

    Note that the directory location of the feature catalog csv file, and Maplets
    are defined in dem_constants.py. These values can be viewed and/or changed
    within the constants file.
"""

import unittest
import numpy as np

from pathlib import Path
import spiceypy as spice
from osgeo import gdal

import dem_constants as constants

from giant.utilities.stereophotoclinometry import Maplet
from dem_to_landmarks import load_feature_csv, unit_to_lat_lon, feature_in_bodyfixed
from dem_to_landmarks import dem_interp, albedo_interp
from typing import cast

import pandas as pd

LOCALDIR = Path(__file__).parent

class TestDemMaplets(unittest.TestCase):

    # set up function
    def setUp(self):
        # Directory name to get MAP files from.
        dirname = constants.MAPLET_DIR

        # Iterate over all files and save to list of maplet objects in sorted order.
        self.maplets = {maplet_path.stem.lstrip('A'): Maplet(maplet_path) for maplet_path in sorted(dirname.glob('*.MAP'))}
        
        # Load in the list of features from a feature csv file.
        csv_file = constants.FEATURE_CATALOG_CSV
        self.feature_df: pd.DataFrame = load_feature_csv(csv_file)
        self.feature_df.drop_duplicates(inplace=True)

        # Define the underlying maplet grid.
        msize = constants.MAPLET_SIZE # 99 x 99 grid.
        self.mgx, self.mgy = np.meshgrid(np.arange(-(msize//2), msize//2+1),
                                         np.arange(-(msize//2), msize//2+1))

        # Load in the DEM shape data.
        lbl = constants.DEM_DATA
        self.dem_data: gdal.Dataset = gdal.Open(lbl)
        band: gdal.Band = self.dem_data.GetRasterBand(1)
        
        # Retrieve the image data and convert to km.
        scale_image = 1000 # km
        self.img_data = (band.ReadAsArray().astype(np.float32) * band.GetScale() + band.GetOffset())
        self.img_data /= scale_image

        # Interpolate the DEM shape lat, lon coordinates into a grid.
        self.interp = dem_interp(self.dem_data, self.img_data)

        # Load the albedo map.
        alb_img = constants.ALBEDO_DATA
        alb_data: gdal.Dataset = gdal.Open(alb_img)
        alb_band: gdal.Band = alb_data.GetRasterBand(1)
        albedo_map = (alb_band.ReadAsArray().astype(np.float32) * alb_band.GetScale() + alb_band.GetOffset())

        # Interpolate the albedo lat/lon coordinates into a grid.
        self.a_interp = albedo_interp(alb_data, albedo_map)
    
    def tearDown(self):
        # unload all kernels
        spice.kclear()
    
    def test_maplet_attributes(self):

        # Test there exists a maplet file for every feature in the csv file.
        self.assertEqual(self.feature_df.shape[0], len(self.maplets))

        # Loop over all maplet files and test their object attributes.
        for map_id, map_obj in self.maplets.items():

            # Get the corresponding csv file feature data as a Series.
            csv_feature: pd.Series = cast(pd.Series, self.feature_df.loc[self.feature_df['# feature ID'] == int(map_id)].squeeze())

            # Test size.
            expected_size = 49 # (99-1)/2
            self.assertEqual(map_obj.size, expected_size)

            # Test scale.
            expected_gsd = float(csv_feature["gsd [m/px]"])
            self.assertEqual(map_obj.scale, expected_gsd)

            # Test rotation_maplet2body.
            # Define the rotation from the maplet frame to the body fixed frame.
            maplet_z = csv_feature["normal_x [km]":"normal_z [km]"].to_numpy()

            # Constrain x to be positive in the direction of +x body fixed.
            maplet_x_const = np.array([1, 0, 0])

            maplet_y = np.cross(maplet_z, maplet_x_const)
            maplet_y /= np.linalg.norm(maplet_y)

            maplet_x = np.cross(maplet_y, maplet_z)
            maplet_x /= np.linalg.norm(maplet_x)

            # Rotate the maplet to the body fixed frame.
            expected_rot_map_bf = np.array([maplet_x, maplet_y, maplet_z]).T

            self.assertTrue((expected_rot_map_bf.astype(">f4") == map_obj.rotation_maplet2body).all())

            # Prepare height data.
            # Test the position_objmap.
            if csv_feature.longitude < 0:
                csv_feature.longitude += 2*np.pi
            feature_lat_lon = csv_feature["latitude":"longitude"]

            # Define the shift from the maplet frame to the body fixed frame in the body fixed frame.
            shift_maplet_to_bf_in_bf = (maplet_z * self.interp(feature_lat_lon)).reshape(3, 1)

            self.assertTrue((shift_maplet_to_bf_in_bf.ravel().astype(">f4") == map_obj.position_objmap).all())

            # Get the true locations of the grid points in the maplet frame.
            mx = self.mgx * csv_feature['gsd [m/px]']
            my = self.mgy * csv_feature['gsd [m/px]']

            # Convert the grid points to locations in the body fixed frame.
            bf_grid_cells = expected_rot_map_bf @ [
                mx.ravel(), my.ravel(), np.zeros(mx.size)] + shift_maplet_to_bf_in_bf
            bf_grid_cell_units = bf_grid_cells / \
                np.linalg.norm(bf_grid_cells, axis=0, keepdims=True)

            # Get the lat, lon for the maplet grid.
            m_latlon = np.vstack(unit_to_lat_lon(bf_grid_cell_units)).T

            # Get the radii for the maplet.
            m_radii = self.interp(m_latlon)

            # Get the vectors for the maplet in the body fixed frame.
            maplet_bf_locs = bf_grid_cell_units * m_radii

            # Rotate and translate the maplet back into the maplet frame.
            maplet_locs_mframe = expected_rot_map_bf.T @ (maplet_bf_locs - shift_maplet_to_bf_in_bf)

            # The heights for the maplet are now just the z values.
            m_heights = maplet_locs_mframe[2].reshape(self.mgx.shape)

            m_heights /= csv_feature["gsd [m/px]"]

            # Test hscale.
            expected_hscale = np.abs(m_heights).max()/np.iinfo('i2').max
            self.assertEqual(map_obj.hscale, expected_hscale.astype(">f4"))

            # Define the map shape for the height and albedo data.
            map_shape = (map_obj.size * 2 + 1, map_obj.size * 2 + 1)

            # Test heights.
            heights = m_heights.T

            # Correct for the maplet object's heights type conversions when
            # writing and reading to a MAP file.
            heights = np.rint(heights / expected_hscale).flatten()
            heights = map_obj.hscale * heights.reshape(map_shape)

            self.assertTrue((heights.astype(">f4") == map_obj.heights).all())

            # Test albedos.
            # Get the albedos for the maplet.
            m_albedos = self.a_interp(m_latlon).reshape(self.mgx.shape)

            # Defines the albedo for the maplet (not shape data).
            albedos = m_albedos.T
            albedos /= albedos.mean()
            albedos = np.clip(albedos, 0, 2.55)

            # Correct for the maplet object's albedo type conversions when
            # writing and reading to a MAP file.
            albedos = np.rint(albedos * 100).flatten()
            albedos = 0.01 * albedos.reshape(map_shape)

            self.assertTrue((albedos == map_obj.albedos).all())
    
    def test_altitudes(self):

        # Loop over all maplet files and get their altitude in the body fixed frame.
        map_latlons = []
        map_alts_bf = []
        for map_id, _ in self.maplets.items():

            # Get the corresponding csv file feature data as a Series.
            csv_feature = cast(pd.Series, self.feature_df.loc[self.feature_df['# feature ID'] == int(map_id)].squeeze())

            # Get the maplet lat, lon corrdinates in the body fixed frame.
            _, _, m_latlon, _ = feature_in_bodyfixed(csv_feature, self.mgx, self.mgy, self.interp)

            map_latlons.append(m_latlon)

            # Get the altitude for the maplet in the body fixed frame.
            m_altitude = self.interp(m_latlon)

            map_alts_bf.append(m_altitude)

        # Get the DEM altitudes in the bodyfixed frame from the img_data.
        dem_alts_bf = []
        for map in map_latlons:
            dem_altitude = []
            for ll_pair in map:
                # Find the indices of the lat, lon locations in the DEM grid.
                lat_index = np.where(self.interp.grid[0] == ll_pair[0])
                lon_index = np.where(self.interp.grid[1] == ll_pair[1])

                # Get the altitude of the DEM data at the corresponding lat, lon.
                dem_altitude.append(self.img_data[lat_index][lon_index])

            dem_alts_bf.append(dem_altitude)
        
        # Compare the dem altitudes to the maplet altitudes.
        for map_id, map_alt in enumerate(map_alts_bf):
            self.assertTrue((map_alt[0] == dem_alts_bf[map_id][0]).all())

if __name__ == '__main__':
    unittest.main()
