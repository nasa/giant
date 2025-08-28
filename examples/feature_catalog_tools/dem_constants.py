"""This file contains all constants needed for the DEM feature catalog tool."""

from pathlib import Path

# Define directories and filepaths.
MAPLET_DIR = Path('./LCRNS_MAPS_NEW')
FEATURE_CATALOG_CSV = './results/dem_feature_list.csv'
DEM_DATA = './test_data/LDEM_128.LBL'
ALBEDO_DATA = './test_data/LDAM_10_FLOAT.LBL'

# Physical constants.
BODY_MEAN_RADIUS = 1.7374 * 10**3 # (km) in PCK file: 'BODY301_RADII'
BODY_MU = 4.9048695 * 10**3 # km^3/s^2
ORBIT_ALT = 500 # km

# Camera constants.
PITCH =  2.2e-3 # mm?
NCOLS = 2592
NROWS = 1944
HORZ_FOV = 50 # deg

# ID and frame constants.
BODY_ID = 'MOON'
BODY_FRAME = 'IAU_MOON'
# (Note: these 2 variables are not used for the moon case so did not redefine)
SC_ID = '-123'
CAM_FRAME = 'CK_CAESAR'

# Maplet constants.
MAPLET_SIZE = 99 # 99 x 99 maplet grid.