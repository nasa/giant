"""
This module provides a class for interfacing with the USNO GNC (Guidance Navigation and Control) star catalog.

The GNC catalog provides high-precision astrometric data based on GAIA DR3, Hipparcos, and USNO's bright star 
catalog for navigation applications.
"""

from pathlib import Path
from typing import Optional, Iterable, Sequence, Union
from datetime import datetime
import hashlib
import requests

import numpy as np
import pandas as pd

from giant.catalogs.meta_catalog import Catalog, GIANT_COLUMNS, GIANT_TYPES
from giant.catalogs.utilities import PARSEC2KM, AVG_STAR_DIST, apply_proper_motion, AVG_STAR_DIST_SIGMA, DEG2MAS
from giant.utilities.spherical_coordinates import radec_to_unit, radec_distance
from giant._typing import DOUBLE_ARRAY, PATH


DEFAULT_GNC_DIR: Path = Path(__file__).resolve().parent / "data"  
"""
This gives the default directory of the GNC catalog file.

The default location is a directory called "data" in the directory containing this source file.
"""

_GNC_COLS: list[str] = ['ra_2016', 'de_2016', 'plx', 'pmra', 'pmde', 'gmag',
                         'e_ra_2016', 'e_de_2016', 'e_plx', 'e_pmra', 'e_pmde']
"""
This specifies the names of the GNC columns that are required in converting the a GIANT star record
"""

# specify the mapping of GNC columns to GIANT columns
_GNC_TO_GIANT: dict[str, str] = dict(zip(_GNC_COLS, GIANT_COLUMNS))
"""
This specifies the mapping of the GAIA column names to the GIANT star record column names
"""


class USNOGNC(Catalog):
    """
    This class provides an interface to the USNO GNC (Guidance, Navigation, and Control) star catalog.
    
    The GNC catalog is based on GAIA DR3, Hipparcos, and USNO Bright Star data and provides 
    high-precision astrometric measurements for navigation applications. The catalog epoch is 2016.0 
    (January 1, 2016 00:00:00).
    
    The catalog is stored as a CSV file that can be downloaded from the US Naval Observatory.
    """
    
    def __init__(self, catalog_file: Optional[PATH] = None, include_proper_motion: bool = True):
        """
        Initialize the GNC catalog interface.
        
        :param catalog_file: Path to the GNC CSV file. If None, will look for default location.
        """
        
        super().__init__(include_proper_motion=include_proper_motion)
        
        if catalog_file is None:
            # Default location
            self.catalog_file = DEFAULT_GNC_DIR / 'gnc_v1_1_mar_2_2023.csv'
        else:
            self.catalog_file = Path(catalog_file)
            
        self._catalog_data = None
        self.catalog_epoch = 2016.0
        
        # Column mapping from GNC CSV to internal names
        self._gnc_columns = {
            'gnc_id': 'gnc_id',
            'ra_2016': 'ra_deg',
            'de_2016': 'dec_deg', 
            'e_ra_2016': 'ra_sigma_mas',
            'e_de_2016': 'dec_sigma_mas',
            'pmra': 'pmra_mas_yr',
            'pmde': 'pmde_mas_yr',
            'e_pmra': 'pmra_sigma_mas_yr',
            'e_pmde': 'pmde_sigma_mas_yr',
            'plx': 'parallax_mas',
            'e_plx': 'parallax_sigma_mas',
            'gmag': 'magnitude'
        }
        
    def _load_catalog(self):
        """
        Load the GNC catalog from the CSV file.
        """
        if self._catalog_data is None:
            if not self.catalog_file.exists():
                print("USNO GNC file not found at {}".format(self.catalog_file), flush=True)
                user_response = input("Would you like to download the USNO GNC data to this location (y/n)?\n"
                                    "    WARNING: THIS REQUIRES AN INTERNET CONNECTION AND USE UP ~400MB OF DISK SPACE!\n    ")
                
                if user_response[:1].lower() == "y":
                    download_gnc(self.catalog_file.parent)
                else:
                    raise FileNotFoundError("The catalog file is missing and you don't want to download it")
                
            # Read the CSV file
            self._catalog_data = pd.read_csv(self.catalog_file, low_memory=False, index_col=0)
            
            # Rename columns for easier access
            # Handle missing values
            self._catalog_data = self._catalog_data.fillna({
                'e_ra_2016': 0.0,
                'e_de_2016': 0.0,
                'pmde': 0.0, 
                'pmra': 0.0,
                'pmde': 0.0, 
                'e_pmra': 0.0,
                'e_pmde': 0.0,
                'plx': 0.0,
                'e_plx': 0.0
            })
    
    def query_catalog(self, ids: Optional[Iterable[tuple[int, int]]] = None, min_ra: float = 0, max_ra: float = 360,
                        min_dec: float = -90, max_dec: float = 90, min_mag: float = -4, max_mag: float = 20,
                        search_center: Optional[Sequence[float] | DOUBLE_ARRAY] = None, search_radius: Optional[float] = None,
                        new_epoch: Optional[Union[datetime, float]] = None) -> pd.DataFrame:
        """
        Query the GNC catalog for stars within a specified region and magnitude range.
        
        :param ra: Right ascension of search center in degrees
        :param dec: Declination of search center in degrees  
        :param radius: Search radius in degrees
        :param max_mag: Maximum magnitude (faintest stars)
        :param min_mag: Minimum magnitude (brightest stars)
        :param epoch: Target epoch for coordinates (default 2000.0)
        :return: DataFrame with star data in GIANT format
        """
        
        self._load_catalog()
        assert self._catalog_data is not None, "something went wrong"
        
        # Filter by magnitude
        mag_filter = ((self._catalog_data.loc[:, 'gmag'] >= min_mag) & 
                     (self._catalog_data.loc[:, 'gmag'] <= max_mag))
        
        # filter by min/max ra and dec
        ra = self._catalog_data.loc[:, "ra_2016"]
        dec = self._catalog_data.loc[:, "de_2016"]
        dir_filter = ((ra >= min_ra) & (ra <= max_ra) & (dec >= min_dec) & (dec <= max_dec))
        
        if search_center is not None:
            assert search_radius is not None, "search radius must be specified if search center is specified"
            
            dir_filter[dir_filter] = radec_distance(np.deg2rad(ra[dir_filter]),
                                                    np.deg2rad(dec[dir_filter]),
                                                    *np.deg2rad(search_center).ravel()) <= np.deg2rad(search_radius)
        
        candidates = self._catalog_data[mag_filter & dir_filter].copy()
        
        if len(candidates) == 0:
            return self._empty_frame()
        
        giant_records = self._convert_to_giant_format(candidates)
            
        # Apply proper motion corrections to target epoch
        if self.include_proper_motion and new_epoch is not None:
            apply_proper_motion(giant_records, new_epoch, copy=False)
            
        return giant_records
        
    def _convert_to_giant_format(self, gnc_records: pd.DataFrame) -> pd.DataFrame:
        """
        Convert GNC records to GIANT format.
        
        :param gnc_records: DataFrame with GNC catalog data
        :return: DataFrame in GIANT format
        """
        gnc_records.index.name = "GNC_ID"
        
        giant_records = gnc_records.loc[:, _GNC_COLS]
        giant_records.rename(columns=_GNC_TO_GIANT, inplace=True)
        giant_records = giant_records.assign(epoch=2016.0)
        giant_records.astype(GIANT_TYPES)
        giant_records.loc[:, "source"] = "USNO GNC V1.1"
        
        # Use parallax for distance calculation where available
        invalid_plx = giant_records.loc[:, 'distance'] <= 0
        # Distance in parsecs = 1000 / parallax_mas, then convert to km
        giant_records.loc[:, 'distance_sigma'] /= giant_records.loc[:, 'distance'] ** 2  # convert parallax std to distance std
        giant_records.loc[:, 'distance_sigma'] *= 1000 * PARSEC2KM  # convert to km
        giant_records.loc[:, 'distance'] /= 1000  # MAS to arcsecond
        giant_records.loc[:, 'distance'] **= -1  # parallax to distance (arcsecond to parsec)
        giant_records.loc[:, 'distance'] *= PARSEC2KM  # parsec to kilometers
        giant_records.loc[invalid_plx, "distance"] = AVG_STAR_DIST
        giant_records.loc[invalid_plx, "distance_sigma"] =  AVG_STAR_DIST_SIGMA
        
        # convert mas to deg
        giant_records.loc[:, "ra_sigma"] /= DEG2MAS
        giant_records.loc[:, "dec_sigma"] /= DEG2MAS
        giant_records.loc[:, "ra_proper_motion"] /= DEG2MAS
        giant_records.loc[:, "dec_proper_motion"] /= DEG2MAS
        giant_records.loc[:, "ra_pm_sigma"] /= DEG2MAS
        giant_records.loc[:, "dec_pm_sigma"] /= DEG2MAS
        
        return giant_records
    
    def _empty_frame(self) -> pd.DataFrame:
        """
        Return an empty DataFrame with the correct GIANT columns.
        
        :return: Empty DataFrame
        """
        empty_data = {col: pd.Series(dtype=GIANT_TYPES[col]) for col in GIANT_COLUMNS}
        empty_data['epoch'] = pd.Series(dtype=GIANT_TYPES['epoch'])
        
        df = pd.DataFrame(empty_data)
        df.index.name = 'gnc_id'
        return df


def download_gnc(target_directory: Path, verify_checksum: bool = True):
    """
    Download the USNO GNC catalog from the official source.
    
    This function downloads the GNC catalog CSV file and optionally verifies its integrity
    using the provided SHA256 checksum.
    
    :param target_directory: Directory to save the catalog file
    :param verify_checksum: Whether to verify the downloaded file's checksum
    """
    
    target_directory = Path(target_directory)
    target_directory.mkdir(exist_ok=True, parents=True)
    
    catalog_url = "https://crf.usno.navy.mil/data_products/ICRF/GNC/2023/gnc_v1_1_mar_2_2023.csv"
    checksum_url = "https://crf.usno.navy.mil/data_products/ICRF/GNC/2023/checksum.sha256"
    
    catalog_file = target_directory / "gnc_v1_1_mar_2_2023.csv"
    
    print(f"Downloading GNC catalog to {catalog_file}")
    
    # Download the catalog file
    response = requests.get(catalog_url, stream=True)
    response.raise_for_status()
    
    with open(catalog_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Download complete. File size: {catalog_file.stat().st_size:,} bytes")
    
    if verify_checksum:
        print("Verifying checksum...")
        
        # Download checksum file
        checksum_response = requests.get(checksum_url)
        checksum_response.raise_for_status()
        
        # Parse checksum (format: "hash  filename")
        checksum_line = checksum_response.text.strip()
        expected_hash = checksum_line.split()[0]
        
        # Calculate actual hash
        sha256_hash = hashlib.sha256()
        with open(catalog_file, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        actual_hash = sha256_hash.hexdigest()
        
        if actual_hash == expected_hash:
            print("✓ Checksum verification passed")
        else:
            print(f"✗ Checksum verification failed!")
            print(f"Expected: {expected_hash}")
            print(f"Actual:   {actual_hash}")
            raise ValueError("Downloaded file checksum does not match expected value")
    
    print("GNC catalog download completed successfully")
