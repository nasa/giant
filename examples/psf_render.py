
from datetime import datetime

import numpy as np
import pandas as pd

import cv2

import matplotlib.pyplot as plt

from giant.point_spread_functions.gaussians import Gaussian

from giant.image import OpNavImage

from giant.catalogs.gaia import Gaia

from giant.utilities.spherical_coordinates import unit_to_radec, radec_to_unit

from giant.camera_models.brown_model import BrownModel
from giant.camera_models import CameraModel

from giant.rotations import Rotation

from giant.ray_tracer.scene import correct_stellar_aberration

def get_motion_blur_kernel(x, y, thickness=1, ksize=21):
    """ Obtains Motion Blur Kernel
        Inputs:
            x - horizontal direction of blur
            y - vertical direction of blur
            thickness - thickness of blur kernel line
            ksize - size of blur kernel
        Outputs:
            blur_kernel
        """
    blur_kernel = np.zeros((ksize, ksize))
    c = int(ksize/2)

    blur_kernel = np.zeros((ksize, ksize))
    blur_kernel = cv2.line(blur_kernel, (c+x//2,c+y//2), (c-x//2,c-x//2), (255,), thickness)
    return blur_kernel


def get_stars_directions_and_pixels(cat: Gaia, image: OpNavImage, model: CameraModel, max_mag: float, min_mag: float = -4) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    This function produces the visible stars in an image, including their records, their inertial unit vectors, and their pixel locations.
    
    The function queries the catalog using information contained in the OpNavImage input and the min/max mag inputs. 
    Important attributes of the OpNavImage are `observation_date`, `rotation_inertial_to_camera`, `temperature`, `position`, and `velocity`.
    
    The function corrects the catalog unit vectors for parallax and for stellar aberration.
    
    :param cat: The catalog instance to query
    :param image: The OpNavImage used to specify metadata about the camera
    :param model: The camera model used to project unit vectors onto the image
    :param max_mag: the maximum magnitude star to query from the catalog
    :param min_mag: the minimum magnitude star to query from the catalog
    :returns: A tuple containing the star records as a pandas DataFrame, the star inertial unit vectors (corrected for aberration and parallax), and the pixels the stars project to
    """
    
    
    # get the ra/dec of the z axis of the camera in the inertial frame
    ra, dec = unit_to_radec(image.rotation_inertial_to_camera.matrix[-1])

    # query the star catalog for stars in the field of view
    stars = cat.query_catalog(search_center=(float(np.rad2deg(ra)), float(np.rad2deg(dec))), 
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
    return stars, camera2stars_inertial, model.project_onto_image(catalog_unit_vectors_camera, temperature=image.temperature)
    

if __name__ == "__main__":
    
    # camera orientation
    inertial_to_camera = Rotation(np.random.randn(3)*np.deg2rad(30))

    image = OpNavImage(np.zeros((1944, 2592), dtype=np.float64), observation_date=datetime(2023, 9, 14), rotation_inertial_to_camera=inertial_to_camera)

    model = BrownModel(fx=13448.168, fy=13447.850, py=971.5, px=1295.5, k1=1.074e-2, k2=3.641e-1, k3=5.287e-2, p1=-4.64e-4, p2=-3.31e-4, n_cols=2592, n_rows=1944)

    cat = Gaia()
    
    # get the stars and pixels
    stars, _, star_pixels = get_stars_directions_and_pixels(cat, image, model, 7)
    
    half_dsize = 20
    delta_pix = np.arange(-half_dsize, half_dsize+1)
    
    motion_blur = get_motion_blur_kernel(int(np.floor(half_dsize/np.sqrt(2))), int(np.floor(half_dsize/np.sqrt(2))), ksize=2*half_dsize+1)

    for pix, mag in zip(star_pixels.T, stars.mag):

        # check to make sure we're in the FOV
        if (pix < 0).any() or (pix > [model.n_cols, model.n_rows]).any():
            continue

        amplitude = 10000*10**(mag/-2.5)

        psf = Gaussian(sigma_x=0.6, sigma_y=0.6, amplitude=amplitude, centroid_x=pix[0], centroid_y=pix[1])
        
        row_subs = delta_pix+pix[1]
        col_subs = delta_pix+pix[0]
        valid_rows = (row_subs > 0) & (row_subs < model.n_rows)
        valid_cols = (col_subs > 0) & (col_subs < model.n_cols)
        
        if not (valid_rows.any() and valid_cols.any()):
            continue
        
        sample_cols, sample_rows = np.meshgrid(col_subs[valid_rows], row_subs[valid_cols])
        

        render_cols = sample_cols.ravel()
        render_rows = sample_rows.ravel()

        intensity = cv2.filter2D(psf.evaluate(render_cols, render_rows).reshape(valid_rows.sum(), valid_cols.sum()), ddepth=-1, kernel=motion_blur).ravel()
        np.add.at(image, (render_rows.astype(int), render_cols.astype(int)), intensity)

    # use log scale just to make things easier to see.
    plt.imshow(np.log(image + 10), cmap='gray', interpolation='None')
    plt.show()