# a utility for retrieving a list of files using glob patterns
import glob

# the warning utility for filtering annoying warnings
import warnings

# a utility for generating plots
import matplotlib.pyplot as plt

# the Framing Camera object we defined before
from dawn_giant import DawnFCCamera, fc2_attitude, \
    vesta_attitude, vesta_position, sun_orientation, sun_position

# the function to load the camera model
from giant.camera_models import load

# The class we will use to perform the stellar opnav
from giant.stellar_opnav.stellar_class import StellarOpNav, StellarOpNavOptions

# tool for visualizing the results of our star identification
from giant.stellar_opnav.visualizers import show_id_results

# the star catalog we will use for our "truth" star locations
from giant.catalogs.gaia import Gaia, DEFAULT_CAT_FILE

# the class we will use to perform the relative navigation
from giant.relative_opnav.relnav_class import RelativeOpNav

# options for the relative navigation
from giant.relative_opnav import XCorrCenterFindingOptions

# the point spread function for the camera
from giant.point_spread_functions.gaussians import Gaussian

# the scene we will use to describe how things are related spatially
from giant.ray_tracer.scene import Scene, SceneObject

# The shape object we will use for the sun
from giant.ray_tracer.shapes import Point

# some utilities from giant for visualizing the relative opnav results
from giant.relative_opnav.visualizers import limb_summary_gif, template_summary_gif, show_center_finding_residuals

# A module to provide access to the NAIF Spice routines
import spiceypy as spice

# python standard library serialization tool
import pickle


if __name__ == "__main__":
    # furnish the meta kernel so we have all of the a priori state information
    spice.furnsh('./meta_kernel.tm')

    # choose the images we are going to process
    # use sorted to ensure they are in time sequential order
    images = sorted(glob.glob('../opnav/2011123_OPNAV_001/*.FIT') +
                    glob.glob('../opnav/2011165_OPNAV_007/*.FIT') +
                    glob.glob('../opnav/2011198_OPNAV_017/*.FIT'))[:-20]

    # load the camera model we are using
    camera_model = load('dawn_camera_models.xml', 'FC2')

    # create the camera instance and load the images
    camera = DawnFCCamera(images=images, model=camera_model, psf=Gaussian(sigma_x=0.75, sigma_y=0.75, size=5),
                          attitude_function=fc2_attitude)

    # now we need to build our scene for the relative navigation.
    # begin by loading the shape model
    with open('../shape_model/kdtree.pickle', 'rb') as tree_file:

        vesta_shape = pickle.load(tree_file)

    # we need to make this into an SceneObject, which essentially allows us to wrap the object with functions that
    # give the state of the object at any given time
    vesta_obj = SceneObject(vesta_shape, position_function=vesta_position,
                            orientation_function=vesta_attitude, name='Vesta')

    # now we need to form the SceneObject for our Sun Object
    sun_obj = SceneObject(Point([0, 0, 0]), position_function=sun_position, orientation_function=sun_orientation)

    # now we can form our scene
    opnav_scene = Scene(target_objs=[vesta_obj], light_obj=sun_obj)

    # do the stellar opnav to correct the attitude
    # build the stellar opnav object, which is very similar to the calibration object but without the ability to do
    # calibration.
    sopnav_options = StellarOpNavOptions()
    # sopnav_options.star_id_options.catalog = Gaia(catalog_file=DEFAULT_CAT_FILE)  # if you built the local catalog, then uncomment this line and comment the next one
    sopnav_options.star_id_options.catalog = Gaia()
    sopnav = StellarOpNav(camera, options=sopnav_options)

    # ensure only the long exposure images are on
    sopnav.camera.only_long_on()

    # set the parameters to get a successful star identification
    # we only need to estimate the attitude here so we can be fairly conservative
    sopnav.star_id.max_magnitude = 8.0
    sopnav.point_of_interest_finder.threshold = 20
    sopnav.star_id.tolerance = 40
    sopnav.star_id.ransac_tolerance = 1
    sopnav.star_id.max_combos = 1000

    # now id the stars and estimate the attitude
    sopnav.id_stars()
    show_id_results(sopnav)
    plt.show()
    sopnav.estimate_attitude()

    # ensure we got a good id
    # show_id_results(sopnav)
    sopnav.sid_summary()

    # now, we need to turn on the short exposure images, and use the updated attitude from the long exposure images to
    # update the attitude for the short exposure images
    sopnav.camera.only_short_on()
    sopnav.camera.update_short_attitude(method='propagate')

    # define the RelativeOpNav instance
    # define the settings for the portions of Relnav
    xcorr_options = XCorrCenterFindingOptions(grid_size=3, search_region=50)

    relnav = RelativeOpNav(camera, opnav_scene,
                           cross_correlation_options=xcorr_options,
                           save_templates=True)
    relnav.auto_estimate()
    
    # close the cartalog in case we opened it
    sopnav_options.star_id_options.catalog.close()

    # show the results
    limb_summary_gif(relnav)
    template_summary_gif(relnav)
    show_center_finding_residuals(relnav)
    plt.show()
