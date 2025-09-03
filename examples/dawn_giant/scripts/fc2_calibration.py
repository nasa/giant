# a utility for retrieving a list of files using glob patterns
import glob

# A numerical library for performing various math and linear algebra implementations in an efficient manner
import numpy as np

# the camera model we will use to map points from the camera frame to the image and a function to save our final model
from giant.camera_models import BrownModel, save

# The class we will use to perform the calibration
from giant.calibration.calibration_class import Calibration, CalibrationOptions

# tools for visualizing the results of our calibration
from giant.calibration.visualizer import plot_distortion_map
from giant.stellar_opnav.visualizers import show_id_results, residual_histograms, residuals_vs_magnitude

# the star catalog we will use for our "truth" star locations
from giant.catalogs.gaia import Gaia, DEFAULT_CAT_FILE

# the Framing Camera object we defined before
from dawn_giant import DawnFCCamera, fc2_attitude

# the tool for reading the PDS label files
import pvl

# A module to provide access to the NAIF Spice routines
import spiceypy as spice


def bin_images(image_files: list[str]):
    """
    A simple utility to bin the images into exposure groupd for DAWN
    :param image_files: The filepaths to be binned
    :return: a list of exposures and a corresponding list of lists where each list corresponds to an exposure group
    """

    # initialize lists to store the exposure for each image
    images, exposures = [], []
    for image in image_files:

        # read the label file for the image
        with open(image.replace('.FIT', '.LBL'), 'r') as label:
            data = pvl.loads(label.read().replace('\\', '/'))
            
            assert data is not None, "we were unable to parse the label file {}".format(image.replace('.FIT', '.LBL'))

        # check if this is a normal image
        if data["DAWN:IMAGE_ACQUIRE_MODE"] == "NORMAL":
            # get the exposure time in seconds
            exposure = data["EXPOSURE_DURATION"].value/1000 # pyright: ignore[reportAttributeAccessIssue]

            # only keep exposure values longer than 1 second
            if exposure > 1:

                # store the image and its exposure length
                images.append(image)
                exposures.append(exposure)

    # if things met our requirements
    if images:

        # choose the unique exposures and sort them
        uniq_exposures = np.sort(np.unique(exposures))

        binned = []

        # loop through each unique exposure and see which images have that exposure length
        for expo in uniq_exposures:

            # store a list of images that have this exposure length
            binned.append(sorted([im for ind, im in enumerate(images) if exposures[ind] == expo]))

        # return the results
        return uniq_exposures, binned
    else:
        return [], []


if __name__ == "__main__":
    # first, we need to define our initial camera model.
    # typically, we do this based off of the prescribed camera specifications.
    # for the dawn framing cameras, this information is available in the DAWN_FC_SIS_20160815.PDF which is available at
    # https://sbnarchive.psi.edu/pds3/dawn/fc/DWNXFC2_1A/DOCUMENT/SIS/DAWN_FC_SIS_20160815.PDF
    # For this analysis we'll use a Brown camera model

    focal_length = 150  # mm
    pitch = 14e-3  # mm/pixel
    focal_length_pixels = focal_length/pitch
    fov = 5  # degrees (corresponds to half of diagonal FOV)
    nrows = 1024  # pixels
    ncols = 1024  # pixels

    px = (ncols - 1)/2  # the center of the detector in pixels
    py = (nrows - 1)/2  # the center of the detector in pixels

    # set the parameters that we want to estimate in the calibration.
    # this tells giant to estimate the x and y focal lengths (fx, fy), the second, fourth, and sixth order
    # radial distortion coefficients (k1-3), the tangential distortion (p1, p2), and a single attitude misalignment for
    # each image.
    estimation_parameters = ['fx', 'fy', 'k1', 'p1', 'p2', 'multiple misalignments']

    camera_model = BrownModel(fx=focal_length_pixels, fy=-focal_length_pixels, px=px, py=py, n_rows=nrows, n_cols=ncols,
                              field_of_view=fov, estimation_parameters=estimation_parameters)

    # load the meta kernel so we have access to the state information
    spice.furnsh('./meta_kernel.tm')

    # now we need to load our images and create our camera instance.
    # first, lets bin our images into exposure groups
    exp, binned_images = bin_images(glob.glob('../cal*/*.FIT'))

    # now, form our camera object with the images from the first exposure group
    camera = DawnFCCamera(images=binned_images[0], model=camera_model, attitude_function=fc2_attitude)

    # we can build our calibration object now, which we'll use to identify the stars and then estimate an update to the
    # camera model
    # for the star id, set the catalog to be the Gaia catalog (which is already the default)
    calib_options = CalibrationOptions()
    # calib_options.star_id_options.catalog = Gaia(catalog_file=DEFAULT_CAT_FILE) # if you built the local catalog, then uncomment this line
    calib_options.star_id_options.catalog = Gaia()
    calib = Calibration(camera, options=calib_options)

    # set the initial parameters for our first star identification
    # typically, we are conservative with the first star identification because we only need about 5 correctly
    # identified stars in each image in order to correct our attitude
    calib.star_id.max_magnitude = 7.0
    calib.point_of_interest_finder.threshold = 20
    calib.point_of_interest_finder.reject_saturation = False
    calib.star_id.tolerance = 40
    calib.star_id.ransac_tolerance = 2
    calib.star_id.max_combos = 5000
    calib.star_id.second_closest_check = False
    calib.star_id.unique_check = False

    # now we can identify our stars and estimate an update to our attitude based on these matched star pairs
    calib.id_stars()
    
    calib.estimate_attitude()

    # now, since we're doing a calibration, we want to extract a lot of stars so we can fully observe the whole
    # field of view.  Therefore, set some less conservative parameters
    calib.star_id.max_magnitude = 9.0
    calib.point_of_interest_finder.threshold = 15
    calib.star_id.tolerance = 2
    calib.star_id.max_combos = 0

    # now we just want to identify stars, since we've already adjusted our attitude
    calib.id_stars()
    
    # show the results if we want to verify them
    # show_id_results(calib)

    # now that we have id'd stars in these images, turn them off so they're no longer processed
    calib.camera.all_off()

    # add the images from our second exposure group
    calib.add_images(binned_images[1])

    # repeat the steps above for the second exposure group.
    calib.star_id.max_magnitude = 8.5
    calib.point_of_interest_finder.threshold = 80
    calib.star_id.tolerance = 40
    calib.star_id.ransac_tolerance = 2
    calib.star_id.max_combos = 1000

    calib.id_stars()

    calib.estimate_attitude()

    calib.star_id.max_magnitude = 9.5
    calib.point_of_interest_finder.threshold = 40
    calib.star_id.tolerance = 2
    calib.star_id.max_combos = 0

    calib.id_stars()

    calib.camera.all_off()

    # and for the final exposure group (we need to throw out the third and fourth groups because the images are
    # windowed and thus not very useful for calibration)
    calib.add_images(binned_images[4])

    calib.star_id.max_magnitude = 9.5
    calib.point_of_interest_finder.threshold = 100
    calib.star_id.tolerance = 20
    calib.star_id.ransac_tolerance = 2
    calib.star_id.max_combos = 1000

    calib.id_stars()
    calib.estimate_attitude()

    calib.star_id.max_magnitude = 11.0
    calib.point_of_interest_finder.threshold = 40
    calib.star_id.tolerance = 2
    calib.star_id.max_combos = 0

    calib.id_stars()

    # now, turn on all of the images so they all get included in our calibration
    calib.camera.all_on()

    # do the calibration once, manually check the outliers, and then do the final calibration
    calib.estimate_geometric_calibration()
    calib.remove_outliers(hard_threshold=0.5)
    calib.estimate_geometric_calibration()

    # show all of our results
    # a table summary of our star identification results for each image printed to stdout
    calib.sid_summary()
    # A summary of the formal uncertainties and correlation coefficients in the post fit camera model printed to stdout
    calib.geometric_calibration_summary(measurement_covariance=0.15)
    # the final camera model printed to stdout
    print(repr(calib.model))
    # plots showing the star indentification results for each image and overall
    show_id_results(calib)
    # the overall post-fit residuals
    residual_histograms(calib)
    # the overall post-fit residuals vs magnitude
    residuals_vs_magnitude(calib)
    # the distortion map for the solved for camera model
    plot_distortion_map(calib.model)

    # now save off our solved for camera model to a file
    save('dawn_camera_models.xml', 'FC2', calib.model)

    # clear all of the spice kernels we loaded
    spice.kclear()
    
    # close the gaia catalog in case we opened it
    calib_options.star_id_options.catalog.close()