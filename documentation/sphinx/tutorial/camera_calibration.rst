Performing Camera Calibration
=============================
One of the first things that we will typically use GIANT for on any mission is to generate the geometric model of the
camera.  This model is important both to GIANT, and to others, because it allows us to map points in the camera frame
onto points in the image itself.

In GIANT, camera modelling is done by processing images of star fields.  The observed stars in these images are matched
with a catalogue of stars that provide their location in the inertial frame.  Through these matches, we can develop a
camera model that minimizes the residuals between the projection of the catalogue defined star locations onto the image
with the extracted star locations in the image (from image processing).

Lets set up a script to do this for the Dawn Framing Camera 2 (FC2).  In the ``dawn_giant/scripts`` directory, create
a script entitled ``fc2_calibration.py`` and open it with your favorite text editor.

Initial Imports
---------------
As with the ``dawn_giant`` module, we want to begin our calibration script with the imports we will need.

.. code::

    # a utility for retrieving a list of files using glob patterns
    import glob

    # the warnings utility for filtering annoying warnings that don't mean things to us right now
    import warnings

    # A numerical library for performing various math and linear algebra implementations in an efficient manner
    import numpy as np

    # the camera model we will use to map points from the camera frame to the image and a function to save our final model
    from giant.camera_models import BrownModel, save

    # The class we will use to perform the calibration
    from giant.calibration.calibration_class import Calibration

    # tools for visualizing the results of our calibration
    from giant.calibration.visualizer import plot_distortion_map
    from giant.stellar_opnav.visualizer import show_id_results, residual_histograms, plot_residuals_vs_magnitude

    # the star catalogue we will use for our "truth" star locations
    from giant.catalogues.giant_catalogue import GIANTCatalogue

    # the Framing Camera object we defined before
    from dawn_giant import DawnFCCamera, fc2_attitude

    # the tool for reading the PDS label files
    import pvl

    # A module to provide access to the NAIF Spice routines
    import spiceypy as spice

Initializing Our Camera Model
-----------------------------
The way that camera calibration works, we need to have an initial guess of the camera model available from the start.
Typically, this initial guess is based off of the prescribed geometry of the camera and assumes a pinhole camera model
with no distortion.  For Dawn, we can get information about the prescribed geometry from the framing camera SIS
available `here <https://sbnarchive.psi.edu/pds3/dawn/fc/DWNXFC2_1A/DOCUMENT/SIS/DAWN_FC_SIS_20160815.PDF>`_.

For this tutorial, we will use the Brown Camera Model (:class:`.BrownModel`) which is a simple yet powerful model that
can accurately model most modern cameras.  For a list of potential other camera models and more information about how
camera models work see the :mod:`.camera_models` documentation.

We can define our initial camera model for FC2 using the following code.

.. code::

    if __name__ == "__main__":
        # filter some annoying warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=DeprecationWarning)

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

All we are doing above is translating the prescribed camera information into the format that the Brown camera model
expects.  We are also setting the camera model parameters that we want to estimate in our calibration.  Choosing these
parameters takes a little bit of skill and is outside the scope of this tutorial, so trust that we have chosen the
correct parameters for now.

Defining a Meta Kernel
----------------------
Before we proceed further, we need to define a meta kernel to load all of the spice files so we have access to the state
information we will need to do the calibration (and the OpNav in a later script).  This tutorial is not a tutorial on
Spice so simply, open a new file called ``meta_kernel.tm`` in the ``dawn_giant/scripts`` directory and paste the
following

.. code::

    KPL/MK
     \begindata

     PATH_VALUES = ( '../kernels' )
     PATH_SYMBOLS = ( 'ROOT' )

     KERNELS_TO_LOAD = ( '$ROOT/lsk/naif0012.tls',
                         '$ROOT/pck/pck00008.tpc',
                         '$ROOT/spk/de432.bsp',
                         '$ROOT/spk/sb_vesta_ssd_120716.bsp',
                         '$ROOT/pck/dawn_vesta_v02.tpc',
                         '$ROOT/fk/dawn_v14.tf',
                         '$ROOT/fk/dawn_vesta_v00.tf',
                         '$ROOT/sclk/DAWN_203_SCLKSCET.00090.tsc',
                         '$ROOT/spk/dawn_rec_070927-070930_081218_v1.bsp',
                         '$ROOT/spk/dawn_rec_070930-071201_081218_v1.bsp',
                         '$ROOT/spk/dawn_rec_071201-080205_081218_v1.bsp',
                         '$ROOT/spk/dawn_rec_100208-100316_100323_v1.bsp',
                         '$ROOT/spk/dawn_rec_100316-100413_100422_v1.bsp',
                         '$ROOT/spk/dawn_rec_100413-100622_100830_v1.bsp',
                         '$ROOT/spk/dawn_rec_100622-100824_100830_v1.bsp',
                         '$ROOT/spk/dawn_rec_100824-101130_101202_v1.bsp',
                         '$ROOT/spk/dawn_rec_101130-110201_110201_v1.bsp',
                         '$ROOT/spk/dawn_rec_101130-110419_pred_110419-110502_110420_v1.bsp',
                         '$ROOT/spk/dawn_rec_101130-110606_pred_110606-110628_110609_v1.bsp',
                         '$ROOT/spk/dawn_rec_110201-110328_110328_v1.bsp',
                         '$ROOT/spk/dawn_rec_110328-110419_110419_v1.bsp',
                         '$ROOT/spk/dawn_rec_110328-110419_110420_v1.bsp',
                         '$ROOT/spk/dawn_rec_110416-110802_110913_v1.bsp',
                         '$ROOT/spk/dawn_rec_110802-110831_110922_v1.bsp',
                         '$ROOT/spk/dawn_rec_110831-110928_111221_v1.bsp',
                         '$ROOT/spk/dawn_rec_110928-111102_111221_v1.bsp',
                         '$ROOT/spk/dawn_rec_110928-111102_120615_v1.bsp',
                         '$ROOT/spk/dawn_rec_111102-111210_120618_v1.bsp',
                         '$ROOT/spk/dawn_rec_111211-120501_120620_v1.bsp',
                         '$ROOT/ck/dawn_fc_v3.bc',
                         '$ROOT/ck/dawn_sc_071203_071209.bc',
                         '$ROOT/ck/dawn_sc_071210_071216.bc',
                         '$ROOT/ck/dawn_sc_071217_071223.bc',
                         '$ROOT/ck/dawn_sc_071224_071230.bc',
                         '$ROOT/ck/dawn_sc_071231_080106.bc',
                         '$ROOT/ck/dawn_sc_100705_100711.bc',
                         '$ROOT/ck/dawn_sc_100712_100718.bc',
                         '$ROOT/ck/dawn_sc_100719_100725.bc',
                         '$ROOT/ck/dawn_sc_100726_100801.bc',
                         '$ROOT/ck/dawn_sc_110502_110508.bc',
                         '$ROOT/ck/dawn_sc_110509_110515.bc',
                         '$ROOT/ck/dawn_sc_110516_110522.bc',
                         '$ROOT/ck/dawn_sc_110523_110529.bc',
                         '$ROOT/ck/dawn_sc_110530_110605.bc',
                         '$ROOT/ck/dawn_sc_110606_110612.bc',
                         '$ROOT/ck/dawn_sc_110613_110619.bc',
                         '$ROOT/ck/dawn_sc_110620_110626.bc',
                         '$ROOT/ck/dawn_sc_110627_110703.bc',
                         '$ROOT/ck/dawn_sc_110704_110710.bc',
                         '$ROOT/ck/dawn_sc_110711_110717.bc',
                         '$ROOT/ck/dawn_sc_110718_110724.bc',
                         '$ROOT/ck/dawn_sc_110725_110731.bc',
                         '$ROOT/ck/dawn_sc_110801_110807.bc',
                         '$ROOT/ck/dawn_sc_110808_110814.bc',
                         '$ROOT/ck/dawn_sc_110815_110821.bc',
                         '$ROOT/ck/dawn_sc_110822_110828.bc',
                         '$ROOT/ck/dawn_sc_110829_110904.bc'
      )

     \begintext

Now, return to ``fc2_calibration.py`` and add the following lines to load this file.

.. code::

    # load the meta kernel so we have access to the state information
    spice.furnsh('./meta_kernel.tm')

Loading the Images and Creating the Camera Instance
---------------------------------------------------
With the camera model defined, we can now load the images we are going to use to do our calibration.  For Dawn,
they took special images of only star fields so they could do a calibration like this.  Therefore we will use those
images.

The way that GIANT processes star images, it is usually best to try and process all images with similar exposure lengths
at the same time.  Doing this allows you to use one set of parameters for processing these images, and typically images
with similar exposure lengths can be processed with the same set of parameters.  Therefore, the first thing we should do
is bin our images into groups that have similar exposure lengths.  We can do this using the following function, which
you should place after the imports but before the ``if __name__ == "__main__"`` section of the code

.. code::

    def bin_images(image_files):
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

            # check if this is a normal image
            if data["DAWN:IMAGE_ACQUIRE_MODE"] == "NORMAL":
                # get the exposure time in seconds
                exposure = data["EXPOSURE_DURATION"].value/1000

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

Now, we can use this function to bin the image and load just the first batch into the camera.  Enter the following code
at the bottom of your file.

.. code::

    # now we need to load our images and create our camera instance.
    # first, lets bin our images into exposure groups
    exp, binned_images = bin_images(glob.glob('../cal*/*.FIT'))

    # now, form our camera object with the images from the first exposure group
    camera = DawnFCCamera(images=binned_images[0], model=camera_model, attitude_function=fc2_attitude)


Note that we are using the ``DawnFCCamera`` class that we defined before, and giving it the list of the paths to the
first group of images we want to process, the initial camera model we defined earlier, and the attitude function for the
framing camera that we defined in our ``dawn_giant`` module.  The initialized camera object will contain all of this
information and have loaded the images that we requested so that we are ready to go.

Creating the Calibration Object
-------------------------------
With our camera defined and images loaded we can now form our :class:`.Calibration` object.  The calibration object is
what we will interact with to perform the calibration.  To initialize the object enter the following

.. code::

    # we can build our calibration object now, which we'll use to identify the stars and then estimate an update to the
    # camera model
    # for the star id key word arguments, set the catalogue to be the GIANT catalogue
    calib = Calibration(camera, star_id_kwargs={'catalogue': GIANTCatalogue()})

We give the calibration object the camera object that we just initialized, as well as a key word argument called
``star_identification_kwargs`` which specifies that we want to use the default GIANT catalogue to get our star
locations.  There are other things you can specify for the :class:`.Calibration` class constructor but they are outside
of the scope for this tutorial and you will need to consult the :class:`.Calibration` documentation for more information
on them.

Identifying Stars in an Image
-----------------------------
It's been a long journey, but we're finally ready to start processing our images.  This is done by specifying some
settings on the calibration class, and then calling the method :meth:`~.Calibration.id_stars` to identify the stars
in the image. Setting parameters that lead to successful star identification is something that takes some practice
and is outside the scope of this tutorial, however, the :mod:`.stellar_opnav` documentation provides a good walk
through of how to understand these parameters and successfully identify stars.

When we're doing camera calibration, we typically identify stars in an image twice.  The first time we are fairly
conservative in trying to only match stars that we are certain are correctly identified.  This is because we need to
correct our initial attitude for the image before we can really try to identify a lot of stars.  Once we have that
initial pairing, and have corrected our attitude for the images, then we can try to match more stars.

For the first set of images, the following parameters work well for both the initial and full identifications

.. code::

    # set the initial parameters for our first star identification
    # typically, we are conservative with the first star identification because we only need about 5 correctly
    # identified stars in each image in order to correct our attitude
    calib.star_id.max_magnitude = 7.0
    calib.image_processing.poi_threshold = 20
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
    calib.image_processing.poi_threshold = 15
    calib.star_id.tolerance = 2
    calib.star_id.max_combos = 0

    # now we just want to identify stars, since we've already adjusted our attitude
    calib.id_stars()

In the above code, we set our initial star identification parameters, then we call the method
:meth:`~.Calibration.id_stars`, then the method :meth:`~.Calibration.estimate_attitude`.  This corrects the attitude
for the images.  Once that is done, we set new parameters and then call :meth:`~.Calibration.id_stars` again (without
calling :meth:`~.Calibration.estimate_attitude` this time.  This leads to GIANT having knowledge about all of the
matched stars for the images we have loaded.  To see the results of our star identification you can add the line

.. code::

    show_id_results(calib)

to your script which will show you the identified stars in each image.

.. note::

    At this point, we encourage you to save the script, run it (``python fc2_calibration.py``) and mess around with the
    parameters to see how they affect the star identification results.  You may even find a better set than what we've
    presented here.  Once you're finished, you should remove the ``show_id_results(calib)`` line so that you don't have
    to close all of the figure windows each time you run the script

With the first group of images out of the way we can move onto our next group of images.  We do this by first turning
off the images we've already processed (so that GIANT doesn't reprocess them but still remembers the stuff we've
already done) and then loading the next image group.

.. code::

    # now that we have id'd stars in these images, turn them off so they're no longer processed
    calib.camera.all_off()

    # add the images from our second exposure group
    calib.add_images(binned_images[1])

With the second exposure group loaded, we are going to repeat the same steps as for the first, but with different
parameter settings.

.. code::

    # repeat the steps above for the second exposure group.
    calib.star_id.max_magnitude = 8.5
    calib.image_processing.poi_threshold = 80
    calib.star_id.tolerance = 40
    calib.star_id.ransac_tolerance = 2
    calib.star_id.max_combos = 1000

    calib.id_stars()

    calib.estimate_attitude()

    calib.star_id.max_magnitude = 9.5
    calib.image_processing.poi_threshold = 40
    calib.star_id.tolerance = 2
    calib.star_id.max_combos = 0

    calib.id_stars()

    calib.camera.all_off()

Now we can load and identify stars in the final exposure group (two of the exposure groups have windowed images which
are not very useful for calibration so we skip over them).

.. code::

    # and for the final exposure group (we need to throw out the third and fourth groups because the images are
    # windowed and thus not very useful for calibration)
    calib.add_images(binned_images[4])

    calib.star_id.max_magnitude = 9.5
    calib.image_processing.poi_threshold = 100
    calib.star_id.tolerance = 20
    calib.star_id.ransac_tolerance = 2
    calib.star_id.max_combos = 1000

    calib.id_stars()
    calib.estimate_attitude()

    calib.star_id.max_magnitude = 11.0
    calib.image_processing.poi_threshold = 40
    calib.star_id.tolerance = 2
    calib.star_id.max_combos = 0

    calib.id_stars()

Performing Calibration
----------------------
With stars identified in all of our images we can now do our estimation.  First, we need to turn all of the images back
on so that GIANT knows to use stars from all images in the calibration

.. code::

    # now, turn on all of the images so they all get included in our calibration
    calib.camera.all_on()

Now, we simply call the method :meth:`~.Calibration.estimate_calibration` and GIANT will solve for the updated camera
model. Typically we do this twice, removing outliers after the first time, to ensure we get a really good fit. The
:meth:`~.Calibration.remove_outliers` function allows us to automatically reject outliers whose post-fit residuals are
greater than some specified tolerance.  We could also manually inspect outliers using the
:meth:`~.Calibration.review_outliers`.

.. code::

    # do the calibration once, manually check the outliers, and then do the final calibration
    calib.estimate_calibration()
    calib.remove_outliers(hard_threshold=0.5)
    calib.estimate_calibration()

And that is it, the calibration is done and we have solved for an update to our camera model

Viewing the Calibration Results
-------------------------------
GIANT provides a number of useful functions and methods to visualize the calibration results.  A few
examples are provided below with comments explaining what each one does.  Be sure to close out of all open
figures to proceed to the next visualization

.. code::

    # show all of our results
    # a table summary of our star identification results for each image printed to stdout
    calib.sid_summary()
    # A summary of the formal uncertainties and correlation coefficients in the post fit camera model printed to stdout
    calib.calib_summary(measurement_covariance=0.15)
    # the final camera model printed to stdout
    print(repr(calib.model))
    # plots showing the star indentification results for each image and overall
    show_id_results(calib)
    # the overall post-fit residuals
    residual_histograms(calib)
    # the overall post-fit residuals vs magnitude
    plot_residuals_vs_magnitude(calib)
    # the distortion map for the solved for camera model
    plot_distortion_map(calib.model)

Saving the Solved for Camera Model
----------------------------------
The final step in our calibration is to save off the results so we can use it again later without having to go through
all of these steps again.  This is done using the :func:`.save` function from the :mod:`.camera_models` module.
Simply specify the file that you want to save the results to, specify the name of the camera the model is for, and
provide the model to be saved.  In the future, you can load the module using the :func:`.load` function.

.. code::

    # now save off our solved for camera model to a file
    save('dawn_camera_models.xml', 'FC2', calib.model)

    # clear all of the spice kernels we loaded
    spice.kclear()

The Full FC2 Calibration Script
-------------------------------
For convenience, the full FC2 calibration script is presented here

.. code::

    # a utility for retrieving a list of files using glob patterns
    import glob

    # the warnings utility for filtering annoying warnings that don't mean things to us right now
    import warnings

    # A numerical library for performing various math and linear algebra implementations in an efficient manner
    import numpy as np

    # the camera model we will use to map points from the camera frame to the image and a function to save our final model
    from giant.camera_models import BrownModel, save

    # The class we will use to perform the calibration
    from giant.calibration.calibration_class import Calibration

    # tools for visualizing the results of our calibration
    from giant.calibration.visualizer import plot_distortion_map
    from giant.stellar_opnav.visualizer import show_id_results, residual_histograms, plot_residuals_vs_magnitude

    # the star catalogue we will use for our "truth" star locations
    from giant.catalogues.giant_catalogue import GIANTCatalogue

    # the Framing Camera object we defined before
    from dawn_giant import DawnFCCamera, fc2_attitude

    # the tool for reading the PDS label files
    import pvl

    # A module to provide access to the NAIF Spice routines
    import spiceypy as spice


    def bin_images(image_files):
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

            # check if this is a normal image
            if data["DAWN:IMAGE_ACQUIRE_MODE"] == "NORMAL":
                # get the exposure time in seconds
                exposure = data["EXPOSURE_DURATION"].value/1000

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


    if __name__ == "__main__":
        # filter some annoying warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=DeprecationWarning)

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
        # for the star id key word arguments, set the catalogue to be the GIANT catalogue
        calib = Calibration(camera, star_id_kwargs={'catalogue': GIANTCatalogue()})

        # set the initial parameters for our first star identification
        # typically, we are conservative with the first star identification because we only need about 5 correctly
        # identified stars in each image in order to correct our attitude
        calib.star_id.max_magnitude = 7.0
        calib.image_processing.poi_threshold = 20
        calib.star_id.tolerance = 40
        calib.star_id.ransac_tolerance = 2
        calib.star_id.max_combos = 5000
        calib.star_id.second_closest_check = False
        calib.star_id.unique_check = False

        # now we can identify our stars and estimate an update to our attitude based on these matched star pairs
        calib.id_stars()
        # show_id_results(calib)

        calib.estimate_attitude()

        # now, since we're doing a calibration, we want to extract a lot of stars so we can fully observe the whole
        # field of view.  Therefore, set some less conservative parameters
        calib.star_id.max_magnitude = 9.0
        calib.image_processing.poi_threshold = 15
        calib.star_id.tolerance = 2
        calib.star_id.max_combos = 0

        # now we just want to identify stars, since we've already adjusted our attitude
        calib.id_stars()

        # now that we have id'd stars in these images, turn them off so they're no longer processed
        calib.camera.all_off()

        # add the images from our second exposure group
        calib.add_images(binned_images[1])

        # repeat the steps above for the second exposure group.
        calib.star_id.max_magnitude = 8.5
        calib.image_processing.poi_threshold = 80
        calib.star_id.tolerance = 40
        calib.star_id.ransac_tolerance = 2
        calib.star_id.max_combos = 1000

        calib.id_stars()

        calib.estimate_attitude()

        calib.star_id.max_magnitude = 9.5
        calib.image_processing.poi_threshold = 40
        calib.star_id.tolerance = 2
        calib.star_id.max_combos = 0

        calib.id_stars()

        calib.camera.all_off()

        # and for the final exposure group (we need to throw out the third and fourth groups because the images are
        # windowed and thus not very useful for calibration)
        calib.add_images(binned_images[4])

        calib.star_id.max_magnitude = 9.5
        calib.image_processing.poi_threshold = 100
        calib.star_id.tolerance = 20
        calib.star_id.ransac_tolerance = 2
        calib.star_id.max_combos = 1000

        calib.id_stars()
        calib.estimate_attitude()

        calib.star_id.max_magnitude = 11.0
        calib.image_processing.poi_threshold = 40
        calib.star_id.tolerance = 2
        calib.star_id.max_combos = 0

        calib.id_stars()

        # now, turn on all of the images so they all get included in our calibration
        calib.camera.all_on()

        # do the calibration once, manually check the outliers, and then do the final calibration
        calib.estimate_calibration()
        calib.remove_outliers(hard_threshold=0.5)
        calib.estimate_calibration()

        # show all of our results
        # a table summary of our star identification results for each image printed to stdout
        calib.sid_summary()
        # A summary of the formal uncertainties and correlation coefficients in the post fit camera model printed to stdout
        calib.calib_summary(measurement_covariance=0.15)
        # the final camera model printed to stdout
        print(repr(calib.model))
        # plots showing the star indentification results for each image and overall
        # show_id_results(calib)
        # the overall post-fit residuals
        residual_histograms(calib)
        # the overall post-fit residuals vs magnitude
        plot_residuals_vs_magnitude(calib)
        # the distortion map for the solved for camera model
        plot_distortion_map(calib.model)

        # now save off our solved for camera model to a file
        save('dawn_camera_models.xml', 'FC2', calib.model)

        # clear all of the spice kernels we loaded
        spice.kclear()

