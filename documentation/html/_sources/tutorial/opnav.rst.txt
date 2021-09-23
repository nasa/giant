Performing Optical Navigation
=============================
With our camera calibration complete, we can now use GIANT to do some relative navigation!  For this, we will process
OpNav images taken during Dawn's approach to the asteroid Vesta.  These images were taken in sets of 40 for each OpNav
window, alternating between long and short exposure lengths.  The long exposure images are used to refine the attitude
knowledge of the camera by observing stars in the image, while the short exposure images are used to perform the
relative navigation with respect to Vesta.

To begin doing OpNav, create a script called ``opnav.py`` in the ``scripts`` directory and open it with your favorite
text editor.

Initial Imports
---------------
As with the previous two scripts, we need to start off with importing the modules and packages we will need.

.. code::

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
    from giant.stellar_opnav.stellar_class import StellarOpNav

    # tool for visualizing the results of our star identification
    from giant.stellar_opnav.visualizer import show_id_results

    # the star catalogue we will use for our "truth" star locations
    from giant.catalogues.giant_catalogue import GIANTCatalogue

    # the class we will use to perform the relative navigation
    from giant.relative_opnav.relnav_class import RelativeOpNav

    # the point spread function for the camera
    from giant.point_spread_functions.gaussians import Gaussian

    # the scene we will use to describe how things are related spatially
    from giant.ray_tracer.scene import Scene, SceneObject

    # The shape object we will use for the sun
    from giant.ray_tracer.shapes import Point

    # the illumination function we will use to predict the image of our model
    from giant.ray_tracer.illumination import McEwenIllumination

    # some utilities from giant for visualizing the relative opnav results
    from giant.relative_opnav.visualizer import limb_summary_gif, template_summary_gif, show_center_finding_residuals

    # A module to provide access to the NAIF Spice routines
    import spiceypy as spice

    # python standard library serialization tool
    import pickle

Loading the Data
----------------
With the imports out of the way, we now need to load our meta kernel, our images, and our camera model that we solved
for in the calibration script and define the function that represents the point spread function for our camera.
The loading is  done using basic utilities from external libraries plus the :func:`.load`
function from the :mod:`.camera_models` module.  The PSF is defined using the OpenCV ``GaussianBlur`` function
with some specified settings.  These settings are camera dependent and are usually determined during the calibration
campaign.  For now, we'll just use a rough guess for the PSF.

.. code::

    if __name__ == "__main__":
        # filter some annoying warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=DeprecationWarning)

        # furnish the meta kernel so we have all of the a priori state information
        spice.furnsh('./meta_kernel.tm')

        # choose the images we are going to process
        # use sorted to ensure they are in time sequential order
        images = sorted(glob.glob('../opnav/2011123_OPNAV_001/*.FIT') +
                        glob.glob('../opnav/2011165_OPNAV_007/*.FIT') +
                        glob.glob('../opnav/2011198_OPNAV_017/*.FIT'))

        # load the camera model we are using
        camera_model = load('dawn_camera_models.xml', 'FC2')

Loading the Images and Creating the Camera Instance
---------------------------------------------------
Now we can ask GIANT to load all of our images and create our camera instance.  This is simply done by initializing our
``DawnFCCamera`` class as we did in the calibration script.  The only difference now is that we also provide our PSF to
the camera initializer so that GIANT knows about it.

.. code::

        # create the camera instance and load the images
        camera = DawnFCCamera(images=images, model=camera_model, psf=Gaussian(sigma_x=0.75, sigma_y=0.75, size=5),
                              attitude_function=fc2_attitude)

Estimating the Rotation Using Star Images
-----------------------------------------
With our camera object created, we can now start estimating the attitude in the long-exposure images using star
observations.  This is extremely similar to how we perform camera calibration, but we use the :class:`.StellarOpNav`
class instead and we only estimate the attitude, not the calibration.  Plus, we only want to look for stars in long
exposure images so we tell GIANT to only use the long exposure images using the :meth:`~.Camera.only_long_on` method
of the :attr:`.StellarOpNav.camera` attribute.

.. code::

    # do the stellar opnav to correct the attitude
    # build the stellar opnav object, which is very similar to the calibration object but without the ability to do
    # calibration.
    sopnav = StellarOpNav(camera, star_id_kwargs={'catalogue': GIANTCatalogue()})

    # ensure only the long exposure images are on
    sopnav.camera.only_long_on()

    # set the parameters to get a successful star identification
    # we only need to estimate the attitude here so we can be fairly conservative
    sopnav.star_id.max_magnitude = 8.0
    sopnav.image_processing.poi_threshold = 20
    sopnav.star_id.tolerance = 40
    sopnav.star_id.ransac_tolerance = 1
    sopnav.star_id.max_combos = 1000

    # now id the stars and estimate the attitude
    sopnav.id_stars()
    sopnav.estimate_attitude()

    # ensure we got a good id
    show_id_results(sopnav)
    sopnav.sid_summary()


If you run the script and save it you should see the id result plots appear (there will be a lot of them) and should see
good results and post-fit residuals around 0.1 pixels in standard deviation.  You can mess around with the various star
identification and image processing parameters if you want or you can just leave them and move on.  When you're ready to
move on then comment out the line with the :func:`.show_id_results` function so that it doesn't pop up every time we run
the script.

Updating the Short Exposure Image Rotation
------------------------------------------
With the long exposure image attitudes corrected, we now want to use this information to update our short-exposure image
attitudes.  This is done in 2 steps.  First, we turn on the only the short exposure images using the
:meth:`~.Camera.only_short_on` method.  Then, we call the :meth:`~.Camera.update_short_attitude` method which
propagates the solved for attitudes in the long-exposure images to the following short-exposure image times using the
:attr:`~.Camera.attitude_function` of the camera instance.

.. code::

    # now, we need to turn on the short exposure images, and use the updated attitude from the long exposure images to
    # update the attitude for the short exposure images
    sopnav.camera.only_short_on()
    sopnav.camera.update_short_attitude()

Defining the OpNav Scene
------------------------
Now that we have updated the attitude for the short-exposure images we need to define the OpNav scene.  The OpNav scene
tells GIANT what objects to expect in the images, as well as their relative position and orientation with respect to
each other.  For the DAWN approach to Vesta, we only have 3 objects we need to worry about in our scene: (1) the camera,
(2) the sun, and (3) Vesta.

Lets begin by considering Vesta. For Vesta, we need a shape model which defines the terrain and shape of the body.
GIANT uses the shape model when predicting what Vesta should look like in the field of view.  To load the shape model,
we use the :mod:`pickle` module from the python standard library to load the data from the ``kdtree.pickle`` file that
we created when downloading our data.  The ``kdtree.pickle`` contains a KDTree representation of the shape model that
GIANT can understand and was created using the ``ingest_shape`` script that is packaged with GIANT.

.. code::

    # now we need to build our scene for the relative navigation.
    # begin by loading the shape model
    with open('../shape_model/kdtree.pickle', 'rb') as tree_file:

        vesta_shape = pickle.load(tree_file)

With the shape model loaded, we need to create an :class:`.SceneObject` instance for Vesta.  The :class:`.SceneObject`
class essentially wraps the shape model with functions that define its position and orientation in a scene at a given
time, along with a name that GIANT can use to distinguish the object.  In this case, the position and orientation
functions we will use are wrappers to spice functions that we defined in our ``dawn_giant`` module before.  The position
function returns the positions of Vesta with respect to the Solar System Bary Center in the inertial frame.  The
orientation function returns the rotation from the Vesta fixed frame to the inertial frame as an :class:`.Rotation`
object, which GIANT uses to rotate the shape model so that the correct side of the asteroid is viewed.

.. code::

    # we need to make this into a SceneObject, which essentially allows us to wrap the object with functions that
    # give the state of the object at any given time
    vesta_obj = SceneObject(vesta_shape, position_function=vesta_position, orientation_function=vesta_attitude, name='Vesta')

We also need to create a :class:`.SceneObject` for the sun.  While the sun won't be imaged directly (so we don't need
a shape model), we do need to know its relative position in the scene so that we can predict the illumination
conditions.  Therefore, we create a :class:`.SceneObject` wrapped around a :class:`.Point` object to represent the sun.

.. code::

    # now we need to form the SceneObject for our Sun Object
    sun_obj = SceneObject(Point([0, 0, 0]), position_function=sun_position, orientation_function=sun_orientation)

Finally, we can define our actual scene.  This is done by creating an :class:`.Scene` instance which includes our
Vesta and Sun objects, as well as our camera instance which provides the scene relative information about the location
and orientation of the camera in the inertial frame.

In this scene, Vesta is the only target we are observing, but GIANT is set up to allow multiple targets to be observed
in the same scene, therefore we wrap the Vesta object in a list.  The sun becomes the light source in the scene.

.. code::

    # now we can form our scene
    opnav_scene = Scene(target_objs=[vesta_obj], light_obj=sun_obj)

Creating the RelNav Instance and extracting the observables
-----------------------------------------------------------
With the scene defined we can now create our :class:`.RelativeOpNav` instance.  The :class:`.RelativeOpNav` class
behaves very similarly to the :class:`.StellarOpNav` and :class:`.Calibration` classes, but exposes methods and settings
for performing Relative Navigation instead of Stellar Navigation and Calibration.

We create the :class:`.RelativeOpNav` class by providing it the camera, the scene, a BRDF to translate viewing geometry
into a predicted brightness, and a set of dictionaries to specify the settings for the various estimators in the
RelNav class (these can also be set as attributes after initialization as with the :class:`.StellarOpNav` and
:class:`.Calibration` classes).

The Vesta approach OpNavs only include images where Vesta is resolved (> 5 pixels in apparent diameter) thus we will
only be using cross-correlation and only need to worry about settings for the :class:`.XCorrCenterFinding` class.
In particular, we only really care about the ``grid_size`` and ``denoise_image`` settings.  The ``grid_size`` setting
specifies the number of rays we want to use to estimate the brightness in each pixel.  GIANT always assumes a square
grid and this number specifies the length of the sides.  Therefore, if you specify a grid-size of 9, then you will use a
9x9 grid of rays for each pixel (which quickly adds up to a lot of rays).  Because the body gets pretty large for our
last day of OpNavs we are going to process, we'll only use a ``grid_size`` of 3 pixels, which creates a 3x3 grid of rays
for each pixel.  The ``denoise_image`` flag specifies whether we want to attempt to decrease the noise in the image
using a Gaussian Smoothing technique.  Whether you set this flag to true or not depends on how noisey the images are.
In general though, it is good to set this to ``True``.  We can also use the ``search_region`` setting to restrict how
many pixels around the predicted location we should look for the correlation peak.  This can be useful for images where
the target is smaller in the field of view to ensure that we don't get any false positives due to noise.

The ``brdf`` keyword argument to the :class:`RelativeOpNav` class specifies the function that will convert viewing
geometry (observation vector, illumination vector, surface normal, surface albedo) into a brightness value.  GIANT has
a number of BRDFs available in the :mod:`.illumination` sub-module and in this case we'll use the familiar
:class:`.McEwenIllumination` BRDF.

.. code::

    # define the RelativeOpNav instance
    # define the settings for the portions of Relnav
    xcorr_kwargs = {"grid_size": 3, "denoise_image": True,
                    'search_region': 50}

    relnav = RelativeOpNav(camera, opnav_scene,
                           xcorr_kwargs=xcorr_kwargs,
                           brdf=McEwenIllumination(),
                           save_templates=True)

With the RelNav instance defined, we can now extract the observables, which take the form of observed pixel locations of
the center-of-figure of the body in each image.  We do this by calling the :meth:`.auto_estimate` method, which loops
through each image, updates the scene to the predicted state at the time of the image, determines whether the body is
resolved or not, and then locates the body in the image using either normalized cross correlation (resolved bodies) or
by performing a Gaussian fit to the illumination data (unresolved bodies).  Alternatively you could apply a specific
relnav technique using :meth:`.ellipse_matching_estimate`, :meth:`.limb_matching_estimate`,
:meth:`.cross_correlation_estimate`, :meth:`.moment_algorithm_estimate`, or :meth:`.unresolved_estimate`.  You can try
playing around with these if you want, though note that not all of the visualization routines will work with all of the
methods.

.. code::

    relnav.auto_estimate()

And that is it, we've used GIANT to extract center-of-figure observables from real images of Dawn's approach to
Vesta.  We can examine our results using the visualization functions we imported from GIANT.  :func:`.limb_summary_gif`
creates a GIF showing the alignment of the limbs in each image after identifying the body, :func:`.template_summary_gif`
creates a GIF showing the actual image of the target and the predicted image of the target for each image and each
target, and :func:`.show_center_finding_residuals` shows the observed-computed center finding resiudals in pixels.

.. code::

    # show the results
    limb_summary_gif(relnav)
    template_summary_gif(relnav)
    show_center_finding_residuals(relnav)
    plt.show()

.. note::
    If you receive an error about ``TypeError: 'NoneType' object is not callable`` then you likely
    need to update matplotlib by doing ``pip install --upgrade matplotlib``

You can finish now, or you can try playing around with images from other OpNav days.

The Complete OpNav Script
-------------------------
For your convenience, the complete ``opnav.py`` script is presented here.

.. code::

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
    from giant.stellar_opnav.stellar_class import StellarOpNav

    # tool for visualizing the results of our star identification
    from giant.stellar_opnav.visualizer import show_id_results

    # the star catalogue we will use for our "truth" star locations
    from giant.catalogues.giant_catalogue import GIANTCatalogue

    # the class we will use to perform the relative navigation
    from giant.relative_opnav.relnav_class import RelativeOpNav

    # the point spread function for the camera
    from giant.point_spread_functions.gaussians import Gaussian

    # the scene we will use to describe how things are related spatially
    from giant.ray_tracer.scene import Scene, SceneObject

    # The shape object we will use for the sun
    from giant.ray_tracer.shapes import Point

    # the illumination function we will use to predict the image of our model
    from giant.ray_tracer.illumination import McEwenIllumination

    # some utilities from giant for visualizing the relative opnav results
    from giant.relative_opnav.visualizer import limb_summary_gif, template_summary_gif, show_center_finding_residuals

    # A module to provide access to the NAIF Spice routines
    import spiceypy as spice

    # python standard library serialization tool
    import pickle


    if __name__ == "__main__":
        # filter some annoying warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=DeprecationWarning)

        # furnish the meta kernel so we have all of the a priori state information
        spice.furnsh('./meta_kernel.tm')

        # choose the images we are going to process
        # use sorted to ensure they are in time sequential order
        images = sorted(glob.glob('../opnav/2011123_OPNAV_001/*.FIT') +
                        glob.glob('../opnav/2011165_OPNAV_007/*.FIT') +
                        glob.glob('../opnav/2011198_OPNAV_017/*.FIT'))

        # load the camera model we are using
        camera_model = load('dawn_camera_models.xml', 'FC2')

        # create the camera instance and load the images
        camera = DawnFCCamera(images=images, model=camera_model, psf=Gaussian(sigma_x=0.75, sigma_y=0.75, size=5),
                              attitude_function=fc2_attitude)

        # do the stellar opnav to correct the attitude
        # build the stellar opnav object, which is very similar to the calibration object but without the ability to do
        # calibration.
        sopnav = StellarOpNav(camera, star_id_kwargs={'catalogue': GIANTCatalogue()})

        # ensure only the long exposure images are on
        sopnav.camera.only_long_on()

        # set the parameters to get a successful star identification
        # we only need to estimate the attitude here so we can be fairly conservative
        sopnav.star_id.max_magnitude = 8.0
        sopnav.image_processing.poi_threshold = 20
        sopnav.star_id.tolerance = 40
        sopnav.star_id.ransac_tolerance = 1
        sopnav.star_id.max_combos = 1000

        # now id the stars and estimate the attitude
        sopnav.id_stars()
        sopnav.estimate_attitude()

        # ensure we got a good id
        # show_id_results(sopnav)
        sopnav.sid_summary()

        # now, we need to turn on the short exposure images, and use the updated attitude from the long exposure images to
        # update the attitude for the short exposure images
        sopnav.camera.only_short_on()
        sopnav.camera.update_short_attitude()

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

        # define the RelativeOpNav instance
        # define the settings for the portions of Relnav
        xcorr_kwargs = {"grid_size": 3, "denoise_image": True,
                        'search_region': 50}

        relnav = RelativeOpNav(camera, opnav_scene,
                               xcorr_kwargs=xcorr_kwargs,
                               brdf=McEwenIllumination(),
                               limb_matching_kwargs={'recenter': False},
                               save_templates=True)

        relnav.auto_estimate()

        # show the results
        limb_summary_gif(relnav)
        template_summary_gif(relnav)
        show_center_finding_residuals(relnav)
        plt.show()

Conclusion
==========
And that's the basics of GIANT.  We successfully generated a camera model from star images and extracted
center-of-figure observables from OpNav images for the DAWN approach to Vesta.
There is certainly much more you can do with GIANT, but this provides a general
overview of how things work and shows how you can quickly get GIANT working for a new mission.  For more details,
read through the rest of the documentation.
