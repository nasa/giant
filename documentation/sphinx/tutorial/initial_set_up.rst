Initial Set-up
==============

GIANT in its base form is also designed to be entirely mission agnostic.  This makes it really easy to use GIANT
for multiple missions without ever having to change the core GIANT code.  Unfortunately this also means that GIANT
isn't 100% ready for use right out of the box.  Instead, we need to complete a few simple classes to tailor things to
our particular setup.

Typically this is done in a separate module so that we can store the updates we make and continue using them in the
future.  To make your setup module, begin by making a new directory (anywhere you want).  For this tutorial, lets
try to set up GIANT for DAWN's approach to Vesta.

.. code::

    mkdir dawn_giant
    cd dawn_giant

Customizing GIANT for DAWN
--------------------------
We need to define our mission specific implementation.  This involves subclassing a few of the GIANT classes to ensure
that they work with our current mission.  The steps to do this are discussed in the following subsections.

Defining Our Imports
--------------------
The first step to creating our GIANT customization module is to import all of the modules and packages we will need.  It
is considered good practice to do this at the top of each file, instead of spreading the imports throughout the code, so
that is how we will build our module here.  To begin, create a new file called `dawn_giant.py` in the `dawn_giant`
directory and open it with your favorite text editor.  Now, enter the following imports so we have everything available
that we'll need later.  You'll see these things in use in a little bit so just trust us that we need them for now.

.. code::

    # the GIANT classes we need to customize
    from giant.image import OpNavImage
    from giant.camera import Camera

    # the attitude object that represents rotations in GIANT
    from giant.rotations import Rotation

    # a GIANT module for working with spice
    import giant.utilities.spice_interface as spint

    # a standard library module that lets us query and manipulate file paths
    import os

    # A module to provide access to the NAIF Spice routines
    import spiceypy as spice
    from spiceypy.utils.support_types import SpiceyError

    # a standard library for representing dates and time deltas
    from datetime import timedelta

    # a library for parsing label files.
    import pvl

Subclassing OpNavImage
----------------------
With the imports out of the way the first object we need to update is the :class:`.OpNavImage` class. While its not
required to personalize the :class:`.OpNavImage` class, it is generally a good idea because it allows you to
automatically fill in much of the required meta data for the image at load time, and also allows you to build special
loaders that can handle non-standard image formats (like raw formats directly from the spacecraft).  There are two
different methods we can override to set these behaviours.  The first is :meth:`.load_image`.  In this method, the image
data is loaded from the file and stored as a 2D numpy array.  The method expects a string to be input that represents
the path to the file to be loaded, and the rest of GIANT expects that the loaded image data is returned as a numpy
array.  If your image is in a basic format then you probably don't need to worry about this (and for this tutorial, that
is the case), but we wanted to make this possibility known to you.

The next method we can override is the :meth:`.parse_data` method.  This method is intended to fill out all of the meta
data from an image from the image file the image was loaded from.  In pure GIANT, this method is not implemented because
there are an incredible number of ways to communicate meta data with an image so it would be impossible for us to
provide even a small subset.  Therefore, if you want this feature you need to implement it yourself.

For DAWN, the images come in the standard FITS format, which GIANT can read by default, so we do not need to update the
:meth:`.load_image` method.  However, we do need to update the :meth:`.parse_data` method to parse the data from the
label files that correspond to each image.  The following code shows how to do this:

.. code::

    class DawnFCImage(OpNavImage):

        def parse_data(self):

            # be sure the image file exists
            if os.path.exists(self.file):

                # get the extension from the file
                _, ext = os.path.splitext(self.file)

                # replace the extension from the file with LBL to find the corresponding label file
                lbl_file = self.file.replace(ext, '.LBL')

                # check to see if the label file exists
                if os.path.exists(lbl_file):

                    # read the label file
                    with open(lbl_file, 'r') as lfile:
                        # pvl treats \ as escape characters so we need to replace them
                        data = pvl.loads(lfile.read().replace('\\', '/'))

                    # extract the exposure time from the label and convert from ms to seconds
                    self.exposure = data["EXPOSURE_DURATION"].value / 1000

                    # set the exposure type based off of the exposure length to make handling long/short opnav sequences
                    # easier.  This is typically camera specific and needs to be set by an analyst
                    if self.exposure > 1:
                        self.exposure_type = "long"
                    else:
                        self.exposure_type = "short"

                    # extract the observation observation_date (middle of the exposure time of the image)
                    self.observation_date = data["START_TIME"].replace(tzinfo=None) + timedelta(seconds=self.exposure / 2)

                    # get the temperature of the camera for this image
                    self.temperature = data["DAWN:T_LENS_BARREL"].value - 273.15  # convert kelvin to celsius

                    # get the quaternion of the rotation from the inertial frmame to the camera frame
                    # store the rotation as an attitude object.  Need to move the scalar term last
                    self.rotation_inertial_to_camera = Rotation(data["QUATERNION"][1:] + data["QUATERNION"][0:1])

                    # get the target
                    self.target = data["TARGET_NAME"]
                    if self.target == "N/A":
                        self.target = None  # notify that we don't have a target here
                    else:
                        self.target = self.target.split()[0]  # throw out the target number and just keep the name

                    # get the instrument name (spice instrument frame name)
                    if data["INSTRUMENT_ID"] == "FC2":
                        self.instrument = "DAWN_FC2"
                    else:
                        self.instrument = "DAWN_FC1"

                    # set the spacecraft the camera belongs to
                    self.spacecraft = "DAWN"

                    # set the saturation value for this image
                    self.saturation = 2 ** 16 - 1

                    # query spice to get the camera position and velocity in the inertial frame,
                    # as well as the sun direction in the camera frame

                    # first convert the observation observation_date to ephemeris time
                    try:
                        et = spice.str2et(self.observation_date.isoformat())
                    except SpiceyError:
                        et = 0
                        print('unable to compute ephemeris time for image at time {}'.format(
                            self.observation_date.isoformat()))

                    # get the position and velocity of the camera (sc) with respect to the solar system bary center
                    try:
                        state, _ = spice.spkezr(self.spacecraft, et, 'J2000', 'NONE', "SSB")
                        self.position = state[:3]
                        self.velocity = state[3:]

                    except SpiceyError:

                        print('Unable to retrieve camera position and velocity \n'
                              'for {0} at time {1}'.format(self.instrument, self.observation_date.isoformat()))

                else:
                    raise ValueError("we can't find the label file for this image so we can't parse the data."
                                     "Looking for file {}".format(lbl_file))

In the above code block, the first thing we do is extract the exposure time and type and store it in the
:attr:`~.OpNavImage.exposure` and :attr:`~.OpNavImage.exposure_type` attributes.  Both of these attributes are required
attributes and should always be set when creating an :class:`.OpNavImage`.  The :attr:`~.OpNavImage.exposure` should
be the exposure time in seconds and the :attr:`~.OpNavImage.exposure_type` should be a string specifying either
``long`` or ``short`` depending on the exposure length.  The :attr:`~.OpNavImage.exposure_type` is used for quickly
turning images on and off when doing long/short pairs of OpNav images to get attitude and center of figure
observations.  Setting this parameter is dependent on the camera being used, and sometimes on the phase of the mission
being considered.

Next we extract the observation date for the image.  This is set to be the mid-point of the exposure interval in UTC
time and is stored in the :attr:`~.OpNavImage.observation_date` attribute.  Again, this is a required attribute and
should always be set to a datetime representation of the observation date in UTC.

Now, we extract the temperature of the camera at the time we captured the image in degrees celsius.  The
:attr:`~.OpNavImage.temperature` parameter is not required, but can be useful as it allows us to use/estimate
temperature dependent focal lengths, which are required for some cameras.

Next, we extract the attitude of the camera with respect to the inertial frame and store it in the
:attr:`~.OpNavImage.rotation_inertial_to_camera` attribute.  This is a required attribute that is used extensively
throughout GIANT.  It also can get updated when performing stellar processing.

Following the attitude, we extract the target that the camera was observing.  This is not a required attribute, but
it can be useful meta-data for the user, and for determining the sun direction in the camera frame as we will see
shortly.  This is stored as a string in the :attr:`~.OpNavImage.target` attribute, and typically is set to the NAIF
Spice identifier of the target for ease of use.

The last information to be extracted from the label file is the instrument.  The :attr:`~.OpNavImage.instrument`
attribute is again not a required attribute, but it can be useful when querying Spice if you set it to the Spice
frame ID for the instrument.

Next, we need to set two attributes which are external to the label file.  This includes the
:attr:`~.OpNavImage.spacecraft` attribute, which should be a string describing the spacecraft that hosts the camera.
While this isn't used directly in GIANT, it can be useful if you use the NAIF Spice id for the spacecraft.  The
second attribute we set is the :attr:`~.OpNavImage.saturation` attribute.  The :attr:`~.OpNavImage.saturation`
attribute specifies the maximum DN value above which a pixel is considered saturated.  This value is important to set
properly because GIANT will dispose of objects that contain saturated pixels when doing stellar processing.

Finally, we need to query some data from spice.  This includes the camera position and velocity in the inertial frame
with respect to the solar system bary center.  These values are used in the relative navigation and stellar
processing portions of GIANT to determine the location of objects in the camera frame.  While it is not required that
they be inertial with respect to the solar system bary center, this is what is typically used in GIANT.  If you want
to use a different convention, that is fine, but you must carefully trace where it is used throughout and be sure you
are consistent.  Similarly, the units are almost always kilometers and kilometers per second in GIANT, but this is not
required. You could conceivably use whatever units you want, so long as you are consistent for both your position,
velocity, and shape models.

Subclassing the Camera Class
----------------------------
With the images themselves handled, we can turn our attention to the :class:`.Camera` class.  The :class:`.Camera` class
works as a container to store all of the images we are currently processing, as well as some information about the
camera itself, including the geometric camera model that maps points in the camera frame to points in an image.  There
are typically two main things we need to update in the camera class.  This is the special :meth:`Camera.__init__` method
and the :meth:`.Camera.preprocessor` method.  We set these methods for the DAWN framing cameras below

.. code::

    class DawnFCCamera(Camera):

        # update the init function to use the new DawnFCImage class instead of the default OpNavImage class
        def __init__(self, images=None, model=None, name=None, spacecraft_name=None,
                     frame=None, parse_data=True, psf=None, attitude_function=None, start_date=None, end_date=None,
                     default_image_class=DawnFCImage):

            super().__init__(images=images, model=model, name=name, spacecraft_name=spacecraft_name,
                             frame=frame, parse_data=parse_data, psf=psf,
                             attitude_function=attitude_function, start_date=start_date, end_date=end_date,
                             default_image_class=default_image_class)

        def preprocessor(self, image):

            # here we might apply corrections to the image (like flat fields and darks) or we can extract extra
            # information about the image and store it as another attribute (like dark_pixels which can be used to
            # compute the noise level in the image). For the DAWN framing cameras though, we don't need to do anything
            # so we just return the image unmodified.
            return image

In the init method, we simply change the default value for the ``default_image_class`` key word argument to point to
our new ``DawnFCImage`` class that we just defined.  We then pass all of these values to the default constructor for the
:class:`.Camera` class and move on.

In the :meth:`~.Camera.preprocessor` method, we don't have to do anything for the DAWN framing cameras except to
return the images as they are.  For other cameras and missions, the preprocessor is where you can put things like
image corrections to remove fixed pattern noise, apply dark and flat field corrections, and extract covered active
pixels into the :attr:`~.OpNavImage.dark_pixels` attribute as a way to extract the noise level for each image.

Defining Functions to Return State Information
----------------------------------------------
One non-required thing we can due is to predefine some functions that return the state (position, velocity, attitude) of
certain objects that we will frequently need.  Since most of this data is coming from spice, we can use the GIANT
:mod:`.spice_interface` module to interface with SPICE and make the functions that we need.

.. code::

    # convenience functions
    def sun_orientation(*args):
        # always set the sun orientation to be the identity rotation (J2000) because it doesn't actually matter
        return Rotation([0, 0, 0])


    # define a function that will return the sun position in the inertial frame wrt SSB for a datetime
    sun_position = spint.SpicePosition('SUN', 'J2000', 'NONE', 'SSB')


    # define a function that will return the framing camera 1 attitude with respect to inertial for an input datetime
    fc1_attitude = spint.SpiceOrientation('J2000', 'DAWN_FC1')
    # define a function that will return the framing camera 2 attitude with respect to inertial for an input datetime
    fc2_attitude = spint.SpiceOrientation('J2000', 'DAWN_FC2')


    # define a function that will return the dawn spacecraft attitude with respect to inertial for an input datetime
    dawn_attitude = spint.SpiceOrientation('J2000', 'DAWN_SPACECRAFT')
    # define a function that will return the spacecraft state in the inertial frame wrt SSB for a datetime
    dawn_state = spint.SpiceState('DAWN', 'J2000', 'NONE', 'SSB')
    # define a function that will return the spacecraft position in the inertial frame wrt SSB for a datetime
    dawn_position = spint.SpicePosition('DAWN', 'J2000', 'NONE', 'SSB')


    # define a function that will return the vesta body fixed attitude with respect to inertial for an input datetime
    # GIANT needs this to be from body fixed to inertial
    vesta_attitude = spint.SpiceOrientation('VESTA_FIXED', 'J2000')
    # define a function that will return vesta's position and velocity in the inertial frame wrt SSB for a datetime
    vesta_state = spint.SpiceState('VESTA', 'J2000', 'NONE', 'SSB')
    # define a function that will return vesta's position in the inertial frame wrt SSB for a datetime
    vesta_position = spint.SpicePosition('VESTA', 'J2000', 'none', 'SSB')

Installing dawn_giant
---------------------
The final step in customizing GIANT is to install our ``dawn_giant`` module to our python path.  While this isn't a
required step, it makes it easier to have access to all of the work we just did from whatever directory we want, so
it is strongly recommended.  The easiest way to perform this step is using setuptools and a setup.py file.

In the ``dawn_giant`` directory, create a file called ``setup.py`` and open it with your favorite text editor.  Then,
place the following code into the file

.. code::

    from setuptools import setup

    setup(
        name='dawn_giant',
        version='1.0',
        description='Dawn Customizations for GIANT',
        py_modules=['dawn_giant'],
        install_requires=[
             'giant',
             'numpy',
             'spiceypy',
             'pvl',
             'bs4',
             'requests'
        ]
    )

This script simply tells python that we want to install our dawn_giant module so that it is always available.  It also
lists the external requirements that need to be installed for this file to work.  If you've been following along to this
point then most of these requirements are already installed, with the exception of `pvl`, which we discussed above.  The
nice thing is, when we run the ``setup.py`` script, python will install pvl for us.

Now, be sure that your ``giant_env`` is activated and then run ``python setup.py develop`` from the ``dawn_giant``
directory in order to install the ``dawn_giant`` package.  To test this install, simply cd to any other directory,
start an interactive python shell, and then try ``import dawn_giant``.  This should complete successfully without any
errors.

And that is it, we have successfully customized GIANT to work for the DAWN mission and now we can move on to doing some
actual processing.

The Full dawn_giant File
------------------------
For your convenience the full ``dawn_giant.py`` file is presented here

.. code::

    # the GIANT classes we need to customize
    from giant.image import OpNavImage
    from giant.camera import Camera

    # the attitude object that represents rotations in GIANT
    from giant.rotations import Rotation

    # a GIANT module for working with spice
    import giant.utilities.spice_interface as spint

    # a standard library module that lets us query and manipulate file paths
    import os

    # A module to provide access to the NAIF Spice routines
    import spiceypy as spice
    from spiceypy.utils.support_types import SpiceyError

    # a standard library for representing dates and time deltas
    from datetime import timedelta

    # a library for parsing label files.
    import pvl


    class DawnFCImage(OpNavImage):

        def parse_data(self):

            # be sure the image file exists
            if os.path.exists(self.file):

                # get the extension from the file
                _, ext = os.path.splitext(self.file)

                # replace the extension from the file with LBL to find the corresponding label file
                lbl_file = self.file.replace(ext, '.LBL')

                # check to see if the label file exists
                if os.path.exists(lbl_file):

                    # read the label file
                    with open(lbl_file, 'r') as lfile:
                        # pvl treats \ as escape characters so we need to replace them
                        data = pvl.loads(lfile.read().replace('\\', '/'))

                    # extract the exposure time from the label and convert from ms to seconds
                    self.exposure = data["EXPOSURE_DURATION"].value / 1000

                    # set the exposure type based off of the exposure length to make handling long/short opnav sequences
                    # easier.  This is typically camera specific and needs to be set by an analyst
                    if self.exposure > 1:
                        self.exposure_type = "long"
                    else:
                        self.exposure_type = "short"

                    # extract the observation observation_date (middle of the exposure time of the image)
                    self.observation_date = data["START_TIME"].replace(tzinfo=None) + timedelta(seconds=self.exposure / 2)

                    # get the temperature of the camera for this image
                    self.temperature = data["DAWN:T_LENS_BARREL"].value - 273.15  # convert kelvin to celsius

                    # get the quaternion of the rotation from the inertial frmame to the camera frame
                    # store the rotation as an attitude object.  Need to move the scalar term last
                    self.rotation_inertial_to_camera = Rotation(data["QUATERNION"][1:] + data["QUATERNION"][0:1])

                    # get the target
                    self.target = data["TARGET_NAME"]
                    if self.target == "N/A":
                        self.target = None  # notify that we don't have a target here
                    else:
                        self.target = self.target.split()[0]  # throw out the target number and just keep the name

                    # get the instrument name (spice instrument frame name)
                    if data["INSTRUMENT_ID"] == "FC2":
                        self.instrument = "DAWN_FC2"
                    else:
                        self.instrument = "DAWN_FC1"

                    # set the spacecraft the camera belongs to
                    self.spacecraft = "DAWN"

                    # set the saturation value for this image
                    self.saturation = 2 ** 16 - 1

                    # query spice to get the camera position and velocity in the inertial frame,
                    # as well as the sun direction in the camera frame

                    # first convert the observation observation_date to ephemeris time
                    try:
                        et = spice.str2et(self.observation_date.isoformat())
                    except SpiceyError:
                        et = 0
                        print('unable to compute ephemeris time for image at time {}'.format(
                            self.observation_date.isoformat()))

                    # get the position and velocity of the camera (sc) with respect to the solar system bary center
                    try:
                        state, _ = spice.spkezr(self.spacecraft, et, 'J2000', 'NONE', "SSB")
                        self.position = state[:3]
                        self.velocity = state[3:]

                    except SpiceyError:

                        print('Unable to retrieve camera position and velocity \n'
                              'for {0} at time {1}'.format(self.instrument, self.observation_date.isoformat()))

                else:
                    raise ValueError("we can't find the label file for this image so we can't parse the data."
                                     "Looking for file {}".format(lbl_file))


    class DawnFCCamera(Camera):

        # update the init function to use the new DawnFCImage class instead of the default OpNavImage class
        def __init__(self, images=None, model=None, name=None, spacecraft_name=None,
                     frame=None, parse_data=True, psf=None, attitude_function=None, start_date=None, end_date=None,
                     default_image_class=DawnFCImage):
            super().__init__(images=images, model=model, name=name, spacecraft_name=spacecraft_name,
                             frame=frame, parse_data=parse_data, psf=psf,
                             attitude_function=attitude_function, start_date=start_date, end_date=end_date,
                             default_image_class=default_image_class)

        def preprocessor(self, image):
            # here we might apply corrections to the image (like flat fields and darks) or we can extract extra
            # information about the image and store it as another attribute (like dark_pixels which can be used to
            # compute the noise level in the image.  For the DAWN framing cameras though, we don't need to do anything
            # so we just return the image unmodified.
            return image


    # convenience functions
    def sun_orientation(*args):
        # always set the sun orientation to be the identity rotation (J2000) because it doesn't actually matter
        return Rotation([0, 0, 0])


    # define a function that will return the sun position in the inertial frame wrt SSB for a datetime
    sun_position = spint.SpicePosition('SUN', 'J2000', 'NONE', 'SSB')

    # define a function that will return the framing camera 1 attitude with respect to inertial for an input datetime
    fc1_attitude = spint.SpiceOrientation('J2000', 'DAWN_FC1')
    # define a function that will return the framing camera 2 attitude with respect to inertial for an input datetime
    fc2_attitude = spint.SpiceOrientation('J2000', 'DAWN_FC2')

    # define a function that will return the dawn spacecraft attitude with respect to inertial for an input datetime
    dawn_attitude = spint.SpiceOrientation('J2000', 'DAWN_SPACECRAFT')
    # define a function that will return the spacecraft state in the inertial frame wrt SSB for a datetime
    dawn_state = spint.SpiceState('DAWN', 'J2000', 'NONE', 'SSB')
    # define a function that will return the spacecraft position in the inertial frame wrt SSB for a datetime
    dawn_position = spint.SpicePosition('DAWN', 'J2000', 'NONE', 'SSB')

    # define a function that will return the vesta body fixed attitude with respect to inertial for an input datetime
    # GIANT needs this to be from body fixed to inertial
    vesta_attitude = spint.SpiceOrientation('VESTA_FIXED', 'J2000')
    # define a function that will return vesta's position and velocity in the inertial frame wrt SSB for a datetime
    vesta_state = spint.SpiceState('VESTA', 'J2000', 'NONE', 'SSB')
    # define a function that will return vesta's position in the inertial frame wrt SSB for a datetime
    vesta_position = spint.SpicePosition('VESTA', 'J2000', 'none', 'SSB')

