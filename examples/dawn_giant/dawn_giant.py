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
from spiceypy import SpiceyError

# a standard library for representing dates and time deltas
from datetime import timedelta

# a library for parsing label files.
import pvl


class DawnFCImage(OpNavImage):

    def parse_data(self, *args):

        # be sure the image file exists
        assert self.file is not None, "we need a file to parse the data from"
        if os.path.exists(self.file):

            # get the extension from the file
            _, ext = os.path.splitext(self.file)

            # replace the extension from the file with LBL to find the corresponding label file
            lbl_file = str(self.file).replace(ext, '.LBL')

            # check to see if the label file exists
            if os.path.exists(lbl_file):

                # read the label file
                with open(lbl_file, 'r') as lfile:
                    # pvl treats \ as escape characters so we need to replace them
                    data = pvl.loads(lfile.read().replace('\\', '/'))
                    
                    assert data is not None, "we were unable to parse the label file {}".format(lbl_file)

                # extract the exposure time from the label and convert from ms to seconds
                self.exposure = data["EXPOSURE_DURATION"].value / 1000 # pyright: ignore[reportAttributeAccessIssue]

                # set the exposure type based off of the exposure length to make handling long/short opnav sequences
                # easier.  This is typically camera specific and needs to be set by an analyst
                assert self.exposure is not None, "we were unable to extract the exposure time from the label file {}".format(lbl_file)
                if self.exposure > 1:
                    self.exposure_type = "long"
                else:
                    self.exposure_type = "short"

                # extract the observation observation_date (middle of the exposure time of the image)
                self.observation_date = data["START_TIME"].replace(tzinfo=None) + timedelta(seconds=self.exposure / 2) # pyright: ignore[reportAttributeAccessIssue]

                # get the temperature of the camera for this image
                self.temperature = data["DAWN:T_LENS_BARREL"].value - 273.15  # pyright: ignore[reportAttributeAccessIssue] # convert kelvin to celsius

                # get the quaternion of the rotation from the inertial frmame to the camera frame
                # store the rotation as an attitude object.  Need to move the scalar term last
                self.rotation_inertial_to_camera = Rotation(data["QUATERNION"][1:] + data["QUATERNION"][0:1])

                # get the target
                self.target = str(data["TARGET_NAME"])
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
                    self.position = state[:3] # pyright: ignore[reportIndexIssue]
                    self.velocity = state[3:] # pyright: ignore[reportIndexIssue]

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
    return Rotation()


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