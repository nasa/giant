# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


r"""
This module provides a number of classes for interfacing with files used in the Stereophotoclinometry (SPC) software
suite for performing shape modelling and surface feature navigation.

This module *does not* provide the same functionality that the SPC software does, instead it just provides the ability
to view, manipulate, and write SPC files and makes it easy to transform SPC data into a format that GIANT
can use.  This module does assume some level of familiarity with the SPC software and files, so descriptions
are brief.  See the SPC wiki for more details at https://web.psi.edu/spc_wiki/HomePage
"""

import numpy as np
import os
import struct  # we need the struct stuff to handle the conversion to binary
from datetime import datetime
from typing import Optional, Union

from dateutil import parser

from ..ray_tracer.shapes import Triangle64, Triangle32
from .._typing import PATH, NONENUM, NONEARRAY, Real

DATE_FMT = '%Y %b %d %H:%M:%S.%f'
"""
This is the observation_date format used in SPC summary and nominal files.  It is used to parse and write the dates
"""


class Summary:
    """
    This class is used to read/write from the SPC summary (.SUM) files.

    The summary files represent meta data about an Image.  The image data itself is stored in a corresponding
    .DAT file which can be read/written with the :class:`Image` class.  When creating an instance of this class,
    you can enter the ``file_name`` argument, in which case the specified SUM file will be opened and read into the
    members of this class (if the sum file exists), or you can specify the individual components of the sum file as
    keyword arguments.  If you specify both an existing SUM file and keyword arguments for the individual
    components of the SUM file, the keyword arguments will override whatever is read from the file.

    The documentation for each instance attribute of this class specifies the corresponding location in the SUM file
    for clarification and reference.
    """

    def __init__(self, file_name: Optional[PATH] = None, image_name: Optional[str] = None,
                 observation_date: Optional[datetime] = None, num_cols: NONENUM = None, num_rows: NONENUM = None,
                 min_illum: NONENUM = None, max_illum: NONENUM = None, focal_length: NONENUM = None,
                 principle_point: NONEARRAY = None, position_camera_to_target: NONEARRAY = None,
                 rotation_target_fixed_to_camera: NONEARRAY = None, direction_target_to_sun: NONEARRAY = None,
                 intrinsic_matrix: NONEARRAY = None, near_dist_params: NONEARRAY = None,
                 sig_pos_camera_to_target: NONEARRAY = None, sig_rot_target_fixed_to_camera: NONEARRAY = None):
        """
        :param file_name: The path to a summary file to read data from
        :param image_name: The name of the image this object corresponds to
        :param observation_date: The utc observation observation_date for the image as a python datetime object
        :param num_cols: The number of columns in the image array
        :param num_rows: The number of rows in the image array
        :param min_illum: The minimum illumination value to allow for an image.  SPC ignores pixels with DN values
                          below this
        :param max_illum: The maximum illumination value to allow for an image.  SPC ignores pixels wiht DN values
                          above this.
        :param focal_length: The focal length of the camera in units of millimeters
        :param principle_point: The principal point of the camera (location where the optical axis pierces the image
                                plane) in units of pixels.
        :param position_camera_to_target: The vector from the spacecraft to the target in the target body fixed frame as
                                          a length 3 numpy array
        :param rotation_target_fixed_to_camera: The 3x3 rotation matrix from the target body fixed frame to the camera
                                                frame
        :param direction_target_to_sun: The unit vector from the target to the sun in the target body fixed frame
        :param intrinsic_matrix: The intrinsic matrix of the camera (see the :mod:`~.pinhole_model` documentation
                                 for details
        :param near_dist_params: This obsolete array contains special distortion values for the cameras that were flown
                                 on NEAR.  This should be all 0 for all modern SPC use.
        :param sig_pos_camera_to_target: This is the formal uncertainty on the ``position_camera_to_target`` vector
                                         (1 sigma)
        :param sig_rot_target_fixed_to_camera: This is the formal uncertainty on the spacecraft pointing
                                               (roll, pitch, yaw?) (1 sigma)
        """

        self.file_name = None  # type: Optional[PATH]
        """
        The path to the summary file that this object was populated from.
        
        This also will be the file that changes are written to if :meth:`.write` is called with no arguments.  
        
        Typically this is named according to SUMFILES/*IMAGENAME*.SUM where *IMAGENAME* is replaced with the SPC name
        of the image.  This can be set to None, a str, or a :class:`Path`.
        """

        self.image_name = None  # type: Optional[str]
        """
        The name of the image this object corresponds to.
        
        This corresponds to the first line of the SUM file.
        """

        self.observation_date = None  # type: Optional[datetime]
        """
        The middle of the time in UTC that the image was exposed during as a python datetime object.
        
        This corresponds to the second line of the SUM file.
        """

        self.num_cols = None  # type: Optional[int]
        """
        The integer number of columns in the image this Summary object corresponds to.
        
        This corresponds to the first value of the line containing ``NPX, NLN, THRSH`` in the SUM file (typically the 
        third line).
        """

        self.num_rows = None  # type: Optional[int]
        """
        The integer number of rows in the image this Summary object corresponds to.

        This corresponds to the second value of the line containing ``NPX, NLN, THRSH`` in the SUM file (typically the 
        third line).
        """

        self.min_illum = None  # type: NONENUM
        """
        The minimum DN value for a pixel to be used in SPC.
        
        Pixels with DN values below this threshold are ignored when computing correlation coefficients and other values
        in SPC.
        
        This corresponds to the third value of the line containing ``NPX, NLN, THRSH`` in the SUM file (typically the 
        third line).
        """

        self.max_illum = None  # type: NONENUM
        """
        The maximum DN value for a pixel to be used in SPC.

        Pixels with DN values above this threshold are ignored when computing correlation coefficients and other values
        in SPC.

        This corresponds to the fourth value of the line containing ``NPX, NLN, THRSH`` in the SUM file (typically the 
        third line).
        """

        self.focal_length = None  # type: NONENUM
        """
        The focal length of the camera in units of mm.
        
        This corresponds to the first value of the line containing ``MMFL, CTR`` in the SUM file (typically the fourth
        line).
        """

        self.princ_point = np.zeros(2)  # type: np.ndarray
        """
        The principle point of the camera as a length 2 numpy array of floats.
        
        This typically corresponds to the :attr:`.PinholeModel.px` and :attr:`.PinholeModel.py` attributes of the
        Pinhole based camera models in GIANT.  The first component is for the x-axis (columns) and the second component
        is for the y-axis (rows).
        
        This corresponds to the second and third values of the line containing ``MMFL, CTR`` in the SUM file (typically 
        the fourth line).
        """

        self.position_camera_to_target = np.zeros(3)  # type: np.ndarray
        """
        The position vector from the spacecraft to the target in target body fixed coordinates at the time of the image
        with units of kilometers as a length 3 numpy array of floats.
        
        This corresponds to all the values of the line containing ``SCOBJ`` in the SUM file (typically the fifth line).
        """

        self.rotation_target_fixed_to_camera = np.zeros((3, 3))  # type: np.ndarray
        """
        The rotation matrix from the target body fixed frame to the camera frame at the time of the image as a
        shape 3x3 numpy array of floats.

        This corresponds to all the values of the lines containing ``CX, CY, CZ`` in the SUM file (typically the 
        sixth through the eighth lines).  The matrix is formed according to ``np.vstack([CX, CY, CZ])`` 
        """

        self.direction_target_to_sun = np.zeros(3)  # type: np.ndarray
        """
        The unit vector from the target to the sun in the target body fixed frame at the time of the image as a length
        3 numpy array of floats.
        
        This corresponds to all the values of the line containing ``SZ`` in the sum file (typically the ninth line).
        """

        self.intrinsic_matrix = np.zeros((2, 3))  # type: np.ndarray
        """
        The intrinsic matrix of the camera as a shape 2x3 numpy array of floats.
        
        This has an extra column that is kept for SPC compatibility reasons but which is not used anymore.  Therefore,
        this does not directly correspond to the :attr:`.PinholeModel.intrinsic_matrix` attribute of the Pinhole model 
        and its subclasses in GIANT, but rather the first 2 columns of this matrix correspond to the first 2 columns of
        the intrinsic matrix in GIANT and the 3rd column should be discarded.
        
        This corresponds to all of the values of the line containing ``K-MATRIX`` in the SUM file (typically the
        tenth line).  Weirdly this is a row major flattening of the intrinsic matrix in the SUM file.
        """

        self.near_dist_params = np.zeros(4)  # type: np.ndarray
        """
        The deprecated NEAR distortion parameters.
        
        Just leave these at 0.
        
        This corresponds to all of the values of the line containing ``DISTORTION`` in the SUM file (typically the 
        eleventh line).
        """

        self.sig_pos_camera_to_target = np.zeros(3)  # type: np.ndarray
        """
        The formal uncertainty on the position_camera_to_target vector as a length 3 numpy array of floats.
        
        The formal uncertainty are the 1 sigma values for each component of the position vector.
        
        This corresponds to all of the values of the line containing ``SIGMA_VSO`` in the SUM file (typically the 
        twelfth line).
        """

        self.sig_rot_target_fixed_to_camera = np.zeros(3)  # type: np.ndarray
        """
        The formal uncertainty on the rotation from the target body frame to the spacecraft frame as a length 3 numpy 
        array of floats.

        The formal uncertainty are the 1 sigma values.

        This corresponds to all of the values of the line containing ``SIGMA_PTG`` in the SUM file (typically the 
        thirteenth line).
        """

        self.landmarks = {}
        """
        A dictionary of observed SPC landmarks and their observed locations in units of pixels.
        
        The keys to the dictionary are the SPC landmark names (typically 6 character alpha numeric names) 
        and the values are the pixel location the landmark was identified at in the image. Each value is the x, y or 
        column, row location as a length 2 list of floats 
        
        This corresponds to all of the lines between ``LANDMARKS`` and ``LIMB FITS`` or ``END FILE`` in the SUM file.
        """

        self.limb_fits = {}
        """
        A dictionary of observed SPC landmarks that occur on the limb of the target in the image and their observed 
        locations in units of pixels and formal uncertainty.

        The keys to the dictionary are the SPC landmark names (typically 6 character alpha numeric names) 
        and the values are the pixel location the landmark was identified at in the image followed by the uncertainty of
        the fit. Each value is the x, y, sigma or 
        column, row, sigma as a length 3 list of floats 

        This corresponds to all of the lines between ``LIMB FITS`` and ``LANDMARKS`` or ``END FILE`` in the SUM file.
        """

        if file_name is not None:
            self.read(file_name)
            self.file_name = file_name

        # override anything that was manually specified
        if image_name is not None:
            self.image_name = image_name

        if observation_date is not None:
            self.observation_date = observation_date

        if num_cols is not None:
            self.num_cols = num_cols

        if num_rows is not None:
            self.num_rows = num_rows

        if min_illum is not None:
            self.min_illum = min_illum

        if max_illum is not None:
            self.max_illum = max_illum

        if focal_length is not None:
            self.focal_length = focal_length

        if principle_point is not None:
            self.princ_point = np.array(principle_point)

        if position_camera_to_target is not None:
            self.position_camera_to_target = np.array(position_camera_to_target)

        if rotation_target_fixed_to_camera is not None:
            self.rotation_target_fixed_to_camera = np.array(rotation_target_fixed_to_camera)

        if direction_target_to_sun is not None:
            self.direction_target_to_sun = np.array(direction_target_to_sun)

        if intrinsic_matrix is not None:
            self.intrinsic_matrix = np.array(intrinsic_matrix)

        if near_dist_params is not None:
            self.near_dist_params = np.array(near_dist_params)

        if sig_pos_camera_to_target is not None:
            self.sig_pos_camera_to_target = np.array(sig_pos_camera_to_target)

        if sig_rot_target_fixed_to_camera is not None:
            self.sig_rot_target_fixed_to_camera = np.array(sig_rot_target_fixed_to_camera)

    def __eq__(self, other):

        if not isinstance(other, self.__class__):
            return False

        comp_dict = {'image_name': self.image_name == other.image_name,
                     'observation_date': self.observation_date == other.observation_date,
                     'num_cols': self.num_cols == other.num_cols,
                     'num_rows': self.num_rows == other.num_rows,
                     'min_illum': self.min_illum == other.min_illum,
                     'max_illum': self.max_illum == other.max_illum,
                     'focal_length': self.focal_length == other.focal_length,
                     'princ_point': (self.princ_point == other.princ_point).all(),
                     'position_camera_to_target': (self.position_camera_to_target ==
                                                   other.position_camera_to_target).all(),
                     'rotation_target_fixed_to_camera': (self.rotation_target_fixed_to_camera ==
                                                         other.rotation_target_fixed_to_camera).all(),
                     'direction_target_to_sun': (self.direction_target_to_sun == other.direction_target_to_sun).all(),
                     'intrinsic_matrix': (self.intrinsic_matrix == other.intrinsic_matrix).all(),
                     'near_dist_params': (self.near_dist_params == other.near_dist_params).all(),
                     'sig_pos_camera_to_target': (self.sig_pos_camera_to_target ==
                                                  other.sig_pos_camera_to_target).all(),
                     "sig_rot_target_fixed_to_camera": (self.sig_rot_target_fixed_to_camera ==
                                                        other.sig_rot_target_fixed_to_camera).all(),
                     "landmarks": self.landmarks == other.landmarks,
                     "limb_fits": self.limb_fits == other.limb_fits}

        if np.all(comp_dict.values()):
            return True
        else:
            print(comp_dict)
            return False

    def read(self, file_name: Optional[PATH] = None):
        """
        This method reads data from a SPC SUM file (normally SUMFILES/*IMAGENAME*.SUM) and populates the attributes of
        this class with that data.

        If ``file_name`` is not specified the :attr:`file_name` attribute of the class is used instead

        :param file_name: The file to load the sum data from
        """

        if file_name is None:
            file_name = self.file_name

        try:
            with open(file_name, 'r') as sum_file:

                # parse the name and observation_date lines (first two lines
                self.image_name = sum_file.readline().strip(' \n\t\r')

                self.observation_date = datetime.strptime(sum_file.readline().strip(' \n\t\r'), DATE_FMT)

                # parse the size of the Image and the dynamic range of the Image values
                line = sum_file.readline().strip(' \n\t\r').replace('D', 'E')

                self.num_cols, self.num_rows, self.min_illum, self.max_illum = np.fromstring(line, dtype=int,
                                                                                             count=4, sep=' ')

                # parse the focal length and the principal point of the camera
                line = sum_file.readline().strip(' \n\t\r').replace('D', 'E')

                self.focal_length, self.princ_point[0], self.princ_point[1] = np.fromstring(line, count=3, sep=' ')

                # parse the spacecraft to object position vector
                line = sum_file.readline().strip(' \n\t\r').replace('D', 'E')

                self.position_camera_to_target[:] = np.fromstring(line, count=3, sep=' ')

                # parse the object to camera rotation matrix
                line = sum_file.readline().strip(' \n\t\r').replace('D', 'E')

                self.rotation_target_fixed_to_camera[0] = np.fromstring(line, count=3, sep=' ')

                line = sum_file.readline().strip(' \n\t\r').replace('D', 'E')

                self.rotation_target_fixed_to_camera[1] = np.fromstring(line, count=3, sep=' ')

                line = sum_file.readline().strip(' \n\t\r').replace('D', 'E')

                self.rotation_target_fixed_to_camera[2] = np.fromstring(line, count=3, sep=' ')

                # parse the sun direction vector
                line = sum_file.readline().strip(' \n\t\r').replace('D', 'E')

                self.direction_target_to_sun[:] = np.fromstring(line, count=3, sep=' ')

                # parse the intrinsic calibration matrix
                line = sum_file.readline().strip(' \n\t\r').replace('D', 'E')

                self.intrinsic_matrix[:] = np.fromstring(line, count=6, sep=' ').reshape(2, 3)

                # parse the near distortion parameters
                line = sum_file.readline().strip(' \n\t\r').replace('D', 'E')

                self.near_dist_params[:] = np.fromstring(line, count=4, sep=' ')

                # parse the sigma_VSO values
                line = sum_file.readline().strip(' \n\t\r').replace('D', 'E')

                self.sig_pos_camera_to_target[:] = np.fromstring(line, count=3, sep=' ')

                # parse the sigma_PTG values
                line = sum_file.readline().strip(' \n\t\r').replace('D', 'E')

                self.sig_rot_target_fixed_to_camera[:] = np.fromstring(line, count=3, sep=' ')

                # read in the landmark and limb data
                if "landmarks" in sum_file.readline().upper().lower():

                    line = sum_file.readline().strip(' \n\t\r')

                    maplet_names = []
                    maplet_image_locs_list = []

                    while ('end' not in line.upper().lower()) and ('limb' not in line.upper().lower()):
                        split_line = line.split()

                        maplet_names.append(split_line[0])

                        maplet_image_locs_list.append([float(split_line[1]), float(split_line[2])])

                        line = sum_file.readline().strip(' \n\t\r')

                    if maplet_names:
                        self.landmarks = dict(zip(maplet_names, maplet_image_locs_list))

                    else:
                        self.landmarks = {}

                # read in any limb fits
                if 'limb' in line.upper().lower():

                    limb_names = []
                    limb_image_locs_list = []

                    line = sum_file.readline().strip(' \n\t\r')

                    while 'end' not in line.upper().lower():
                        split_line = line.split()

                        limb_names.append(split_line[0])

                        limb_image_locs_list.append([float(split_line[1]), float(split_line[2]),  # location
                                                     float(split_line[3])])  # sigma

                        line = sum_file.readline().strip(' \n\t\r')

                    if limb_names:
                        self.limb_fits = dict(zip(limb_names, limb_image_locs_list))

                    else:
                        self.limb_fits = {}

        except FileNotFoundError:
            raise FileNotFoundError('The file {} does not exist.'.format(file_name))

    def write(self, file_name: Optional[PATH] = None):
        """
        This function writes the data contained in the current instance of the class into the specified file

        If the ``file_name`` argument is not specified then it writes to the file stored in the
        :attr:`~.Summary.file_name` attribute.

        :param file_name: the full or relative path to the file to be written.  If left as ``None`` then writes to the
                          file stored in the :attr:`~.Summary.file_name` attribute
        """

        if file_name is None:
            if self.file_name is not None:
                file_name = self.file_name
            else:
                raise ValueError('file_name must be specified')

        vector_format = '{0:20.10E} {1:19.10E} {2:19.10E}   '

        name_line = '{0:s}\n'.format(self.image_name)

        date_line = self.observation_date.strftime(DATE_FMT)[:-3].upper() + '\n'

        image_line = '  {0:4d}  {1:4d}   {2:3d} {3:5d}'.format(
            self.num_cols, self.num_rows, self.min_illum, self.max_illum
        ) + ' ' * 39 + 'NPX, NLN, THRSH\n'

        camera_line = vector_format.format(self.focal_length,
                                           *self.princ_point).replace('E', 'D') + 'MMFL, CTR\n'

        position_line = vector_format.format(*self.position_camera_to_target).replace('E', 'D') + 'SCOBJ\n'

        rot_x_line = vector_format.format(*self.rotation_target_fixed_to_camera[0]).replace('E', 'D') + 'CX\n'
        rot_y_line = vector_format.format(*self.rotation_target_fixed_to_camera[1]).replace('E', 'D') + 'CY\n'
        rot_z_line = vector_format.format(*self.rotation_target_fixed_to_camera[2]).replace('E', 'D') + 'CZ\n'

        sun_direction_line = vector_format.format(*self.direction_target_to_sun).replace('E', 'D') + 'SZ\n'

        intrinsic_line = '{0:10.5f} {1:9.5f} {2:9.5f} {3:9.5f} {4:9.5f} {5:9.5f}   '.format(
            *self.intrinsic_matrix.flatten()
        ).replace('E', 'D') + 'K-MATRIX\n'

        distortion_line = '{0:15.5E} {1:14.5E} {2:14.5E} {3:14.5E}   DISTORTION\n'

        distortion_line = distortion_line.format(*self.near_dist_params).replace('E', 'D')

        sig_vso_line = vector_format.format(*self.sig_pos_camera_to_target).replace('E', 'D') + 'SIG_VSO\n'
        sig_ptg_line = vector_format.format(*self.sig_rot_target_fixed_to_camera).replace('E', 'D') + 'SIG_PTG\n'

        fit_formats = '{0:6s} {1:9.2f} {2:10.2f}\n'

        maplet_lines = []
        limb_lines = []

        for map_name, location in self.landmarks.items():
            maplet_lines.append(fit_formats.format(map_name, *location))

        for limb_name, location in self.limb_fits.items():
            limb_lines.append(fit_formats.format(limb_name, *location))

        # open/create the file for writing
        with open(file_name, 'w') as sum_file:

            sum_file.write(name_line)
            sum_file.write(date_line)
            sum_file.write(image_line)
            sum_file.write(camera_line)
            sum_file.write(position_line)
            sum_file.write(rot_x_line)
            sum_file.write(rot_y_line)
            sum_file.write(rot_z_line)
            sum_file.write(sun_direction_line)
            sum_file.write(intrinsic_line)
            sum_file.write(distortion_line)
            sum_file.write(sig_vso_line)
            sum_file.write(sig_ptg_line)
            sum_file.write('LANDMARKS\n')
            sum_file.writelines(maplet_lines)
            sum_file.write('LIMB FITS\n')
            sum_file.writelines(limb_lines)
            sum_file.write('END FILE')


class Image:
    """
    This class is used to read and write from SPC Image files (.DAT).

    The Image files contain the 2D image data for SPC processing.

    When creating an instance of this class you can enter the ``file_name`` argument and the data will automatically
    be read from that file.  Alternatively you can specify individual components of the object through key word
    arguments.  If you provide both key word arguments and a file name then the key word arguments you specified will
    overwrite anything read from the file.

    Because the image files are a minimal format, you must specify the number of columns, number of rows, and maximum
    illumination value in addition to the file name.  These can be determined by reading the corresponding summary file
    for the image (:class:`.Summary`).  As an alternative to the maximum illumination you can specify the integer size
    and endianess arguments if known.
    """

    def __init__(self, file_name: Optional[PATH] = None, n_cols: Optional[int] = None, n_rows: Optional[int] = None,
                 maximum_illumination: NONENUM = None,
                 integer_size: Optional[int] = None, endianess: Optional[str] = None,
                 data: NONEARRAY = None):
        """
        :param file_name: The path to a summary file to read data from
        :param n_cols: The number of columns in the image
        :param n_rows: The number of rows in the image
        :param maximum_illumination: The maximum illumination value that is possible in the image.  This is used
                                     to attempt to discern the endianess of the file, so it typically should be set to
                                     either 255 for 8 bit or 65535 for 16 bit.
        """

        self.file_name = file_name  # type: Optional[PATH]
        """
        The path to the image file that this object was populated from.

        This also will be the file that changes are written to if :meth:`.write` is called with no arguments.  

        Typically this is named according to IMAGEFILES/*IMAGENAME*.DAT where *IMAGENAME* is replaced with the SPC name
        of the image.  This can be set to None, a str, or a :class:`Path`.
        """

        self.n_cols = n_cols  # type: Optional[int]
        """
        The number of columns in the image. 
        
        This must be specified before the image file is read.  It can be determined by the :attr:`.Summary.n_cols` 
        attribute from the corresponding summary file.
        """

        self.n_rows = n_rows  # type: Optional[int]
        """
        The number of rows in the image. 

        This must be specified before the image file is read.  It can be determined by the :attr:`.Summary.n_rows` 
        attribute from the corresponding summary file.
        """

        self.maximum_illumination = maximum_illumination
        """
        The maximum possible illumination for an image.
        
        This is used to try to discern the endianess of the file if it is not known and the file has 16 bit integers, 
        but it is not foolproof.  If you know the endianess of the file you should specify that instead.
        """

        self.integer_size = integer_size  # type: Optional[int]
        """
        The number of bytes for each integer in the file.
        
        This is discerned from the total number of bytes in the file vs the total number of pixels computed from 
        :attr:`n_rows` and :attr:`n_cols`. A value of 1 indicates an 8 bit integer while a value of 2 indicates a 16 
        bit integer.  If you know this already then you can specify it to increase reading speed slightly.  
        
        This is required to be not ``None`` before a call to write is made.
        """

        self.endianess = endianess  # type: Optional[str]
        """
        The endianess of the data in the file if it is stored in 16 bit integers.  
        
        This may be able to be inferred if it is not known and the :attr:`maximum_illumination` value is specified; 
        however, this is not foolproof and may not work.  Therefore, if you know the endianess for 16 bit files you 
        should specify it.  
        
        This should be either '>' for big-endian or '<' for little-endian'
        
        This must be specified before a call to :meth:`write` if the integer size is 16 bit.  This doesn't matter if the
        integer size is 8 bit.
        """

        self.data = None  # type: NONEARRAY
        """
        The actual image data as a 2D numpy array
        """

        if file_name is not None:
            self.read()

        if data is not None:
            self.data = data

    def read(self, file_name: Optional[PATH] = None):
        """
        This method reads data from a SPC Image file (normally ``IMAGEFILES/*IMAGENAME*.DAT``) and populates the
        attributes of this class with that data.

        If ``file_name`` is not specified the :attr:`file_name` attribute of the class is used instead

        :param file_name: The file to load the sum data from
        """

        if file_name is None:
            file_name = self.file_name

        if (self.n_rows is None) or (self.n_cols is None):
            raise ValueError('The number of rows and columns in the image must be specified before calling read')

        with open(file_name, 'r') as ifile:

            if self.integer_size is None:
                # get the number of bytes by going to the end of the file, getting the position,
                # and then rewinding to the beginning of the file
                ifile.seek(0, 2)

                num_bytes = ifile.tell()

                ifile.seek(0, 0)

                # determine whether this is 1 or 2 byte integers
                self.integer_size = 2 if (num_bytes / self.n_rows == 2 * self.n_cols) else 1

            # if the data is 2byte
            if self.integer_size == 2:

                if self.endianess is None:

                    # read in the data as big endian
                    illums_big_e = np.fromfile(ifile, dtype=">u2", count=-1).reshape(self.n_rows, self.n_cols)

                    # reset back to the beginning of the file
                    ifile.seek(0, 0)

                    # read in the data as little endian
                    illums_little_e = np.fromfile(ifile, dtype="<u2", count=-1).reshape(self.n_rows, self.n_cols)

                    if np.max(illums_big_e) <= self.maximum_illumination <= np.max(illums_little_e):
                        # if the data is big endian

                        self.data = illums_big_e.astype(np.uint16)

                        self.endianess = '>'

                    elif np.max(illums_little_e) <= self.maximum_illumination <= np.max(illums_big_e):
                        # if the data is little endian

                        self.data = illums_little_e.astype(np.uint16)

                        self.endianess = '<'

                    else:
                        # otherwise something went wrong
                        raise IOError("It appears like the file has been corrupted (Can't determine endianness")

                else:
                    self.data = np.fromfile(ifile, dtype=self.endianess + "u2", count=-1).reshape(self.n_rows,
                                                                                                  self.n_cols)

            else:

                self.data = np.fromfile(ifile, dtype="u1", count=-1).reshape(self.n_rows, self.n_cols)

    def write(self, file_name=None):
        """
        This function writes the file contained in the Image object to a dat file

        If the ``file_name`` argument is not specified then it writes to the file stored in the
        :attr:`~.Image.file_name` attribute.

        :param file_name: the full or relative path to the file to be written.  If left as ``None`` then writes to the
                          file stored in the :attr:`~.Nominal.file_name` attribute
        """

        if file_name is None:
            if self.file_name is not None:
                file_name = self.file_name
            else:
                raise ValueError('file_name must be specified')

        if self.integer_size is None:
            raise ValueError("Integer size must be specified to write to a Image file.")

        with open(file_name, 'wb') as ofile:

            if self.integer_size == 1:

                ofile.write(self.data.astype('u1').tostring())

            else:

                if self.endianess is None:
                    raise ValueError('Endianess must be specified for a 2 byte integer')

                ofile.write(self.data.astype(self.endianess + 'u2').tostring())


class Nominal:
    """
    This class is used to read/write from the SPC Nominal (.NOM) files.

    The nominal files represent meta data about an Image as original loaded with updated values being stored in the
    Summary (.SUM) files.  The image data itself is stored in a corresponding .DAT file which can be read/written with
    the :class:`Image` class.  When creating an instance of this class, you can enter the ``file_name`` argument, in
    which case the specified NOM file will be opened and read into the members of this class (if the file exists), or
    you can specify the individual components of the file as keyword arguments.  If you specify both an existing
    NOM file and keyword arguments for the individual components of the NOM file, the keyword arguments will override
    whatever is read from the file.

    The documentation for each instance attribute of this class specifies the corresponding location in the NOM file
    for clarification and reference.
    """

    def __init__(self, file_name: Optional[PATH] = None, image_name: Optional[str] = None,
                 position_camera_to_target: NONEARRAY = None, velocity: NONEARRAY = None,
                 rotation_target_fixed_to_camera: NONEARRAY = None,
                 sig_pos_camera_to_target: NONEARRAY = None,
                 sig_rot_target_fixed_to_camera: NONEARRAY = None,
                 frame: Optional[str] = None, ending: Optional[str] = None):
        """
        :param file_name: The path to a summary file to read data from
        :param image_name: The name of the image this object corresponds to
        :param position_camera_to_target: The vector from the spacecraft to the target in the target body fixed frame as
                                          a length 3 numpy array as originally specified
        :param velocity: The velocity of the camera
        :param rotation_target_fixed_to_camera: The 3x3 rotation matrix from the target body fixed frame to the camera
                                                frame
        :param sig_pos_camera_to_target: This is the formal uncertainty on the ``position_camera_to_target`` vector
                                         (1 sigma) as originally specified
        :param sig_rot_target_fixed_to_camera: This is the formal uncertainty on the spacecraft pointing
                                               (roll, pitch, yaw?) (1 sigma) as originally specified
        :param frame: This specifies the frame the position uncertainty is experessed in
        :param ending: This specifies stuff that goes at the end of the nominal file.
        """

        # predefine the attributes of the class for readability
        self.file_name = None  # type: Optional[PATH]
        """
        The path to the nominal file that this object was populated from.

        This also will be the file that changes are written to if :meth:`.write` is called with no arguments.  

        Typically this is named according to NOMINALS/*IMAGENAME*.NOM where *IMAGENAME* is replaced with the SPC name
        of the image.  This can be set to None, a str, or a :class:`Path`.
        """

        self.image_name = None  # type: Optional[str]
        """
        The name of the image this object corresponds to.

        This corresponds to the first line of the NOM file.
        """

        self.position_camera_to_target = np.zeros(3)  # type: np.ndarray
        """
        The position vector from the spacecraft to the target in target body fixed coordinates at the time of the image
        with units of kilometers as a length 3 numpy array of floats.  This represents the vector as originally loaded 
        into SPC (usually from SPICE).  That means this typically is not updated.

        This corresponds to all the values of the line containing ``SCOBJ`` in the NOM file, usually the 3rd line.
        """

        self.velocity = np.zeros(3)  # type: np.ndarray
        """
        The velocity of the camera with units of kilometers per second as a length 3 numpy array of floats.
        
        This corresponds to the second line in the NOM file.
        """

        self.rotation_target_fixed_to_camera = np.zeros((3, 3))  # type: np.ndarray
        """
        The rotation matrix from the target body fixed frame to the camera frame at the time of the image as a
        shape 3x3 numpy array of floats. This represents the matrix as originally read into SPC (usually from spice)

        This corresponds to all the values of the lines containing ``CX, CY, CZ`` in the NOM file.  The matrix is formed 
        according to ``np.vstack([CX, CY, CZ])`` 
        """

        self.sig_pos_camera_to_target = np.zeros(3)  # type: np.ndarray
        """
        The formal uncertainty on the position_camera_to_target vector as a length 3 numpy array of floats.

        The formal uncertainty are the 1 sigma values for each component of the position vector.

        This corresponds to all of the values of the line containing ``SIGMA_VSO`` in the NOM file.
        """

        self.sig_rot_target_fixed_to_camera = np.zeros(3)  # type: np.ndarray
        """
        The formal uncertainty on the rotation from the target body frame to the spacecraft frame as a length 3 numpy 
        array of floats.

        The formal uncertainty are the 1 sigma values.

        This corresponds to all of the values of the line containing ``SIGMA_PTG`` in the NOM file.
        """

        self.frame = ''  # type: str
        """
        This specifies the frame the sig_pos_camera_to_target is specified in.
        
        This corresponds to the label on the velocity line in the NOM file.
        """

        self.ending = ''  # type: str
        """
        This corresponds to the end of the Nominal file (essentially anything that wasn't captured) and is a fall back 
        for future additions.
        """

        if file_name is not None:
            self.read(file_name)
            self.file_name = file_name

        if image_name is not None:
            self.image_name = image_name
        if position_camera_to_target is not None:
            self.position_camera_to_target = position_camera_to_target
        if velocity is not None:
            self.velocity = velocity
        if rotation_target_fixed_to_camera is not None:
            self.rotation_target_fixed_to_camera = rotation_target_fixed_to_camera
        if sig_pos_camera_to_target is not None:
            self.sig_pos_camera_to_target = sig_pos_camera_to_target
        if sig_rot_target_fixed_to_camera is not None:
            self.sig_rot_target_fixed_to_camera = sig_rot_target_fixed_to_camera
        if frame is not None:
            self.frame = frame
        if ending is not None:
            self.ending = ending

    def __eq__(self, other):

        if not isinstance(other, self.__class__):
            return False

        comp_dict = {'image_name': self.image_name == other.image_name,
                     'position_camera_to_target': (self.position_camera_to_target ==
                                                   other.position_camera_to_target).all(),
                     'velocity': (self.velocity == other.velocity).all(),
                     'rotation_target_fixed_to_camera': (self.rotation_target_fixed_to_camera ==
                                                         other.rotation_target_fixed_to_camera).all(),
                     'sig_pos_camera_to_target': (self.sig_pos_camera_to_target ==
                                                  other.sig_pos_camera_to_target).all(),
                     'sig_rot_target_fixed_to_camera': (self.sig_rot_target_fixed_to_camera ==
                                                        other.sig_rot_target_fixed_to_camera).all()}

        if np.all(comp_dict.values()):
            return True

        else:
            return comp_dict

    def read(self, file_name: Optional[PATH] = None):
        """
        This method reads data from a SPC NOM file (normally ``NOMINALS/*IMAGENAME*.NOM``) and populates the
        attributes of this class with that data.

        If ``file_name`` is not specified the :attr:`file_name` attribute of the class is used instead

        :param file_name: The file to load the sum data from
        """

        if file_name is None:
            file_name = self.file_name

        with open(file_name, 'r') as nominal:
            # parse the name line
            self.image_name = nominal.readline().strip(' \n\t\r')

            # read in the spacecraft velocity vector and frame specifier for the uncertainty
            line = nominal.readline().strip(' \n\t\r')
            self.frame = line.split()[-1]
            self.velocity[:] = np.fromstring(line.replace('D', 'E'), sep=' ', count=3)

            # read in the spacecraft position vector
            line = nominal.readline().strip(' \n\t\r').replace('D', 'E')

            self.position_camera_to_target[:] = np.fromstring(line, sep=' ', count=3)

            # read in the sigma_VSO values (whatever that is)
            line = nominal.readline().strip(' \n\t\r').replace('D', 'E')

            self.sig_pos_camera_to_target[:] = np.fromstring(line, sep=' ', count=3)

            # read in the rotation matrix from the body frame to the camera frame
            line = nominal.readline().strip(' \n\t\r').replace('D', 'E')

            self.rotation_target_fixed_to_camera[0] = np.fromstring(line, sep=' ', count=3)

            line = nominal.readline().strip(' \n\t\r').replace('D', 'E')

            self.rotation_target_fixed_to_camera[1] = np.fromstring(line, sep=' ', count=3)

            line = nominal.readline().strip(' \n\t\r').replace('D', 'E')

            self.rotation_target_fixed_to_camera[2] = np.fromstring(line, sep=' ', count=3)

            # read in the sigma_PTG values (whatever that is)
            line = nominal.readline().strip(' \n\t\r').replace('D', 'E')

            self.sig_rot_target_fixed_to_camera[:] = np.fromstring(line, sep=' ', count=3)

            self.ending = nominal.read()

    def write(self, file_name: Optional[PATH] = None):
        """
        This function writes the data contained in the current instance of the class into the specified file

        If the ``file_name`` argument is not specified then it writes to the file stored in the
        :attr:`~.Nominal.file_name` attribute.

        :param file_name: the full or relative path to the file to be written.  If left as ``None`` then writes to the
                          file stored in the :attr:`~.Nominal.file_name` attribute
        """

        if file_name is None:
            if self.file_name is not None:
                file_name = self.file_name
            else:
                raise ValueError('file_name must be specified')

        vector_format = '{0:20.10E} {1:19.10E} {2:19.10E}   '

        name_line = '{0:s}\n'.format(self.image_name)

        velocity_line = vector_format.format(*self.velocity).replace('E', 'D') + self.frame + '\n'

        position_line = vector_format.format(*self.position_camera_to_target).replace('E', 'D') + 'SCOBJ\n'

        sig_vso_line = vector_format.format(*self.sig_pos_camera_to_target).replace('E', 'D') + "SIGMA_VSO\n"

        rot_x_line = vector_format.format(*self.rotation_target_fixed_to_camera[0]).replace('E', 'D') + 'CX\n'
        rot_y_line = vector_format.format(*self.rotation_target_fixed_to_camera[1]).replace('E', 'D') + 'CY\n'
        rot_z_line = vector_format.format(*self.rotation_target_fixed_to_camera[2]).replace('E', 'D') + 'CZ\n'

        sig_ptg_line = vector_format.format(*self.sig_rot_target_fixed_to_camera).replace('E', 'D') + "SIGMA_PTG\n"

        # open/create the file for writing
        with open(file_name, 'w') as nominal:
            nominal.write(name_line)
            nominal.write(velocity_line)
            nominal.write(position_line)
            nominal.write(sig_vso_line)
            nominal.write(rot_x_line)
            nominal.write(rot_y_line)
            nominal.write(rot_z_line)
            nominal.write(sig_ptg_line)
            nominal.write(self.ending)


class Regress:
    """
    This class is used to read from SPC regress files.

    The regress files summarise the results of performing AUTOREGISTER on a batch of images in a single text file.
    Typically this is used to ingest SPC landmark observables into navigation software.

    When creating an instance of this class you can enter the ``file_name`` argument and the data will automatically
    be read from that file.  Alternatively you can specify individual components of the object through key word
    arguments.  If you provide both key word arguments and a file name then the key word arguments you specified will
    overwrite anything read from the file.

    Currently this class does not support writing regress files.
    """

    def __init__(self, file_name: Optional[PATH] = None, spc_image_id: Optional[str] = None,
                 orig_image_name: Optional[str] = None, camera_name: Optional[str] = None,
                 spacecraft: Optional[str] = None, target_name: Optional[str] = None,
                 observation_date: Optional[datetime] = None, ephemeris_time: NONENUM = None,
                 n_rows: Optional[int] = None, n_cols: Optional[int] = None, focal_length: NONENUM = None,
                 px: NONENUM = None, py: NONENUM = None, kx: NONENUM = None, ky: NONENUM = None, kxy: NONENUM = None,
                 kyx: NONENUM = None, titv_mat: Optional[list] = None, tpmi_mat: Optional[list] = None,
                 sf_pos_sc2obj: NONEARRAY = None, sf_camera_ra: NONENUM = None, sf_camera_dec: NONENUM = None,
                 sf_camera_twist: NONENUM = None, pos_sigma: NONENUM = None, pointing_sigma: NONENUM = None,
                 twist_sigma: NONENUM = None, nom_pos_sc2obj: NONENUM = None, nom_camera_ra: NONENUM = None,
                 nom_camera_dec: NONENUM = None, nom_camera_twist: NONENUM = None, sun_dir: NONEARRAY = None,
                 landmarks: Optional[list] = None, lmk_body_fixed_vecs: Optional[list] = None,
                 lmk_body_fixed_vec_sigmas: Optional[list] = None, lmk_observed_pixlines: Optional[list] = None,
                 lmk_pred_pixlines: Optional[list] = None, lmk_jacobians: Optional[list] = None,
                 kernel_list: Optional[list] = None):
        """
        :param file_name: The name of the regress file to read from
        :param spc_image_id: The name of the image file according to SPC
        :param orig_image_name: The name of original image file
        :param camera_name: The name of the camera
        :param spacecraft: The name of the spacecraft that hosts the camera
        :param target_name: The name of the observed target
        :param observation_date: The observation date in UTC
        :param ephemeris_time: The observation in ephemeris seconds since J2000
        :param n_rows: the number of rows in the image
        :param n_cols: the number of columns in the image
        :param focal_length: the focal length of the camera
        :param px: The x component of the principal point of the camera
        :param py: The y component of the principal point of the camera
        :param kx: 1/pixel pitch in the x direction
        :param ky: 1/pixel pitch in the y direction
        :param kxy: skewness term xy
        :param kyx: skewness term yx
        :param titv_mat: I don't know
        :param tpmi_mat: Still don't know
        :param sf_pos_sc2obj: the solved for position from the spacecraft (camera) to the target in km
        :param sf_camera_ra: The solved for right ascension of the camera boresight in inertial?
        :param sf_camera_dec: The solved for declination of the camera boresight in inertial?
        :param sf_camera_twist: The solved for twist about the camera boresight in inertial?
        :param pos_sigma: The 1 sigma uncertainty on the position (not sure if pre or post fit)
        :param pointing_sigma: The 1 sigma uncertainty on the ra/dec pointing (not sure if pre or post fit)
        :param twist_sigma: The 1 sigma uncertainty on the twist about the boresight (not sure if pre or post fit)
        :param nom_pos_sc2obj: The nominal (original) position vector from the spacecraft to the target in km
        :param nom_camera_ra: The nominal (original) right ascension of the camera boresight in intertial?
        :param nom_camera_dec: The nominal (original) declination of the camera boresight in intertial?
        :param nom_camera_twist: The nominal (original) twist about the camera boresight in intertial?
        :param sun_dir: The direction from the target to the sun in the camera frame
        :param landmarks: A list of landmarks identified in the image
        :param lmk_body_fixed_vecs:  A list of landmark body fixed locations corresponding to the landmarks list in km
        :param lmk_body_fixed_vec_sigmas: A list of landmark body fixed location sigmas corresponding to the landmarks
                                          list
        :param lmk_observed_pixlines: A list of observed pixel and line (column, row) values for each landmark in the
                                      landmarks list
        :param lmk_pred_pixlines: A list of predicted pixel and line (column, row) values for each landmark in the
                                  landmarks list
        :param lmk_jacobians: Jacobian matrices of the landmark observables. (with respect to a change in the
                              spacecraft state?)
        :param kernel_list: A list of spice kernels that were loaded by spc
        """

        self.file_name = file_name
        """
        The name of the Regress file this data was read from
        """

        self.spc_image_id = ''
        """
        The name of the image file according to SPC
        
        Corresponds to IMAGE ID in the REGRESS file
        """

        self.orig_image_name = ''
        """
        The name of original image file
        
        Corresponds to ORIGINAL NAME in the REGRESS file
        """

        self.camera_name = ''
        """
        The name of the camera
        
        Corresponds to CAMERA ID in the REGRESS file
        """

        self.spacecraft = ''
        """
        The name of the spacecraft that hosts the camera
        
        Corresponds to SPACECRAFT ID in the REGRESS file
        """

        self.target_name = ''
        """
        The name of the observed target
        
        Corresponds to TARGET ID in the REGRESS file
        """

        self.observation_date = None
        """
        The observation date in UTC
        
        Corresponds to UTC in the REGRESS file
        """

        self.ephemeris_time = None
        """
        The observation in ephemeris seconds since J2000
        
        Corresponds to ET SEC PAST J2000 in the REGRESS file
        """

        self.n_rows = None
        """
        the number of rows in the image
        
        Corresponds to NLN in the REGRESS file
        """

        self.n_cols = None
        """
        the number of columns in the image
        
        Corresponds to NPX in the REGRESS file
        """

        self.focal_length = None
        """
        the focal length of the camera
        
        Corresponds to MMFL in the REGRESS file
        """

        self.px = None
        """
        The x component of the principal point of the camera
        
        Corresponds to CTR[0] in the REGRESS file
        """

        self.py = None
        """
        The y component of the principal point of the camera
        
        Corresponds to CTR[1] in the REGRESS file
        """

        self.kx = None
        """
        1/pixel pitch in the x direction
        
        Corresponds to K-MATRIX[0][0] in the REGRESS file
        """

        self.ky = None
        """
        1/pixel pitch in the y direction
        
        Corresponds to K-MATRIX[1][1] in the REGRESS file
        """

        self.kxy = None
        """
        skewness term xy
        
        Corresponds to K-MATRIX[0][1] in the REGRESS file
        """

        self.kyx = None
        """
        skewness term yx
        
        Corresponds to K-MATRIX[1][0] in the REGRESS file
        """

        self.titv_mat = []
        """
        I don't know
        
        Corresponds to TITV MATRIX in the REGRESS file
        """

        self.tpmi_mat = []
        """
        Still don't know
        
        Corresponds to TPMI MATRIX in the REGRESS file
        """

        self.sf_pos_sc2obj = None
        """
        the solved for position from the spacecraft (camera) to the target in km
        
        Corresponds to SC - OBJ VECTOR in the LITHOS section of the REGRESS file
        """

        self.sf_camera_ra = None
        """
        The solved for right ascension of the camera boresight in inertial?
        
        Corresponds to CAMERA RA in the LITHOS section of the REGRESS file
        """

        self.sf_camera_dec = None
        """
        The solved for declination of the camera boresight in inertial?
        
        Corresponds to CAMERA DC in the LITHOS section of the REGRESS file
        """

        self.sf_camera_twist = None
        """
        The solved for twist about the camera boresight in inertial?
        
        Corresponds to CAMERA TW in the LITHOS section of the REGRESS file
        """

        self.pos_sigma = None
        """
        The 1 sigma uncertainty on the position (not sure if pre or post fit)
        
        Corresponds to S/C POSITION SIG in the LITHOS section of the REGRESS file
        """

        self.pointing_sigma = None
        """
        The 1 sigma uncertainty on the ra/dec pointing (not sure if pre or post fit)
        
        Corresponds to PNT SIGMA (DEG) in the LITHOS section of the REGRESS file
        """

        self.twist_sigma = None
        """
        The 1 sigma uncertainty on the twist about the boresight (not sure if pre or post fit)
        
        Corresponds to TWIST SIGMA (DEG) in the LITHOS section of the REGRESS file
        """

        self.nom_pos_sc2obj = None
        """
        The nominal (original) position vector from the spacecraft to the target in km
        
        Corresponds to SC - OBJ VECTOR in the NOMINAL STATE section of the REGRESS file
        """

        self.nom_camera_ra = None
        """
        The nominal (original) right ascension of the camera boresight in intertial?
        
        Corresponds to CAMERA RA in the NOMINAL STATE section of the REGRESS file
        """

        self.nom_camera_dec = None
        """
        The nominal (original) declination of the camera boresight in intertial?
        
        Corresponds to CAMERA DC in the NOMINAL STATE section of the REGRESS file
        """

        self.nom_camera_twist = None
        """
        The nominal (original) twist about the camera boresight in intertial?
        
        Corresponds to CAMERA TW in the NOMINAL STATE section of the REGRESS file
        """

        self.sun_dir = None
        """
        The direction from the target to the sun in the camera frame
        
        Corresponds to SOLAR UNIT VECTOR in the NOMINAL STATE section of the REGRESS file
        """

        self.landmarks = []
        """
        A list of landmarks identified in the image 
        
        Corresponds to the lines with only 1 entry in the LANDMARKS: section of the REGRESS file
        """

        self.lmk_body_fixed_vecs = []
        """
        A list of landmark body fixed locations corresponding to the landmarks list in km
        
        Corresponds to BODY-FIXED VLM in the LANDMARKS: section of the REGRESS file
        """

        self.lmk_body_fixed_vec_sigmas = []
        """
        A list of landmark body fixed location sigmas corresponding to the landmarks list
        
        Corresponds to B-F VLM SIGMA in the LANDMARKS: section of the REGRESS file
        """

        self.lmk_observed_pixlines = []
        """
        A list of observed pixel and line (column, row) values for each landmark in the landmarks list
        
        Corresponds to PX, LN OBSERVED in the LANDMARKS: section of the REGRESS file
        """

        self.lmk_pred_pixlines = []
        """
        A list of predicted pixel and line (column, row) values for each landmark in the landmarks list
        
        Corresponds to PX, LN PREDICTED in the LANDMARKS: section of the REGRESS file
        """

        self.lmk_jacobians = []
        """
        Jacobian matrices of the landmark observables. (with respect to a change in the spacecraft state?)
        
        Corresponds to DPX/ in the LANDMARKS: section of the REGRESS file
        """

        self.kernel_list = []
        """
        A list of spice kernels that were loaded by spc
        
        Corresponds to all entries following KERNEL LIST in the REGRESS file
        """

        if file_name is not None:
            self.read(file_name)

        if spc_image_id is not None:
            self.spc_image_id = spc_image_id
        if orig_image_name is not None:
            self.orig_image_name = orig_image_name
        if camera_name is not None:
            self.camera_name = camera_name
        if spacecraft is not None:
            self.spacecraft = spacecraft
        if target_name is not None:
            self.target_name = target_name
        if observation_date is not None:
            self.observation_date = observation_date
        if ephemeris_time is not None:
            self.ephemeris_time = ephemeris_time
        if n_rows is not None:
            self.n_rows = n_rows
        if n_cols is not None:
            self.n_cols = n_cols
        if focal_length is not None:
            self.focal = focal_length
        if px is not None:
            self.px = px
        if py is not None:
            self.py = py
        if kx is not None:
            self.kx = kx
        if ky is not None:
            self.ky = ky
        if kxy is not None:
            self.kxy = kxy
        if kyx is not None:
            self.kyx = kyx
        if titv_mat is not None:
            self.titv_mat = titv_mat
        if tpmi_mat is not None:
            self.tpmi_mat = tpmi_mat
        if sf_pos_sc2obj is not None:
            self.sf_pos_sc2obj = sf_pos_sc2obj
        if sf_camera_ra is not None:
            self.sf_camera_ra = sf_camera_ra
        if sf_camera_dec is not None:
            self.sf_camera_dec = sf_camera_dec
        if sf_camera_twist is not None:
            self.sf_camera_twist = sf_camera_twist
        if pos_sigma is not None:
            self.pos_sigma = pos_sigma
        if pointing_sigma is not None:
            self.pointing_sigma = pointing_sigma
        if twist_sigma is not None:
            self.twist_sigma = twist_sigma
        if nom_pos_sc2obj is not None:
            self.nom_pos_sc2obj = nom_pos_sc2obj
        if nom_camera_ra is not None:
            self.nom_camera_ra = nom_camera_ra
        if nom_camera_dec is not None:
            self.nom_camera_dec = nom_camera_dec
        if nom_camera_twist is not None:
            self.nom_camera_twist = nom_camera_twist
        if sun_dir is not None:
            self.sun_dir = sun_dir
        if landmarks is not None:
            self.landmarks = landmarks
        if lmk_body_fixed_vecs is not None:
            self.lmk_body_fixed_vecs = lmk_body_fixed_vecs
        if lmk_body_fixed_vec_sigmas is not None:
            self.lmk_body_fixed_vec_sigmas = lmk_body_fixed_vec_sigmas
        if lmk_observed_pixlines is not None:
            self.lmk_observed_pixlines = lmk_observed_pixlines
        if lmk_pred_pixlines is not None:
            self.lmk_pred_pixlines = lmk_pred_pixlines
        if lmk_jacobians is not None:
            self.lmk_jacobians = lmk_jacobians
        if kernel_list is not None:
            self.kernel_list = kernel_list

    def read(self, file_name: Optional[PATH] = None):
        """
        This method reads data from a SPC REGRESS file and populates the attributes of
        this class with that data.

        If ``file_name`` is not specified the :attr:`file_name` attribute of the class is used instead

        :param file_name: The file to load the sum data from
        """

        # initialize all the lists
        self.titv_mat = []
        self.tpmi_mat = []
        self.landmarks = []
        self.lmk_jacobians = []
        self.lmk_pred_pixlines = []
        self.lmk_observed_pixlines = []
        self.lmk_body_fixed_vec_sigmas = []
        self.lmk_body_fixed_vecs = []
        self.kernel_list = []

        if file_name is None:
            file_name = self.file_name

        with open(file_name, 'r') as rfile:

            for line in rfile:
                if line:

                    upperline = line.upper()

                    if 'IMAGE ID' in upperline:
                        self.spc_image_id = line.split()[0]
                    if 'ORIGINAL NAME' in upperline:
                        self.orig_image_name = line.split()[0]
                    if 'CAMERA ID' in upperline:
                        self.camera_name = line.split()[0]
                    if 'SPACECRAFT ID' in upperline:
                        self.spacecraft = line.split()[0]
                    if 'TARGET ID' in upperline:
                        self.target_name = line.split()[0]
                    if 'UTC' in upperline:
                        self.observation_date = parser.parse(line).replace(tzinfo=None)
                    if 'ET SEC PAST J2000' in upperline:
                        self.ephemeris_time = float(line.split()[0].replace('D', 'e'))
                    if 'NPX, NLN' in upperline:
                        self.n_rows = int(line.split()[1])
                        self.n_cols = int(line.split()[0])
                    if 'MMFL, CTR' in upperline:
                        self.focal = float(line.split()[0].replace('D', 'e'))
                        self.px = float(line.split()[1].replace('D', 'e'))
                        self.py = float(line.split()[2].replace('D', 'e'))
                    if 'K-MATRIX' in upperline:
                        kmat = [float(p) for p in line.replace('D', 'e').split()[:-1]]

                        self.kx = kmat[0]
                        self.kxy = kmat[1]
                        self.kyx = kmat[3]
                        self.ky = kmat[4]
                    if 'TITV MATRIX' in upperline:
                        row = [float(p) for p in line.replace('D', 'e').split()[:3]]
                        self.titv_mat.append(row)
                    if 'TPMI MATRIX' in upperline:
                        row = [float(p) for p in line.replace('D', 'e').split()[:3]]
                        self.tpmi_mat.append(row)
                    if 'LITHOS' in upperline:
                        while 'NOMINAL STATE' not in upperline:
                            if 'SC - OBJ VECTOR' in upperline:
                                self.sf_pos_sc2obj = np.array([float(p) for p in line.replace('D', 'e').split()[:3]])
                            if 'CAMERA RA, DC, TW' in upperline:
                                self.sf_camera_ra, self.sf_camera_dec, self.sf_camera_twist = \
                                    [float(p) for p in line.replace('D', 'e').split()[:3]]
                            if 'S/C POSITION SIG' in upperline:
                                self.pos_sigma = float(line.split()[0].replace('D', 'e'))
                            if 'PNT SIGMA (DEG)' in upperline:
                                self.pointing_sigma = float(line.split()[0].replace('D', 'e'))
                            if 'TWIST SIGMA (DEG)' in upperline:
                                self.twist_sigma = float(line.split()[0].replace('D', 'e'))
                            line = next(rfile)
                            upperline = line.upper()
                    if 'NOMINAL STATE' in upperline:
                        while 'NOMINAL STATE' not in upperline:
                            if 'SC - OBJ VECTOR' in upperline:
                                self.nom_pos_sc2obj = np.array([float(p) for p in line.replace('D', 'e').split()[:3]])
                            if 'CAMERA RA, DC, TW' in upperline:
                                self.nom_camera_ra, self.nom_camera_dec, self.nom_camera_twist = \
                                    [float(p) for p in line.replace('D', 'e').split()[:3]]
                            if 'SOLAR UNIT VECTOR' in upperline:
                                self.sun_dir = np.array([float(p) for p in line.replace('D', 'e').split()[:3]])
                            line = next(rfile)
                            upperline = line.upper()

                    if 'LANDMARKS:' in upperline:
                        line = next(rfile)
                        upperline = line.upper()

                        while 'END LANDMARKS' not in upperline:

                            splitline = line.split()

                            if len(splitline) == 1:
                                self.landmarks.append(splitline[0])
                                self.lmk_jacobians.append([])
                            if 'BODY-FIXED VLM' in upperline:
                                self.lmk_body_fixed_vecs.append(np.array([float(p.replace('D', 'e'))
                                                                          for p in splitline[:3]]))
                            if 'B-F VLM SIGMA' in upperline:
                                self.lmk_body_fixed_vec_sigmas.append(np.array([float(p.replace('D', 'e'))
                                                                                for p in splitline[:3]]))
                            if 'PX, LN OBSERVED' in upperline:
                                self.lmk_observed_pixlines.append(np.array([float(p.replace('D', 'e'))
                                                                            for p in splitline[:2]]))
                            if 'PX, LN PREDICTED' in upperline:
                                self.lmk_pred_pixlines.append(np.array([float(p.replace('D', 'e'))
                                                                        for p in splitline[:2]]))

                            if 'DPX/' in upperline:
                                self.lmk_jacobians[-1].append(np.array([float(p.replace('D', 'e'))
                                                                        for p in splitline[:2]]))

                            line = next(rfile)
                            upperline = line.upper()
                    if 'KERNEL LIST' in upperline:
                        while 'END FILE' not in upperline:

                            if line:
                                self.kernel_list.append(line.strip())

                            line = next(rfile)
                            upperline = line.upper()


class Landmark:
    """
    This class is used to read and write from SPC Landmark files.

    The Landmark files specify information about a landmark (or center of a Maplet).

    When creating an instance of this class you can enter the ``file_name`` argument and the data will automatically
    be read from that file.  Alternatively you can specify individual components of the object through key word
    arguments.  If you provide both key word arguments and a file name then the key word arguments you specified will
    overwrite anything read from the file.
    """

    # TODO: ability to read/write PICTURES, MAP OVERLAYS, and LIMB FITS sections
    # TODO: update the attribute names to be more descriptive

    def __init__(self, file_name: Optional[PATH] = None, name: Optional[str] = None, size: Optional[int] = None,
                 scale: NONENUM = None, sigkm: NONENUM = None, rmslmk: NONENUM = None,
                 vlm: NONEARRAY = None, rot_map2body: NONEARRAY = None, sigma_lmk: NONEARRAY = None):
        """
        :param file_name: The name of the landmark file
        :param name: the name of the landmark
        :param size: half the size of the corresponding maplet
        :param scale: The ground sample distance of the corresponding maplet in units of km
        :param sigkm: The uncertainty of something
        :param rmslmk: The rms of something
        :param vlm: the body-fixed landmark vectore
        :param rot_map2body: the rotation matrix from the corresponding maplet frame to the body fixed frame
        :param sigma_lmk: the uncertainty on the body fixed landmark vector
        """

        self.file_name = file_name  # type: Optional[PATH]
        """
        The name of the landmark file
        """

        self.name = ''  # type: str
        """
        The name of the landmark.  
        
        This should be a 6 character string, typically alpha-numeric.
        
        This corresponds to the first line of the LMK file.
        """

        self.size = 49  # type: int
        """
        Half the number of grid cells on a side for the corresponding Maplet minus 1.
        
        This is computed as to size=(n_rows - 1)/2 where n_rows is the number of rows in the corresponding 
        Maplet (or number of columns since Maplets are always square).
        
        This corresponds to the first component of the SIZE, SCALE line in the LMK file, usually the second line.
        """

        self.scale = None  # type: Optional[float]
        """
        The ground sample distance of each grid cell for the corresponding Maplet in units of KM.

        This corresponds to the second component of the SIZE, SCALE line in the LMK file, usually the second line.
        """

        self.sigkm = 0.15e-3  # type: float
        """
        The 1 sigma uncertainty on the landmark body fixed vector in units of km?
        
        this corresponds to the first component of the SIGKM, RMSLMK line in the LMK file, usually the fourth line.
        """

        self.rmslmk = 0.15e-3  # type: float
        """
        The residual RMS of the landmark body fixed vector in units of km?

        This corresponds to the second component of the SIGKM, RMSLMK line in the LMK file, usually the fourth line.
        """

        self.vlm = np.zeros(3)  # type: np.ndarray
        """
        The body-fixed landmark vector in units of km as a length 3 numpy array of floats
        
        This corresponds to all of the components of the VLM line in the LMK file, usually the fifth line.
        """

        self.rot_map2bod = np.eye(3)  # type: np.ndarray
        """
        The rotation matrix from the corresponding Maplet local frame to the body-fixed frame as a 3x3 numpy array of 
        floats.
        
        This corresponds to all of the components of the UX, UY, and UZ lines in the LMK file, usually lines 6-9.  This 
        matrix is formed according to np.vstack([UX, UY, UZ]).
        """

        self.sigma_lmk = 0.15e-3 * np.ones(3)  # type: np.ndarray
        """
        The 1 sigma uncertainty on all of the components of the landmark body fixed vector in units of km.

        This corresponds all of the second component of the SIMGA_LMK line in the LMK file, usually the tenth line.
        """

        if file_name is not None:
            self.read()

        if name is not None:
            self.name = name
        if size is not None:
            self.size = size
        if scale is not None:
            self.scale = scale
        if sigkm is not None:
            self.sigkm = sigkm
        if rmslmk is not None:
            self.rmlmk = rmslmk
        if vlm is not None:
            self.vlm = vlm
        if rot_map2body is not None:
            self.rot_map2bod = rot_map2body
        if sigma_lmk is not None:
            self.sigma_lmk = None

    def write(self, file_name: Optional[PATH] = None):
        """
        This function writes the data contained in the current instance of the class into the specified file

        If the ``file_name`` argument is not specified then it writes to the file stored in the
        :attr:`~.Landmark.file_name` attribute.

        :param file_name: the full or relative path to the file to be written.  If left as ``None`` then writes to the
                          file stored in the :attr:`~.Landmark.file_name` attribute
        """

        if file_name is None:
            if self.file_name is not None:
                file_name = self.file_name
            else:
                raise ValueError('file_name must be specified')

        size_scale_format = '     {:>3d}   {:.7f}                                           SIZE, SCALE(KM)\n'
        sig_format = '    {:16e}   {:16e}                       SIGKM, RMSLMK\n'
        vector_format = '    {:16e}   {:16e}   {:16e}   {}\n'

        with open(file_name, 'w') as ofile:
            ofile.write('{}   T\n'.format(self.name))
            ofile.write(size_scale_format.format(self.size, self.scale))
            ofile.write('        -1        -1        -1        -1                       HORIZON\n')
            ofile.write(sig_format.format(self.sigkm, self.rmslmk).replace('e', 'D'))
            ofile.write(vector_format.format(*self.vlm, 'VLN').replace('e', 'D'))
            ofile.write(vector_format.format(*self.rot_map2bod[0], 'UX').replace('e', 'D'))
            ofile.write(vector_format.format(*self.rot_map2bod[1], 'UY').replace('e', 'D'))
            ofile.write(vector_format.format(*self.rot_map2bod[2], 'UZ').replace('e', 'D'))
            ofile.write(vector_format.format(*self.sigma_lmk, 'SIMGA_LMK').replace('e', 'D'))
            ofile.write('PICTURES\n')
            ofile.write('MAP OVERLAPS\n')
            ofile.write('LIMB FITS\n')
            ofile.write('END FILE')

    def read(self, file_name: Optional[PATH] = None):
        """
        This method reads data from a SPC LMK file (normally ``LMKFILES/*MAPLETNAME*.LMK``) and populates the
        attributes of this class with that data.

        If ``file_name`` is not specified the :attr:`file_name` attribute of the class is used instead

        :param file_name: The file to load the LMK data from
        """

        if file_name is None:
            if self.file_name is not None:
                file_name = self.file_name
            else:
                raise ValueError('file_name must be specified')

        with open(file_name, 'r') as ifile:

            # parse the name and observation_date lines (first two lines
            self.name = ifile.readline().strip(' \n\t\r').split()[0]

            # parse the size and scale
            line = ifile.readline().strip(' \n\t\r').replace('D', 'E').split()
            self.size, self.scale = int(line[0]), float(line[1])

            # get sigkm and rmslmk
            ifile.readline()  # skip the HORIZON line
            line = ifile.readline().strip(' \n\t\r').replace('D', 'E').split()
            self.sigkm, self.rmlmk = float(line[0]), float(line[1])

            # parse the body to landmark position vector
            line = ifile.readline().strip(' \n\t\r').replace('D', 'E')
            self.vlm = np.fromstring(line, count=3, sep=' ')

            # parse the landmark to body rotation matrix
            line = ifile.readline().strip(' \n\t\r').replace('D', 'E')
            self.rot_map2bod[0] = np.fromstring(line, count=3, sep=' ')
            line = ifile.readline().strip(' \n\t\r').replace('D', 'E')
            self.rot_map2bod[1] = np.fromstring(line, count=3, sep=' ')
            line = ifile.readline().strip(' \n\t\r').replace('D', 'E')
            self.rot_map2bod[2] = np.fromstring(line, count=3, sep=' ')

            # parse the landmark uncertainty
            line = ifile.readline().strip(' \n\t\r').replace('D', 'E')
            self.sigma_lmk = np.fromstring(line, count=3, sep=' ')


class Maplet:
    """
    This class is used to read and write from SPC Maplet files.

    The Maplet files specify local terrain and albedo data for a landmark.

    When creating an instance of this class you can enter the ``file_name`` argument and the data will automatically
    be read from that file.  Alternatively you can specify individual components of the object through key word
    arguments.  If you provide both key word arguments and a file name then the key word arguments you specified will
    overwrite anything read from the file.
    """

    # TODO: update the attribute names to be more descriptive

    def __init__(self, file_name: Optional[PATH] = None, name: Optional[str] = None, size: Optional[int] = None,
                 scale: NONENUM = None, position_objmap: NONEARRAY = None, rotation_maplet2body: NONEARRAY = None,
                 hscale: NONENUM = None, heights: NONEARRAY = None, albedos: NONEARRAY = None):
        """
        :param file_name: The name of the Maplet file
        :param name: the name of the Maplet
        :param size: half the size of the maplet
        :param scale: The ground sample distance of the maplet in units of km
        :param position_objmap: the body-fixed center of the maplet (Landmark)
        :param rotation_maplet2body: The rotation from the local maplet frame to the body fixed frame
        :param hscale: The scaling term used to convert maplet heights between a float in units of scale and an integer
                       for file storage.
        :param heights: The height data for the maplet expressed in the local maplet frame as a 2*size+1 by 2*size+1 2D
                        numpy array
        :param albedos: the relative albedo data for the maplet expressed in the local maplet frame as a 2*size+1 by
                        2*size+1 2D numpy array
        """

        self.file_name = file_name  # type: Optional[PATH]
        """
        The name of the landmark file
        """

        self.name = ''  # type: str
        """
        The name of the Maplet.  

        This should be a 6 character string, typically alpha-numeric.

        This corresponds to the name of the maplet file, not including the the extension.  It is provided for 
        convenience and is not used in either reading or writing.
        """

        self.size = 49  # type: int
        """
        Half the number of grid cells on a side for the Maplet minus 1.

        This is computed as to size=(n_rows - 1)/2 where n_rows is the number of rows in the 
        Maplet (or number of columns since Maplets are always square).

        This corresponds to the 16bit unsigned integer in the 11th and 12th bytes of the maplet file.
        """

        self.scale = None  # type: Optional[float]
        """
        The ground sample distance of each grid cell for the Maplet in units of KM.

        This corresponds to the 32bit float in the 7th through 10th bytes of the maplet file.
        """

        self.position_objmap = np.zeros(3)  # type: np.ndarray
        """
        The body-fixed landmark vector in units of km as a length 3 numpy array of floats

        This corresponds to the 3 32bit floats in the 16th through 28th bytes of the maplet file.
        """

        self.rotation_maplet2body = np.zeros((3, 3))  # type: np.ndarray
        """
        The rotation matrix from the Maplet local frame to the body-fixed frame as a 3x3 numpy array of 
        floats.

        This corresponds to the 9 32bit floats in the 29th-55th bytes of the maplet file..
        """

        self.hscale = None  # type: NONENUM
        """
        The scale used to convert heights from a float to a 16byte signed integer in the maplet file.
        
        This corresponds to the 32bit float in the 56th byte of the maplet file.
        """

        self.heights = np.zeros((2 * 49 + 1, 2 * 49 + 1))  # type: np.ndarray
        """
        The height profile for the maplet as a 2*size+1 by 2*size+1 2D numpy array
        
        This corresponds to the 16bit signed integers in the remaining data in the maplet file, but have been converted
        to floats using hscale.
        """

        self.albedos = np.zeros((2 * 49 + 1, 2 * 49 + 1))  # type: np.ndarray
        """
        The relative albedo profile for the maplet as a 2*size+1 by 2*size+1 2D numpy array.
        
        this corresponds to the 8 bit unsigned integers in the remaining data in the maplet file, but have been 
        converted to floats
        """

        if file_name is not None:
            self.read()
            self.name = os.path.split(self.file_name)[1].split('.')[0]

        if name is not None:
            self.name = name
        if size is not None:
            self.size = size
        if scale is not None:
            self.scale = scale
        if position_objmap is not None:
            self.position_objmap = position_objmap
        if rotation_maplet2body is not None:
            self.rotation_maplet2body = rotation_maplet2body
        if hscale is not None:
            self.hscale = hscale
        if heights is not None:
            self.heights = heights
        if albedos is not None:
            self.albedos = albedos

    def __eq__(self, other):

        if not isinstance(other, self.__class__):
            return False

        comp_dict = {'scale': self.scale == other.scale,
                     'size': self.size == other.size,
                     'position_objmap': (self.position_objmap == other.position_objmap).all(),
                     'rotation_maplet2body': (self.rotation_maplet2body == other.rotation_maplet2body).all(),
                     'hscale': self.hscale == other.hscale,
                     'heights': (self.heights == other.heights).all(),
                     'albedos': (self.albedos == other.albedos).all()}

        if np.all(comp_dict.values()):
            return True

        else:
            return comp_dict

    def read(self, file_name: Optional[PATH] = None):
        """
        This method reads data from a SPC Maplet file (normally ``MAPFILES/*MAPLETNAME*.MAP``) and populates the
        attributes of this class with that data.

        If ``file_name`` is not specified the :attr:`file_name` attribute of the class is used instead

        :param file_name: The file to load the MAP data from
        """

        if file_name is None:
            if self.file_name is not None:
                file_name = self.file_name
            else:
                raise ValueError('file_name must be specified')

        with open(file_name, 'rb') as map_fileobj:
            # READ IN THE MAPLET HEADER

            # skip the first six bytes
            map_fileobj.seek(6, 0)

            # read in the scale using numpy's from file
            # note that here the > in the dtype indicates big endianess
            self.scale = np.fromfile(map_fileobj, dtype=">f4", count=1)[0]

            # read in the Maplet size using little endian order because SPC...
            self.size = np.fromfile(map_fileobj, dtype='<u2', count=1)[0]

            map_shape = (self.size * 2 + 1, self.size * 2 + 1)

            # skip the next three bytes from the current position
            map_fileobj.seek(3, 1)

            # read in the body-maplet vector expressed in the body frame
            self.position_objmap = np.fromfile(map_fileobj, dtype=">f4", count=3)

            # read in the maplet2body frame rotation matrix.
            # Need to use reshape to form into a matrix given the vector of data
            self.rotation_maplet2body = np.fromfile(map_fileobj, dtype=">f4", count=9).reshape(3, 3).T

            # read in the height scale
            self.hscale = np.fromfile(map_fileobj, dtype=">f4", count=1)[0]

            # skip five bytes from the current position
            map_fileobj.seek(5, 1)

            # read in the height and albedo data
            # read in the data as a structured array where the first element in each tuple is the height data
            # and the second element in each tuple is the albedo data
            data = np.fromfile(map_fileobj, dtype=[('height', '>i2'), ('albedo', '>u1')])

            # get how much of that is extraneous data by comparing with the total number of elements in the Maplet
            num_pad = data.size - map_shape[0] * map_shape[1]

            # remove the extraneous data from the end
            data = data[:-num_pad]

            # extract the height data from the structured array
            height_data = data['height']

            # extract the albedo data from the structured array
            albedo_data = data['albedo']

            # set the height data to 0 wherever the albedo data is zero
            height_data[albedo_data == 0] = 0

            # multiply by the height data scale and reshape into the appropriate Maplet size
            self.heights = self.hscale * height_data.reshape(map_shape)

            # reshape the albedo data into the appropriate Maplet size and change to decimal form
            self.albedos = 0.01 * albedo_data.reshape(map_shape)

    def write(self, file_name: Optional[PATH] = None):
        """
        This function writes the data contained in the current instance of the class into the specified file

        If the ``file_name`` argument is not specified then it writes to the file stored in the
        :attr:`~.Maplet.file_name` attribute.

        :param file_name: the full or relative path to the file to be written.  If left as ``None`` then writes to the
                          file stored in the :attr:`~.Maplet.file_name` attribute
        """

        if file_name is None:
            if self.file_name is not None:
                file_name = self.file_name
            else:
                raise ValueError('file_name must be specified')

        with open(file_name, 'wb') as map_fileobj:
            # mark that the first six bytes are not used
            map_fileobj.write(b'UNUSED')

            # write the Maplet scale as a 4 byte float using big endian encoding
            map_fileobj.write(struct.pack('>f', self.scale))

            # write the Maplet size to file using little endian encoding... because SPC...
            map_fileobj.write(struct.pack('<H', self.size))

            # write the Maplet position and placeholders for the formal uncertainty
            map_fileobj.write(struct.pack('>xxxfff', *self.position_objmap))

            # write the Maplet to body rotation matrix
            map_fileobj.write(self.rotation_maplet2body.T.astype('>f4').tostring())

            # write in the scaling term for the Maplet height
            map_fileobj.write(struct.pack('>f' + 'x' * 5, self.hscale))

            # write in the Maplet height and albedo data
            # first determine the total number of pixels we have to store
            num_pixels = (2 * self.size + 1) ** 2

            # now determine how many records of 72 bytes that is going to take
            num_records = int(np.ceil((num_pixels * 3) / 72))

            # get the number of bytes that we will need to pad to fill up these records
            num_pads = 72 * num_records - num_pixels * 3

            # reshape the matrices into the required format and put the data in the proper format
            # first we need to scale the height data and round the scaled data to the nearest integers
            # (because this helps with storage requirements).  Then we use the convenient .reshape(-1)
            # to form a vector from the matrix, but we can only do this as an array type so we need to
            # use the asarray method
            height_data = np.rint(self.heights / self.hscale).flatten()

            # do the same thing for the albedo data
            # noinspection PyArgumentList
            albedo_data = np.rint(self.albedos * 200 / self.albedos.max()).flatten()

            # combine into a structured numpy array for creating the binary representation
            map_data = np.core.records.fromarrays([height_data, albedo_data],
                                                  dtype=[('heights', '>i2'), ('albedos', 'u1')])

            # write to the file -- note that for the format we write num_pixels pairs of short signed 2 byte integers
            # with an unsigned character and then pad with num_pads empty space
            map_fileobj.write(map_data.tostring() + b'\x00' * num_pads)

    def get_triangles(self, me: bool = False) -> Union[Triangle64, Triangle32]:
        """
        This method returns GIANT triangle objects from the maplet data to be used in the GIANT ray tracer

        :param me: A flag specifying whether to return memory efficient triangles (``True``) or not
        :return: The GIANT triangles
        """

        # form the column and row labels
        cols, rows = np.meshgrid(*[np.arange(-int(self.size), int(self.size) + 1)] * 2)

        # get the vectors in the maplet frame
        map_vecs = self.scale * np.vstack([cols.flatten(), rows.flatten(), self.heights.T.flatten()])

        # rotate and translate into the body fixed frame
        body_vecs = (self.position_objmap.reshape(3, 1) + np.matmul(self.rotation_maplet2body, map_vecs))

        # compute the length of a side
        length = self.size * 2 + 1

        # make indicies for the tesselation
        tchunk0a = np.hstack([np.arange(length * i + 1, length * (i + 1), dtype=np.uint64) for i in range(length - 1)])
        tchunk0b = np.hstack([np.arange(length * (i + 1) + 1, length * (i + 2), dtype=np.uint64)
                              for i in range(length - 1)])

        tess0 = np.hstack([tchunk0a, tchunk0b])

        tchunk1a = tchunk0b
        tchunk1b = tchunk0b + 1

        tess1 = np.hstack([tchunk1a, tchunk1b])

        tchunk2a = tchunk0a + 1
        tchunk2b = tchunk2a

        tess2 = np.hstack([tchunk2a, tchunk2b])

        tess = np.vstack([tess0, tess2, tess1]) - 1

        # get the albedos in the proper format
        albs = self.albedos.T.ravel()

        # form the triangles and return
        if me:
            return Triangle32(body_vecs.T.copy(), albs.copy(), tess.T.astype(np.uint32),
                              compute_reference_ellipsoid=False)
        else:
            return Triangle64(body_vecs.T.copy(), albs.copy(), tess.T.astype(np.uint32),
                              compute_reference_ellipsoid=False)


def _fortran_float_converter(in_string: bytes) -> bytes:
    """
    This utility converts fortran double precision float strings to python float strings by replacing D with e

    :param in_string: The fortran string containing double precision floats
    :return: The string ready for ingest by python float utilities
    """

    return in_string.replace(b'D', b'e')


class ShapeModel:
    """
    This class is used to read and write from SPC Shape files (Implicitly connected Quadrilateral, ICQ format).

    The Shape files specify the global terrain for SPC.

    When creating an instance of this class you can enter the ``file_name`` argument and the data will automatically
    be read from that file.  Alternatively you can specify individual components of the object through key word
    arguments.  If you provide both key word arguments and a file name then the key word arguments you specified will
    overwrite anything read from the file.
    """

    def __init__(self, file_name: Optional[PATH] = None, conversion: Real = 1.0, grid_size: NONEARRAY = None,
                 vertices: NONEARRAY = None, albedos: NONEARRAY = None):
        """
        :param file_name: The name of the file to load the data from
        :param conversion: A conversion factor to convert the units of the shape model into the units you want.
                           Official SPC shape models are in km, which are the same units GIANT usually uses so this
                           should typically remain 1.
        :param grid_size: The number of cells on each side of each face for the ICQ format.  This corresponds to the
                          number of vertices by ``6*grid_size**2``
        :param vertices: The vertices for the ICQ format as a nx3 array.  The ordering of these is complicated.
        :param albedos: The albedos corresponding to the vertices as a nx1 array.  Alternatively can be None if no
                        albedo data is available.
        """

        self.file_name = file_name  # type: Optional[PATH]
        """
        The name of the shape file
        """

        self.grid_size = None  # type: Optional[int]
        """
        The length of one side of the grid for the ICQ format
        
        This corresponds to the first number in the shape file
        """

        self.vertices = None  # type: NONEARRAY
        """
        The ICQ vertices and a nx3 array
        """

        self.albedos = None  # type: NONEARRAY
        """
        The albedo for each vertex as a nx1 array.
        
        This is optional and only included in some shape files.  It is the fourth float for each vertex in the shape 
        file if available.
        """

        if file_name is not None:
            self.read(conversion=conversion)

        if grid_size is not None:
            self.grid_size = grid_size
        if vertices is not None:
            self.vertices = vertices
        if albedos is not None:
            self.albedos = albedos

    def read(self, file_name: Optional[PATH] = None, conversion: Real = 1.0):
        """
        This method reads data from a SPC Shape file (normally ``SHAPEFILES/*.TXT``) and populates the attributes
        of this class with that data.

        If ``file_name`` is not specified the :attr:`file_name` attribute of the class is used instead.

        The argument conversion can be used to convert from the SPC units (nominally km) to another unit for the
        vertices.

        :param file_name: The file to load the Shape data from
        :param conversion: The conversion to apply to the verticies to change their units
        """

        if file_name is None:
            if self.file_name is not None:
                file_name = self.file_name
            else:
                raise ValueError('file_name must be specified')

        with open(file_name, 'r') as shape_file:

            # get the grid size for each cube face
            self.grid_size = int(shape_file.readline().strip().split()[0]) + 1

            if len(shape_file.readline().strip().split()) == 4:
                shape_file.seek(0)

                dtype = np.dtype([('vec', np.float64, (3,)), ('albedo', np.float64)])

                # noinspection PyTypeChecker
                data = np.loadtxt(shape_file, dtype=dtype, skiprows=1,
                                  converters={0: _fortran_float_converter, 1: _fortran_float_converter,
                                              2: _fortran_float_converter, 3: _fortran_float_converter})

                self.vertices = data['vec'] * conversion
                self.albedos = data['albedo']

            else:
                shape_file.seek(0)

                # noinspection PyTypeChecker
                data = np.loadtxt(shape_file, dtype=np.float64, skiprows=1,
                                  converters={0: _fortran_float_converter,
                                              1: _fortran_float_converter,
                                              2: _fortran_float_converter})

                self.vertices = data
                self.albedos = None

    def write(self, file_name: Optional[PATH] = None):
        """
        This function writes the data contained in the current instance of the class into the specified file

        If the ``file_name`` argument is not specified then it writes to the file stored in the
        :attr:`~.ShapeModel.file_name` attribute.

        :param file_name: the full or relative path to the file to be written.  If left as ``None`` then writes to the
                          file stored in the :attr:`~.ShapeModel.file_name` attribute
        """

        if file_name is None:
            if self.file_name is not None:
                file_name = self.file_name
            else:
                raise ValueError('file_name must be specified')

        with open(file_name, 'w') as ofile:
            vec_fmt = '{:>12.6f}{:>12.6f}{:>12.6f}\n'
            alb_fmt = '{:>12.6f}{:>12.6f}{:>12.6f}{:>12.6f}\n'

            ofile.write('{:>12d}\n'.format(self.grid_size))

            if self.albedos is not None:
                for v, a in zip(self.vertices, self.albedos):
                    ofile.write(alb_fmt.format(*v, a))
            else:
                for v in self.vertices:
                    ofile.write(vec_fmt.format(*v))

    def get_triangles(self, me: bool = True) -> Union[Triangle64, Triangle32]:
        """
        This method returns GIANT triangle objects from the maplet data to be used in the GIANT ray tracer

        :param me: A flag specifying whether to return memory efficient triangles (``True``) or not
        :return: The GIANT triangles
        """

        if self.albedos is None:
            albedos = 1.0
        else:
            albedos = self.albedos

        indices = np.arange(self.vertices.shape[0])

        icq_indices = indices.reshape((6, self.grid_size, self.grid_size))
        triangulated_icq_indices = np.concatenate([icq_indices[:, :-1, :-1].reshape(-1, 1),
                                                   icq_indices[:, :-1, 1:].reshape(-1, 1),
                                                   icq_indices[:, 1:, 1:].reshape(-1, 1),
                                                   icq_indices[:, 1:, :-1].reshape(-1, 1)], axis=-1)

        facets = np.concatenate([triangulated_icq_indices[..., [0, 3, 1]],
                                 triangulated_icq_indices[..., [1, 3, 2]]], axis=0)

        if me:
            return Triangle32(self.vertices, albedos, facets.astype(np.uint32))
        else:
            return Triangle64(self.vertices, albedos, facets.astype(np.uint32))


def get_distortion(image_name: str, lithos_file: PATH) -> np.ndarray:
    """
    This function gets the distortion value from the lithos file pertaining to a specified image

    :param image_name:  The name of the Image that the distortion belongs to
    :param lithos_file:  The lithos file to read from
    :return distortion: A numpy matrix vector containing the distortion model coefficients.
    """

    distortion = np.nan

    with open(lithos_file, 'r') as lithos:

        lithos_lines = lithos.readlines()

        for index, line in enumerate(lithos_lines):

            if image_name in line:
                distortion = np.fromstring(lithos_lines[index + 1].strip(' \n\t\r').replace('D', 'E'),
                                           sep=' ')

    return distortion


'''
# TODO: Move these to unit tests
if __name__ == "__main__":

    import tempfile

    from contextlib import contextmanager

    import difflib


    @contextmanager
    def tempinput(data):

        temp = tempfile.NamedTemporaryFile(delete=False)

        temp.write(data)

        temp.close()

        yield temp.name

        os.unlink(temp.name)


    sample_sum = b"""N623721668F1
2019 OCT 07 12:00:00.008
  2592  1944     5  4095                                       NPX, NLN, THRSH
    0.7680000000D+01    0.1296500000D+04    0.9725000000D+03   MMFL, CTR
    0.2351649185D+00   -0.2809283686D+00    0.8868377173D+00   SCOBJ
   -0.6317653990D+00    0.6516609256D+00    0.4197743664D+00   CX
   -0.6953193648D+00   -0.7157809166D+00    0.6471986115D-01   CY
    0.3426418853D+00   -0.2509894769D+00    0.9053180772D+00   CZ
    0.9046070715D+00    0.4179848898D+00   -0.8351453790D-01   SZ
 454.54000   0.00000   0.00000   0.00000 454.54000   0.00000   K-MATRIX
    0.00000D+00    0.00000D+00    0.00000D+00    0.00000D+00   DISTORTION
    0.3129597577D-03    0.3131765583D-03    0.3130389276D-03   SIGMA_VSO
    0.5226008432D-03    0.5425981305D-03    0.1994887106D-02   SIGMA_PTG
LANDMARKS
BM0071   2118.75    734.66
BM0072   1859.93    495.69
BM0073   1555.02    379.77
BM0085   1756.97    711.28
BM0086   1388.49    551.32
BM0087   1010.84    563.61
BM0088    735.15    947.80
BM0095   1373.05    861.46
BM0096   1036.84    783.76
LIMB FITS
END FILE"""

    with tempinput(sample_sum) as in_file:

        sum_in = Summary(in_file)

        with tempinput(b'') as out_file:

            sum_in.write(out_file)

            sum_out = Summary(out_file)

            comp = sum_in == sum_out

            if not isinstance(comp, bool):

                print(comp)

                with open(out_file, 'r') as out_data:

                    out_text = out_data.read().splitlines(keepends=True)

                    diff = difflib.Differ()

                    print(''.join(diff.compare(sample_sum.decode().splitlines(keepends=True), out_text)))

            else:
                print('Summary Success')

    sample_nom = b"""N623728868F1
   -0.2811837848D+00    0.5804346431D+00   -0.7642194084D+00   BOD_FRAME
   -0.5044891352D+00    0.6056100257D+00    0.5819377357D+00   SCOBJ
    0.1000000000D-02    0.1000000000D-02    0.1000000000D-02   SIGMA_VSO
    0.5393528366D+00   -0.2528761844D+00    0.8032136410D+00   CX
    0.5932654478D+00    0.7910347948D+00   -0.1493320524D+00   CY
   -0.5976074180D+00    0.5570615665D+00    0.5766695632D+00   CZ
    0.3000000000D-02    0.3000000000D-02    0.3000000000D-02   SIGMA_PTG
END FILE"""

    with tempinput(sample_nom) as in_file:

        nom_in = Nominal(in_file)

        with tempinput(b'') as out_file:

            nom_in.write_nom(out_file)

            nom_out = Nominal(out_file)

            comp = nom_in == nom_out

            if not isinstance(comp, bool):

                print(comp)

                with open(out_file, 'r') as out_data:

                    out_text = out_data.read().splitlines(keepends=True)

                    diff = difflib.Differ()

                    print(''.join(diff.compare(sample_nom.decode().splitlines(keepends=True), out_text)))

            else:
                print('Nominal Success')

    map_file = '/Users/aliounis/Documents/ODTT3B_filestructure/Run1/MAPFILES/BM0007.MAP'

    maplet = Maplet(map_file)

    with tempinput(b'') as out_file:

        maplet.write(out_file)

        map_out = Maplet(out_file)

        comp = maplet == map_out

        if not isinstance(comp, bool):

            print(comp)

        else:

            print('Maplet Success')


    # shape_file = '/Users/aliounis/GDrive/Work/DAWN_VESTA_File_Structure/Vesta/SHAPEFILES/SHAPE0.TXT'
    shape_file = '/Users/aliounis/Repos/spcmc/realdata/CleanDir/SHAPEFILES/SHAPE.TXT'

    shape_model = ShapeModel(shape_file)

    print('stop')

    # import matplotlib.pyplot as plt

    # shape_obj.shapeobj.plot()

    # plt.show()
'''
