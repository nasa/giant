# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


r"""
This module provides a subclass of the :class:`.OpNav` class for performing stellar OpNav and camera calibration.

Interface Description
---------------------

In GIANT, calibration refers primarily to the process of estimating a model to map points in the 3D
world to the observed points in a 2D image.  This is done by estimating both a geometric
:mod:`camera model <.camera_models>` along with an optional pointing alignment between the camera frame and the base
frame the knowledge of the camera attitude is tied to (for instance the spacecraft bus frame).  For both of these, we
use observations of stars to get highly accurate models.

The :class:`Calibration` class is the main interface for performing calibration in GIANT, and in general is all the user
will need to interact with.  It is a subclass of the :class:`.StellarOpNav` class and as such provides a very similar
interface with only a few additional features. It provides direct access to the :class:`.ImageProcessing`,
:class:`.StarID`, :mod:`.stellar_opnav.estimators`, and :mod:`.calibration.estimators` objects and automatically
preforms the required data transfer between the objects for you.  To begin you simply provide the :class:`.StellarOpNav`
constructor a :class:`.Camera` instance, either a :class:`.ImageProcessing` instance or the keyword arguments to create
one, a :class:`.StarID` instance or the keyword arguments to creation one, the attitude estimation object you wish to
use to perform the attitude estimation, the static alignment estimation object you wish to use to perform the static
alignment estimation, the temperature dependent alignment estimation object you wish to use to perform the temperature
dependent alignment, and the calibration estimation object you wish to use to perform the calibration. You can then use
the :class:`Calibration` instance to perform all of the aspects of stellar OpNav and calibration with never having to
interact with the internal objects again.

For example, we could do something like the following (from the directory containing ``sample_data``):

    >>> import pickle
    >>> from giant.calibration import Calibration
    >>> from giant.rotations import Rotation
    >>> with open('sample_data/camera.pickle', 'rb') as ifile:
    ...     camera = pickle.load(ifile)
    >>> # Returns the identity to signify the base frame is the inertial frame
    >>> def base_frame(*args):
    ...     return Rotation([0, 0, 0, 1])
    >>> cal = Calibration(camera, alignment_base_frame_func=base_frame)
    >>> cal.id_stars()  # id the stars for each image
    >>> cal.sid_summary()  # print out a summary of the star identification success for each image
    >>> cal.estimate_attitude()  # estimate an updated attitude for each image
    >>> cal.estimate_calibration()  # estimate an updated camera model
    >>> cal.calib_summary()  # print out a summary of the star identification success for each image
    >>> cal.estimate_static_alignment()  # estimate the alignment between the camera frame and hte base frame
    >>> cal.estimate_temperature_dependent_alignment()  # estimate the temperature dependent alignment

For a more general description of the steps needed to perform calibration, refer to the :mod:`.calibration` package.
For a more in-depth examination of the :class:`Calibration` class see the following API Reference.
"""

import warnings

from copy import deepcopy

from typing import Optional, Sequence, Callable

import numpy as np

from . import estimators as est
from ..stellar_opnav.stellar_class import StellarOpNav
from ..image_processing import ImageProcessing
from ..stellar_opnav.star_identification import StarID
from ..stellar_opnav import estimators as sopnavest
from ..camera import Camera
from ..rotations import Rotation

from .._typing import NONEARRAY, SCALAR_OR_ARRAY


def _print_lr(labels: Sequence, values: np.ndarray):
    """
    This pretty prints the lower left triangle of a matrix with labels.

    This is used to print covariance and correlation matrices.

    :param labels: The labels for each row/column of the matrix
    :param values: The matrix to print
    """

    # get the maximum length of the labels, with a minimum size of 10
    max_label = max(max(map(len, labels)), 10)

    # make the label format string based on the maximum label length
    label_format = '{:<' + str(max_label) + 's}'

    # make the value format string based on the maximum label length
    value_format = '{:>' + str(max_label) + '.' + str(max_label-7) + 'e}'

    # loop through the rows
    for rind, rlabel in enumerate(labels):
        # print the label format at the beginning of each new row.  Don't use a new line after
        print(label_format.format(rlabel), end='  ')
        # loop through the columns
        for cind, clabel in enumerate(labels):
            # skip the upper right triangle
            if cind > rind:
                print('\n', end='')
                break
            # print out the value using the format string.  Don't use a new line after
            print(value_format.format(values[rind, cind]), end='  ')

    # print out a space to get the column labels in the right place
    print('\n' + label_format.format(''), end='')

    # change the label format to be right aligned
    label_format = label_format.replace('<', '>')

    # print out a row of column labels
    for clabel in labels:

        print(label_format.format(clabel), end='  ')

    # print a new line
    print('')

    return max_label


class Calibration(StellarOpNav):
    """
    This class serves as the main user interface for performing geometric camera calibration and camera frame attitude
    alignment.

    The class acts as a container for the :class:`.Camera`, :class:`.ImageProcessing`, and
    :mod:`.stellar_opnav.estimators`, :mod:`.calibration.estimators` objects and also passes the correct and up-to-date
    data from one object to the other. In general, this class will be the exclusive interface to the mentioned objects
    and models for the user.

    This class provides a number of features that make doing stellar OpNav and camera calibration/alignment easy.  The
    first is it provides aliases to the image processing, star id, attitude estimation, calibration estimation, and
    alignment estimation objects. These aliases make it easy to quickly change/update the various tuning parameters that
    are necessary to make star identification and calibration a success. In addition to providing convenient access to
    the underlying settings, some of these aliases also update internal flags that specify whether individual images
    need to be reprocessed, saving computation time when you're trying to find the best tuning.

    This class also provides simple methods for performing star identification, attitude estimation, camera calibration,
    and aligment estimation after you have set the tuning parameters. These methods (:meth:`id_stars`,
    :meth:`sid_summary`, :meth:`estimate_attitude`, :meth:`estimate_calibration`, :meth:`calib_summary`,
    :meth:`estimate_static_alignment`, and :meth:`estimate_temperature_dependent_alignment`) combine all of the
    required steps into a few simple calls, and pass the resulting data from one object to the next. They also store off
    the results of the star identification in the :attr:`queried_catalogue_star_records`,
    :attr:`queried_catalogue_image_points`, :attr:`queried_catalogue_unit_vectors`, :attr:`ip_extracted_image_points`,
    :attr:`ip_image_illums`, :attr:`ip_psfs`, :attr:`ip_stats`, :attr:`ip_snrs`,
    :attr:`unmatched_catalogue_image_points`, :attr:`unmatched_image_illums`,
    :attr:`unmatched_psfs`, :attr:`unmatched_stats`, :attr:`unmatched_snrs`
    :attr:`unmatched_catalogue_star_records`,
    :attr:`unmatched_catalogue_unit_vectors`,
    :attr:`unmatched_extracted_image_points`,
    :attr:`matched_catalogue_image_points`, :attr:`matched_image_illums`,
    :attr:`matched_psfs`, :attr:`matched_stats`, :attr:`matched_snrs`
    :attr:`matched_catalogue_star_records`,
    :attr:`matched_catalogue_unit_vectors_inertial`,
    :attr:`matched_catalogue_unit_vectors_camera`, and
    :attr:`matched_extracted_image_points` attributes, enabling more advanced analysis to be performed external to the
    class.

    This class stores the updated attitude solutions in the image objects themselves, allowing you to directly
    pass your images from stellar OpNav to the :mod:`.relative_opnav` routines with updated attitude solutions. It also
    stores the estimated camera model in the original camera model itself, and store the estimated alignments in the
    :attr:`static_alignment` and :attr:`temperature_dependent_alignment` attributes. Finally, this class
    respects the :attr:`.image_mask` attribute of the :class:`.Camera` object, only considering images that are
    currently turned on.

    When initializing this class, most of the initial options can be set using the ``*_kwargs`` inputs with
    dictionaries specifying the keyword arguments and values. Alternatively, you can provide already initialized
    instances of the :class:`.ImageProcessing`, :class:`.AttitudeEstimator`, :class:`.StarID`,
    :class:`.CalibrationEstimator`, :class:`.StaticAlignmentEstimator`, or
    :class:`.TemperatureDependentAlignmentEstimator` classes or subclasses
    if you want a little more control.  You should see the documentation for these classes for more details on what you
    can do with them.
    """

    def __init__(self, camera: Camera, use_weights: bool = False,
                 image_processing: Optional[ImageProcessing] = None, image_processing_kwargs: Optional[dict] = None,
                 star_id: Optional[StarID] = None, star_id_kwargs: Optional[dict] = None,
                 alignment_base_frame_func: Optional[Callable] = None,
                 attitude_estimator: Optional[sopnavest.AttitudeEstimator] = None,
                 attitude_estimator_kwargs: Optional[dict] = None,
                 static_alignment_estimator: Optional[est.StaticAlignmentEstimator] = None,
                 static_alignment_estimator_kwargs: Optional[dict] = None,
                 temperature_dependent_alignment_estimator: Optional[est.TemperatureDependentAlignmentEstimator] = None,
                 temperature_dependent_alignment_estimator_kwargs: Optional[dict] = None,
                 calibration_estimator: Optional[est.CalibrationEstimator] = None,
                 calibration_estimator_kwargs: Optional[dict] = None):
        """
        :param camera: The :class:`.Camera` object containing the camera model and images to be utilized
        :param use_weights: A flag specifying whether to use weighted estimation for attitude, alignment, and
                            calibration
        :param alignment_base_frame_func: A callable object which returns the orientation of the base frame with respect
                                          to the inertial frame the alignment of the camera frame is to be done with
                                          respect to for a given date.
        :param image_processing: An already initialized instance of :class:`.ImageProcessing` (or a subclass).  If not
                                 ``None`` then ``image_processing_kwargs`` are ignored.
        :param image_processing_kwargs: The keyword arguments to pass to the :class:`.ImageProcessing` class
                                        constructor.  These are ignored if argument ``image_processing`` is not ``None``
        :param star_id: An already initialized instance of :class:`.StarID` (or a subclass).  If not
                        ``None`` then ``star_id_kwargs`` are ignored.
        :param star_id_kwargs:  The keyword arguments to pass to the :class:`.StarID` class constructor as
                                a dictionary.  These are ignored if argument ``star_id`` is not ``None``.
        :param attitude_estimator: An already initialized instance of :class:`.AttitudeEstimator` (or a subclass).  If
                                   not ``None`` then ``attitude_estimator_kwargs`` are ignored.
        :param attitude_estimator_kwargs: The keyword arguments to pass to the :class:`.DavenportQMethod`
                                          constructor as a dictionary.  If argument ``attitude_estimator`` is not
                                          ``None`` then this is ignored.
        :param static_alignment_estimator: An already initialized instance of :class:`.StaticAlignmentEstimator` (or a
                                           subclass).  If not ``None`` then ``static_alignment_estimator_kwargs`` are
                                           ignored.
        :param static_alignment_estimator_kwargs: The keyword arguments to pass to the
                                                  :class:`.StaticAlignmentEstimator` constructor as a dictionary.  If
                                                  argument ``static_alignment_estimator`` is not ``None`` then this is
                                                  ignored.
        :param temperature_dependent_alignment_estimator: An already initialized instance of
                                                          :class:`.TemperatureDependentAlignmentEstimator` (or a
                                                          subclass).  If not ``None`` then
                                                          ``temperature_dependent_alignment_estimator_kwargs`` are
                                                          ignored.
        :param temperature_dependent_alignment_estimator_kwargs: The keyword arguments to pass to the
                                                                 :class:`.TemperatureDependentAlignmentEstimator`
                                                                 constructor as a dictionary.  If argument
                                                                 ``temperature_dependent_alignment_estimator`` is not
                                                                 ``None`` then this is ignored.
        :param calibration_estimator: An already initialized instance of :class:`.CalibrationEstimator` (or a
                                      subclass).  If not ``None`` then ``calibration_estimator_kwargs`` are ignored.
        :param calibration_estimator_kwargs: The keyword arguments to pass to the :class:`.IterativeNonlinearLSTSQ`
                                             constructor as a dictionary.  If argument ``static_alignment_estimator is
                                             not ``None`` then this is ignored.
        """

        # initialize the StellarOpNav super class
        super().__init__(camera, use_weights=use_weights,
                         image_processing=image_processing, image_processing_kwargs=image_processing_kwargs,
                         star_id=star_id, star_id_kwargs=star_id_kwargs,
                         attitude_estimator=attitude_estimator, attitude_estimator_kwargs=attitude_estimator_kwargs)

        if calibration_estimator is None:
            if calibration_estimator_kwargs is not None:
                self._calibration_est = est.IterativeNonlinearLSTSQ(model=self._camera.model,
                                                                    **calibration_estimator_kwargs)
            else:
                self._calibration_est = est.IterativeNonlinearLSTSQ(model=self._camera.model)
        else:
            self._calibration_est = calibration_estimator

        self.alignment_base_frame_func = alignment_base_frame_func  # type: Optional[Callable]
        """
        A callable object which returns the orientation of the base frame with respect  
        to the inertial frame the alignment of the camera frame is to be done with      
        respect to for a given date.                                                    
        
        This is used on calls to :meth:`estimate_static_alignment` and :meth`estimate_temperature_dependent_alignment` 
        to determine the base frame the alignment is being done with respect to.  Typically this returns something like 
        the spacecraft body frame with respect to the inertial frame (inertial to spacecraft body) or another camera 
        frame.
        """

        if static_alignment_estimator is None:
            if static_alignment_estimator_kwargs is not None:
                self._static_alignment_est = est.StaticAlignmentEstimator(**static_alignment_estimator_kwargs)
            else:
                self._static_alignment_est = est.StaticAlignmentEstimator()
        else:
            self._static_alignment_est = static_alignment_estimator

        self.static_alignment = None  # type: Optional[Rotation]
        """
        The static alignment as a :class:`.Rotation` object.
        
        This will be none until the :meth:`estimate_static_alignment` method is called at which point it will contain 
        the estimated alignment.
        """

        if temperature_dependent_alignment_estimator is None:
            if temperature_dependent_alignment_estimator_kwargs is not None:
                self._temperature_dependent_alignment_est = est.TemperatureDependentAlignmentEstimator(
                    **temperature_dependent_alignment_estimator_kwargs
                )
            else:
                self._temperature_dependent_alignment_est = est.TemperatureDependentAlignmentEstimator()
        else:
            self._temperature_dependent_alignment_est = temperature_dependent_alignment_estimator

        self.temperature_dependent_alignment = None  # type: NONEARRAY
        """
        The temperature dependent alignment as a 3x2 numpy array.

        The temperature dependent alignment array is stored such that the first column is the
        static offset for the alignment, the second column is the temperature dependent slope, and each row represents
        the euler angle according to the requested order (so if the requested order is ``'xyx'`` then the rotation from
        the base frame to the camera frame at temperature ``t`` can be computed using:

            >>> from giant.rotations import euler_to_rotmat, Rotation
            >>> import numpy as np
            >>> temperature_dependent_alignment = np.arange(6).reshape(3, 2)  # temp array just to demonstrate
            >>> t = -22.5  # temp temperature just to demonstrate
            >>> angles =temperature_dependent_alignment@[1, t]
            >>> order = 'xyx'
            >>> rotation_base_to_camera = Rotation(euler_to_rotmat(angles, order))
            
        """

        self._initial_calibration_est = self._calibration_est.__class__
        self._initial_calibration_est_kwargs = calibration_estimator_kwargs
        self._initial_static_alignment_est = self._static_alignment_est.__class__
        self._initial_static_alignment_est_kwargs = static_alignment_estimator_kwargs
        self._initial_temperature_dependent_alignment_est = self._temperature_dependent_alignment_est.__class__
        self._initial_temperature_dependent_alignment_est_kwargs = temperature_dependent_alignment_estimator_kwargs

    # update the model setter to also update the sid model
    @StellarOpNav.model.setter
    def model(self, val):
        # dispatch to the super setter
        super().model.__set__(val)
        self._calibration_est.model = val

    @property
    def calibration_estimator(self) -> est.CalibrationEstimator:
        """
        The calibration estimator to use when estimating the geometric calibration

        This should typically be a subclass of the :class:`.CalibrationEstimator` meta class.

        See the :mod:`~.calibration.estimators` documentation for more details.
        """

        return self._calibration_est

    @calibration_estimator.setter
    def calibration_estimator(self, val):
        if isinstance(val, est.CalibrationEstimator):
            self._calibration_est = val
        else:
            warnings.warn("The calibration_estimator object should probably subclass the CalibrationEstimator "
                          "metaclass. We'll assume you know what you're doing for now, but see the "
                          "calibration.estimator documentation for details")

            self._calibration_est = val

    @property
    def static_alignment_estimator(self) -> est.StaticAlignmentEstimator:
        """
        The static alignment estimator to use when estimating the static alignment

        This should typically be a subclass of the :class:`.StaticAlignmentEstimator` class.

        See the :mod:`~.calibration.estimators` documentation for more details.
        """

        return self._static_alignment_est

    @static_alignment_estimator.setter
    def static_alignment_estimator(self, val: est.StaticAlignmentEstimator):
        if isinstance(val, est.StaticAlignmentEstimator):
            self._static_alignment_est = val
        else:
            warnings.warn("The static alignment_estimator object should probably subclass the "
                          "StaticAlignmentEstimator class. We'll assume you know what you're doing for "
                          "now, but see the calibration.estimator documentation for details")

            self._static_alignment_est = val

    @property
    def temperature_dependent_alignment_estimator(self) -> est.TemperatureDependentAlignmentEstimator:
        """
        The temperature_dependent_alignment estimator to use when estimating the temperature_dependent_alignment

        This should typically be a subclass of the :class:`.TemperatureDependentAlignmentEstimator` class.

        See the :mod:`~.calibration.estimators` documentation for more details.
        """

        return self._temperature_dependent_alignment_est

    @temperature_dependent_alignment_estimator.setter
    def temperature_dependent_alignment_estimator(self, val: est.TemperatureDependentAlignmentEstimator):
        if isinstance(val, est.StaticAlignmentEstimator):
            self._temperature_dependent_alignment_est = val
        else:
            warnings.warn("The temperature_dependent_alignment_estimator object should probably subclass the "
                          "TemperatureDependentAlignmentEstimator class. We'll assume you know what you're doing for "
                          "now, but see the calibration.estimator documentation for details")

            self._temperature_dependent_alignment_est = val

    # ____________________________________________________METHODS________________________________________________

    def estimate_calibration(self) -> None:
        """
        This method estimates an updated camera model using all stars identified in all images that are turned on.

        For each turned on image in the :attr:`camera` attribute, this method provides the :attr:`calibration_estimator`
        with the :attr:`matched_extracted_image_points`, the :attr:`matched_catalogue_unit_vectors_camera`, and
        optionally the :attr:`matched_weights_picture` if :attr:`use_weights` is ``True``. The
        :meth:`~.CalibrationEstimator.estimate` method is then called and the resulting updated camera model is stored
        in the :attr:`model` attribute.  Finally, the updated camera model is used to update the following:

        * :attr:`matched_catalogue_image_points`
        * :attr:`queried_catalogue_image_points`
        * :attr:`unmatched_catalogue_image_points`

        For a more thorough description of the calibration estimation routines see the :mod:`.calibration.estimators`
        documentation.

        .. warning::
            This method overwrites the camera model information in the :attr:`camera` attribute and
            does not save old information anywhere.  If you want this information saved be sure to store it yourself.
        """
        # reset things to make sure we don't mix information
        self._calibration_est.reset()

        # prepare the inputs
        use_pois = [[[], []]]*len(self.camera.images)
        use_vecs = [[[], [], []]]*len(self.camera.images)
        big_weights = [[[], []]]*len(self.camera.images)
        temperatures = [0]*len(self.camera.images)
        for ind, image in self.camera:
            pois = self._matched_extracted_image_points[ind]
            if pois is not None:
                use_pois[ind] = pois
            vecs = self._matched_catalogue_unit_vectors_camera[ind]
            if vecs is not None:
                use_vecs[ind] = vecs
            if self.use_weights:
                weights = self._matched_weights_picture[ind]
                if weights is not None:
                    big_weights[ind] = weights

            temperatures[ind] = image.temperature

        # update the attributes for the calibration estimator
        if self.use_weights:
            big_weights = np.diag(np.concatenate(big_weights).ravel())
            self._calibration_est.weighted_estimation = True
            self._calibration_est.measurement_covariance = big_weights
        else:
            self._calibration_est.weighted_estimation = False

        self._calibration_est.measurements = np.concatenate(use_pois, axis=1)

        self._calibration_est.camera_frame_directions = use_vecs

        self._calibration_est.temperatures = temperatures

        self._calibration_est.model = self.model.copy()

        # do the estimation
        self._calibration_est.estimate()

        # store the updated camera model
        self._camera.model.overwrite(self._calibration_est.model)

        # update the catalogue locations
        self.reproject_stars()

    def estimate_static_alignment(self) -> None:
        """
        This method estimates a static (not temeprature dependent) alignment between a base frame and the camera frame
        over multiple images.

        This method uses the :attr:`alignment_base_frame_func` to retrieve the rotation from the inertial frame to the
        base frame the alignment is to be done with respect to for each image time. The inertial matched catalogue unit
        vectors are then rotated into the base frame. Then, the matched image points-of-interest are converted to unit
        vectors in the camera frame. These 2 sets of unit vectors are then provided to the
        :attr:`static_alignment_estimator` and its :meth:`~.StaticAlignmentEstimator.estimate` method is called to
        estimate the alignment between the frames.  The resulting alignment is stored in the :attr:`static_alignment`
        attribute.

        Note that to do alignment, the base frame and the camera frame should generally be fixed with respect to one
        another.  This means that you can't do alignment with respect to something like the inertial frame in general,
        unless your camera is magically fixed with respect to the inertial frame.

        Generally, this method should be called after you have estimated the geometric camera model, because the
        geometric camera model is used to convert the observed pixel locations in the image to unit vectors in the
        camera frame (using :meth:`~.CameraModel.pixels_to_unit`).

        .. Note::
            This method will attempt to account for misalignment estimated along with the camera model when performing
            the estimation; however, this is not recommended. Instead, once you have performed your camera model
            calibration, you should consider resetting the camera model misalignment to 0 and then calling
            :meth:`estimate_attitude` before a call to this function.
        """

        # prepare the inputs
        base_uvecs = []
        cam_uvecs = []

        for ind, image in self.camera:
            if self._matched_catalogue_unit_vectors_inertial[ind] is not None:

                # rotate the inertial catalogue directions into the base frame
                rot_inertial2base = self.alignment_base_frame_func(image.observation_date)

                base_uvecs.append(np.matmul(rot_inertial2base.matrix,
                                            self._matched_catalogue_unit_vectors_inertial[ind]))

                # get the unit vectors in the camera frame using the camera model
                cam_uvecs.append(self.model.pixels_to_unit(self._matched_extracted_image_points[ind],
                                                           temperature=image.temperature, image=ind))

        self._static_alignment_est.frame1_unit_vecs = base_uvecs
        self._static_alignment_est.frame2_unit_vecs = cam_uvecs

        # do the static alignment
        self._static_alignment_est.estimate()

        # store the results
        self.static_alignment = self._static_alignment_est.alignment

    def estimate_temperature_dependent_alignment(self) -> None:
        """
        This method estimates a temperature dependent (not static) alignment between a base frame and the camera frame
        over multiple images.

        This method uses the :attr:`alignment_base_frame_func` to retrieve the rotation from the inertial frame to the
        base frame the alignment is to be done with respect to for each image time. Then, the rotation from the
        inertial frame to the camera frame is retrieved for each image from the
        :attr:`.Image.rotation_inertial_to_camera` attribute for each image (which is updated by a call to
        :meth:`estimate_attitude`).  These frame definitions are then provided to the
        :attr:`temperature_dependent_alignment_estimator` whose
        :meth:`~.TemperatureDependentAlignmentEstimator.estimate` method is then called to estimate the temperature
        dependent alignment.  The estimated alignment is then stored as a 3x2 numpy array where the first column is the
        static offset for the alignment, the second column is the temperature dependent slope, and each row represents
        the euler angle according to the requested order (so if the requested order is ``'xyx'`` then the rotation from
        the base frame to the camera frame at temperature ``t`` can be computed using:

            >>> from giant.rotations import euler_to_rotmat, Rotation
            >>> from giant.calibration.calibration_class import Calibration
            >>> cal = Calibration()
            >>> cal.estimate_temperature_dependent_alignment()
            >>> t = -22.5
            >>> angles = cal.temperature_dependent_alignment@[1, t]
            >>> order = cal.temperature_dependent_alignment_estimator.order
            >>> rotation_base_to_camera = Rotation(euler_to_rotmat(angles, order))

        This example is obviously incomplete but gives the concept of how things could be used.

        Note that to do alignment, the base frame and the camera frame should generally be fixed with respect to one
        another (with the exception of small variations with temperature).  This means that you can't do alignment with
        respect to something like the inertial frame in general, unless your camera is magically fixed with respect to
        the inertial frame.

        Generally, this method should be called after you have estimated the attitude for each image, because the
        estimated image pointing is used to estimate the alignment.  As such, only images where there are successfully
        matched stars are used in the estimation.

        .. Note::
            This method will attempt to account for misalignment estimated along with the camera model when performing
            the estimation; however, this is not recommended. Instead, once you have performed your camera model
            calibration, you should consider resetting the camera model misalignment to 0 and then calling
            :meth:`estimate_attitude` before a call to this function.
        """

        # prepare the inputs
        base_frame_rotations = []
        camera_frame_rotations = []
        temperatures = []

        for ind, image in self.camera:
            # only consider images where we have matched stars
            if self._matched_catalogue_unit_vectors_inertial[ind] is not None:

                # rotate the inertial catalogue directions into the base frame
                base_frame_rotations.append(self.alignment_base_frame_func(image.observation_date))

                # get the unit vectors in the camera frame using the camera model
                camera_rotation = image.rotation_inertial_to_camera
                # handle the misalignment if it exists
                if hasattr(self.camera.model, 'get_misalignment'):
                    camera_rotation = self.camera.model.get_misalignment(ind)*camera_rotation

                camera_frame_rotations.append(camera_rotation)

                temperatures.append(image.temperature)

        self._temperature_dependent_alignment_est.frame_1_rotations = base_frame_rotations
        self._temperature_dependent_alignment_est.frame_2_rotations = camera_frame_rotations
        self._temperature_dependent_alignment_est.temperatures = temperatures

        # do the static alignment
        self._temperature_dependent_alignment_est.estimate()

        # store the results
        self.temperature_dependent_alignment = np.array(
            [[self._temperature_dependent_alignment_est.angle_m_offset,
              self._temperature_dependent_alignment_est.angle_m_slope],
             [self._temperature_dependent_alignment_est.angle_n_offset,
              self._temperature_dependent_alignment_est.angle_n_slope],
             [self._temperature_dependent_alignment_est.angle_p_offset,
              self._temperature_dependent_alignment_est.angle_p_slope]
             ]
        )

    def reset_calibration_estimator(self):
        """
        This method resets the existing calibration estimator instance with a new instance using the initial
        ``calibration_estimator_update`` argument passed to the constructor.

        A new instance of the object is created, therefore there is no backwards reference whatsoever to the state
        before a call to this method.
        """
        if self._initial_calibration_est_kwargs is not None:
            self._calibration_est = self._initial_calibration_est(self._camera.model,
                                                                  **self._initial_calibration_est_kwargs)
        else:
            self._calibration_est = self._initial_calibration_est(self._camera.model)

    def update_calibration_estimator(self, calibration_estimator_update: Optional[dict] = None):
        """
        This method updates the attributes of the :attr:`calibration_estimator` attribute.

        See the :mod:`.calibration.estimators` documentation for accepted attribute values.

        If a supplied attribute is not found in the :attr:`calibration_estimator` attribute then this will print a
        warning and ignore the attribute. Any attributes that are not supplied are left alone.

        :param calibration_estimator_update: A dictionary of attribute->value pairs to update the
                                             :attr:`calibration_estimator` attribute with
        """
        if calibration_estimator_update is not None:
            for key, val in calibration_estimator_update.items():
                if hasattr(self._calibration_est, key):
                    setattr(self._calibration_est, key, val)
                else:
                    warnings.warn("The attribute {0} was not found.\n"
                                  "Cannot update calibration estimation instance".format(key))

    def reset_static_alignment_estimator(self):
        """
        This method replaces the existing static alignment estimator instance with a new instance
        using the initial ``static_alignment_estimator_kwargs`` argument passed to the constructor.

        A new instance of the object is created, therefore there is no backwards reference whatsoever to the state
        before a call to this method.
        """
        if self._initial_static_alignment_est_kwargs is not None:
            self._static_alignment_est = self._initial_static_alignment_est(**self._initial_static_alignment_est_kwargs)
        else:
            self._static_alignment_est = self._initial_static_alignment_est()

    def update_static_alignment_estimator(self, alignment_estimator_update: Optional[dict] = None):
        """
        This method updates the attributes of the :attr:`static_alignment_estimator` attribute.

        See the :mod:`.calibration.estimators` documentation for accepted attribute values.

        If a supplied attribute is not found in the :attr:`static_alignment_estimator` attribute then this will print a
        warning and ignore the attribute. Any attributes that are not supplied are left alone.

        :param alignment_estimator_update: A dictionary of attribute->value pairs to update the
                                           :attr:`static_alignment_estimator` attribute with
        """

        if alignment_estimator_update is not None:
            for key, val in alignment_estimator_update.items():
                if hasattr(self._static_alignment_est, key):
                    setattr(self._static_alignment_est, key, val)
                else:
                    warnings.warn("The attribute {0} was not found.\n"
                                  "Cannot update static alignment estimation instance".format(key))

    def reset_temperature_dependent_alignment_estimator(self):
        """
        This method replaces the existing temperature_dependent_alignment estimator instance with a new instance
        using the initial ``temperature_dependent_alignment_estimator_kwargs`` argument passed to the constructor.

        A new instance of the object is created, therefore there is no backwards reference whatsoever to the state
        before a call to this method.
        """
        if self._initial_temperature_dependent_alignment_est_kwargs is not None:
            self._temperature_dependent_alignment_est = self._initial_temperature_dependent_alignment_est(
                **self._initial_temperature_dependent_alignment_est_kwargs
            )
        else:
            self._temperature_dependent_alignment_est = self._initial_temperature_dependent_alignment_est()

    def update_temperature_dependent_alignment_estimator(self,
                                                         temperature_dependent_alignment_estimator_update:
                                                         Optional[dict] = None):
        """
        This method updates the attributes of the :attr:`temperature_dependent_alignment_estimator` attribute.

        See the :mod:`.calibration.estimators` documentation for accepted attribute values.

        If a supplied attribute is not found in the :attr:`temperature_dependent_alignment_estimator` attribute then 
        this will print a warning and ignore the attribute. Any attributes that are not supplied are left alone.

        :param temperature_dependent_alignment_estimator_update: A dictionary of attribute->value pairs to update the
                                                                 :attr:`temperature_dependent_alignment_estimator` 
                                                                 attribute with
        """

        if temperature_dependent_alignment_estimator_update is not None:
            for key, val in temperature_dependent_alignment_estimator_update.items():
                if hasattr(self._temperature_dependent_alignment_est, key):
                    setattr(self._temperature_dependent_alignment_est, key, val)
                else:
                    warnings.warn("The attribute {0} was not found.\n"
                                  "Cannot update temperature_dependent_alignment estimation instance".format(key))

    def reset_settings(self):
        """
        This method resets all settings to their initially provided values (at class construction)

        Specifically, the following are reset

        * :attr:`star_id`
        * :attr:`image_processing`
        * :attr:`attitude_estimator`
        * :attr:`calibration_estimator`
        * :attr:`static_alignment_estimator`
        * :attr:`temperature_dependent_alignment_estimator`

        In each case, a new instance of the object is created supplying the corresponding ``_kwargs`` argument supplied
        when this class what initialized.

        This is simply a shortcut to calling the ``reset_XXX``` methods individually.
        """
        self.reset_star_id()
        self.reset_image_processing()
        self.reset_attitude_estimator()
        self.reset_calibration_estimator()
        self.reset_static_alignment_estimator()
        self.reset_temperature_dependent_alignment_estimator()

    def update_settings(self, star_id_update: Optional[dict] = None,
                        image_processing_update: Optional[dict] = None,
                        attitude_estimator_update: Optional[dict] = None,
                        calibration_estimator_update: Optional[dict] = None,
                        static_alignment_estimator_update: Optional[dict] = None,
                        temperature_dependent_alignment_estimator_update: Optional[dict] = None):
        """
        This method updates all settings to their provided values

        Specifically, the following are updated depending on the input

        * :attr:`star_id`
        * :attr:`image_processing`
        * :attr:`attitude_estimator`
        * :attr:`calibration_estimator`
        * :attr:`static_alignment_estimator`

        In each case, the existing instance is modified in place with the attributes provided.  Any attributes that are
        not specified are left as is.

        This is simply a shortcut to calling the ``update_XXX`` methods individually.

        :param star_id_update: The updates to :attr:`star_id`.
        :param attitude_estimator_update: The updates to :attr:`attitude_estimator`.
        :param image_processing_update: The updates to :attr:`image_processing`.
        :param calibration_estimator_update: The updates to :attr:`calibration_estimator`.
        :param static_alignment_estimator_update: The updates to :attr:`static_alignment_estimator`.
        :param temperature_dependent_alignment_estimator_update: The updates to
                                                                 :attr:`temperature_dependent_alignment_estimator`.
        """
        self.update_star_id(star_id_update)
        self.update_image_processing(image_processing_update)
        self.update_attitude_estimator(attitude_estimator_update)
        self.update_calibration_estimator(calibration_estimator_update)
        self.update_static_alignment_estimator(static_alignment_estimator_update)
        self.update_temperature_dependent_alignment_estimator(temperature_dependent_alignment_estimator_update)

    def calib_summary(self, measurement_covariance: Optional[SCALAR_OR_ARRAY] = None):
        """
        This prints a summary of the results of calibration to the screen

        The resulting summary displays the labeled covariance matrix, followed by the labeled correlation coefficients,
        followed by the state parameters and their formal uncertainty.

        One optional inputs can be used to specify the uncertainty on the measurements if weighted estimation wasn't
        already used to ensure the post-fit covariance
        has the proper scaling.

        Note that if multiple misalignments were estimated in the calibration, only the first is printed in the
        correlation and covariance matrices.  For all misalignments, the values are replaced with NaN.

        :param measurement_covariance: The covariance for the measurements either as a nxn matrix or as a scalar.
        """

        if measurement_covariance is not None:
            self._calibration_est.weighted_estimation = True
            self._calibration_est.measurement_covariance = measurement_covariance
            covariance = self.calibration_estimator.postfit_covariance
        else:
            covariance = self.calibration_estimator.postfit_covariance

        # get the uncertainty for each parameter
        sigmas = np.sqrt(np.diag(covariance))

        # compute the correlation coefficients
        coefficients = covariance / np.outer(*[np.sqrt(np.diag(covariance))] * 2)

        # get the labels for each element
        labels = self.model.get_state_labels()  # type: list

        # if misalignment is in labels we need to do some fancy manipulation
        if 'misalignment' in labels:
            labels.remove('misalignment')
            labels.append('align_x')
            labels.append('align_y')
            labels.append('align_z')

        # print the covariance matrix
        print('Covariance:')
        _print_lr(labels, covariance)

        # print the correlation coefficient matrix and get the maximum label size
        print('Correlation coefficients:')
        max_label = _print_lr(labels, coefficients)

        # build the format strings for the parameter/formal uncertainty pairs
        label_format = '{:<' + str(max_label) + 's}'
        number_format = '{:>' + str(max_label) + '.' + str(max_label - 7) + 'e}'
        fmt = label_format + ' ' + number_format + ' ' + number_format
        state_vector = self.model.state_vector
        print('Parameter value and formal uncertainty:')
        for ind, label in enumerate(labels):
            if 'align' not in labels:
                print(fmt.format(label, state_vector[ind], sigmas[ind]))
            else:
                print(fmt.format(label, np.nan, sigmas[ind]))

    def limit_magnitude(self, min_magnitude: float, max_magnitude: float, in_place=False) -> 'Calibration':
        """
        This method removes stars from the ``matched_...`` attributes that are not within the provided magnitude bounds.

        This method should be used rarely, as you can typically achieve the same functionality by use the
        :attr:`.StarID.max_magnitude` and :attr:`.StarID.min_magnitude` attributes before calling :meth:`id_stars`.  The
        most typical use case for this method is when you have already completed a full calibration and you now either
        want to filter out some of the stars for plotting purposes, or you want to filter out some of the stars to do an
        alignment analysis, where it is generally better to use only well exposed stars since fewer are needed to
        fully define the alignment.

        When you use this method, by default it will edit and return a copy of the current instance to preserve the
        current instance.  if you are using many images with many stars in them this can use a large amount of memory;
        however, so you can optionally specify ``in_place=True`` to modify the current instance in place.  Note however
        that this not a reversible operation (that is you cannot get back to the original state) so be cautious about
        using this option.

        :param min_magnitude: The minimum star magnitude to accept (recall that minimum magnitude limits the brightest
                              stars)
        :param max_magnitude: The maximum star magnitude to accept (recall that maximum magnitude limits the dimmest
                              stars)
        :param in_place: A flag specifying whether to work on a copy or the original
        :return: The edited Calibration instance (either a copy or a reference)
        """

        if in_place:
            out = self
        else:
            out = deepcopy(self)

        for ind, _ in out.camera:
            # test which stars don't meet the requirements
            mag_test = (out.matched_catalogue_star_records[ind].mag.values >= max_magnitude) | \
                       (out.matched_catalogue_star_records[ind].mag.values <= min_magnitude)

            if mag_test.any():
                indicies = np.argwhere(mag_test).ravel()

                out.remove_matched_stars(ind, indicies)

        return out
