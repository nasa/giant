


r"""
This module provides a subclass of the :class:`.OpNav` class for performing stellar OpNav and camera calibration.

Interface Description
---------------------

In GIANT, calibration refers primarily to the process of estimating a model to map points in the 3D
world to the observed points in a 2D image.  This is done by estimating both a geometric (intrinsic)
:mod:`camera model <.camera_models>` along with an optional pointing alignment between the camera frame and the base
frame the knowledge of the camera attitude is tied to (for instance the spacecraft bus frame).  For both of these, we
use observations of stars to get highly accurate models.

The :class:`Calibration` class is the main interface for performing calibration in GIANT, and in general is all the user
will need to interact with.  It is a subclass of the :class:`.StellarOpNav` class and as such provides a very similar
interface with only a few additional features. It provides direct access to the :class:`.PointOfInterestFinder`,
:class:`.StarID`, :mod:`.stellar_opnav.estimators` objects, and :mod:`.calibration.estimators` objects and automatically
preforms the required data transfer between the objects for you.  To begin you simply provide the :class:`.Calibration`
constructor a :class:`.Camera` instance and a :class:`.CalibrationOptions` instance to configure the class with. You 
can then use the :class:`Calibration` instance to perform all of the aspects of stellar OpNav and calibration with never 
having to interact with the internal objects again.

For example, we could do something like the following (from the directory containing ``sample_data``):

    >>> import pickle
    >>> from giant.calibration import Calibration, CalibrationOptions
    >>> from giant.rotations import Rotation
    >>> with open('sample_data/camera.pickle', 'rb') as ifile:
    ...     camera = pickle.load(ifile)
    >>> # Returns the identity to signify the base frame is the inertial frame
    >>> def base_frame(*args):
    ...     return Rotation([0, 0, 0, 1])
    >>> cal_opts = CalibrationOptions
    >>> cal_opts.alignment_base_frame_func = base_frame
    >>> cal = Calibration(camera, options=cal_opts)
    >>> cal.id_stars()  # id the stars for each image
    >>> cal.sid_summary()  # print out a summary of the star identification success for each image
    >>> cal.estimate_attitude()  # estimate an updated attitude for each image
    >>> cal.estimate_geometric_calibration()  # estimate an updated camera model
    >>> cal.geometric_calibration_summary()  # print out a summary of the star identification success for each image
    >>> cal.estimate_static_alignment()  # estimate the alignment between the camera frame and hte base frame
    >>> cal.estimate_temperature_dependent_alignment()  # estimate the temperature dependent alignment

For a more general description of the steps needed to perform calibration, refer to the :mod:`.calibration` package.
For a more in-depth examination of the :class:`Calibration` class see the following API Reference.
"""

from copy import deepcopy

from typing import Optional, Sequence, Callable, Self

from dataclasses import dataclass

import numpy as np

from giant.calibration import estimators as est
from giant.stellar_opnav.stellar_class import StellarOpNav, StellarOpNavOptions
from giant.camera import Camera
from giant.rotations import Rotation

from giant.utilities.print_llt import print_llt
from giant._typing import DatetimeLike, EULER_ORDERS, DOUBLE_ARRAY


@dataclass
class CalibrationOptions(StellarOpNavOptions):
    
    alignment_base_frame_func: Callable[[DatetimeLike], Rotation] | None = None
    """
    A callable object which returns the orientation of the base frame with respect  
    to the inertial frame the alignment of the camera frame is to be done with      
    respect to for a given date.                                                    
    
    This is used on calls to :meth:`estimate_static_alignment` and :meth`estimate_temperature_dependent_alignment` 
    to determine the base frame the alignment is being done with respect to.  Typically this returns something like 
    the spacecraft body frame with respect to the inertial frame (inertial to spacecraft body) or another camera 
    frame.
    """
    
    temperature_dependent_alignment_euler_order: EULER_ORDERS = "xyz"
    """
    The order of euler angles to use for the temperature dependent alignment.
    """
    
    geometric_estimator_options: est.GeometricEstimatorOptions | est.LMAEstimatorOptions | est.IterativeNonlinearLstSqOptions | None = None
    """
    The options to use to initialize the geometric estimator
    """
    
    custom_geometric_estimator_class: type[est.GeometricEstimator] | None = None
    """
    A custom geometric camera model estimator class to use instead of one of the standard ones
    """
    
    geometric_estimator_type: est.geometric.GeometricEstimatorImplementations = est.geometric.GeometricEstimatorImplementations.ITERATIVE_NONLINEAR_LSTSQ
    """
    Which geometric estimator to use.
    
    For a custom implementation choose CUSTOM
    """


class Calibration(StellarOpNav, CalibrationOptions):
    """
    This class serves as the main user interface for performing geometric camera calibration and camera frame attitude
    alignment.

    The class acts as a container for the :class:`.Camera`, :class:`.PointOfInterestFinder`, 
    :mod:`.stellar_opnav.estimators`, :mod:`.calibration.estimators`, and :class:`.StarId` objects and also passes the 
    correct and up-to-date data from one object to the other. In general, this class will be the exclusive interface to 
    the mentioned objects and models for the user.

    This class provides a number of features that make doing stellar OpNav and camera calibration/alignment easy.  The
    first is it provides aliases to the image processing, star id, attitude estimation, calibration estimation, and
    alignment estimation objects. These aliases make it easy to quickly change/update the various tuning parameters that
    are necessary to make star identification and calibration a success. In addition to providing convenient access to
    the underlying settings, some of these aliases also update internal flags that specify whether individual images
    need to be reprocessed, saving computation time when you're trying to find the best tuning.

    This class also provides simple methods for performing star identification, attitude estimation, camera calibration,
    and aligment estimation after you have set the tuning parameters. These methods (:meth:`id_stars`,
    :meth:`sid_summary`, :meth:`estimate_attitude`, :meth:`estimate_geometric_calibration`, :meth:`geometric_calibration_summary`,
    :meth:`estimate_static_alignment`, and :meth:`estimate_temperature_dependent_alignment`) combine all of the
    required steps into a few simple calls, and pass the resulting data from one object to the next. They also store off
    the results of the star identification in the :attr:`queried_catalog_star_records`,
    :attr:`queried_catalog_image_points`, :attr:`queried_catalog_unit_vectors`, :attr:`extracted_image_points`,
    :attr:`extracted_image_illums`, :attr:`extracted_psfs`, :attr:`extracted_stats`, :attr:`extracted_snrs`,
    :attr:`unmatched_catalog_image_points`, :attr:`unmatched_image_illums`,
    :attr:`unmatched_psfs`, :attr:`unmatched_stats`, :attr:`unmatched_snrs`
    :attr:`unmatched_catalog_star_records`,
    :attr:`unmatched_catalog_unit_vectors`,
    :attr:`unmatched_extracted_image_points`,
    :attr:`matched_catalog_image_points`, :attr:`matched_image_illums`,
    :attr:`matched_psfs`, :attr:`matched_stats`, :attr:`matched_snrs`
    :attr:`matched_catalog_star_records`,
    :attr:`matched_catalog_unit_vectors_inertial`,
    :attr:`matched_catalog_unit_vectors_camera`, and
    :attr:`matched_extracted_image_points` attributes, enabling more advanced analysis to be performed external to the
    class.

    This class stores the updated attitude solutions in the image objects themselves, allowing you to directly
    pass your images from stellar OpNav to the :mod:`.relative_opnav` routines with updated attitude solutions. It also
    stores the estimated camera model in the original camera model itself, and store the estimated alignments in the
    :attr:`static_alignment` and :attr:`temperature_dependent_alignment` attributes. Finally, this class
    respects the :attr:`.image_mask` attribute of the :class:`.Camera` object, only considering images that are
    currently turned on.
    """

    def __init__(self, camera: Camera, options: CalibrationOptions | None = None):
        """
        :param camera: The :class:`.Camera` object containing the camera model and images to be utilized
        :param options: The options dataclass to use to configure
        """
        
        if options is None:
            # need to do this because of how the base class is set up
            options = CalibrationOptions()

        # initialize the StellarOpNav super class
        super().__init__(camera, options=options)
        
        if self.geometric_estimator_type is not est.geometric.GeometricEstimatorImplementations.CUSTOM:
            self._geometric_estimator = est.geometric.get_estimator(self.geometric_estimator_type, camera.model, self.geometric_estimator_options)
        else:
            assert self.custom_geometric_estimator_class is not None, "The custom_geometric_estimator_class must not be None if CUSTOM is chosen as the estimator type"
            self._geometric_estimator = self.custom_geometric_estimator_class(camera.model, self.geometric_estimator_options)
            
        self.static_alignment: Optional[Rotation] = None
        """
        The estimated static alignment. 
        
        This is set to None until :meth:`estimate_static_alignment` is called
        """
        
        self.temperature_dependent_alignment: Optional[est.TemperatureDependentResults] = None
        """
        The estimated temperature dependent alignment. 
        
        This is set to None until :meth:`estimate_temperature_dependent_alignment` is called
        """
        

    # update the model setter to also update the sid model
    @StellarOpNav.model.setter
    def model(self, val):
        # dispatch to the super setter
        super().model.__set__(val) # type: ignore
        self._geometric_estimator.model = val

    @property
    def geometric_estimator(self) -> est.GeometricEstimator:
        """
        The estimator to use when estimating the geometric calibration

        This must implement the :class:`.GeometricEstimator` interface

        See the :mod:`~.calibration.estimators` documentation for more details.
        """

        return self._geometric_estimator

    @geometric_estimator.setter
    def geometric_estimator(self, val: est.GeometricEstimator):
        if not isinstance(val, est.GeometricEstimator):
            raise TypeError("The geometric_estimator object must implement the GeometricEstimator interface")

        self._geometric_estimator = val

    # ____________________________________________________METHODS________________________________________________

    def estimate_geometric_calibration(self) -> None:
        """
        This method estimates an updated camera model using all stars identified in all images that are turned on.

        For each turned on image in the :attr:`camera` attribute, this method provides the :attr:`geometric_estimator`
        with the :attr:`matched_extracted_image_points`, the :attr:`matched_catalog_unit_vectors_camera`, and
        optionally the :attr:`matched_weights_picture` if :attr:`use_weights` is ``True``. The
        :meth:`~.GeometricEstimator.estimate` method is then called and the resulting updated camera model is stored
        in the :attr:`model` attribute.  Finally, the updated camera model is used to update the following:

        * :attr:`matched_catalog_image_points`
        * :attr:`queried_catalog_image_points`
        * :attr:`unmatched_catalog_image_points`

        For a more thorough description of the calibration estimation routines see the :mod:`.calibration.estimators.geometric`
        documentation.

        .. warning::
            This method overwrites the camera model information in the :attr:`camera` attribute and
            does not save old information anywhere.  If you want this information saved be sure to store it yourself.
        """
        # reset things to make sure we don't mix information
        self._geometric_estimator.reset()

        # prepare the inputs
        use_pois: list[DOUBLE_ARRAY | list[list]] = [[[], []]]*len(self.camera.images)
        use_vecs: list[DOUBLE_ARRAY | list[list]] = [[[], [], []]]*len(self.camera.images)
        l_big_weights: list[DOUBLE_ARRAY | list[list]] = [[[], []]]*len(self.camera.images)
        temperatures: list[float] = [0.0]*len(self.camera.images)
        for ind, image in self.camera:
            pois = self._matched_extracted_image_points[ind]
            if pois is not None:
                use_pois[ind] = pois
            vecs = self._matched_catalog_unit_vectors_camera[ind]
            if vecs is not None:
                use_vecs[ind] = vecs
            if self.use_weights:
                weights = self._matched_weights_picture[ind]
                if weights is not None:
                    l_big_weights[ind] = weights

            temperatures[ind] = image.temperature

        # update the attributes for the geometric estimator
        if self.use_weights:
            big_weights = np.diag(np.concatenate(l_big_weights).ravel())
            self._geometric_estimator.weighted_estimation = True
            self._geometric_estimator.measurement_covariance = big_weights
        else:
            self._geometric_estimator.weighted_estimation = False

        self._geometric_estimator.measurements = np.concatenate(use_pois, axis=1)

        self._geometric_estimator.camera_frame_directions = use_vecs

        self._geometric_estimator.temperatures = temperatures

        self._geometric_estimator.model = self.model.copy()

        # do the estimation
        self._geometric_estimator.estimate()

        # store the updated camera model
        self._camera.model.overwrite(self._geometric_estimator.model)

        # update the catalog locations
        self.reproject_stars()

    def estimate_static_alignment(self) -> Rotation:
        """
        This method estimates a static (not temeprature dependent) alignment between a base frame and the camera frame
        over multiple images.

        This method uses the :attr:`alignment_base_frame_func` to retrieve the rotation from the inertial frame to the
        base frame the alignment is to be done with respect to for each image time. The inertial matched catalog unit
        vectors are then rotated into the base frame. Then, the matched image points-of-interest are converted to unit
        vectors in the camera frame. These 2 sets of unit vectors are then provided to the
        :func:`.static_alignment_estimator` optionally along with weights, to estimate the alignment between the frames.  
        The resulting alignment is returned as a :class:`.Rotation` object and assigned to the :attr:`static_alignment` 
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
            
        :return: The static alignment results as a :class:`.Rotation` from the base frame to the camera frame
        """

        # prepare the inputs
        base_uvecs = []
        cam_uvecs = []
        weights: list[float] = []
        
        assert self.alignment_base_frame_func is not None, "the alignment base frame function must be specified to perform static alignment"

        for ind, image in self.camera:
            if (inertial_vecs := self._matched_catalog_unit_vectors_inertial[ind]) is not None:

                # rotate the inertial catalog directions into the base frame
                rot_inertial2base = self.alignment_base_frame_func(image.observation_date)

                base_uvecs.append(rot_inertial2base.matrix @ inertial_vecs)

                # get the unit vectors in the camera frame using the camera model
                assert (image_points := self._matched_extracted_image_points[ind]) is not None, "The image points shouldn't be None if the unit vectors aren't None"
                cam_uvecs.append(self.model.pixels_to_unit(image_points, temperature=image.temperature, image=ind))
                
                if self.use_weights:
                    assert (inertial_weights := self._matched_weights_inertial[ind]) is not None, "The weights shouldn't be None if use_weights is set to True"
                    # use the trace of the inertial weights
                    weights.append(np.trace(inertial_weights))
                    
        if not self.use_weights:
            provided_weights = None
        else:
            provided_weights = np.array(weights)

        # compute the results
        self.static_alignment = est.static_alignment_estimator(np.hstack(base_uvecs), np.hstack(cam_uvecs), provided_weights)
        
        return self.static_alignment.copy()

    def estimate_temperature_dependent_alignment(self) -> est.alignment.temperature_dependent.TemperatureDependentResults:
        """
        This method estimates a temperature dependent (not static) alignment between a base frame and the camera frame
        over multiple images.

        This method uses the :attr:`alignment_base_frame_func` to retrieve the rotation from the inertial frame to the
        base frame the alignment is to be done with respect to for each image time. Then, the rotation from the
        inertial frame to the camera frame is retrieved for each image from the
        :attr:`.Image.rotation_inertial_to_camera` attribute for each image (which is updated by a call to
        :meth:`estimate_attitude`).  These frame definitions are then provided to the
        :func:`.temperature_dependent_alignment_estimator` to estimate the temperature
        dependent alignment.  The estimated alignment is returned and can be queried for a specific temeprature
        using :func:`.evaluate_temperature_dependent_alignment`.  It is also stored in the 
        :attr:`temperature_dependent_alignment` attribute.
        
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
            
        :return: The temeprature dependent alignment as linear (in temperature) euler angle equations in radians to go from
                 the base frame to the camera frame 
        """
        assert self.alignment_base_frame_func is not None, "the alignment base frame function must be specified to perform temperature_dependent_alignment"

        # prepare the inputs
        base_frame_rotations: list[Rotation] = []
        camera_frame_rotations: list[Rotation] = []
        temperatures: list[float] = []

        for ind, image in self.camera:
            # only consider images where we have matched stars
            if self._matched_catalog_unit_vectors_inertial[ind] is not None:

                # rotate the inertial catalog directions into the base frame
                base_frame_rotations.append(self.alignment_base_frame_func(image.observation_date))

                # get the unit vectors in the camera frame using the camera model
                camera_rotation = image.rotation_inertial_to_camera
                # handle the misalignment if it exists
                if hasattr(self.camera.model, 'get_misalignment'):
                    camera_rotation = self.camera.model.get_misalignment(ind)*camera_rotation # type: ignore

                camera_frame_rotations.append(camera_rotation)

                temperatures.append(image.temperature)
                
        self.temperature_dependent_alignment = est.temperature_dependent_alignment_estimator(base_frame_rotations, camera_frame_rotations, temperatures, self.temperature_dependent_alignment_euler_order)
        
        return self.temperature_dependent_alignment


    def reset_geometric_estimator(self):
        """
        This method resets the existing calibration estimator instance with a new instance using the initial
        settings provided.
        """
        
        self._geometric_estimator.reset_settings()

    def update_geometric_estimator(self, geometric_estimator_update: Optional[est.GeometricEstimatorOptions] = None):
        """
        This method updates the attributes of the :attr:`geometric_estimator` attribute.
        
        Note that to update only specific settings it is best to initialize the settings structure with 
        `update = cal.geometric_estimator.original_options.copy()` and then modify the specific attributes

        :param geometric_estimator_update: An instance of the GeometricEstimatorOptions or a subclass to update settings with
        """
        
        if geometric_estimator_update is not None:
            geometric_estimator_update.apply_options(self._geometric_estimator)

    def reset_settings(self):
        """
        This method resets all settings to their initially provided values (at class construction)

        Specifically, the following are reset

        * :attr:`star_id`
        * :attr:`point_of_interest_finder`
        * :attr:`attitude_estimator`
        * :attr:`geometric_estimator`
        
        along with direct attributes of the class

        This is simply a shortcut to calling the ``reset_XXX``` methods individually.
        """
        super().reset_settings()
        self.reset_star_id()
        self.reset_attitude_estimator()
        self.reset_geometric_estimator()
        self.reset_point_of_interest_finder()

    def geometric_calibration_summary(self, measurement_covariance: Optional[float | DOUBLE_ARRAY] = None):
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
            self._geometric_estimator.weighted_estimation = True
            self._geometric_estimator.measurement_covariance = measurement_covariance
            covariance = self.geometric_estimator.postfit_covariance
        else:
            covariance = self.geometric_estimator.postfit_covariance
            
        assert covariance is not None, "estimate_geometric_calibration must be called before geometric_calibration_summary"

        # get the uncertainty for each parameter
        sigmas = np.sqrt(np.diag(covariance))

        # compute the correlation coefficients
        coefficients = covariance / np.outer(*[np.sqrt(np.diag(covariance))] * 2)

        # get the labels for each element
        labels: list = self.model.get_state_labels() 

        # if misalignment is in labels we need to do some fancy manipulation
        if 'misalignment' in labels:
            labels.remove('misalignment')
            labels.append('align_x')
            labels.append('align_y')
            labels.append('align_z')

        # print the covariance matrix
        print('Covariance:')
        print_llt(labels, covariance)

        # print the correlation coefficient matrix and get the maximum label size
        print('Correlation coefficients:')
        max_label = print_llt(labels, coefficients)

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

    def limit_magnitude(self, min_magnitude: float, max_magnitude: float, in_place=False) -> Self:
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
            if (matched_records := out.matched_catalog_star_records[ind]) is not None:
                mag_test = (matched_records.mag.to_numpy() >= max_magnitude) | \
                           (matched_records.mag.to_numpy() <= min_magnitude)

                if mag_test.any():
                    indicies = np.argwhere(mag_test).ravel()

                    out.remove_matched_stars(ind, indicies)

        return out
