

"""
This module provides photometric modelling for calculations in GIANT.

Description
-----------

In GIANT, a Photometry object is used to describe the photometric attributes of a target. These attributes are:
    - whether the target is resolved
    - apparent magnitude of the target
    - I over F of the target
    - DNrate of the target 
    - SNR of the target
    
The PhotometricCameraModel object is used to define camera parameters specifically used for photometry and planning.

Use
---

To use the Photometry class in GIANT you simply create :class:`.SceneObject` instances for any targets and the light source for
your scene (currently only a single light source is allowed).  You then create a :class:`.Scene` around these objects and input this
scene into the photometry class along with the associated PhotometryCameraModel. 

    ***Note: If using this class outside the ObservationPlanner class, the scene must be defined in the camera frame.

This then gives you the ability to determine the photometry of the target bodies relative to the camera frame. This determines
if a target is resolved, dnrate, and Signal-to-Noise ratio. 
"""

import numpy as np
from typing import Optional, Callable

from giant.camera_models.camera_model import CameraModel
from giant.ray_tracer.scene import Scene
from giant.photometry.magnitude import PhaseMagnitudeModel
from giant.photometry.scattered_light_class import ScatteredLight
from giant.photometry.utilities import au


class PhotometricCameraModel:
    """
    This class is used to set parameters for a Photometric Camera 
    
    Required inputs are used to estimate and plan photometric targeting with the camera. Additional parameters can be set to define the camera model
    """

    def __init__(self, gain: float, transfer_time: float,
                 standard_mag: float, bin_mode: int, resolved_rows_threshold: int,
                 dn_readnoise: float, dn_rate_standard: float, dark_current: float,
                 psf_factor: float, camera_model: CameraModel, name: str | None = None,
                 max_exposure: float = 2e5, **kwargs):
        """
        :param gain: A float representing the camera gain in dn
        :param transfer_time: A float representing the camera transfer time in seconds
        :param standard_mag: standard magnitude
        :param bin_mode: bin mode of camera
        :param coadd_num: coadded images 
        :param resolved_rows_threshold: number of pixel rows to define a target as resolved
        :param dn_readnoise: background dn from the camera noise
        :param dark_current: a float representing the dark current 
        :param dn_rate_standard: DN/s for I/F = 1 at 1AU
        :param psf_factor: factor for the point spread function
        :param camera_model: giant camera model
        :param name: a name for the camera model
        :param max_exposure: the maximum exposure time the camera can manage in seconds
        """
        self.gain = gain
        self.transfer_time = transfer_time
        self.standard_mag = standard_mag
        self.resolved_rows_threshold = resolved_rows_threshold
        self.bin_mode = bin_mode
        self.resolved_rows_threshold = resolved_rows_threshold
        self.dn_readnoise = dn_readnoise
        self.dn_rate_standard = dn_rate_standard
        self.dark_current = dark_current
        self.psf_factor = psf_factor
        self.camera_model = camera_model
        self.name = name
        self.max_exposure = max_exposure
        self.__dict__.update(**kwargs)

    @property
    def dnrate_dark(self):
        '''
        DN Rate due to dark current of the camera
        '''

        return self.dark_current / self.gain


class Photometry:
    """
    This is most useful when you would like to determine the required photometry inputs for opnavs given
    a certain date. In addition to the Scene methods, this class contains calculations for iof, dnrate, 
    snr, etc. This class can also determine target pariapse given an estimated time and move the scene 
    to that time, as well as determine if the target is resolved during the scene. 

    This is also useful because it can calculate the apparent location of objects in the scene (if they have the
    :attr:`.position_function` and :attr:`.orientation_function` attributes defined) while applying corrections for
    light time and stellar aberration.  This can be done for all objects in a scene using method :meth:`update` or for
    individual object using :meth:`calculate_apparent_position`.  Any objects that do not define the mentioned
    attributes will likely not be placed correctly in the scene when using these methods and thus warnings will be
    printed.
    """

    def __init__(self, scene: Scene,
                 photometric_camera_model: Optional[PhotometricCameraModel] = None,
                 scattered_light: Optional[ScatteredLight] = None):
        """
        :param scene: Scene containing at least one target and lighting object 
        :param photometric_camera_model: PhotometricCameraModel object
        :param scattered_light: scatteredLight object used to define how scattered light is handled. If none provided, 
                            scattered light is not taken into account
        """
        self.scene = scene
        """
        Scene containing at least one target and lighting object 
        """
        self.photometric_camera_model = photometric_camera_model
        """
        Photometric Camera model object used to define constants of the camera
        """

        self.scattered_light = scattered_light
        """
        scatteredLight object used to define how scattered light is handled
        """

        # preallocate the magnitude and I/f parameter to save magnitude per target object
        self.mag: list[None | float] = [None for x in range(len(self.scene.target_objs))]
        self.i_over_f: list[None | float] = [None for x in range(len(self.scene.target_objs))]

    def sea_angle(self, target_index: int) -> float:
        r"""
        This method computes the solar elongation angle defined as the angle between the :attr:`light_obj`, 
        the observer and the target at ``target_index``.

        The sea_angle is define as the interior angle between the vector from light object to the observer
        and the vector from the target to the observer.  The SEA angle is computed as the angular separation 
        of these two vectors in radians.

        This method assumes that the scene has already been put in the observer frame,
        that is, the observer is located at the origin and the :attr:`light_obj` and the target
        are also defined with respect to that origin.

        The SEA will always be between 0 and :math:`\pi` radians (0-180 degrees) TODO: CHECK IF TRUE

        :param target_index: the index into the :attr:`target_objs` list for which to compute the SEA angle for
        :return: the SEA angle in radians
        """
        
        assert self.scene.light_obj is not None
        vsep = np.arccos((np.dot(self.scene.light_obj.position, self.scene.target_objs[target_index].position)) /
                         (np.linalg.norm(self.scene.light_obj.position) * (
                             np.linalg.norm(self.scene.target_objs[target_index].position))))
        return vsep

    def resolved(self, target_index: int, photometric_camera_model: PhotometricCameraModel | None = None) -> bool:
        """
        This method determines if the the target at ``target_index`` is resolved by the camera.

        Whether a target is resolved is determined by the apparent diameter and position of the the target
        at ``target_index`` and the resolution of the camera.

        This method requires that a PhotometricCameraModel has been set as the :param:`photometric_camera_model` in 
        the scene, or one is provided when calling :meth:`resolved`. 

        :param target_index: the index into the :attr:`target_objs` list for which to compute the SEA angle for
        :param photometric_camera_model: PhotometricCameraModel used to determine if target is resolved
        :return: Boolean True/False for resolved/unresolved respectively
        """
        if photometric_camera_model is not None:  # override camera_model with new input
            self.photometric_camera_model = photometric_camera_model
            
        if self.photometric_camera_model is None:
            raise ValueError('the photometric camera model has not been provided')

        return self.scene.target_objs[target_index].get_apparent_diameter(
            self.photometric_camera_model.camera_model) > self.photometric_camera_model.resolved_rows_threshold

    def _fraction_rows(self, target_index: int, photometric_camera_model: PhotometricCameraModel | None = None):
        """
        This method determines what fraction of rows the target covers in the FOV.

        This parameter is only used internally for resolved targets

        This method assumes that a PhotometricCameraModel has been set as the :param:`photometric_camera_model` in 
        the scene, or one is provided when calling :meth:`resolved`. 

        :param target_index: the index into the :attr:`target_objs` list for which to compute the SEA angle for
        :param photometric_camera_model: PhotometricCameraModel used to determine if target is resolved
        :return: float representing the fraction of camera FOV covered by the target
        """
        if photometric_camera_model is not None:  # override camera_model with new input
            self.photometric_camera_model = photometric_camera_model

        if self.photometric_camera_model is None:
            raise ValueError('the photometric camera model has not been provided')
        
        return (self.scene.target_objs[target_index].get_apparent_diameter(
            self.photometric_camera_model.camera_model) / self.photometric_camera_model.camera_model.n_rows) * self.photometric_camera_model.bin_mode

    def apparent_mag(self, target_index: int, magnitude_model: PhaseMagnitudeModel) -> float:
        """
        This method determines the apparent magnitude of the the target at ``target_index`` given the 
        position during the scene.

        The magnitude is based on an input PhaseMagnitudeModel Class that calculates the magnitude of 
        the target at ``target_index`` based on the current phase angle at the scene, in radians.
         
        It is up to the user to determine the proper magnitude model to used based on if the target is resolved or
        unresolved.

        :param target_index: the index into the :attr:`target_objs` list for which to compute the phase angle for
        :param magnitude_model: magnitude_model used to determine apparent magnitude of target
        :return: apparent magnitude of target. 
        """
        m = magnitude_model.magnitude_function(self.scene, self.scene.target_objs[target_index])
        self.mag[target_index] = m
        return m

    def iof(self, target_index: int, luminosity_function: Callable[[int, 'Photometry'], float], **kwargs) -> float:
        """
        This method determines the Luminosity I/F of the the target at ``target_index`` given the 
        position during the scene.

        The I/F is based on a provided luminosity_function that calculates the I/F of  the target at ``target_index``
        based on the current phase angle of the target in the scene, in radians. The luminosity_function can be 
        dependent on other attributes of the :obj:`PhotometricScene`
        
        Additional kwargs can be input as key word arguments to the luminosity function. 

        :param target_index: the index into the :attr:`target_objs` list for which to compute the SEA angle for
        :param luminosity_function: Luminosity function used to determine IoverF of target
        :param kwargs: passed to the luminosity function
        :return: I/F
        """
        i_over_f = luminosity_function(target_index, self, **kwargs)
        self.i_over_f[target_index] = i_over_f  # unitless
        return i_over_f

    def _dnrate_unresolved(self, target_index: int) -> float:
        """
        This method determines the DN rate of the the target at ``target_index`` given its magnitude 
        and parameters of the photometric camera. 

        This calculation for DN rate should only be used for unresolved targets, the :meth:`dn_rate` method will 
        take this into account before calculating DN rate. This calculates the peak signal from the target.
        
        This method requires that the magnitude of the target has already been calculated using :meth:`apparent_mag`
        and that the :attr:`.photometric_camera_model` is not None.

        :param target_index: the index into the :attr:`target_objs` list for which to compute the magnitude for
        :return: DN rate of target in DN/s. 
        """
        assert self.photometric_camera_model is not None
        assert (use_mag := self.mag[target_index]) is not None
        rate = (10 ** ((self.photometric_camera_model.standard_mag - use_mag) / 2.5)) * self.photometric_camera_model.psf_factor
        return rate

    def _dnrate_resolved(self, target_index: int) -> float:
        """
        This method determines the DN rate of the the target at ``target_index`` given its I/F and 
        and parameters of the photometric camera. 

        This calculation for DN rate should only be used for resolved targets, the :meth:`dn_rate` method will 
        take this into account before calculating DN rate.
        
        This method requires that the I/F of the target has already been calculated using :meth:`apparent_mag`
        and that the :attr:`.photometric_camera_model` is not None.

        :param target_index: the index into the :attr:`target_objs` list for which to compute the magnitude for
        :return: DN rate of target in dn/s
        """
        assert self.photometric_camera_model is not None
        assert (use_iof := self.i_over_f[target_index]) is not None
        assert self.scene.light_obj is not None
        targ_sun_distance = float(np.linalg.norm(
            self.scene.target_objs[target_index].position - self.scene.light_obj.position))
        rate = self.photometric_camera_model.dn_rate_standard * (
                    use_iof / (au(targ_sun_distance) ** 2))  # scaling DNrate at 1AU
        return rate

    def _dnrate_transfer(self, target_index: int) -> float:
        """
        This method determines the DN rate of the the target at ``target_index`` based on the fraction of rows 
        the target is covering in the camera FOV. 

        This calculation for DN rate should only be used for resolved targets, the :meth:`dn_rate` method will 
        take this into account before calculating DN rate.
        
        This method requires that the I/F of the target has already been calculated using :meth:`iof`
        and that the :attr:`.photometric_camera_model` is not None.

        :param target_index: the index into the :attr:`target_objs` list for which to compute the magnitude for
        :return: DN rate of target in dn/s
        """
        return self._dnrate_resolved(target_index) * self._fraction_rows(target_index)

    def dnrate(self, target_index: int) -> float:
        """
        This method determines the DN rate of the the target at ``target_index`` and determines which model 
        to used based on if the requested target is resolved

        This method assumes that the magnitude of the target has already been calculated using :meth:`iof`

        :param target_index: the index into the :attr:`target_objs` list for which to compute the magnitude for
        :return: DN rate of target in dn/s. 
        """
        if not self.resolved(target_index):
            dnrate = self._dnrate_unresolved(target_index)
        else:
            dnrate = self._dnrate_resolved(target_index)
        return dnrate

    def snr(self, target_index: int, exposure_time: float, coadded_images: int = 1) -> float:
        """
        This method determines the Signal to Noise Ratio of the the target at ``target_index`` based on the input exposure time.
        
        This method requires that the magnitude of the target has already been calculated using :meth:`apparent_mag`
        and that the 

        :param target_index: the index into the :attr:`target_objs` list for which to compute the magnitude for
        :param exposure_time: the exposure time of the camera in seconds
        :param coadded_images: the number of coadded images 
        :return: SNR
        """

        # calculate the DN from the target 
        target_dn = exposure_time * self.dnrate(target_index)
        
        assert self.photometric_camera_model is not None

        # calculate the dn from the camera transfer time and point spread function
        transfer_dn = self._dnrate_transfer(
            target_index) * self.photometric_camera_model.transfer_time * coadded_images if self.resolved(
            target_index) else 0

        # determine the background dn 
        if self.scattered_light is not None:
            dn_background = (exposure_time * (
                        self.photometric_camera_model.dnrate_dark + self.scattered_light.dnrate(target_index, self,
                                                                                                dn_rate_standard=self.photometric_camera_model.dn_rate_standard))) + transfer_dn
        else:
            dn_background = (exposure_time * (self.photometric_camera_model.dnrate_dark)) + transfer_dn
        dn_total_nonstochastic = target_dn + dn_background
        dn_shotnoise = np.sqrt(dn_total_nonstochastic / self.photometric_camera_model.gain)
        dn_total_noise = np.sqrt(dn_shotnoise ** 2 + self.photometric_camera_model.dn_readnoise ** 2)

        return target_dn / dn_total_noise

