

import numpy as np
from abc import ABCMeta, abstractmethod
from math import log10

from giant.ray_tracer.scene import SceneObject, Scene
from giant.ray_tracer.shapes import Ellipsoid, Point
from giant.photometry.utilities import au


class PhaseMagnitudeModel(metaclass=ABCMeta):
    '''
    This is a parent class used to define the magnitude function of a body 
    by the solar phase angle. A custom magnitude model should follow this 
    template.
    
    The class must contain a magnitude_function definition that inputs a scene
    and scene object. 
    '''

    @abstractmethod
    def magnitude_function(self, scene: Scene, target: SceneObject, phase_angle: float | None = None) -> float:
        """
        This method should return the apparent magnitude of the target at the provided phase angle or scene setup.
        
        :param scene: used to define the geometry of the scene, primarily phase angle
        :param target: used to get physical attributes of the target such as diameter
        :param phase_angle: optional input for phase angle in radians used to override real phase_angle
                            in the scene. This is mainly used for plotting purposes. 
        """
        pass
    

class LinearPhaseMagnitudeModel(PhaseMagnitudeModel):
    '''
    General Phase-Slope Model used to determine Magnitude of target
    '''

    def __init__(self, phase_slope: float, albedo: float | None = None):
        """
        :param phase_slope: phase slope of object brightness
        :param albedo: object light reflectance 
        """
        self.phase_slope = phase_slope
        self.albedo = albedo

    def magnitude_function(self, scene: Scene, target: SceneObject, phase_angle: float | None = None) -> float:
        """
        :param scene: used to define the geometry of the scene, primarily phase angle
        :param target: used to get physical attributes of the target such as diameter
        :param phase_angle: optional input for phase angle in radians used to override real phase_angle
                            in the scene. This is mainly used for plotting purposes. 
        """
        if scene.light_obj is None:
            raise ValueError('The scene must contain an illumination source to compute the magnitude')
        
        # get the solar phase angle
        if phase_angle is None:
            try:
                target_index = scene.target_objs.index(target)
            except:
                if len(scene.target_objs) == 1:
                    target_index = 0
                else:
                    raise ValueError('The phase angle was not provided and we cannot locate the provided target in the scene')
            phase_angle = scene.phase_angle(target_index)

        # determine the diameter of the target
        if isinstance(target.shape, Ellipsoid):
            diameter = target.shape.principal_axes.mean()
        elif isinstance(target.shape, Point):
            diameter = 0
        else:
            diameter = 0

        # this model needs the target albedo
        albedo = self.albedo
        if albedo is None:
            if (s_albedo := getattr(target, 'albedo', None)) is None:
                raise ValueError('Albedo is required to calculate magnitude using a Linear Phase Slope Model')
            albedo = s_albedo

        magnitude = -5 * np.log10(np.sqrt(albedo) * au(diameter)) + \
                    5 * np.log10((au(target.distance) - au(scene.light_obj.distance)) * \
                    au(target.distance)) + self.phase_slope * np.rad2deg(phase_angle) - 26.75
        return magnitude


class HGPhaseMagnitudeModel(PhaseMagnitudeModel):
    r"""
    Phase-Slope Model Typically used for Asteroids. This model takes the absolute magnitude and 
    phase-slope brightness of the target into consideration. 
    
    This takes the form of
    
    .. math::
        m(\alpha) &= H-2.5\log\left[[1-G]\Phi_1(\alpha)+G\Phi_2(\alpha)\right] \\
        W &= \exp(-90.56\tan^2(\alpha/2)) \\
        \Phi_1 &=W\phi_{1S}+(1-@)\phi_{1L} \\ 
        \phi_{1S} &= 1-\frac{C_1\sin\alpha}{0.119+1.341\sin\alha-0.754\sin^2\alpha} \\
        \phi_{1L} &= \exp(-A_1\left(tan\frac{\alpha}{2}\right)^{B_1}) \\
        \Phi_2 &= W\phi_{2S}+(1-W)\phi_{2L} \\
        \phi_{2S} &=1-\frac{C_2\sin\alpha}{0.119+1.341\sin\alpha-0.754\sin^2\alpha}
        \phi_{2L} &= \exp(-A_2\left(tan\frac{\alpha}{2}\right)^{B_2}) 
            
    where :math:`\alpha` is the phase angle in radian, :math:`H` and :math:`G` are the absolute visual magnitude and 
    phase-slope of brightness respectively, and the cooeficiens take the following values:
    
        :math:`A_1` = 3.332      :math:`A_2` = 1.862
        :math:`B_1` = 0.631      :math:`B_2` = 1.218
        :math:`C_1` = 0.986      :math:`C_2` = 0.238
        
    This model comes from https://adsabs.harvard.edu/full/2010SASS...29..101B
    
    For most targets, H and G can be retrieved from JPL's Horizons database.
    """

    def __init__(self, abs_visual_mag: float, phase_slope: float):
        """
        :param abs_visual_mag: Object absolute visual magnitude (H)
        :param phase_slope: Phase slope of brightness (G)
        """
        self.abs_visual_mag = abs_visual_mag
        self.phase_slope = phase_slope

    def magnitude_function(self, scene: Scene, target: SceneObject, phase_angle: float | None = None) -> float:
        """
        :param scene: used to define the geometry of the scene, primarily phase angle
        :param target: used to get physical attributes of the target such as diameter
        :param phase_angle: optional input for phase angle in radians used to override real phase_angle
                            in the scene. This is mainly used for plotting purposes. 
        """

        if scene.light_obj is None:
            raise ValueError('The scene must contain an illumination source to compute the magnitude')
        
        # get the solar phase angle
        if phase_angle is None:
            try:
                target_index = scene.target_objs.index(target)
            except:
                if len(scene.target_objs) == 1:
                    target_index = 0
                else:
                    raise ValueError('The phase angle was not provided and we cannot locate the provided target in the scene')
            phase_angle = scene.phase_angle(target_index)

        # apply the model to get magnitude
        W = np.exp(-90.56 * (np.tan(phase_angle / 2)) ** 2)
        A1, B1, C1 = 3.332, 0.631, 0.986  # coefficient
        th_1s = 1 - ((C1 * np.sin(phase_angle)) / (
                    0.119 + (1.341 * np.sin(phase_angle)) - (0.754 * np.sin(phase_angle) ** 2)))
        th_1l = np.exp(-A1 * (np.tan(phase_angle / 2)) ** B1)
        th1 = W * th_1s + (1 - W) * th_1l

        A2, B2, C2 = 1.862, 1.218, 0.238  # coefficients
        th_2s = 1 - ((C2 * np.sin(phase_angle)) / (
                    0.119 + (1.341 * np.sin(phase_angle)) - (0.754 * np.sin(phase_angle) ** 2)))
        th_2l = np.exp(-A2 * (np.tan(phase_angle / 2)) ** B2)
        th2 = W * th_2s + (1 - W) * th_2l

        phaseH = self.abs_visual_mag - 2.5 * np.log10(((1 - self.phase_slope) * th1) + (self.phase_slope * th2))

        return phaseH + 5 * np.log10(au(float(np.linalg.norm(target.position - scene.light_obj.position))) * au(target.distance))