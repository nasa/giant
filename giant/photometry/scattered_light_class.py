# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.

import numpy as np

from typing import Callable, Optional

from giant.photometry.utilities import au


class ScatteredLight:
    """
    This class is used to determine the I/F and DNrate of a target due to scattered light
    
    """

    def __init__(self, luminosity_function: Callable, factor: Optional[float] = 1):
        """
        :param luminosity_function: function that inputs scene and scene object and returns IoverF
        :param factor: scaling factor to apply to DNRate
        """

        self.luminosity_function = luminosity_function
        self.factor = factor

    def iof(self, target_index: int, photometry, function: Optional[Callable] = None, **kwargs) -> float:
        """ 
        Calculate i_over_f due to scattered light of the target defined in photometry
        
        :param target_index: the index into the :attr:`target_objs` list for which to compute i_over_f due to scattered light
        :param photometry: Photometry object containing the scene and PhotometricCameraModel
        :param function: i_over_f function to use if not provided when initializing/overwriting self.luminocity_function
        """
        if function is not None:
            self.i_over_f = function(target_index, photometry, **kwargs)
        else:
            self.i_over_f = self.luminosity_function(target_index, photometry, **kwargs)
        return self.i_over_f

    def dnrate(self, target_index, photometry, dn_rate_standard=None):
        """
        Calculate DNrate due to scattered light of the target defined in photometry
        
        :param target_index: the index into the :attr:`target_objs` list for which to compute i_over_f due to scattered light
        :param photometry: Photometry object containing the scene and PhotometricCameraModel
        :param dn_rate_standard: standard dnrate used by the camera
        """

        if not hasattr(self, 'i_over_f'):
            self.iof(target_index, photometry)
            
        dn = dn_rate_standard * self.i_over_f / (au(np.linalg.norm(
            photometry.scene.target_objs[target_index].position - photometry.scene.light_obj.position)) ** 2)

        return self.factor * dn


