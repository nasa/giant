from giant.camera_models import CameraModel

from giant.calibration.estimators.geometric.geometric_estimator import GeometricEstimator, GeometricEstimatorOptions
from giant.calibration.estimators.geometric.iterative_nonlinear_lstsq import IterativeNonlinearLSTSQ, IterativeNonlinearLstSqOptions
from giant.calibration.estimators.geometric.lma import LMAEstimator, LMAEstimatorOptions

from enum import Enum, auto

class GeometricEstimatorImplementations(Enum):
    """
    An enum specifying the available geometric camera model estimators implemented.
    
    For a non-standard implementation, choose CUSTOM
    """
    
    ITERATIVE_NONLINEAR_LSTSQ = auto()
    """
    Use the iterative nonlinear least squares estimator
    """
    
    LMA = auto()
    """
    Use the Levenberg-Marquadt algorithm estimator
    """
    
    CUSTOM = auto()
    """
    A custom implementation which implements the GeometricEstimator interface
    """
    
    
def get_estimator(type: GeometricEstimatorImplementations, model: CameraModel, options: IterativeNonlinearLstSqOptions | GeometricEstimatorOptions | None = None) -> LMAEstimator | IterativeNonlinearLSTSQ:
    """
    Returns an instance of the appropriate estimator per the provided type.
    
    :returns: the requested estimator initialized with options
    :raises: ValueError if CUSTOM is chosen
    """
    
    match type:
        case GeometricEstimatorImplementations.ITERATIVE_NONLINEAR_LSTSQ:
            assert isinstance(options, IterativeNonlinearLstSqOptions) or options is None, "Options must be ESOQ2Options or None"
            return IterativeNonlinearLSTSQ(model, options)
        case GeometricEstimatorImplementations.LMA:
            assert isinstance(options, LMAEstimatorOptions) or options is None, "Options must be ESOQ2Options or None"
            return LMAEstimator(model, options)
        case _:
            raise ValueError('Cannot return a custom attitude implementation')



__all__ = ["IterativeNonlinearLSTSQ", "IterativeNonlinearLstSqOptions",
           "LMAEstimator", "LMAEstimatorOptions",
           "GeometricEstimator", "GeometricEstimatorOptions"]
