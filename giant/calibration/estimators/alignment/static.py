
from giant.rotations import Rotation
from giant.stellar_opnav.estimators.davenport_q_method import DavenportQMethod
from giant._typing import DOUBLE_ARRAY


def static_alignment_estimator(frame1_unit_vecs: DOUBLE_ARRAY, frame2_unit_vecs: DOUBLE_ARRAY, weights: DOUBLE_ARRAY | None = None) -> Rotation:
    """
    This function estimates a static attitude alignment between one frame and another.

    The static alignment is estimated using Davenport's Q-Method solution to Wahba's problem, using the
    :class:`.DavenportQMethod` class.  To use, simply provide the unit vectors from the base frame and the unit vectors
    from the target frame.  The estimated alignment from frame 1 to frame 2 will be returned as a :class:`.Rotation` 
    object from frame 1 to frame 2

    In general this function should not be used by the user, and instead you should use the :class:`.Calibration` class and
    its :meth:`~.Calibration.estimate_static_alignment` method.

    For more details about the algorithm used see the :class:`.DavenportQMethod` documentation.
    
    :param frame1_unit_vecs: Unit vectors in the base frame as a 3xn array where each column is a unit vector.
    :param frame2_unit_vecs: Unit vectors in the destination (camera) frame as a 3xn array where each column is a
                                unit vector
    :param weights: Option length n array of weighting to use for each unit vector pair in the estimation process
    """

    solver = DavenportQMethod()
    solver.weighted_estimation = weights is not None

    return solver.estimate(frame2_unit_vecs, frame1_unit_vecs, weights)
