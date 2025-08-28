import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from copy import deepcopy
from math import radians, degrees
from typing import Union

from giant.ray_tracer.scene import SceneObject, Scene
from giant.photometry.magnitude import PhaseMagnitudeModel


def plot_magnitude_by_phase(magnitude_model: PhaseMagnitudeModel, scene: Scene,
                            scene_object: Union[SceneObject, SceneObject]) -> tuple[Figure, Axes]:
    """
    Plot a Phase Magnitude Model. Resulting figure will plot phase angle (deg) vs. magnitude with an x representing the current geometry.
    
    :param magnitude_model: PhaseMagnitudeModel instance for plotting
    :param scene: Scene representing the current geometry 
    :param scene_object: SceneObject or Scene object to find magnitude
    :returns: matplotlib figure and axes objects containing plot
    """
    target_copy = deepcopy(scene_object)
    scene_copy = deepcopy(scene)
    magnitudes = []

    # plot the magnitude at varying phase angles (geometry is static)
    angles = np.linspace(radians(0), radians(120), 1000)
    for angle in angles:
        magnitudes.append(magnitude_model.magnitude_function(scene_copy, target_copy, phase_angle=angle))
    fig, ax = plt.subplots()
    plt.plot(
        [degrees(a) for a in angles],
        magnitudes,
    )

    # plot current phase angle and magnitude 
    try:
        target_index = scene.target_objs.index(scene_object)
    except:
        if len(scene.target_objs) == 1:
            target_index = 0
        else:
            raise ValueError("we can't find the target in the scene")
    phase_angle = scene.phase_angle(target_index)
    target_magnitude = magnitude_model.magnitude_function(scene, scene_object, phase_angle=phase_angle)
    plt.plot(
        degrees(phase_angle),
        target_magnitude,
        marker='x',
    )

    # plot labels and formatting
    ax.set_ylabel('Apparent Magnitude')
    ax.set_xlabel('Phase Angle, deg')
    ax.grid()
    ax.set_title(f'{scene_object.name} : {magnitude_model.__class__.__name__}({magnitude_model.__dict__})')

    return fig, ax
