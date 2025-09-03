


r"""
This module provides utilities for visually inspecting calibration and alignment results.

Throughout this module, the visualizations created can either be displayed interactively on-screen or saved directly to
a PDF file.  To save to a PDF file, simply provide the optional argument ``pdf_name`` to each function.  To display the
plot interactively, leave the ``pdf_name`` argument alone or set it to ``None``.

All of the functions in this module assume that at minimum the :meth:`.Calibration.id_stars` has been called. This
means you can use many of these functions to generate prefit plots for comparison with post-fit plots if you so desire.

This module imports the pertinent functions from the :mod:`.stellar_opnav.visualizer` module for convenience
"""

from typing import Optional
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from giant.calibration.calibration_class import Calibration
from giant.calibration.estimators.alignment.temperature_dependent import evaluate_temperature_dependent_alignment
from giant.rotations import quaternion_to_euler
from giant.camera_models import CameraModel
from giant._typing import PATH

from giant.stellar_opnav.visualizers import (show_id_results, residual_histograms, residuals_vs_magnitude,
                                             residuals_vs_temperature)


def plot_focal_length_temperature_dependence(calib: Calibration, show_individual_focal_lengths: bool = True,
                                             pdf_name: Optional[PATH] = None, show: bool = True):
    """
    This function generates a figure of the camera model's focal length temperature dependence.

    The temperature dependence is shown over the range of temperatures of the camera for the images currently contained
    in the images in the camera as a line.  Optionally, if ``show_individual_focal_lengths`` is set to ``True``, this
    function will go through and estimate just the focal length with no temperature dependence for each image
    individually (holding all other parameters of the camera model fixed) and plot these individual focal lengths as a
    scatter plot on the figure.  If this is requested, then :meth:`~.Calibration.id_stars` must have been called at
    least once.

    Note that this function assumes that the camera model being estimated is a subclass of :class:`.PinholeModel` here
    and that it follows the same convention for the temperature dependence as the :class:`.PinholeModel`.  If the camera
    model being estimated does not implement a :meth:`~.PinholeModel.get_temperature_scale` then this function will
    raise an exception.  Additionally, if it deviates significantly from the :class:`.PinholeModel` and its subclasses
    in the way it handles focal length/temperature dependence this function will likely run into issues, therefore it
    is recommended to either use one of the existing GIANT :class:`.PinholeModel` subclasses or be extra careful in how
    you implement your own camera model to mimic the :class:`.PinholeModel`

    :raises ValueError: if the camera model doesn't implement a :meth:`~.PinholeModel.get_temperature_scale` method.

    :param calib: The calibration object to plot the results for
    :param show_individual_focal_lengths: A boolean flag specifying whether to plot the individual focal lengths for
                                          each image
    :param pdf_name: Save the plot to this file name as a pdf
    :param show: a flag indicating whether to bring uip the live display (only checked if pdf_name is None)
    """

    if not hasattr(calib.model, 'get_temperature_scale'):
        raise ValueError('The camera model must implement temperature dependence to use this function through the '
                         'get_temperature_scale method to use this function.')

    # get the list of temperatures under consideration
    temperatures = np.array([image.temperature for _, image in calib.camera])

    # get the temperature scales
    assert (gts := getattr(calib.model, "get_temperature_scale", None)) is not None, "the camera model must support temperature dependence"
    temp_scale = gts(temperatures)

    # make the figure and axes
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # check if this is a single focal length model or a double focal length model and handle accordingly
    double_focal_length = False
    if (fx := getattr(calib.model, "fx", None)) is not None and (fy := getattr(calib.model, "fy", None)) is not None:  # double focal length:
        fx_temp_fit = abs(fx) * temp_scale
        fy_temp_fit = abs(fy) * temp_scale
        ax.plot(temperatures, fx_temp_fit, color='red', label='focal_x')
        ax.plot(temperatures, fy_temp_fit, color='blue', label='focal_y')
        plt.ylabel('Abs(Focal Length), pix')
        double_focal_length = True
    elif (fl := getattr(calib.model, "focal_length", None)) is not None:  # single focal length
        
        f_temp_fit = fl * temp_scale
        ax.plot(temperatures, f_temp_fit, color='red', label='focal_length')
        plt.ylabel('Focal Length, mm')

    plt.xlabel('Temperature, (deg C)')

    if show_individual_focal_lengths:
        # work on a copy of the model
        orig_model = calib.model
        model_copy = orig_model.copy()
        calib.model = model_copy

        # only estimate the focal length parameters
        if double_focal_length:
            calib.model.estimation_parameters = ['fx', 'fy']
        else:
            calib.model.estimation_parameters = ['focal_length']
        # reset the temperature coefficients
        calib.model.temperature_coefficients[:] = 0 # type: ignore

        # get a copy of the current image mask from the camera
        orig_mask = deepcopy(calib.camera.image_mask)

        focal_lengths = []

        # loop through each turned on image
        for ind, _ in calib.camera:

            # if stars were found for this image, call estimate_calibration to get an updated focal length
            if ((mcsr := calib.matched_catalog_star_records[ind]) is not None) and (not mcsr.empty):
                # turn off all but this image
                calib.camera.all_off()
                calib.camera.image_mask[ind] = True
                # estimate the focal length
                calib.estimate_geometric_calibration()
                # reset the image mask to its original value
                calib.camera.image_mask = deepcopy(orig_mask)

            # store the updated focal length
            if double_focal_length:
                focal_lengths.append([abs(calib.model.fx), abs(calib.model.fy)]) # type: ignore
            else:
                focal_lengths.append(calib.model.focal_length) # type: ignore
        else:
            if double_focal_length:
                focal_lengths.append([None, None])
            else:
                focal_lengths.append(None)

        # scatter the individual focal lengths
        if double_focal_length:
            fx, fy = np.array(focal_lengths).T
            ax.scatter(temperatures, fx, color='red')
            ax.scatter(temperatures, fy, color='blue')
        else:
            ax.scatter(temperatures, np.array(focal_lengths), color='red')

        # put things back the way it was
        calib.model = orig_model

    plt.legend()

    if pdf_name:
        pdf = PdfPages(pdf_name)
        pdf.savefig(fig)
        pdf.close()
        plt.close(fig)
    elif show:
        plt.show()


def plot_distortion_map(model: CameraModel, pdf_name: Optional[PATH] = None, show: bool = True):
    """
    This function produces a distortion map for the provided camera model.

    If the `pdf_name` param is provided, the figure will be saved to a pdf file of the same name, and the figure
    will not be displayed to the user.

    :param model: The camera model to generate the distortion map for
    :param pdf_name: Used as the file name for saving the figure to pdf
    :param show: a flag indicating whether to bring uip the live display (only checked if pdf_name is None)
    """
    
    r, c, d = model.distortion_map(None, 100)
    fig = plt.figure()
    cs = plt.contour(c, r, np.linalg.norm(d, axis=0).reshape(r.shape))
    plt.gca().quiver(c, r, *d, angles='xy', scale_units='xy',
                     color='black', width=0.0005, headwidth=20, headlength=20)
    plt.clabel(cs, inline=True, fontsize=10)
    plt.xlabel('Column, pix')
    plt.ylabel('Row, pix')

    if pdf_name:
        pdf = PdfPages(pdf_name)
        pdf.savefig(fig)
        pdf.close()
        plt.close(fig)
    else:
        plt.show()


def plot_alignment_residuals(calib: Calibration, pdf_name: Optional[PATH] = None, show: bool = True):

    """
    This function plots the residual alignment errors per image as roll/pitch/yaw for both the estimated static and
    temperature alignments (if done).

    The residual plots are generated by comparing the current values of :attr:`Image.rotation_inertial_to_camera` to
    ``calib...._alignment*calib.alignment_base_frame_func(image.observation_date)`` and converting the residual into
    roll, pitch, yaw Tait-Bryan angles (xyz) rotation.  This function will generate 1 figure for each type of alignment
    that has been performed, plotting the residuals versus camera temperature.  The residuals are computed as the
    rotation from the computed frame using the estimated alignment to the observed frame for each image.

    .. Note::

        This function will take into account misalignment estimated in the camera model itself if it finds one, however,
        it is recommended that before calling this function you reset the misalignment to 0 in the camera model and
        then call :meth:`~.Calibration.estimate_attitude` again before using this function.

    You should have called at least one of :meth:`~.Calibration.estimate_static_alignment` or
    :meth:`~.Calibration.estimate_temperature_dependent_alignment` before using this function.  In addition, you should
    have called method :meth:`~.Calibration.estimate_attitude` for the results generated from this function to be
    meaningful.  Finally, the :attr:`~.Calibration.alignment_base_frame_func` must not be ``None``.

    If the ``pdf_name`` param is provided, the figures will be saved to a pdf file of the same name, and will not be
    displayed interactively.

    :raises ValueError: If both :attr:`~.Calibration.static_alignment` and
                        :attr:`~.Calibration.temperature_dependent_alignment` are ``None`` or if
                        :attr:`~.Calibration.alignment_base_frame_func` is ``None``.

    :param calib: The :class:`.Calibration` instance
    :param pdf_name: Used as the file name for saving the figures to a pdf
    :param show: a flag indicating whether to bring uip the live display (only checked if pdf_name is None)
    """

    if (calib.static_alignment is None) and (calib.temperature_dependent_alignment is None):
        raise ValueError('One form of alignment must have been estimated before using this function.'
                         'Please call estimate_static_alignment or estimate_temperature_dependent_alignment from the '
                         'calibration object before using this function')

    if calib.alignment_base_frame_func is None:
        raise ValueError('The alignment base frame function must not be None to use this function.')

    temperatures = []
    static_residuals = [[], [], []]
    temperature_residuals = [[], [], []]

    for ind, image in calib.camera:
        # check if any stars were identified for this image
        if (mcsr := calib.matched_catalog_star_records[ind]) is not None and not mcsr.empty:
            temperatures.append(image.temperature)

            measured = image.rotation_inertial_to_camera * calib.alignment_base_frame_func(image.observation_date).inv()
            # take into account the camera model misalignment
            if hasattr(calib.model, 'get_misalignment'):
                measured = calib.model.get_misalignment(ind)*measured  # type: ignore

            if calib.static_alignment is not None:
                static_err = quaternion_to_euler((measured * calib.static_alignment.inv()).quaternion)

                static_residuals[0].append(static_err[0]*1000)  # in milliradians
                static_residuals[1].append(static_err[1]*1000)  # in milliradians
                static_residuals[2].append(static_err[2]*1000)  # in milliradians

            if calib.temperature_dependent_alignment is not None:
                temperature_err = quaternion_to_euler(
                    (measured * evaluate_temperature_dependent_alignment(calib.temperature_dependent_alignment, 
                                                                         image.temperature).inv()).quaternion
                )

                temperature_residuals[0].append(temperature_err[0]*1000)  # in milliradians
                temperature_residuals[1].append(temperature_err[1]*1000)  # in milliradians
                temperature_residuals[2].append(temperature_err[2]*1000)  # in milliradians

    if pdf_name is not None:
        pdf = PdfPages(pdf_name)
    else:
        pdf = None

    if calib.static_alignment is not None:
        fig = plt.figure()
        plt.scatter(temperatures, static_residuals[0], color='red', label=r'pitch, $\mathbf{x}_C$')
        plt.scatter(temperatures, static_residuals[1], color='blue', label=r'yaw, $\mathbf{y}_C$')
        plt.scatter(temperatures, static_residuals[2], color='green', label=r'roll, $\mathbf{z}_C$')
        plt.title('Static Alignment Residuals vs Temperature\n'
                  'Standard Deviation ({:.3g}, {:.3g}, {:.3g})'.format(np.std(static_residuals[0]),
                                                                       np.std(static_residuals[1]),
                                                                       np.std(static_residuals[2])))

        plt.xlabel('temperature, deg C')
        plt.ylabel('Computed to Measured Residuals, mrad')

        plt.legend().set_draggable(True)

        if pdf:
            pdf.savefig(fig)
            plt.close(fig)

    if calib.temperature_dependent_alignment is not None:
        fig = plt.figure()
        plt.scatter(temperatures, temperature_residuals[0], color='red', label=r'pitch, $\mathbf{x}_C$')
        plt.scatter(temperatures, temperature_residuals[1], color='blue', label=r'yaw, $\mathbf{y}_C$')
        plt.scatter(temperatures, temperature_residuals[2], color='green', label=r'roll, $\mathbf{z}_C$')
        plt.title('Temperature Dependent Alignment Residuals vs Temperature\n'
                  'Standard Deviation ({:.3g}, {:.3g}, {:.3g})'.format(np.std(temperature_residuals[0]),
                                                                       np.std(temperature_residuals[1]),
                                                                       np.std(temperature_residuals[2])))

        plt.xlabel('temperature, deg C')
        plt.ylabel('Computed to Measured Residuals, mrad')
        plt.legend().set_draggable(True)

        if pdf:
            pdf.savefig(fig)
            plt.close(fig)

    if pdf is not None:
        pdf.close()
    elif show:
        plt.show()
