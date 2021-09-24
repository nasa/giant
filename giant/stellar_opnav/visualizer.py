# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


r"""
This module provides utilities for visually inspecting star identification and attitude estimation results.

In general, the only functions a user will directly interface with from this module are the :func:`show_id_results`
which shows the results of performing star identification and attitude estimation, :func:`residual_histograms` which
shows histograms of the residuals, and :func:`plot_residuals_vs_magnitude` which generates a scatter plot of residuals
as a function of star magnitude.  The other contents of this model are used for manual outlier inspection, which is
typically done by using the :meth:`~.StellarOpNav.review_outliers` method.
"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from giant._typing import PATH

from giant.stellar_opnav.stellar_class import StellarOpNav


def show_id_results(sopnav: StellarOpNav, pdf_name: Optional[PATH] = None,
                    flattened_image: bool = False, log_scale: bool = False):
    """
    This function generates a figure for each turned on image in ``sopnav`` showing the star identification results, as
    well as a couple figures showing the residuals between the predicted catalogue star locations and the image star
    locations for all images combined.

    For each individual figure, the matched catalogue projected star locations are shown with one marker, the matched
    image points of interest are shown with another marker, the unmatched catalogue projected star locations in the
    field of view are shown with another marker, and the unmatched image points of interest are shown with another
    marker.  In addition, an array is drawn indicating the residuals for the match star pairs.

    You must have called :meth:`~.StellarOpNav.id_stars` at least once before calling this function.

    If ``pdf_name`` param is not ``None``, the figures will be saved to a pdf file of the same name.

    If ``flattened_image`` is set to True, then the individual figures will show the flattened image that is used for
    initial detection of possible stars in the image.  Typically this should be left to ``False`` unless you are having
    issues finding stars and want to inspect the image to see what is going on.

    If ``log_scale`` is set to ``True`` then the individual images are shown using a logarithmic scale to help make
    stars stand out more from the background.  This is generally a fairly useful feature and is used frequently.

    .. warning::
        This function generates 1 figure for every image in your :class:`.StellarOpNav` instance, which can really slow
        down your computer if you have a lot of images loaded and turned on.  To avoid this problem, try turning some of
        the images off using the :attr:`~.Camera.image_mask` attribute before using this function, or save to
        pdf instead.

    :param sopnav: The :class:`.StellarOpNav` instance containing the star identification results
    :param pdf_name: Label of the star id results to be shown. Used as the file name for saving figures to pdf
    :param flattened_image: A boolean flag specifying whether to show the flattened image instead of the raw image
    :param log_scale: A boolean flag specifying whether to use a logarithmic scale for the image intensity values.
    """

    all_resids_fig = plt.figure()
    ax = all_resids_fig.add_subplot(111)
    big_resids = []

    if pdf_name:
        pdf = PdfPages(pdf_name)
    else:
        pdf = None

    for ind, image in sopnav.camera:
        cat_locations = sopnav.matched_catalogue_image_points[ind]

        if flattened_image:
            image, _ = sopnav.image_processing.flatten_image_and_get_noise_level(image)
            # make the min of the image 100
            image -= image.min() - 100

        fig = plt.figure()
        axt = plt.gca()
        if log_scale:
            image = image.astype(np.float32) - image.min() + 100
            axt.imshow(np.log(image), cmap='gray', interpolation='none')
        else:
            axt.imshow(image, cmap='gray', interpolation='none')

        if (cat_locations is not None) and cat_locations.size:
            resids = cat_locations - sopnav.matched_extracted_image_points[ind]

            # # Plot Matched Pairs
            axt.scatter(*sopnav.matched_extracted_image_points[ind], color='none', edgecolors='yellow',
                        linewidths=1.5, label='Matched Centroids')
            axt.scatter(*cat_locations, marker='x', color='red', linewidths=1.5, label='Matched Catalog')
            c = np.arange(sopnav.matched_extracted_image_points[ind].shape[1])

            for x, y, label in zip(*cat_locations, c.astype(np.str_)):
                axt.text(x, y, label, color="white", fontsize=8)

            big_resids.append(resids)

            ax.quiver(*sopnav.matched_extracted_image_points[ind], *resids, angles='xy', scale_units='xy',
                      color='black', width=0.0005, headwidth=20, headlength=20)

        in_fov = ((sopnav.unmatched_catalogue_image_points[ind] >= 0) &
                  (sopnav.unmatched_catalogue_image_points[ind] <= [[sopnav.model.n_cols],
                                                                    [sopnav.model.n_rows]])).all(axis=0)

        # Plot Unmatched Pairs
        axt.scatter(*sopnav.unmatched_extracted_image_points[ind], marker='d', color='none', edgecolors='green',
                    linewidths=0.5, label='Unmatched Centroids')
        axt.scatter(*(sopnav.unmatched_catalogue_image_points[ind][:, in_fov]), marker='d', color='none',
                    edgecolors='cyan',
                    linewidths=0.5, label='Unmatched Catalog')
        axt.legend().set_draggable(True)

        plt.title('Observation Date: {}'.format(image.observation_date))

        if pdf_name:
            pdf.savefig(fig)
            fig.clear()
            plt.close(fig)

    if big_resids:
        big_resids = np.concatenate(big_resids, axis=1)
        ax.set_xlim([0, sopnav.model.n_cols])
        ax.set_ylim([0, sopnav.model.n_rows])
        ax.invert_yaxis()
        ax.set_xlabel('columns, pix')
        ax.set_ylabel('rows, pix')
        ax.set_title('Residuals All Images\n '
                     'std = ({0:5.3f}, {1:5.3f}) '
                     'mean = ({2:5.3f}, {3:5.3f}) pixels'.format(*np.std(big_resids, axis=1),
                                                                 *np.mean(big_resids, axis=1)))
        print(*np.std(big_resids, axis=1))
        print(*np.mean(big_resids, axis=1))
    if pdf_name:
        pdf.savefig(all_resids_fig)
        pdf.close()
        all_resids_fig.savefig(pdf_name + '_combined_resids.pdf')
        all_resids_fig.clear()
        plt.close(all_resids_fig)
    else:
        plt.show()


def residual_histograms(sopnav: StellarOpNav, individual_images: bool = False, pdf_name: Optional[PATH] = None):
    """
    This function generates histograms of the matched star residuals for a given stellar opnav object.

    Typically, 3 histograms are created in a single figure.  The first shows the histogram of all residuals (both column
    and row) for all images. The second shows the histogram of just the column residuals for all images. The third shows
    the histogram of just the row residuals for all images. Optionally, if you specify ``individual_images`` to be
    ``True`` then this function will generate a single figure for each image turned on in ``sopnav`` showing the three
    histograms described previously.  If using this option it is recommended to save the plots to a PDF instead of
    showing interactively because many figures would be opened, possibly causing your compute to slow down.

    You must have called :meth:`~.StellarOpNav.id_stars` at least once before calling this function.

    If the ``pdf_name`` param is provided, the figures will be saved to a pdf file of the same name, and will not be
    displayed interactively.

    :param sopnav: The stellar opnav object to plot the histograms for
    :param individual_images: A flag specifying whether to generate a single histogram for all images or no
    :param pdf_name: Used as the file name for saving the figures to a pdf
    """

    if pdf_name:
        pdf = PdfPages(pdf_name)
    else:
        pdf = None

    column_residuals = []
    row_residuals = []
    all_residuals = []

    image_residuals = []

    for ind, _ in sopnav.camera:
        residuals = sopnav.matched_star_residuals(ind)
        if individual_images:
            image_residuals.append(residuals)
        if (residuals is not None) and residuals.size:
            column_residuals.extend(residuals[0])
            row_residuals.extend(residuals[1])
            all_residuals.extend(residuals[0])
            all_residuals.extend(residuals[1])

    fig = plt.figure()
    plt.hist(all_residuals, bins='auto', histtype='bar', ec='white', label='all')
    plt.hist(column_residuals, bins='auto', histtype='bar', ec='white', alpha=0.5, label='column')
    plt.hist(row_residuals, bins='auto', histtype='bar', ec='white', alpha=0.5, label='row')
    plt.title('Residual Histogram All Images\n'
              'STD (Total (column, row)) = {:.3g} ({:.3g}, {:.3g})'.format(np.std(all_residuals),
                                                                           np.std(column_residuals),
                                                                           np.std(row_residuals)))
    plt.xlabel('Residuals, pix')
    plt.ylabel('Count')
    try:
        plt.legend().draggable()
    except AttributeError:
        plt.legend().set_draggable(True)

    if pdf_name:
        pdf.savefig(fig)
        plt.close(fig)

    if individual_images:
        for ind, image in sopnav.camera:
            if image_residuals[ind]:
                fig = plt.figure()
                residuals = image_residuals[ind]
                plt.hist(residuals.ravel(), bins='auto', histtype='bar', ec='white', label='all')
                plt.hist(residuals[0], bins='auto', histtype='bar', ec='white', alpha=0.5, label='column')
                plt.hist(residuals[1], bins='auto', histtype='bar', ec='white', alpha=0.5, label='row')
                plt.title(
                    'Residual Histogram Images {}\n'
                    'STD (Total (column, row)) = {:.3g} ({:.3g}, {:.3g})'.format(image.observation_date.isoformat(),
                                                                                 np.std(residuals.ravel()),
                                                                                 np.std(residuals[0]),
                                                                                 np.std(residuals[1])))
                plt.xlabel('Residuals, pix')
                plt.ylabel('Count')
                try:
                    plt.legend().draggable()
                except AttributeError:
                    plt.legend().set_draggable(True)

                if pdf_name:
                    pdf.savefig(fig)
                    plt.close(fig)

    if pdf_name:
        pdf.close()
    else:
        plt.show()


def plot_residuals_vs_magnitude(sopnav: StellarOpNav, individual_images: bool = False, pdf_name: Optional[PATH] = None):
    """
    This function generates a scatter plot of x and y residuals versus star magnitudes from the matched catalogue
    stars for a given stellar opnav object.

    Generally, this function will generate a single scatter plot showing the residuals vs magnitude across all images,
    however, if you specify ``individual_images`` as ``True``, then in addition to the summary plot, a single plot will
    be made showing the residuals vs magnitude for each image.

    You must have called :meth:`~.StellarOpNav.id_stars` at least once before calling this function.

    If the ``pdf_name`` param is provided, the figures will be saved to a pdf file of the same name, and will not be
    displayed interactively.

    :param sopnav: The stellar opnav object to plot the scatters for
    :param individual_images: A flag specifying whether to generate individual plots for each image in addition to the
                              plot spanning all images
    :param pdf_name: Used as the file name for saving the figures to a pdf
    """

    if pdf_name:
        pdf = PdfPages(pdf_name)
    else:
        pdf = None

    column_residuals = []
    row_residuals = []
    all_residuals = []

    image_residuals = []
    star_magnitudes = []

    for ind, _ in sopnav.camera:
        residuals = sopnav.matched_star_residuals(ind)
        if individual_images:
            image_residuals.append(residuals)
        if (residuals is not None) and residuals.size:
            column_residuals.extend(residuals[0])
            row_residuals.extend(residuals[1])
            all_residuals.extend(residuals[0])
            all_residuals.extend(residuals[1])
            star_magnitudes.extend(sopnav.matched_catalogue_star_records[ind].loc[:, 'mag'])

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    ax.scatter(star_magnitudes, column_residuals, color='blue', label='column')
    ax.scatter(star_magnitudes, row_residuals, color='red', label='row')
    try:
        ax.legend().draggable()
    except AttributeError:
        ax.legend().set_draggable(True)

    plt.title('Residuals vs. Star Magnitude')
    plt.xlabel('Star Magnitude')
    plt.ylabel('Residual, pix')

    if pdf_name:
        pdf.savefig(fig)
        plt.close(fig)

    if individual_images:
        for ind, image in sopnav.camera:
            if image_residuals[ind]:
                fig = plt.figure()
                residuals = image_residuals[ind]
                mags = sopnav.matched_catalogue_star_records[ind].loc[:, "mag"]
                plt.scatter(mags, residuals[0], label='column')
                plt.scatter(mags, residuals[1], label='row')
                plt.title('Residual vs Star Magnitude Image {}'.format(image.observation_date.isoformat()))
                plt.xlabel('Star Magnitude')
                plt.ylabel('Residual, pix')
                try:
                    plt.legend().draggable()
                except AttributeError:
                    plt.legend().set_draggable(True)

                if pdf_name:
                    pdf.savefig(fig)
                    plt.close(fig)

    if pdf_name:
        pdf.close()
    else:
        plt.show()


def plot_residuals_vs_temperature(sopnav: StellarOpNav, pdf_name: Optional[PATH] = None):

    """
    This function generates a scatter plot of x and y residuals versus image temperature from the matched catalogue
    stars for a given stellar opnav object.

    You must have called :meth:`~.StellarOpNav.id_stars` at least once before calling this function.

    If the ``pdf_name`` param is provided, the figures will be saved to a pdf file of the same name, and will not be
    displayed interactively.

    :param sopnav: The stellar opnav object to plot the scatters for
    :param pdf_name: Used as the file name for saving the figures to a pdf
    """

    x_residuals = []
    y_residuals = []
    temperatures = []

    for ind, image in sopnav.camera:
        residuals = sopnav.matched_star_residuals(ind)

        if (residuals is not None) and residuals.size:
            x_residuals.extend(residuals[0])
            y_residuals.extend(residuals[1])

            temperatures.extend([image.temperature]*len(residuals[0]))

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    ax.scatter(temperatures, x_residuals, color='blue', label='column')
    ax.scatter(temperatures, y_residuals, color='red', label='row')
    try:
        ax.legend().draggable()
    except AttributeError:
        ax.legend().set_draggable(True)

    plt.title('Residuals vs. Star Magnitude')
    plt.xlabel('Temperature, deg C')
    plt.ylabel('Residual, pix')

    if pdf_name:
        pdf = PdfPages(pdf_name)
        pdf.savefig(fig)
        pdf.close()
        plt.close(fig)
    else:
        plt.show()


class OutlierCallback:
    """
    The class is used to represent an outlier shown to the user for review via the function :func:`show_outlier`
    and to store the user's choice whether or not to remove the outlier from the matched star pairs.

    This typically is not used by the user directly.  See :meth:`~.StellarOpNav.review_residuals` or
    :func:`show_outlier` instead.
    """

    def __init__(self, outlier_number: int, image_number: int, centroid: np.ndarray, plot: Figure):
        """
        :param outlier_number: The index of the outlier in the ``matched_`` properties of the :class:`StellarOpNav`
                               class
        :param image_number: The index of the image in the :attr:`camera` attribute of a :class:`.StellarOpNav`
                             instance where the outlier represented by :class:`OutlierCallback` was found
        :param centroid: The pixel coordinates of the outlier in the image
        :param plot: The plot of the outlier
        """

        self.outlier_number = outlier_number  # type: int
        """
        The number of the outlier in the image
        """

        self.image_number = image_number  # type: int
        """
        Image number
        """

        self.centroid = centroid  # type: np.ndarray
        """
        Observed location
        """

        self.plot = plot  # type: Figure
        """
        The plot which shows the outlier in question
        """

        self.removed = False  # type: bool
        """
        A flag specifying whether the outlier has been removed by the user
        """

    def remove(self, _):
        """
        This method sets the removed flag to 1 and closes the plot.
        """
        self.removed = True
        print('You have Removed Star {} ({:.2f}, {:.2f}) from Consideration in Image {}'.format(self.outlier_number,
                                                                                                *self.centroid,
                                                                                                self.image_number))
        plt.close(self.plot)

    def keep(self, _):
        """
        This method closes the plot.
        """
        plt.close(self.plot)


def show_outlier(sopnav: StellarOpNav, index: int,
                 image_num: int, residuals: np.ndarray) -> OutlierCallback:
    """
    This function generates a figure for a specified outlier in the given image number in ``sopnav``.

    In the generated figure the matched catalogue projected star location is shown with one marker, and the matched
    image point of interest is shown with another. In addition, a vector is drawn indicating the residual
    for the matched star pair.

    You must have called :meth:`~.StellarOpNav.id_stars` at least once before calling this function.

    :param sopnav: The :class:`.StellarOpNav` instance containing the star identification results
    :param index: The index of the outlier
    :param image_num: The index of the image in the :attr:`.StellarOpNav.camera` attribute
    :param residuals: np.ndarray of the residuals between the matched catalogue projected star locations and the
                      matched image points of interest for the given image number in the :attr:`.StellarOpNav.camera`
                      attribute
    """

    w = 10  # image zoom window size...num pixels to left and right of centroid

    cat_locations = sopnav.matched_catalogue_image_points[image_num]
    outlier_image = sopnav.camera.images[image_num]
    centroid = sopnav.matched_extracted_image_points[image_num][:, index]
    c0 = centroid[1]

    c1 = centroid[0]
    outlier_fig = plt.figure()
    a = plt.gca()
    a.imshow(np.log(outlier_image.astype(np.float64)), cmap='gray', interpolation='none')
    a.scatter(*cat_locations[:, index], color='none', edgecolors='red', linewidths=1, label='Catalog')
    a.scatter(c1, c0, color='none', edgecolors='green', linewidths=1, label='Centroid')
    a.text(c1 + 5, c0 - 5, str(np.round(residuals[:, index], 2).astype(str)), color="yellow", fontsize=8)
    a.text(c1 - 5, c0 - 5, (np.round(c1).astype(str), np.round(c0).astype(str)), color='green', fontsize=8)
    a.quiver(c1, c0, residuals[0, index], residuals[1, index], angles='xy', scale_units='xy', scale=1 / 200,
             color='yellow', label='Residual')
    a.set_xlim([(c1 - w), (c1 + w)])
    a.set_ylim([(c0 - w), (c0 + w)])
    a.invert_yaxis()
    a.set_title('Potential Outliers: Image {0}'.format(image_num))
    try:
        a.legend().draggable()
    except AttributeError:
        a.legend().set_draggable(True)
    ax1 = plt.axes([0.15, 0.1, .1, .05])
    ax2 = plt.axes([0.75, 0.1, .1, .05])

    callback = OutlierCallback(index, image_num, centroid, outlier_fig)
    b1 = Button(ax1, 'Remove')
    b1.on_clicked(callback.remove)

    b2 = Button(ax2, 'Keep')
    b2.on_clicked(callback.keep)
    plt.show()

    return callback
