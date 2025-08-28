import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.figure import Figure

from giant.stellar_opnav.stellar_class import StellarOpNav



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

        self.outlier_number: int = outlier_number 
        """
        The number of the outlier in the image
        """

        self.image_number: int = image_number 
        """
        Image number
        """

        self.centroid: np.ndarray = centroid 
        """
        Observed location
        """

        self.plot: Figure = plot 
        """
        The plot which shows the outlier in question
        """

        self.removed: bool = False 
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

    In the generated figure the matched catalog projected star location is shown with one marker, and the matched
    image point of interest is shown with another. In addition, a vector is drawn indicating the residual
    for the matched star pair.

    You must have called :meth:`~.StellarOpNav.id_stars` at least once before calling this function.

    :param sopnav: The :class:`.StellarOpNav` instance containing the star identification results
    :param index: The index of the outlier
    :param image_num: The index of the image in the :attr:`.StellarOpNav.camera` attribute
    :param residuals: np.ndarray of the residuals between the matched catalog projected star locations and the
                      matched image points of interest for the given image number in the :attr:`.StellarOpNav.camera`
                      attribute
    """

    w = 10  # image zoom window size...num pixels to left and right of centroid

    cat_locations = sopnav.matched_catalog_image_points[image_num]
    outlier_image = sopnav.camera.images[image_num]
    meip = sopnav.matched_extracted_image_points[image_num]
    assert meip is not None and cat_locations is not None
    centroid: np.ndarray = meip[:, index]
    c0: float = float(centroid[1])

    c1: float = float(centroid[0])
    outlier_fig = plt.figure()
    a = plt.gca()
    a.imshow(np.log(outlier_image.astype(np.float64)), cmap='gray', interpolation='none')
    a.scatter(*cat_locations[:, index], color='none', edgecolors='red', linewidths=1, label='Catalog')
    a.scatter(c1, c0, color='none', edgecolors='green', linewidths=1, label='Centroid')
    a.text(c1 + 5, c0 - 5, str(np.round(residuals[:, index], 2).astype(str)), color="yellow", fontsize=8)
    a.text(c1 - 5, c0 - 5, str((str(np.round(c1)), str(np.round(c0)))), color='green', fontsize=8)
    a.quiver(c1, c0, residuals[0, index], residuals[1, index], angles='xy', scale_units='xy', scale=1 / 200,
             color='yellow', label='Residual')
    a.set_xlim((c1 - w), (c1 + w))
    a.set_ylim((c0 - w), (c0 + w))
    a.invert_yaxis()
    a.set_title('Potential Outliers: Image {0}'.format(image_num))
    a.legend().set_draggable(True)
    ax1 = plt.axes((0.15, 0.1, .1, .05))
    ax2 = plt.axes((0.75, 0.1, .1, .05))

    callback = OutlierCallback(index, image_num, centroid, outlier_fig)
    b1 = Button(ax1, 'Remove')
    b1.on_clicked(callback.remove)

    b2 = Button(ax2, 'Keep')
    b2.on_clicked(callback.keep)
    plt.show()

    return callback
