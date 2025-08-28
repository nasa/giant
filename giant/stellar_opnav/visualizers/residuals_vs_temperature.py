from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from giant._typing import PATH

from giant.stellar_opnav.stellar_class import StellarOpNav


def residuals_vs_temperature(sopnav: StellarOpNav, pdf_name: Optional[PATH] = None, show: bool = True):

    """
    This function generates a scatter plot of x and y residuals versus image temperature from the matched catalog
    stars for a given stellar opnav object.

    You must have called :meth:`~.StellarOpNav.id_stars` at least once before calling this function.

    If the ``pdf_name`` param is provided, the figures will be saved to a pdf file of the same name, and will not be
    displayed interactively.

    :param sopnav: The stellar opnav object to plot the scatters for
    :param pdf_name: Used as the file name for saving the figures to a pdf
    :param show: whether to call plt.show (only checked if pdf_name is None)
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
    ax.legend().set_draggable(True)

    plt.title('Residuals vs. Star Magnitude')
    plt.xlabel('Temperature, deg C')
    plt.ylabel('Residual, pix')

    if pdf_name:
        pdf = PdfPages(pdf_name)
        pdf.savefig(fig)
        pdf.close()
        plt.close(fig)
    elif show:
        plt.show()


