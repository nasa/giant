from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from giant._typing import PATH

from giant.stellar_opnav.stellar_class import StellarOpNav


def residual_histograms(sopnav: StellarOpNav, individual_images: bool = False, pdf_name: Optional[PATH] = None, show: bool = True):
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
    :param show: whether to call plt.show (only checked if pdf_name is None)
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
    plt.legend().set_draggable(True)

    if pdf is not None:
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
                plt.legend().set_draggable(True)

                if pdf is not None:
                    pdf.savefig(fig)
                    plt.close(fig)

    if pdf is not None:
        pdf.close()
    elif show:
        plt.show()


