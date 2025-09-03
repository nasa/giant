from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from giant._typing import PATH

from giant.stellar_opnav.stellar_class import StellarOpNav



def residuals_vs_magnitude(sopnav: StellarOpNav, individual_images: bool = False, pdf_name: Optional[PATH] = None,
                                show: bool = True):
    """
    This function generates a scatter plot of x and y residuals versus star magnitudes from the matched catalog
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
            star_magnitudes.extend(sopnav.matched_catalog_star_records[ind].loc[:, 'mag']) # type: ignore

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    ax.scatter(star_magnitudes, column_residuals, color='blue', label='column')
    ax.scatter(star_magnitudes, row_residuals, color='red', label='row')
    ax.legend().set_draggable(True)

    plt.title('Residuals vs. Star Magnitude')
    plt.xlabel('Star Magnitude')
    plt.ylabel('Residual, pix')

    if pdf is not None:
        pdf.savefig(fig)
        plt.close(fig)

    if individual_images:
        for ind, image in sopnav.camera:
            if image_residuals[ind]:
                fig = plt.figure()
                residuals = image_residuals[ind]
                mags = sopnav.matched_catalog_star_records[ind].loc[:, "mag"] # type: ignore
                plt.scatter(mags, residuals[0], label='column')
                plt.scatter(mags, residuals[1], label='row')
                plt.title('Residual vs Star Magnitude Image {}'.format(image.observation_date.isoformat()))
                plt.xlabel('Star Magnitude')
                plt.ylabel('Residual, pix')
                plt.legend().set_draggable(True)

                if pdf is not None:
                    pdf.savefig(fig)
                    plt.close(fig)

    if pdf is not None:
        pdf.close()
    elif show:
        plt.show()


