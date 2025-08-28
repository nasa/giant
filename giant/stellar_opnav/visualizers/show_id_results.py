from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from giant._typing import PATH

from giant.stellar_opnav.stellar_class import StellarOpNav


def show_id_results(sopnav: StellarOpNav, pdf_name: Optional[PATH] = None,
                    flattened_image: bool = False, log_scale: bool = False, 
                    show: bool = True):
    """
    This function generates a figure for each turned on image in ``sopnav`` showing the star identification results, as
    well as a couple figures showing the residuals between the predicted catalog star locations and the image star
    locations for all images combined.

    For each individual figure, the matched catalog projected star locations are shown with one marker, the matched
    image points of interest are shown with another marker, the unmatched catalog projected star locations in the
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
    :param show: A boolean flag specifying whether to "show" the figure
    """

    all_resids_fig = plt.figure()
    ax = all_resids_fig.add_subplot(111)
    big_resids = []

    if pdf_name is not None:
        pdf = PdfPages(pdf_name)
    else:
        pdf = None

    for ind, image in sopnav.camera:
        cat_locations = sopnav.matched_catalog_image_points[ind]

        if flattened_image:
            image = sopnav.point_of_interest_finder.image_flattener(image).flattened
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
            matched_extracted_points = sopnav.matched_extracted_image_points[ind]
            assert matched_extracted_points is not None
            resids = cat_locations - matched_extracted_points

            # # Plot Matched Pairs
            axt.scatter(*matched_extracted_points, color='none', edgecolors='yellow',
                        linewidths=1.5, label='Matched Centroids')
            axt.scatter(*cat_locations, marker='x', color='red', linewidths=1.5, label='Matched Catalog')
            c = np.arange(matched_extracted_points.shape[1])

            for x, y, label in zip(*cat_locations, c.astype(np.str_)):
                axt.text(x, y, label, color="white", fontsize=8)

            big_resids.append(resids)

            ax.quiver(matched_extracted_points, *resids, angles='xy', scale_units='xy',
                      color='black', width=0.0005, headwidth=20, headlength=20)
            
        # Plot Unmatched Pairs
        if (unmatched_cat_points:=sopnav.unmatched_catalog_image_points[ind]) is not None:

            in_fov = ((unmatched_cat_points >= 0) &
                      (unmatched_cat_points <= [[sopnav.model.n_cols], [sopnav.model.n_rows]])).all(axis=0)
            axt.scatter(*(unmatched_cat_points[:, in_fov]), marker='d', color='none',
                        edgecolors='cyan',
                        linewidths=0.5, label='Unmatched Catalog')
            
        if (unmatched_image_points := sopnav.unmatched_extracted_image_points[ind]) is not None:
            axt.scatter(*unmatched_image_points, marker='d', color='none', edgecolors='green',
                        linewidths=0.5, label='Unmatched Centroids')
        axt.legend().set_draggable(True)

        plt.title('Observation Date: {}'.format(image.observation_date))

        if pdf is not None:
            pdf.savefig(fig)
            fig.clear()
            plt.close(fig)

    if big_resids:
        big_resids = np.concatenate(big_resids, axis=1)
        ax.set_xlim(0.0, float(sopnav.model.n_cols))
        ax.set_ylim(0.0, sopnav.model.n_rows)
        ax.invert_yaxis()
        ax.set_xlabel('columns, pix')
        ax.set_ylabel('rows, pix')
        ax.set_title('Residuals All Images\n '
                     'std = ({0:5.3f}, {1:5.3f}) '
                     'mean = ({2:5.3f}, {3:5.3f}) pixels'.format(*np.std(big_resids, axis=1),
                                                                 *np.mean(big_resids, axis=1)))
        print(*np.std(big_resids, axis=1))
        print(*np.mean(big_resids, axis=1))
        
    if pdf is not None and pdf_name is not None:
        pdf.savefig(all_resids_fig)
        pdf.close()
        all_resids_fig.savefig(str(pdf_name) + '_combined_resids.pdf')
        all_resids_fig.clear()
        plt.close(all_resids_fig)
    elif show:
        
        plt.show()
