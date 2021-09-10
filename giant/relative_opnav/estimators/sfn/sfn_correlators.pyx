# cython: language_level=3
# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This module defines the spatial correlator used by :class:`.SurfaceFeatureNavigation` for subpixel locations of features
in an image.

The function provided by this module, :func:`sfn_correlator`, provides a spatial correlation routine using pearson
product moments to generate a correlation surface between the template and the image.  It does this about a predicted
location of the center of the template in the image, and takes into account 2 masks that specify when to include pixels
in the computation of a pixel score.  The computation is parallelized for speed, though, with it being a spatial
correlator, it still is slower than FFT based correlation for large seach distances.

Typically you will not use this function for anything besides doing surface feature navigation, however, if you have
another application full documentation is provided.  Even when doing surface feature navigation the user typically
doesn't interact with this class directly.
"""

from typing import Optional

import numpy as np

import cython
from cython.parallel import prange, parallel
from libc.math cimport sqrt


@cython.boundscheck(False)
cdef double compute_cor_score(double[:, :] image, double[:, :] template, unsigned char[:, :] image_mask,
                              unsigned char[:, :] template_mask, int center_row, int center_col) nogil:
    """
    This c function computes the pearson product moment correlation coefficient between the image and template overlay
    taking into account the provided masks
    """
    cdef double dot=0, temp_sum=0, temp_sum2=0, img_sum=0, img_sum2=0, denom=0
    cdef long n_valid=0
    cdef int template_row, template_col, img_row, img_col
    cdef int template_center_row = template.shape[0] // 2, template_center_col = template.shape[1] // 2
    cdef double coef=0
    cdef int template_delta_row = template.shape[0] // 2
    cdef int template_delta_col = template.shape[1] // 2
    cdef double temp_val, img_val

    cdef double img_row_max = image.shape[0]-0.5
    cdef double img_col_max = image.shape[1]-0.5

    # loop through each template row
    for template_row in range(template.shape[0]):
        # figure out the corresponding image row
        img_row = center_row - template_delta_row + template_row

        # if we're outside the image don't consider this row in the correlation score
        if (img_row <= -0.5) or (img_row >= img_row_max):
            continue

        # loop through each template column
        for template_col in range(template.shape[1]):

            # figure out the corresponding image column
            img_col = center_col - template_delta_col + template_col

            # if we're outside the image don't consider this column
            if (img_col < -0.5) or (img_col >= img_col_max):
                continue

            # check the logial or between our 2 masks
            if template_mask[template_row, template_col] or image_mask[img_row, img_col]:

                temp_val = template[template_row, template_col] 
                img_val = image[img_row, img_col]

                # increase the number of pixels included in this sum
                n_valid += 1

                # add the dot product
                dot += temp_val * img_val

                # add the other sums for the normalization
                temp_sum += temp_val
                temp_sum2 += temp_val*temp_val
                img_sum += img_val
                img_sum2 += img_val*img_val

    # as long as we considered at least 1 pixel
    if n_valid:
        # figure out the denominator for the coefficient
        denom = (temp_sum2/n_valid - temp_sum*temp_sum/n_valid/n_valid) * (img_sum2/n_valid - img_sum*img_sum/n_valid/n_valid)

        # if denom <= 0 something went wrong so don't proceed
        if denom > 0:

            # compute the pearson product moment coefficient
            coef = (dot/n_valid - temp_sum*img_sum/n_valid/n_valid)/sqrt(denom)

    return coef

@cython.boundscheck(False)
def sfn_correlator(image: np.ndarray, template: np.ndarray, space_mask: Optional[np.ndarray] = None,
                   intersects: Optional[np.ndarray] = None, search_dist: int = 10,
                   center_predicted: Optional[np.ndarray] = None) -> np.ndarray:
    """
    sfn_correlator(image, template, space_mask=None, intersects=None, search_dist=10, center_predicted=None)

    This function performs normalized cross correlation in the spatial domain over a given search distance about a
    center between an image and a template using masks.

    The correlation is performed by aligning the center of the template with various gridded points in the image
    centered on the predicted location of the center of the template in the image and extending to +/- ``search_dist``
    in both the rows and columns.  At each alignment, the Pearson Product moment correlation between the template and
    the overlaid portion of the image is computed and stored into a ``(2*search_dist+1, 2*search_dist+1)`` correlation
    surface. During the Pearson Product moment correlation computation, 2 masks are checked with a logical or to see if
    the pixel should be included in the computation. Anywhere that the template and the image do not overlap we also
    do not include in the computation of the Pearson Product moment correlation coefficient.

    The first mask, ``space_mask`` should be the same size as the image and have
    a value of ``True`` for anywhere in the image that contains empty space, and a value of ``False`` for anywhere in
    the image that contains an extended body.  The second mask, ``intersects`` should be the same size as the template
    and should have a value of ``True`` for anywhere in the template where a ray shot through the pixel intersected a
    surface and a value of ``False`` for anywhere in the template where no rays shot through the pixel intersected a
    surface.  The purpose of these masks is to exclude regions of the template that were not rendered, since we
    normally render a few pixels around the edge of the template to ensure we capture all of it, while still including
    regions of the image which included empty space.  This largely only affects templates that are very near the limb
    of the observed body.

    For the correlation surface that is returned, the center pixel (``np.array(surface.shape)//2``) represents a shift
    of 0 from the predicted template center location in the image.  Any location with a row/column less than the center
    pixel is a negative shift, a vice-versa for anywhere greater.  Therefore, the shift from the nominal location can be
    notionally found using ``shift = np.unravel_index(surface.argmax(), surface.shape) - np.array(surface.shape)//2``

    Typically this function is only used by the :class:`.SurfaceFeatureNavigation` class.  In fact, it should not be
    used as the correlator for most other classes, which assume the correlator does a global search instead of a local
    search.

    We here provide an example

    >>> import numpy as np
    >>> from giant.relative_opnav.estimators.sfn.sfn_correlators import sfn_correlator
    >>> example_image = np.random.randn(200, 200)
    >>> example_template = example_image[30:60, 45:60]
    >>> # set the intersect mask to just include part of the interior
    >>> intersect_mask = np.zeros(example_template.shape, dtype=bool)
    >>> intersect_mask[5:-5, 2:-3] = True
    >>> set the space_mask to be all False
    >>> space_mask = np.zeros(example_image.shape, dtype=bool)
    >>> surface = sfn_correlator(example_image, example_template, space_mask=space_mask, intersects=intersect_mask,
    ...                          search_dist=20, center_predicted = [40, 35])
    >>> shift = np.unravel_index(surface.argmax(), surface.shape) - np.array(surface.shape)//2
    >>> # find the location of the template center in the image
    >>> [35, 40] + shift
    array([45, 52])

    :param image: The image we are correlating with as a numpy array
    :type image: numpy.ndarray
    :param template: The template we are trying to find in the image
    :type template: numpy.ndarray
    :param space_mask: A numpy boolean array specifying which portions of the image are space and thus should be
                       included in computing the correlation score.  If ``None`` then it is assumed to all be ``True``
    :type space_mask: Optional[numpy.ndarray]
    :param intersects: A numpy boolean array specifying which portions of the template were actually intersected by rays
                       and which were empty space.  Only points intersected are included in the computation of the
                       correlation score.  If ``None`` then all points are considered.
    :type intersects: Optional[numpy.ndarray]
    :param search_dist: The number of pixels around the predicted center of the template in the image to search.
    :type search_dist: int
    :param center_predicted: The predicted center of the template in the image.  If ``None`` then it is assumed the
                             center of the template is near the center of the image.  Note that this is (x, y) or
                             (col, row)
    :type center_predicted: Optional[numpy.ndarray]
    :return: A correlation surface of shape ``(search_dist*2+1, search_dist*2+1)`` as a numpy array with values between
             -1 and 1 where 1 indicates perfect positive correlation and -1 indicates perfect negative correlation.  The
             center of the correlation surface corresponds to a shift of 0 between the expected template location and
             the found template location.  Values past the center indicate a postive shift and values before the center
             indicate a negative shift
    :rtype: numpy.ndarray
    """


    if center_predicted is None:
        center_predicted = np.array(image.shape, dtype=np.float64)/2

    cdef int center_col = int(center_predicted[0])
    cdef int center_row = int(center_predicted[1])
   
    cdef int rind, cind, row, col, n_steps_cols, n_steps_rows
    n_steps_rows = 2*search_dist + 1
    n_steps_cols = n_steps_rows

    # Generate the correlation surface to return
    cdef double[:, :] cor = np.zeros((n_steps_rows, n_steps_cols), dtype=np.float64)

    if intersects is None:
        intersects = np.ones(template.shape, dtype=np.uint8)
    if space_mask is None:
        space_mask = np.ones(image.shape, dtype=np.uint8)

    cdef double[:, :] img_mview = image.astype(np.float64)
    cdef double[:, :] templ_mview = template.astype(np.float64)
    cdef unsigned char[:, :] int_mview = intersects.astype(np.uint8)
    cdef unsigned char[:, :] spm_mview = space_mask.astype(np.uint8)

    cdef int sdist = search_dist
    cdef double temp

    # Perform correlation across image:
    with nogil, parallel():
        for rind in prange(n_steps_rows, schedule='dynamic'):
            # determine the current row we are working on
            # this corresponds to the location in the image where the center of the template is currently overlaid
            row = center_row + (-sdist + rind)
            for cind in range(n_steps_cols):
                # determine the current col we are working on
                # this corresponds to the location in the image where the center of the template is currently overlaid
                col = center_col + (-sdist + cind)

                # compute the correlation score and store it
                temp = compute_cor_score(img_mview, templ_mview, spm_mview, int_mview, row, col)
                cor[rind, cind] = temp

    return np.asarray(cor)
