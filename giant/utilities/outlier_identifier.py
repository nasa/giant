# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This module provides a function for calculating outliers in a 1 dimensional data set using Median Absolute Deviation.

This is useful for rejecting gross outliers from large data sets and is used fairly frequently internal to GIANT.  See
the :func:`.get_outliers` documentation for more details.
"""
import numpy as np

from giant._typing import ARRAY_LIKE


def get_outliers(samples: ARRAY_LIKE, sigma_cutoff: float = 4) -> np.ndarray:
    r"""
    This function can be used to identify outliers in a 1 dimensional set of data.

    It is based on the median absolute deviation algorithm:

    .. math::
        \widetilde{\mathbf{x}}=\text{median}(\mathbf{x}) \\
        mad = \text{median}(\left|\mathbf{x}-\widetilde{\mathbf{x}}\right|)

    where :math:`\widetilde{\mathbf{x}}` is the median of the data set :math:`\mathbf{x}` and :math:`mad` is the median
    absolute deviation.  Outliers are then identified by dividing the absolute deviation from the median by the median
    absolute deviation, multiplying by 1.4826 to represent a normal distribution, and then dividing by the median
    absolute deviation to compute the median absolute deviation "sigma".  This is then compared against a user specified
    sigma threshold and anything greater than or equal to this value is labeled as an outlier

    .. math::
        \sigma_{mad} = \frac{\left|\mathbf{x}-\widetilde{\mathbf{x}}\right|}{1.4826 mad}

    To use this function, simply enter a 1 dimensional data set and optionally the desired sigma threshold and you will
    get out a numpy boolean array which is True where the identified outliers are

        >>> from giant.utilities.outlier_identifier import get_outliers
        >>> import numpy as np
        >>> data = np.random.randn(5)
        >>> data[2] = data.max()*10000
        >>> get_outliers(data, sigma_cutoff=10)
        array([False, False,  True, False, False])

    To subsequently get inliers, just use the NOT operator ~

        >>> inliers = ~get_outliers(data, sigma_cutoff=10)

    :param samples: The 1 dimensional data set to identify outliers in
    :param sigma_cutoff: The sigma threshold to use when labelling outliers
    :return: A numpy boolean array with True where outliers are present in the data and False otherwise
    """

    # compute the distance each sample is from the median of the samples
    asamples = np.asanyarray(samples)
    median_distances = np.abs(asamples - np.median(asamples))

    # the median of the median distances
    median_distance = np.median(median_distances)

    # compute the median distance sigma score for each point
    median_sigmas = median_distances / (1.4826 * median_distance) if median_distance else \
        median_distances / (1.4826 * np.mean(median_distances))

    # find outliers based on the specified sigma level
    outliers = median_sigmas >= sigma_cutoff

    return outliers
