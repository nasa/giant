"""
This module includes a helper function for identifying local maxima inside of a 2d array.
"""

import numpy as np

from numpy.typing import NDArray

from giant._typing import ARRAY_LIKE


def local_maxima(data_grid: ARRAY_LIKE) -> NDArray[np.bool_]:
    """
    This function returns a boolean mask selecting all local maxima from a 2d array.

    A local maxima is defined as any value that is greater than or equal to all of the values surrounding it.  That is,
    given:

    .. code::

        +---+---+---+
        | 1 | 2 | 3 |
        +---+---+---+
        | 4 | 5 | 6 |
        +---+---+---+
        | 7 | 8 | 9 |
        +---+---+---+

    value 5 is a local maxima if and only if it is greater than or equal to values 1, 2, 3, 4, 6, 7, 8, 9.

    For edge cases, only the valid cells are checked (ie value 1 would be checked against values 2, 4, 5 only).

    >>> from giant.image_processing import local_maxima
    >>> im = [[0, 1, 2, 20, 1],
    ...       [5, 2, 1, 3, 1],
    ...       [0, 1, 2, 10, 1],
    ...       [1, 2, -1, -2, -5]]
    >>> local_maxima(im)
    array([[False, False, False,  True, False],
           [ True, False, False, False, False],
           [False, False, False,  True, False],
           [False,  True, False, False, False]], dtype=bool)

    :param data_grid: The grid of values to search for local maximas
    :return: A 2d boolean array with `True` where the data_grid values are local maximas
    """

    # make sure the array is numpy
    array2d = np.atleast_2d(data_grid)

    # check the interior points
    test = ((array2d >= np.roll(array2d, 1, 0)) &
            (array2d >= np.roll(array2d, -1, 0)) &
            (array2d >= np.roll(array2d, 1, 1)) &
            (array2d >= np.roll(array2d, -1, 1)) &
            (array2d >= np.roll(np.roll(array2d, 1, 0), 1, 1)) &
            (array2d >= np.roll(np.roll(array2d, -1, 0), 1, 1)) &
            (array2d >= np.roll(np.roll(array2d, 1, 0), -1, 1)) &
            (array2d >= np.roll(np.roll(array2d, -1, 0), -1, 1))
            )

    # test the edges
    # test the top
    test[0] = array2d[0] >= array2d[1]
    test[0, :-1] &= (array2d[0, :-1] >= array2d[0, 1:]) & (array2d[0, :-1] >= array2d[1, 1:])
    test[0, 1:] &= (array2d[0, 1:] >= array2d[0, :-1]) & (array2d[0, 1:] >= array2d[1, :-1])

    # test the left
    test[:, 0] = array2d[:, 0] >= array2d[:, 1]
    test[:-1, 0] &= (array2d[:-1, 0] >= array2d[1:, 0]) & (array2d[:-1, 0] >= array2d[1:, 1])
    test[1:, 0] &= (array2d[1:, 0] >= array2d[:-1, 0]) & (array2d[1:, 0] >= array2d[:-1, 1])

    # test the right
    test[:, -1] = array2d[:, -1] >= array2d[:, -2]
    test[:-1, -1] &= (array2d[:-1, -1] >= array2d[1:, -1]) & (array2d[:-1, -1] >= array2d[1:, -2])
    test[1:, -1] &= (array2d[1:, -1] >= array2d[:-1, -1]) & (array2d[1:, -1] >= array2d[:-1, -2])

    # test the bottom
    test[-1] = array2d[-1] >= array2d[-2]
    test[-1, :-1] &= (array2d[-1, :-1] >= array2d[-1, 1:]) & (array2d[-1, :-1] >= array2d[-2, 1:])
    test[-1, 1:] &= (array2d[-1, 1:] >= array2d[-1, :-1]) & (array2d[-1, 1:] >= array2d[-2, :-1])

    # send out the results
    return test