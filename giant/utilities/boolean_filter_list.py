from typing import TypeVar, Sequence
from numpy.typing import NDArray
import numpy as np

ContentsT = TypeVar("ContentsT")

def boolean_filter_list(inlist: list[ContentsT], boolean_filter: NDArray[np.bool_] | Sequence[bool]) -> list[ContentsT]:
    """
    Filter a list based on a boolean sequence or NumPy array.

    :param inlist: The input list to be filtered.
    :param boolean_filter: A boolean sequence or NumPy array used for filtering. Must have the same length as inlist.

    :returns: A new list containing only the elements from inlist
              where the corresponding boolean in boolean_filter is True.

    :raises ValueError: If the lengths of inlist and boolean_filter do not match.
    """
    
    if len(inlist) != len(boolean_filter):
        raise ValueError('the filter must be the same length as the provided list')
    
    return [inlist[ind] for ind, test in enumerate(boolean_filter) if test]