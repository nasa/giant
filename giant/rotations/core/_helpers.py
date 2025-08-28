import copy

import numpy as np

from giant._typing import ARRAY_LIKE, DOUBLE_ARRAY

def _check_array_and_shape(input: ARRAY_LIKE, 
                           return_copy: bool = False,
                           first_axis_length: int | None = None, 
                           second_last_axis_length: int | None = None, 
                           last_axis_length: int | None = None) -> DOUBLE_ARRAY:
    in_shape = np.shape(input)
    
    if not in_shape:
        raise ValueError('The input must be shaped')
    
    if first_axis_length is not None and in_shape[0] != first_axis_length:
        raise ValueError(f'The length of the first axis must be {first_axis_length}')
    
    if second_last_axis_length is not None:
        if len(in_shape) < 2 or in_shape[-2] != second_last_axis_length:
            raise ValueError(f'The length of the second to last axis must be {second_last_axis_length}')
    
    if last_axis_length is not None and in_shape[-1] != last_axis_length:
        raise ValueError(f'The length of the last axis must be {last_axis_length}')
    
    if return_copy:
        input = copy.deepcopy(input)
    
    # ensure the value is an array and break mutability
    return np.asanyarray(input, dtype=np.float64)


def _check_quaternion_array_and_shape(quaternion: ARRAY_LIKE, return_copy: bool = False) -> DOUBLE_ARRAY:
    return _check_array_and_shape(quaternion, return_copy, first_axis_length=4)


def _check_vector_array_and_shape(vector: ARRAY_LIKE, return_copy: bool = False) -> DOUBLE_ARRAY:
    return _check_array_and_shape(vector, return_copy, first_axis_length=3)
    
def _check_matrix_array_and_shape(matrix: ARRAY_LIKE, return_copy: bool = False) -> DOUBLE_ARRAY:
    return _check_array_and_shape(matrix, return_copy, second_last_axis_length=3, last_axis_length=3)

