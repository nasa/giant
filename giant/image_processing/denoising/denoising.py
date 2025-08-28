from typing import Callable
from numpy.typing import NDArray


DENOISING_TYPE = Callable[[NDArray], NDArray]
"""
The type for a denoising function in GIANT.

Essentially it should be a callable object that takes in a grayscale 2d array and returns a denoised grayscale 2d array.

The input can be of any type and the output should either be the same as the input type or a floating type (most common).
The output should have the same shape as the input.
"""