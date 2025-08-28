from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from giant.ray_tracer.kdtree import KDTree, get_ignore_inds
from giant.ray_tracer.shapes.surface import RawSurface

from giant.coverage.utilities.project_triangles_latlon import project_triangles_latlon

from giant._typing import DOUBLE_ARRAY


def prepare_shape(shape: KDTree | RawSurface) -> tuple[Sequence[NDArray[np.integer]], tuple[DOUBLE_ARRAY, DOUBLE_ARRAY]]:
    """
    This function can prepare requisite data for coverage analysis and visualization of coverage analysis from a shape model.
    
    You can use this to get both the ignore indices you should use for a shape model (to avoid self shadowing) and 
    the projection of the shape model's triangles into a 2d lat/lon spherical projection.
    
    Note that you can easily use this function to compute these things once (they can take a while, particularly for large 
    shape models) and then save the results to a pickle file to load in future analysis to speed things up.
    
    :param shape: the shape you want to get the requisite information for 
    :returns: a tuple where the first element is a sequence of NDArrays containing the ignore indices for each vertice of the 
              shape model, and the second element is a tuple of nx3 double arrays containing the projected triangles as 
              lat, lon in degrees.
    
    """
    
    if isinstance(shape, KDTree):
        surf = shape.surface
    else:
        surf = shape
        
    facets = surf.facets
    vertices = surf.vertices
    
    if isinstance(shape, KDTree):
        ignore_inds = [np.array(get_ignore_inds(shape.root, v), dtype=np.int64).ravel() for v in range(vertices.shape[0])]
    else:
        ignore_inds = [np.argwhere((facets == v).any(axis=-1)).astype(np.int64).ravel() for v in range(vertices.shape[0])]
        
    return ignore_inds, project_triangles_latlon(vertices, facets)
        