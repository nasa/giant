import numpy as np
from numpy.typing import NDArray

from giant._typing import DOUBLE_ARRAY

        
def project_triangles_latlon(vertices: DOUBLE_ARRAY, facets: NDArray[np.integer]) -> tuple[DOUBLE_ARRAY, DOUBLE_ARRAY]:
    """
    This helper function project 3d triangles into 2d triangles on a lat/lon projection.
    
    :param vertices: The vertices of the shape model
    :param facets: the facet map for the triangles
    :returns: The triangles projected using lattitude/longitude accounting for wrapping as a tuple of 2 nx3 arrays lat, lon
    """
    
    assert vertices.shape[-2] == 3, "Vertices must be a ...x3xn array"
    
    r = np.linalg.norm(vertices, axis=0)
    
    lat = np.rad2deg(np.arcsin(vertices[..., 2, :]/r))
    lon = np.rad2deg(np.arctan2(vertices[..., 1, :], vertices[..., 0, :]))
    
    wrapped_lon, wrapped_lat = [], []

    for llat, llon in zip(lat[facets], lon[facets]):

        llat_sign = np.sign(llat)
        count = (llat_sign >= 0).sum()

        if 0 < count < 3:
            if -(llat[llat_sign == -1] - llat[llat_sign != -1]).max() > 50:
                print(llat)
                llat[llat_sign == -1] += 180
        elif -3 < count < 0:
            if (llat[llat_sign == 1] - llat[llat_sign != 1]).max() > 50:
                llat[llat_sign == 1] -= 180

        llon_sign = np.sign(llon)
        count = (llon_sign >= 0).sum()

        if 0 < count < 3:
            if -(llon[llon_sign == -1] - llon[llon_sign != -1]).max() > 90:
                llon[llon_sign == -1] += 360
        elif -3 < count < 0:
            if (llon[llon_sign == 1] - llon[llon_sign != 1]).max() > 90:
                llon[llon_sign == 1] -= 360

        wrapped_lat.append(llat)
        wrapped_lon.append(llon)
        
    return np.array(wrapped_lat), np.array(wrapped_lon)
