from typing import Sequence, Literal, Callable

import copy

import matplotlib.cm as cm
import matplotlib.colors as mcol
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from giant.utilities.spherical_coordinates import unit_to_radec
from giant.rotations import rot_z
from giant.coverage.coverage_class import Coverage
from giant.coverage.utilities.project_triangles_latlon import project_triangles_latlon

from giant._typing import DOUBLE_ARRAY

STAT_SETS = Literal['v_count', 'd_albedo', 'd_x_slope', 'd_y_slope', 'd_total']
"""
Available stats to visualize.

* `v_count` is the number of times each facet was visible (in the FOV and illumintated) in the imaging plan
* `d_albedo` is the albedo dilution of precision value
* `d_x_slope` is the slope in the x direction dilution of precision value
* `d_y_slope` is the slope in the z direction dilution of precision value
* `d_total` is the RSS of the dilution of precision values
"""


def polar_plot(dop, ra, dec, observations, ax=None, cmap='coolwarm'):
    """
    This function serves to visualize the observed geometries of a single target surface element
    given a specific dilution of precision metric and the observations where that
    surface element was visible.
    
    It will generate a polar plot with the ra and dec values serving as the polar grid
    with the observation data overlayed to see the ra/dec pairs of the surface element's
    orientation when it was observed and the DOP data overlayed to see the confidence
    level in the observations at each surface element orientation.
    
    :param dop: the DOP metric of the surface element evaluated at each permutation of ra and dec values
    :param ra: np.ndarray with dtype np.float64 and shape (n, m)
               where n is the number of distinct RA values the surface normal of the element permutated through
               and m is the number of distinct Dec values the surface normal of the element permutated through
    :param dec: np.ndarray with dtype np.float64 and shape (n, m)
                where n is the number of distinct RA values the surface normal of the element permutated through
                and m is the number of distinct Dec values the surface normal of the element permutated through
    :param observations: np.array of visibility parameters for all observations where the
                         surface element was visible
    """

    z_axis = observations['normal'][0]

    east_dir = rot_z(-0.1) @ z_axis

    north = np.cross(z_axis, east_dir)
    north /= np.linalg.norm(north)

    east = np.cross(north, z_axis)
    east /= np.linalg.norm(east)

    rotation2enu = np.array([east, north, z_axis])

    valid_obs = observations[observations['visible']]
    local_inc = -rotation2enu @ valid_obs['incidence'].T
    local_emi = rotation2enu @ valid_obs['exidence'].T

    ra_sun, dec_sun = unit_to_radec(local_inc)
    ra_cam, dec_cam = unit_to_radec(local_emi)

    dop = dop.copy()

    if np.isnan(dop).any() and np.isfinite(dop).any():
        dop[~np.isnan(dop)] = np.max(dop[np.isfinite(dop)])

    norm = mcol.Normalize(vmin=0, vmax=10, clip=True)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
    mappable = ax.pcolor(ra, dec * 180 / np.pi,
                         dop.reshape(ra.shape), cmap=cmap, norm=norm, shading='auto')

    ax.scatter(ra_sun, 90 - dec_sun * 180 / np.pi, color='black', marker='*')
    ax.scatter(ra_cam, 90 - dec_cam * 180 / np.pi, color='black', marker='D')

    plt.colorbar(mappable)


class UpdateableColorScale:
    """
    This class is used to create a color scale to respresent different numerical
    data assigned to facets on either a 2D or 3D visualization of a target body.
    """

    def __init__(self, collection, color_data, cmap='hot', location='bottom', integer=True, fig=None, ax=None,
                 xbounds=(-180, 180), ybounds=(-90, 90), zbounds=None,
                 xlabel='longitude, degrees', ylabel='latitude, degrees', zlabel=None,
                 title='dop', absolute_max=100, projection=None, nan_to_inf=True,
                 slider=True, show_cbar=True):
        """
        :param collection: A collection of facets on the target body's surface
                           that are used to visualize the target's surface in the plot
        :param color_data: An np.array containing a value for each facet, which will
                           be used to determine which color the facet will be shaded
                           on the plot based on where it falls on the color scale.
                           Note that any values above the maximum of the color scale
                           will take on the maximal color, and similarly, any values below
                           the minimum of the color scale will take on the minimal color
        :param cmap: An optional string representing a color gradient to use for
                     mapping to values to the color scale
        :param location: An optional string representing the location of the
                         interactive slider
        :param integer: An optional flag to force the threshold value on the slider
                        to be rounded to the nearest integer
        :param fig: An optional pre-existing figure to provide as a destination for
                    the color scale to be applied
        :param ax: An optional pre-existing axes to provide as a destination for
                   the color scale to be applied
        :param xbounds: An optional tuple containing bounds for x-values to show in the plot
        :param ybounds: An optional tuple containing bounds for y-values to show in the plot
        :param zbounds: An optional tuple containing bounds for z-values to show in the plot
        :param xlabel: An optional string representing a label for the x-axis to show in the plot
        :param ylabel: An optional string representing a label for the y-axis to show in the plot
        :param zlabel: An optional string representing a label for the z-axis to show in the plot
        :param title: An optional string to set as the title of the plot
        :param absolute_max: An optional number to provide as the highest value the
                             color bar (and slider) can display. If the absolute_max
                             exceeds the largest value in the :attr:`color_data`,
                             the color bar (and slider) will be limited to that
                             largest value.
        :param nan_to_inf: An optional flag to convert np.nan values to np.inf in the
                           :attr:`color_data` for consistent plotting results
        :param projection: An optional string representing a matplotlib projection type
        :param slider: An optional flag to display an interactive slider in the figure.
                       The value of the slider represents the maximum value of the
                       color scale
        :param show_cbar: An optional flag to display the color bar representing
                          the numerical value associated with each color in the plot
        """
        if (fig is None) and (ax is None):
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection=projection)
        elif fig is None:
            self.fig = ax.figure # type: ignore
            self.ax = ax
        elif ax is None:
            self.fig = fig
            self.ax = self.fig.add_subplot(111, projection=projection)
        else:
            self.fig = fig
            self.ax = ax
        assert self.ax is not None
        assert self.fig is not None

        self.collection = collection

        self.color_data = color_data

        self.integer = integer

        if nan_to_inf:
            self.color_data[np.isnan(self.color_data)] = np.inf

        if slider:
            if location.lower() == 'bottom':
                self.fig.subplots_adjust(bottom=0.25)
            elif location.lower() == 'top':
                self.fig.subplots_adjust(top=0.75)
            elif location.lower() == 'left':
                self.fig.subplots_adjust(left=0.25)
            elif location.lower() == 'right':
                self.fig.subplots_adjust(left=0.75)

        if np.isfinite(self.color_data).any():
            cmax = min(self.color_data[np.isfinite(self.color_data)].max(), absolute_max)
        else:
            cmax = 1.e20

        self.norm = mcol.Normalize(vmin=0, vmax=cmax, clip=True)
        self.mapper = cm.ScalarMappable(norm=self.norm, cmap=plt.get_cmap(cmap))
        self.rgbcols = self.mapper.to_rgba(self.color_data)

        self.collection.set_facecolor(self.rgbcols)

        self.mapper.set_array(self.color_data)
        if show_cbar:
            self.cbar = self.fig.colorbar(self.mapper, ax=self.ax)
            self.cbar.update_normal(self.mapper)
        self.ax.add_collection(self.collection)

        self.ax.set_xlim(xbounds)
        self.ax.set_ylim(ybounds)

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if zlabel is not None:
            assert isinstance(self.ax, Axes3D)
            self.ax.set_zlim(zbounds)
            self.ax.set_zlabel(zlabel)
        self.ax.set_title(title)

        if slider:
            self.axcolor = self.fig.add_axes((0.15, 0.1, 0.6, 0.03))

            self.scolor = Slider(self.axcolor, 'Threshold', 0, cmax, valinit=cmax)
            self.scolor.drawon = False

            self.scolor.on_changed(self._update_slider)

    def _update_slider(self, val):
        """
        This method takes input via an interactive slider in a matplotlib figure
        and uses this input to update the color scale of the figure.
        
        :param val: A numeric value automatically input based on the position
                    of the slider. It will update the maximum value of the color
                    scale to be equal to val, which can change how the target
                    will be shaded.
        """
        if self.integer:
            cmax = int(np.round(self.scolor.val))
            self.scolor.val = cmax
            self.scolor.valtext.set_text('{}'.format(cmax))
        else:
            cmax = self.scolor.val

        self.norm.vmax = cmax
        self.mapper.norm = self.norm
        self.collection.set_facecolor(self.mapper.to_rgba(self.color_data))

        self.cbar.update_normal(self.mapper)

        self.fig.canvas.draw_idle()


def create_patches_latlon(lat: DOUBLE_ARRAY, lon: DOUBLE_ARRAY) -> list[Polygon]:
    """
    This helper function creates matplotlib Polygon patches for triangles from lat/lon arrays
    
    :param lat: the lattitude values as a 2d array.  Generally should be from the :func:`.project_triangles_latlon` function
    :param lon: the longitude values as a 2d array.  Generally should be from the :func:`.project_triangles_latlon` function
    :returns: a list of matplotlib Polygons as specified in the input
    """
    patches = []
    
    for llat, llon in zip(lat, lon):
        patches.append(Polygon(np.vstack([llon, llat]).T, closed=True, edgecolor='gray'))
    
    return patches


def percent_below_threshold_reducer(threshold: float) -> Callable[[DOUBLE_ARRAY], float]:
    """
    This helper function returns a callable which computes the percentage of values that is below some specified 
    threshold.
    
    :param threshold: the threshold to use
    :returns: a callable
    """
    def percent_below_threshold(values: DOUBLE_ARRAY) -> float:
        return (np.asarray(values, dtype=np.float64) < threshold).sum() / np.size(values) * 100
    return percent_below_threshold


def get_coloring(cov: Coverage, stat_set: STAT_SETS = "v_count", label: str | None = None, 
                 reduction_function: Callable[[DOUBLE_ARRAY], float] = percent_below_threshold_reducer(1)) -> DOUBLE_ARRAY:
    """
    This helper function chooses the appropriate values from the coverage object for displaying, applying the reduction function
    if necessary.
    
    :param cov: the :class:`.Coverage` object
    :param stat_set: the stats to get the coloring for
    :param label: the label to use if labeled analysis was performed
    :param reduction_function: a callabe to reduce all the DOP values for a facet to a single value for display
    :returns: a numpy array containing the values to color map in the display
    """
    
    match stat_set:
        case "v_count":
            if isinstance(cov.observation_count, dict):
                if label is not None:
                    coloring = cov.observation_count[label]
                else:
                    raise ValueError('labeled analyis was performed but you did not specify the label to visualize')
            else:
                coloring = np.array(cov.observation_count, dtype=np.float64)
                
        case "d_albedo":
            
            if isinstance(cov.albedo_dop, dict):
                if label is not None:
                    coloring = reduction_function(np.array(cov.albedo_dop[label])) 
                else:
                    raise ValueError('labeled analyis was performed but you did not specify the label to visualize')
            else:
                coloring = reduction_function(np.array(cov.albedo_dop))
        
        case "d_x_slope":
            
            if isinstance(cov.x_terrain_dop, dict):
                if label is not None:
                    coloring = reduction_function(np.array(cov.x_terrain_dop[label])) 
                else:
                    raise ValueError('labeled analyis was performed but you did not specify the label to visualize')
            else:
                coloring = reduction_function(np.array(cov.x_terrain_dop))
        
        case "d_y_slope":
            
            if isinstance(cov.y_terrain_dop, dict):
                if label is not None:
                    coloring = reduction_function(np.array(cov.y_terrain_dop[label])) 
                else:
                    raise ValueError('labeled analyis was performed but you did not specify the label to visualize')
            else:
                coloring = reduction_function(np.array(cov.y_terrain_dop))
    
        case "d_total":
            
            if isinstance(cov.total_dop, dict):
                if label is not None:
                    coloring = reduction_function(np.array(cov.total_dop[label])) 
                else:
                    raise ValueError('labeled analyis was performed but you did not specify the label to visualize')
            else:
                coloring = reduction_function(np.array(cov.total_dop))
                
    return np.array(coloring, dtype=np.float64)
        
        
def visualize2d(cov: Coverage, patches: None | Sequence[Polygon],  stat_set: STAT_SETS = 'v_count', label: str | None = None,
                reduction_function: Callable[[DOUBLE_ARRAY], float] = percent_below_threshold_reducer(1),
                fig: Figure | None = None, ax: Axes | None = None, cmap: str = 'hot') -> UpdateableColorScale:
    """
    This function provides a simplified interface to generate a 2D, lat/lon projected map of the statistics.
    
    For more advanced usage, use this function as a template and use the :class:`.UpdateableColorScale` directly.
    
    :param cov: the coverage object containing the coverage results
    :param patches: an option list of Polygons to use (if None, they'll be created for you)
    :param stat_set: a string specifying what stat to show
    :param label: the label to visualize (if labeled analysis was performed)
    :param reduction_function: a callabe to reduce all the DOP values for a facet to a single value for display
    :param fig: the matplitlib Figure to add the plot to (if None a new figure will be created)
    :param ax: the matplitlib Axes to add the plot to (if None a new Axes will be created)
    :param cmap: The color map to use to color the results.
    :returns: A :class:`.UpdateableColorScale` object set for display
    """
    
    if patches is None:    
        lat_tris, lon_tris = project_triangles_latlon(cov.targetvecs.T, cov.targetfacets)
        patches = create_patches_latlon(lat_tris, lon_tris)
        
    coloring = get_coloring(cov, stat_set=stat_set, label=label, reduction_function=reduction_function)
    
    return UpdateableColorScale(PatchCollection(copy.copy(patches), edgecolors='gray', linewidths=0.1), coloring, cmap=cmap, 
                                xlabel='longitude, deg', ylabel='latitude, deg', 
                                ax=ax, fig=fig, integer=stat_set == "v_count",
                                title=stat_set)
        
    
def visualize3d(cov: Coverage,  stat_set: STAT_SETS = 'v_count', label: str | None = None,
                reduction_function: Callable[[DOUBLE_ARRAY], float] = percent_below_threshold_reducer(1),
                fig: Figure | None = None, ax: Axes | None = None, cmap: str = 'hot') -> UpdateableColorScale:
    """
    This function provides a simplified interface to generate a 2D, lat/lon projected map of the statistics.
    
    For more advanced usage, use this function as a template and use the :class:`.UpdateableColorScale` directly.
    
    :param cov: the coverage object containing the coverage results
    :param stat_set: a string specifying what stat to show
    :param label: the label to visualize (if labeled analysis was performed)
    :param reduction_function: a callabe to reduce all the DOP values for a facet to a single value for display
    :param fig: the matplitlib Figure to add the plot to (if None a new figure will be created)
    :param ax: the matplitlib Axes to add the plot to (if None a new Axes will be created)
    :param cmap: The color map to use to color the results.
    :returns: A :class:`.UpdateableColorScale` object set for display
    """
    
        
    coloring = get_coloring(cov, stat_set=stat_set, label=label, reduction_function=reduction_function)
    
    collection = Poly3DCollection(cov.targetvecs[cov.targetfacets], edgecolors='gray', linewidths=0.1)
    
    shape_bounds = np.abs(cov.targetvecs).max()
    
    return UpdateableColorScale(collection, coloring, cmap=cmap,  projection='3d',
                                xbounds=(-shape_bounds, shape_bounds),
                                ybounds=(-shape_bounds, shape_bounds),
                                zbounds=(-shape_bounds, shape_bounds),
                                xlabel='x, km', ylabel='y, km', zlabel='z, km',
                                ax=ax, fig=fig, integer=stat_set == "v_count",
                                title=stat_set)
        
    
                
        
        