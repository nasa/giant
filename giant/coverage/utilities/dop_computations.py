

import numpy as np
from numpy.typing import NDArray

from giant.ray_tracer.illumination import IlluminationModel
from giant.rotations.core.elementals import rot_z
from giant.utilities.spherical_coordinates import radec_to_unit

from giant.coverage.typing import DOP_TYPE, JACOBIAN_TYPE

from giant._typing import DOUBLE_ARRAY



class DOPComputations:
    """
    This class serves as a container for methods that compute dilution of precision (DOP) metrics for given
    illumination observation data.
    
    This is used to make it easy to distribute the DOP computations across multiple cores using multiprocessing.
    Generally, you should not use this as the :class:`.Coverage` class should set everything up for you.
    """

    def __init__(self, az_grid: DOUBLE_ARRAY, elev_grid: DOUBLE_ARRAY,
                 brdf: IlluminationModel, visibility: NDArray | None = None):
        """
        :param az_grid: An array of azimuth angles in radians with shape (n, m)
                        where n is the number of distinct azimuth values the facet's surface normal will permutate through,
                        and m is the number of distinct elevation values the facet's surface normal will permutate through
        :param elev_grid: An array of elevation angles in radians with shape (n, m)
                         where n is the number of distinct azimuth values the facet's surface normal will permutate through,
                         and m is the number of distinct elevation values the facet's surface normal will permutate through
        :param brdf: An :class:`IlluminationModel` representing a Bidirectional
                     Reflectance Distribution Function (BRDF) used to compute the
                     jacobian matrix of the change in the illumination values
                     given a change in the surface normal and/or albedo.
        :param visibility: An optional array of observations of the target body
                          :data:`ILLUM_DTYPE` values with shape (i, f)
                          where i is the number of images and f is the number of
                          surface elements for the shape model being used
                          (surface elements can be vertices or facets).
                          Each entry characterizes the illumination geometry
                          of a surface element at a specific time so that
                          DOP metrics can be computed from the surface normal vector
                          provided by each entry.
        """

        self.visibility = visibility
        self.az_grid = az_grid
        self.elev_grid = elev_grid
        self.brdf = brdf

    @staticmethod
    def _combine_results(jacobians: dict[str, JACOBIAN_TYPE]) \
            -> tuple[JACOBIAN_TYPE, DOP_TYPE, DOP_TYPE, DOP_TYPE, DOP_TYPE]:
        """
        This method combines jacobians for different time labels together and extracts
        DOP metrics from the combined jacobians.

        The combined DOP values for albedo, x slope, y slope, and rss are computed from the diagonal
        of the inverse of the observability Gramian matrix (H^T*W*H)^(-1) where H is the jacobian and
        W is a weighting matrix (in this code W is just the identity matrix)
        
        :param jacobians: A dictionary with the keys being labels for imaging times,
                          and the values being 2D lists of illumination jacobians
        
        :return: A tuple of the following data, each having the length of the number
                 of permutations the facet's surface normal vector was oriented through:\n
                 The combined jacobians,\n
                 The albedo DOP values,\n
                 The x slope DOP values,\n
                 The y slope DOP values, and\n
                 The RSS of the DOP values\n
        """

        big_jacs = []
        alb_dops = []
        xt_dops = []
        yt_dops = []
        rss_dops = []

        for jacs in zip(*jacobians.values()):

            jacs = [jac for jac in jacs if jac is not None]

            if jacs:
                jacs.append(np.eye(3, dtype=np.float64) * np.sqrt(5e-3))
                current_jac = np.vstack(jacs).astype(np.float64)
                big_jacs.append(current_jac)

                dop = np.linalg.inv(current_jac.T @ current_jac)

                alb_dops.append(np.sqrt(dop[2, 2]))
                xt_dops.append(np.sqrt(dop[0, 0]))
                yt_dops.append(np.sqrt(dop[1, 1]))
                rss_dops.append(np.sqrt(np.trace(dop)))

            else:
                big_jacs.append(None)
                alb_dops.append(np.inf)
                xt_dops.append(np.inf)
                yt_dops.append(np.inf)
                rss_dops.append(np.inf)

        return big_jacs, alb_dops, xt_dops, yt_dops, rss_dops

    def _compute_dop(self, visibility: NDArray) \
            -> tuple[int, JACOBIAN_TYPE, DOP_TYPE, DOP_TYPE, DOP_TYPE, DOP_TYPE]:
        """
        This method takes a list of observations for a certain facet on the target's
        surface where the facet is determined to be visible and computes DOP metrics.
        
        It permutates the facet's surface normal vector through every combination
        of azimuth and elevation values as specified in :attr:`.az_grid` and :attr:`.elev_grid`,
        and uses these orientations to compute the jacobian using the
        :meth:`.compute_photoclinometry_jacobian` method for the specified :attr:`.brdf`.
        
        Once it has a jacobian, it computes DOP values for albedo, x slope, y slope, and rss
        from the diagonal of the inverse of the observability Gramian matrix (H^T*W*H)^(-1)
        where H is the jacobian and W is a weighting matrix (in this code W is just the identity matrix)
        
        :param visibility: An array of :data:`ILLUM_DTYPE` values representing the
                           illumination geometry of a certain facet, with length i
                           where i is the number of observations where the facet is visible
        
        :return: A tuple of the following data, each having the length of the number
                 of permutations the facet's surface normal vector was oriented through:\n
                 The number of usable observations for the given facet,\n
                 The jacobians,\n
                 The albedo DOP values,\n
                 The x slope DOP values,\n
                 The y slope DOP values, and\n
                 The RSS of the DOP values\n
        """

        obs_count: int = visibility.shape[0]

        jacobians = []

        if obs_count > 0:
            current_hx, current_hy, current_alb, current_tot = [], [], [], []

            # define the local ENU frame
            # define the z axis to be in line with the normal vector for the facet in the body-fixed frame
            z_axis = visibility['normal'][0].copy()

            # as long as we aren't at the north or south pole
            if (np.abs(z_axis.ravel()) != [0, 0, 1]).all():
                # rotate about the body-fixed z axis to get the vector constraining the east direciton
                east_dir = rot_z(0.1) @ z_axis

                # cross the z axis and the east direction constraint vector to get the north vector
                north = np.cross(z_axis, east_dir)
                north /= np.linalg.norm(north)

                # complete the right handed system
                east = np.cross(north, z_axis)
                east /= np.linalg.norm(east)

                # form the rotation matrix from the local frame to the body-fixed frame
                local_enu_to_body_fixed = np.array([east, north, z_axis]).T

            else:  # if the local normal is the north or south pole then use the body-fixed frame
                # complete the right handed system
                x_axis = [1, 0, 0]
                y_axis = [0, 1, 0] if z_axis.ravel()[2] > 0 else [0, -1, 0]

                # form the rotation matrix from the local frame to the body-fixed frame
                local_enu_to_body_fixed = np.array([x_axis, y_axis, z_axis]).T

            counts = []

            for az, elev in zip(self.az_grid.ravel(), self.elev_grid.ravel()):

                # get the shifted normal vector in the body-fixed frame
                visibility['normal'] = local_enu_to_body_fixed @ radec_to_unit(az, np.pi / 2 - elev).ravel()

                current_jac, valid = self.brdf.compute_photoclinometry_jacobian(visibility, local_enu_to_body_fixed)
                counts.append(valid.sum())

                if counts[-1] >= 3:
                    current_jac = current_jac[valid]

                    jacobians.append(current_jac[:-3])
                    try:
                        current_dop = np.linalg.inv(current_jac.T @ current_jac)
                    except np.linalg.linalg.LinAlgError:
                        current_alb.append(np.inf)
                        current_hx.append(np.inf)
                        current_hy.append(np.inf)
                        current_tot.append(np.inf)
                        jacobians.append(None)
                        continue

                    current_total_dop = np.sqrt(current_dop.trace())

                    current_alb.append(np.sqrt(current_dop[2, 2]))
                    current_hx.append(np.sqrt(current_dop[0, 0]))
                    current_hy.append(np.sqrt(current_dop[1, 1]))
                    current_tot.append(current_total_dop)

                else:
                    current_alb.append(np.inf)
                    current_hx.append(np.inf)
                    current_hy.append(np.inf)
                    current_tot.append(np.inf)
                    jacobians.append(None)

        else:

            jacobians = [None] * self.az_grid.size
            current_alb = [np.inf] * self.az_grid.size
            current_hx = [np.inf] * self.az_grid.size
            current_hy = [np.inf] * self.az_grid.size
            current_tot = [np.inf] * self.az_grid.size

        return obs_count, jacobians, current_alb, current_hx, current_hy, current_tot

    def compute_target_dop_facet(self, visibility: NDArray) \
            -> tuple[int, JACOBIAN_TYPE, DOP_TYPE, DOP_TYPE, DOP_TYPE, DOP_TYPE] | \
               tuple[dict[str, int], dict[str, JACOBIAN_TYPE], dict[str, DOP_TYPE], dict[str, DOP_TYPE], dict[str, DOP_TYPE], dict[str, DOP_TYPE]]:
        """
        This method takes a list of observations for a certain facet on the target's
        surface and computes DOP metrics based on the observations where the facet is visible.
        
        :param visibility: An array of :data:`ILLUM_DTYPE` values representing the
                           illumination geometry of a certain facet, with length i
                           where i is the number of images. Each entry characterizes
                           the illumination geometry of the facet at a specific time
                           so that DOP metrics can be computed from the surface
                           normal vector provided by each entry
        
        Note that if labels are being used to handle multiple sets of imaging times,
        the DOP metrics will be calculated for each label and then a set of combined
        DOP metrics will be computed for all imaging times. Each of these computed
        metrics will be stored as values in a dictionary with each label being the
        keys along with an "all" label for the combined DOP metrics.
        
        :return: A tuple of the following data, each having the length of the number
                 of permutations the facet's surface normal vector was oriented through:\n
                 The number of usable observations for the given facet,\n
                 The jacobians,\n
                 The albedo DOP values,\n
                 The x slope DOP values,\n
                 The y slope DOP values, and\n
                 The RSS of the DOP values\n
        """
        if 'label' in visibility.dtype.names:  # type: ignore

            labels = np.unique(visibility['label'])

            jacobians: dict[str, JACOBIAN_TYPE] = {}
            observation_count: dict[str, int] = {}
            x_terrain_dop: dict[str, DOP_TYPE] = {}
            y_terrain_dop: dict[str, DOP_TYPE] = {}
            albedo_dop: dict[str, DOP_TYPE] = {}
            rss_dop: dict[str, DOP_TYPE] = {}

            # compute dop for each label
            for label in labels:
                test = visibility['visible'] & (visibility['label'] == label)
                (observation_count[label], jacobians[label], albedo_dop[label],
                 x_terrain_dop[label], y_terrain_dop[label], rss_dop[label]) = self._compute_dop(visibility[test])

            # compute overall dop

            observation_count['all'] = sum(observation_count.values())
            if observation_count['all'] > 0:
                (jacobians['all'], albedo_dop['all'], x_terrain_dop['all'],
                 y_terrain_dop['all'], rss_dop['all']) = self._combine_results(jacobians)
            else:
                jacobians['all'] = [None] * self.az_grid.size
                albedo_dop['all'] = [np.inf] * self.az_grid.size
                x_terrain_dop['all'] = [np.inf] * self.az_grid.size
                y_terrain_dop['all'] = [np.inf] * self.az_grid.size
                rss_dop['all'] = [np.inf] * self.az_grid.size
                
            return observation_count, jacobians, albedo_dop, x_terrain_dop, y_terrain_dop, rss_dop
                
        else:
             return self._compute_dop(visibility[visibility['visible']])


    def compute_target_dop(self, index: int, normal: NDArray[np.float64]) \
            -> tuple[int, JACOBIAN_TYPE, DOP_TYPE, DOP_TYPE, DOP_TYPE, DOP_TYPE] | \
               tuple[int, JACOBIAN_TYPE, None, None, None, None]:
        """
        This method takes a surface element index and a surface normal vector to
        compute the DOP metrics for the surface element at various potential orientations.
        
        The surface element index is used to access sorted data in a visibility matrix,
        which corresponds to a specific vertex or facet on the surface.
        
        The normal vector is permutated through every combination of azimuth and elevation values as specified in
        :attr:`.az_grid` and :attr:`.elev_grid`, and these orientations are used to compute the jacobian using the
        :meth:`compute_photoclinometry_jacobian` method for the specified :attr:`.brdf`.
        
        Once it has a jacobian, it computes DOP values for albedo, x slope, y slope, and rss
        from the diagonal of the inverse of the observability Gramian matrix (H^T*W*H)^(-1)
        where H is the jacobian and W is a weighting matrix (in this code W is just an identity matrix)
        
        :param index: An integer index representing which surface element of the visibility
                      matrix is being used to compute DOP metrics
        :param normal: A surface normal vector corresponding to the specified surface
                       element that will be rotated through various orientations
        
        :return: A tuple of the following data, each having the length of the number
                 of permutations the normal vector was oriented through:\n
                 The number of usable observations for the given surface element,\n
                 The jacobians,\n
                 The albedo DOP values,\n
                 The x slope DOP values,\n
                 The y slope DOP values, and\n
                 The RSS of the DOP values\n
        """

        assert self.visibility is not None, "must call compute_visibility before compute_target_dop"
        visibility = self.visibility[:, index]

        observation_count: int = visibility['visible'].any(axis=-1).sum()

        jacobians: JACOBIAN_TYPE = []

        if visibility['visible'].any():

            current_hx, current_hy, current_alb, current_tot = [], [], [], []

            visibility = visibility[visibility['visible']]

            # determine the local frame
            # use the normal vector for the target as the z axis
            z_axis = normal

            if (np.abs(z_axis.ravel()) != [0, 0, 1]).all():
                east_dir = rot_z(0.1) @ z_axis

                north = np.cross(z_axis, east_dir)
                north /= np.linalg.norm(north)

                east = np.cross(north, z_axis)
                east /= np.linalg.norm(east)

                # form the rotation matrix from the local frame to the body-fixed frame
                local_enu_to_body_fixed = np.array([east, north, z_axis]).T

            else:  # if the local normal is the north or south pole then use the body-fixed frame
                # complete the right handed system
                x_axis = [1, 0, 0]
                y_axis = [0, 1, 0] if z_axis.ravel()[2] > 0 else [0, -1, 0]

                # form the rotation matrix from the local frame to the body-fixed frame
                local_enu_to_body_fixed = np.array([x_axis, y_axis, z_axis]).T

            for az, elev in zip(self.az_grid.ravel(), self.elev_grid.ravel()):

                # get the normal shifted normal vector in the body-fixed frame
                visibility['normal'] = local_enu_to_body_fixed @ radec_to_unit(az, 90 - elev).ravel()

                current_jac, valid = self.brdf.compute_photoclinometry_jacobian(visibility, local_enu_to_body_fixed)

                if valid.any():
                    current_jac = current_jac[valid]

                    jacobians.append(current_jac)

                    current_dop = np.linalg.pinv(current_jac.T @ current_jac)

                    current_total_dop = np.sqrt(current_dop.trace())

                    current_alb.append(np.sqrt(current_dop[2, 2]))
                    current_hx.append(np.sqrt(current_dop[0, 0]))
                    current_hy.append(np.sqrt(current_dop[1, 1]))
                    current_tot.append(current_total_dop)

                else:
                    current_alb.append(np.inf)
                    current_hx.append(np.inf)
                    current_hy.append(np.inf)
                    current_tot.append(np.inf)

            albedo_dop: DOP_TYPE = current_alb
            x_terrain_dop: DOP_TYPE = current_hx
            y_terrain_dop: DOP_TYPE = current_hy
            all_total_dop: DOP_TYPE = current_tot
            
            return observation_count, jacobians, albedo_dop, x_terrain_dop, y_terrain_dop, all_total_dop

        else:

            return observation_count, jacobians, None, None, None, None
