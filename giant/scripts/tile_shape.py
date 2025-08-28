# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
Tile a shape model into SurfaceFeatures in a FeatureCatalog.

This script tiles an existing shape model in a GIANT format (:mod:`.shapes`, :mod:`.kdtree`, :mod:`.surface_feature`,
etc) into a :class:`.FeatureCatalgoue` of :class:`.SurfaceFeatures`.  The features are sampled uniformly from the
existing shape as square grids with a specified size and ground sample distance. Multiple GSDs can be requested in a
single run.  The resulting features can also optionally be exported to OBJs or SPC Maplets/Landmarks.

Typically this script will be used in 2 ways.  First, if there is a global high resolution of model of the target being
observed (such as from LiDAR data) then this script can be used to tile the high resolution model into features with
varying ground sample distances for direct use in :mod:`.sfn` or SPC.  Second, this script may be used when building a
shape model using only optical data.  After constructing an initial low resolution shape model using limbs, this script
can then be used to tile the low reduction model into initial features that will subsequently be refined through SPC or
other shape modelling techniques.

.. warning::

    For large number of features or when starting with a high-resolution shape model this script can take a long time to
    run.  Therefore we encourage you to use nohup to allow it to complete unhindered.

.. warning::

    This script load/saves some results from/to python pickle files.  Pickle files can be used to execute arbitrary
    code, so you should never open one from an untrusted source.
"""

from pathlib import Path
from argparse import ArgumentParser

# Added warning to documentation
import pickle  # nosec

import time
from string import ascii_uppercase
from itertools import product

import numpy as np

from giant.ray_tracer.rays import Rays
from giant.ray_tracer.shapes import Triangle64, Triangle32
from giant.ray_tracer.kdtree import KDTree
from giant.relative_opnav.estimators.sfn import SurfaceFeature, FeatureCatalog
from giant.utilities.spherical_coordinates import unit_to_radec
from giant.rotations import rot_z, Rotation
from giant.utilities.stereophotoclinometry import Landmark, Maplet


LMK_LETTERS = map(lambda x: ''.join(x), product(ascii_uppercase, ascii_uppercase))


def _get_parser():

    warning = "WARNING: This script loads/saves some results from/to python pickle files.  " \
              "Pickle files can be used to execute arbitrary code, " \
              "so you should never open one from an untrusted source."

    parser = ArgumentParser(description='Generate a feature catalog for Surface Feature Navigation (SFN) containing '
                                        'by tiling a global shape model', epilog=warning)

    parser.add_argument('shape', help='path to the shape file directory')
    parser.add_argument('-f', '--feature_output', help='The directory to save the feature results to',
                        default='./features')
    parser.add_argument('-c', '--catalog_output', help='The directory to save the feature results to',
                        default='./feature_catalog.pickle')
    parser.add_argument('-m', '--memory_efficient', help='use memory efficient triangles', action='store_true')
    parser.add_argument('-p', '--spc', help='Make spc stuff', action='store_true')
    parser.add_argument('-o', '--objs', help='directory to output objs for each feature to this location '
                                             '(leave none to not make objs)', default=None)
    parser.add_argument('-g', '--gsds', help='The gsds to build features at in meters', nargs='+',
                        default=[1.5, 1, 0.75, 0.35, 0.15], type=float)
    parser.add_argument('-s', '--size', help='The size of each side of a feature', default=201, type=int)
    parser.add_argument('-r', '--shape_radius', help='The average radius of the shape in meters.  '
                                                     'Used to determine the number of tiles to make.',
                        default=250, type=float)
    parser.add_argument('-v', '--overlap', help='The minimum overlap between each tile as a faction in  [0, 1).  '
                                                '0 indicates no minimum overlap (default) and 1 would indicate full '
                                                'overlap though that is not possible',
                        default=0, type=float)

    return parser


def fibonacci_sphere(n: int = 100) -> np.ndarray:
    """
    Generate nearly uniformally distributed points on the surface of an unit sphere using a Fibonacci sequence.

    :param n: The number of points to generate
    :return: A numpy array of the 3D unit vectors
    """

    step_size = np.sqrt(n*np.pi)

    samples = np.arange(1, n+1)

    z = np.ones(samples.shape) - (2*samples-np.ones(samples.shape))/samples.size
    phi = np.arccos(z)
    theta = step_size*phi

    sin_phi = np.sin(phi)
    x = sin_phi*np.cos(theta)
    y = sin_phi*np.sin(theta)

    points = np.vstack([x, y, z])

    return points/np.linalg.norm(points, axis=0, keepdims=True)


def determine_centers(gsd, feature_size, body_radius, overlap):

    # determine the area of each feature
    feature_area = (gsd*feature_size)**2

    # determine the surface area of the body
    surface_area = 4*np.pi*body_radius**2

    # determine the number of features we need to make
    nfeatures = int((surface_area//feature_area)/(1-overlap))+1
    print(1/(1-overlap), surface_area//feature_area, surface_area, feature_area, nfeatures)

    # get the feature locations
    feature_locations = fibonacci_sphere(nfeatures)

    return feature_locations.T


def build_feature(gsd, size, center, shape, me, odir, spc=False):
    """
    :param gsd:
    :param size:
    :param center:
    :param shape:
    :param me:
    :param odir:
    :param spc:
    :return:
    :rtype: tuple
    """

    # determine the name for this feature
    ra, dec = unit_to_radec(center)

    name = "SF_{:.2f}_{:.2f}_{:g}".format(ra*180/np.pi, dec*180/np.pi, gsd)

    # determine the initial rotation into the feature frame assuming a sphere and a east north up frame
    mz = center

    # check to be sure we aren't at a pole
    if np.abs(center[:2]).sum() > 1e-6:
        # rotate about the pole in the positive direction to get east
        mx_const = rot_z(0.01)@mz
        my = np.cross(mz, mx_const)
        my /= np.linalg.norm(my)
        mx = np.cross(my, mz)
        mx /= np.linalg.norm(mx)

    else:
        # just use body fixed
        mx = np.array([1, 0, 0])
        my = np.array([0, 1, 0])

    rotation_body_to_feature = Rotation(np.vstack([mx, my, mz]))

    # determine the rays that need to be traced
    half_size = (size - 1) // 2

    grid_x, grid_y = np.meshgrid(*[np.arange(-half_size, half_size+1)*gsd/1000]*2)

    rays = Rays(np.vstack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)]), np.array([0, 0, 1]))

    if isinstance(shape, FeatureCatalog):
        shape.include_features = None

    # rotate the shape into the feature frame
    shape.rotate(rotation_body_to_feature)

    # trace the rays to get the topography for the feature
    intersects = shape.trace(rays)

    # rotate the shape back into the body fixed frame
    shape.rotate(rotation_body_to_feature.inv())

    if not intersects['check'].all():
        print("something didn't hit", flush=True)
        return None, None, None, None

    # determine the tesselation of the feature
    indices = np.arange(grid_x.size).reshape(grid_x.shape)

    half_no_tris = indices[:-1, :-1].size
    facets = np.zeros((2*half_no_tris, 3), dtype=np.uint32)
    facets[:half_no_tris, 0] = indices[:-1, :-1].ravel()
    facets[:half_no_tris, 1] = indices[:-1, 1:].ravel()
    facets[:half_no_tris, 2] = indices[1:, :-1].ravel()
    facets[half_no_tris:, 0] = indices[1:, :-1].ravel()
    facets[half_no_tris:, 1] = indices[:-1, 1:].ravel()
    facets[half_no_tris:, 2] = indices[1:, 1:].ravel()

    # rotate the vertices back into the body fixed frame
    vertices = (rotation_body_to_feature.matrix.T@intersects['intersect'].T).T

    # determine the center of the feature frame
    feature_center = vertices[vertices.shape[0]//2]

    # build the shape
    if me:
        feature_shapes = Triangle32(vertices, intersects['albedo'], facets, compute_reference_ellipsoid=False)
    else:
        feature_shapes = Triangle64(vertices, intersects['albedo'], facets, compute_reference_ellipsoid=False)

    # build the tree
    tree = KDTree(feature_shapes, max_depth=18)

    tree.build(force=False, print_progress=False)

    # write the kdtree to the output directory
    ofile = odir / (name + ".pickle")
    with ofile.open('wb') as kfile:
        # Added warning to documentation
        pickle.dump(tree, kfile)  # nosec

    # make the feature and return it
    feature_normal = feature_shapes.normals.mean(axis=0)
    feature = SurfaceFeature(ofile.resolve(), feature_normal/np.linalg.norm(feature_normal),
                             feature_center, name, ground_sample_distance=gsd/1000)

    map_info = {'order': tree.order,
                'bounds': tree.bounding_box.vertices}

    if spc:
        lmk = Landmark()
        lmk.name = spc
        lmk.size = half_size
        lmk.scale = gsd/1000
        lmk.vlm = feature_center.ravel()
        lmk.rot_map2bod = rotation_body_to_feature.matrix.T
        
        lmkdir = Path('.') / 'LMKFILES'
        lmkdir.mkdir(parents=True, exist_ok=True)
        lmk.write(lmkdir / (spc + '.LMK'))

        maplet = Maplet()
        maplet.scale = gsd/1000
        maplet.size = half_size
        maplet.position_objmap = feature_center.ravel()
        maplet.rotation_maplet2body = rotation_body_to_feature.matrix.T
        
        heights = intersects['intersect'][:, 2] - (rotation_body_to_feature.matrix @ feature_center.ravel())[2]
        heights /= maplet.scale

        maplet.hscale = np.abs(heights).max()/np.iinfo('i2').max

        maplet.heights = heights.reshape(2*half_size+1, 2*half_size+1).T

        maplet.albedos = intersects['albedo'].reshape(2*half_size+1, 2*half_size+1).T

        maplet.albedos /= maplet.albedos.mean()

        maplet.albedos = np.clip(maplet.albedos, 0, 2.55)

        mapdir = Path('.') / 'MAPFILES'
        mapdir.mkdir(parents=True, exist_ok=True)
        maplet.write(mapdir / (spc + '.MAP'))

    return feature, vertices, facets, map_info


def main():
    parser = _get_parser()

    args = parser.parse_args()

    # make the feature directory
    feature_dir = Path(args.feature_output)
    feature_dir.mkdir(parents=True, exist_ok=True)

    # make the obj directory if requested
    obj_dir = args.objs
    if obj_dir:
        obj_dir = Path(obj_dir)
        obj_dir.mkdir(parents=True, exist_ok=True)

    # load the shape file
    with open(args.shape, 'rb') as pfile:
        # Added warning to documentation
        shape = pickle.load(pfile)  # nosec

    facet_str = 'f {} {} {}\n'
    vertex_str = 'v {} {} {}\n'
    features = []
    feature_info = []
    numbers = iter(range(1, 9999))
    letter = next(LMK_LETTERS)

    lmks = []


    # loop through each requested gsd
    for gind, gsd in enumerate(args.gsds):

        gstart = time.time()

        # determine the centers for the landmarks for this gsd
        centers = determine_centers(gsd, args.size, args.shape_radius, max(min(args.overlap, 0.9999), 0))

        # loop through each center and build a feature
        for ind, center in enumerate(centers):

            fstart = time.time()

            if args.spc:
                try:
                    spc = letter + '{:04d}'.format(next(numbers))

                except StopIteration:
                    letter = next(LMK_LETTERS)
                    numbers = iter(range(1, 9999))
                    spc = letter + '{:04d}'.format(next(numbers))

                lmks.append(spc)

            else:
                spc = False

            # build the feature
            feature, verts, tris, info = build_feature(gsd, args.size, center, shape, args.memory_efficient,
                                                       feature_dir, spc=spc)

            if feature is None:
                continue

            # if we are to save the obj
            if obj_dir is not None:
                with (obj_dir / (feature.name + ".obj")).open('w') as objfile:
                    for vec in verts:
                        objfile.write(vertex_str.format(*vec))
                    for facet in tris:
                        objfile.write(facet_str.format(*facet+1))
            features.append(feature)
            feature_info.append(info)

            print('feature {} of {} done in {:.3f} secs'.format(ind+1, len(centers), time.time()-fstart), flush=True)
        print('gsd {} of {} done in {:.3f} secs'.format(gind+1, len(args.gsds), time.time()-gstart), flush=True)

    if args.spc:
        with Path('./LMRKLIST.TXT').open('w') as ofile:
            for lmk in lmks:
                ofile.write(lmk + '\n')
            ofile.write('END')
    catalog = FeatureCatalog(features, map_info=feature_info)

    with open(args.catalog_output, 'wb') as ofile:
        # Added warning to documentation
        pickle.dump(catalog, ofile)  # nosec


if __name__ == '__main__':
    main()
