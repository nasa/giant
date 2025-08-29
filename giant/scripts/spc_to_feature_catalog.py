"""
This script provides a utility for converting a SPC set of Maplets into a GIANT feature catalog for use with GIANT
Surface Feature Navigation (SFN).

.. warning::

    This script saves some results to python pickle files.  Pickle files can be used to execute arbitrary
    code, so you should never open one from an untrusted source.
"""

# added warning
import pickle  # nosec
import time
from argparse import ArgumentParser
import os
from itertools import repeat
from typing import Tuple, List, Literal
from glob import glob
from multiprocessing import Pool, cpu_count
import pathlib

from giant._typing import PATH
from giant.ray_tracer.kdtree import KDTree
from giant.relative_opnav.estimators.sfn import SurfaceFeature, FeatureCatalog
from giant.utilities.stereophotoclinometry import Maplet


def _get_parser():

    warning = "WARNING: This script saves some results to python pickle files.  " \
              "Pickle files can be used to execute arbitrary code, " \
              "so you should never open one from an untrusted source."

    parser = ArgumentParser(
        description='Generate a feature catalog for Surface Feature Navigation (SFN) '
                    'containing locations of maplet topography files.', epilog=warning)

    parser.add_argument('shape', help='path to the shape file directory')
    parser.add_argument('-f', '--filter', help='a list of the landmark subset to be used', default=None)
    parser.add_argument('-o', '--output', help='The file to save the results to', default='./spc_maps.pickle')
    parser.add_argument('-d', '--output_dir', help='The directory to save the feature files to',
                        default=None)
    parser.add_argument('--triangles_32',
                        help='Use the memory efficient (32 bit) triangles instead of the 64 bit ones', action='store_true')
    parser.add_argument('-u', '--update',
                        help='use existing kdtree if available', action='store_true')


    return parser


def build_feature(inp: Tuple[int, Tuple[PATH, int, Literal[32, 64], PATH, bool]]) -> Tuple[SurfaceFeature, dict]:
    """
    Load a maplet and convert it into a GIANT SurfaceFeature, returning the created feature.
    :param inp: the inputs that are needed as a tuple (current_index, (maplet_file, number_of_maplets,
                memory_efficient_flag, output_directory))
    :return: The surface feature and a dictionary containing the keys order and bounds about the feature
    """

    ind, (file, n_maps, precision, output, update) = inp

    start = time.time()

    maplet = Maplet(file_name=file)

    print(f'{file} -- loaded', flush=True)

    # make the output path
    shape_path = pathlib.Path(output) / (maplet.name + '.pickle')

    if update:
        if os.path.exists(shape_path):
            try:
                with open(shape_path, 'rb') as ifile:
                    kd = pickle.load(ifile)

                map_info = {'order': kd.order,
                            'bounds': kd.bounding_box.vertices}

                # Store path to pickle into SurfaceFeature:
                feat = SurfaceFeature(shape_path.resolve(), 
                                      maplet.rotation_maplet2body[:, 2], 
                                      maplet.position_objmap, 
                                      maplet.name,
                                      maplet.scale)

                print('map {} of {} finished in {:.3f} seconds'.format(ind, n_maps, time.time() - start), flush=True)

                return feat, map_info

            except:
                pass

    tris = maplet.get_triangles(triangle_precision=precision)

    print(f'{file} -- tessellated', flush=True)

    kd = KDTree(tris, max_depth=11)

    kd.build(print_progress=False, force=False)

    print(f'{file} -- built', flush=True)

    # Write KD tree as .pickle:
    map_info = {'order': kd.order,
                'bounds': kd.bounding_box.vertices}

    with open(shape_path, 'wb') as f:
        pickle.dump(kd, f)

    # Store path to pickle into SurfaceFeature:
    feat = SurfaceFeature(shape_path.resolve(), 
                          maplet.rotation_maplet2body[:, 2], 
                          maplet.position_objmap, 
                          maplet.name,
                          maplet.scale)

    print('map {} of {} finished in {:.3f} seconds'.format(ind, n_maps, time.time() - start), flush=True)

    return feat, map_info


def main():
    """
    The main code that is run
    """

    parser = _get_parser()

    args = parser.parse_args()

    shape_path = args.shape
    if args.filter is not None:
        map_files: List[str] = []
        filter_file = args.filter
        try:
            with open(filter_file, mode='r') as infile:
                for line in infile:
                    if 'END' not in line:
                        # noinspection SpellCheckingInspection
                        temp = shape_path + '/MAPFILES/' + line.strip() + '.MAP'
                        map_files.append(temp)
        except FileNotFoundError:
            # noinspection SpellCheckingInspection
            map_files = sorted(glob(shape_path + '/MAPFILES/*.MAP'))[::int(args.filter)]

    else:
        # noinspection SpellCheckingInspection
        map_files = glob(shape_path + '/MAPFILES/*.MAP')

    if args.output_dir is None:

        output_dir = (pathlib.Path(shape_path) / 'pickle_files')
    else:
        output_dir = pathlib.Path(args.output_dir)

    output_dir.mkdir(exist_ok=True)

    n_maps = len(map_files)
    precision: Literal[32, 64] = 32 if args.triangles_32 else 64

    with Pool(cpu_count()//2) as pool:
        res = pool.map(build_feature, enumerate(zip(map_files,
                                                    repeat(n_maps), repeat(precision), repeat(output_dir),
                                                    repeat(args.update)))) # pyright: ignore[reportArgumentType]

    sfs = [r[0] for r in res]
    map_info = [r[1] for r in res]

    fc = FeatureCatalog(sfs, map_info=map_info)

    start = time.time()

    out_bytes = pickle.dumps(fc, protocol=pickle.HIGHEST_PROTOCOL)

    print('serialized in {:.3f} seconds'.format(time.time() - start))

    start = time.time()

    chunk_size = 2 ** 30
    n_chunks = len(out_bytes) // chunk_size
    with open(args.output, 'wb') as feature_catalog_file:

        for n, idx in enumerate(range(0, len(out_bytes), chunk_size)):
            local_start = time.time()

            feature_catalog_file.write(out_bytes[idx:idx + chunk_size])

            print('chunk {} of {} written in {} seconds'.format(n, n_chunks, time.time() - local_start))

    print('written in {} seconds'.format(time.time() - start))


if __name__ == "__main__":
    main()
