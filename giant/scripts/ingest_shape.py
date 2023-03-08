# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
Ingest a tessellated shape model and store it in a GIANT KDTree for use in GIANT relative OpNav processes.

In addition to building a KDTree for the ingested shape, this script also makes a "shape_info.txt" file, which contains
statistics and information about the shape model.  This file is not used explicitly in GIANT, but can be
useful in operations environments to ensure that observations are tied to the proper shape model (see the "relnav.py"
script from :ref:`getting started <getting-started>` for an example).

If your shape model is in a format not understood by this script (currently basic OBJ, ICQ, and DSK formats) then you
can either convert to one of these formats, or you can modify this script to add in the ability to read in your
preferred format.  Someday we hope to make a configurable option to make this easier, but we're running out of time for
version 1.0 right now...

.. warning::

    This script saves the results to a python pickle files.  Pickle files can be used to execute arbitrary code, so you
    should never open one from an untrusted source.  While this scrip does not open a pickle file, we wanted to
    warn you about the results of using this script.

.. warning::

    This script makes a shell call to dskexp, which may be automatically downloaded by this function if it doesn't
    exist on your path.  Unfortunately, NAIF provides no way to verify the dskexp executable, therefore, there is a
    small security risk if the naif website is hacked, however, this risk is the same as if you manually downloaded
    the dskexp function yourself.
"""

# only use pickle for dumping data here, so no real security threat in this module
import pickle  # nosec

from argparse import ArgumentParser

import os

from typing import Union

from datetime import datetime
# added check to ensure that the run call is safe
from subprocess import run  # nosec
import sys
import platform
import urllib.request

import time

import numpy as np

from giant.ray_tracer.shapes import Triangle64, Triangle32

from giant.ray_tracer.kdtree import KDTree, describe_tree

from giant.ray_tracer.utilities import compute_com

from giant.utilities.stereophotoclinometry import ShapeModel
from giant.utilities.tee import Tee

from giant.scripts.shape_stats import describe_shape, GCOEF

from giant._typing import PATH, Real


def read_obj(file: PATH, conv: Real = 1., me: bool = False) -> Union[Triangle64, Triangle32]:
    """
    Reads simply formatted obj files into the GIANT triangle format.

    Currently this is only capable of handling obj formats that only specify facets
    and vertices as 3 elements each.

    :param file:  The file to read the shape model from
    :param conv:  The conversion factor used to convert the units to kilometers
    :param me:  Use memory efficient triangles
    :return: The triangle object representing the shape
    """

    dtype = np.dtype([('type', 'U1'), ('value', float, (3, ))])

    contents = np.loadtxt(file, dtype=dtype)

    vecs = contents[contents["type"] == 'v']["value"] * conv

    facets = contents[contents["type"] == 'f']["value"].astype(int) - 1

    if me:
        return Triangle32(vecs.astype(np.float64), 1, facets.astype(np.uint32))
    else:
        return Triangle64(vecs.astype(np.float64), 1, facets.astype(np.uint32))


def read_tab(file: PATH, conv: Real = 1., me: bool = False) -> Union[Triangle64, Triangle32]:
    """
    Reads vertice/facet tables from PDS into the GIANT triangle format.
    
    Currently this has only been tested on the HELENE model as is not guarnateed to work on other project shape models.
    
    :param file:  The file to read the shape model from
    :param conv:  The conversion factor used to convert the units to kilometers
    :param me:  Use memory efficient triangles
    :return: The triangle object representing the shape
    """

    with open(file, 'r') as f:
        n_vertices, n_facets = [int(x) for x in f.readline().split()] 

        if me:
            vertices = np.loadtxt(f, dtype=np.float32, max_rows=n_vertices).reshape(n_vertices, 3) * conv
        else:
            vertices = np.loadtxt(f, dtype=np.float64, max_rows=n_vertices).reshape(n_vertices, 3) * conv

        facets = np.loadtxt(f, dtype=np.uint32, max_rows=n_facets).reshape(n_facets, 3)

    if me:
        return Triangle32(vertices, 1, facets)
    else:
        return Triangle64(vertices, 1, facets)


def process_dsk(dsk_file: PATH, conv: Real = 1., me: bool = False) -> Union[Triangle64, Triangle32]:
    """
    Ingest a DSK file into a GIANT KDTree.

    This is done by using the dskexp command from spice to convert the shape model to an OBJ first, and then using
    :func:`read_obj` to read in the generated file and finish the import.

    .. warning::

        This function makes a shell call to dskexp, which may be automatically downloaded by this function if it doesn't
        exist on your path.  Unfortunately, NAIF provides no way to verify the dskexp executable, therefore, there is a
        small security risk if the naif website is hacked, however, this risk is the same as if you manually downloaded
        the dskexp function yourself.

    :param dsk_file: The SPICE DSK file to ingest
    :param conv: A conversion scale to use to change units.  Since SPICE uses kilometers (as does GIANT usually) this
                 should typically be 1.0
    :param me:  Use memory efficient triangles
    :return: The triangle object representing the shape
    """

    # make a temporary file to store the obj file to
    tmp_file = os.path.join(os.path.curdir,
                            'tmp_shape.{}'.format(datetime.now().isoformat().replace(':', '').replace('-', '')))

    # figure out the name of the dskexp executable
    if sys.platform == "win32":
        executable = 'dskexp.exe'
    elif sys.platform == "cygwin":
        executable = 'dskexp.exe'
    else:
        executable = 'dskexp'

    try:
        # try running dskexp, catching a file not found error for if the executable is not on the path
        # put the filename in quotes to ensure that it doesn't include anything naughty
        print(' '.join([executable, '-dsk', str(dsk_file), '-text', tmp_file, '-format', 'obj']))
        run([executable, '-dsk', str(dsk_file), '-text', tmp_file, '-format', 'obj'])  # nosec
    except FileNotFoundError:
        # check to see if the user wants us to download
        print('dskexp is not available on your path')
        resp = input('Would you like to download dskexp now?\n    ')

        if resp.lower().startswith('y'):
            # download the appropriate executable
            root = "https://naif.jpl.nasa.gov/pub/naif/utilities/"

            arch64 = "64" in platform.architecture()[0]

            chmod = False

            if sys.platform == "win32":
                executable = 'dskexp.exe'
                if arch64:
                    address = root + "PC_Windows_64bit/dskexp.exe"
                else:
                    address = root + "PC_Windows_32bit/dskexp.exe"

            elif sys.platform == "cygwin":
                executable = 'dskexp.exe'
                if arch64:
                    address = root + "PC_Cygwin_64bit/dskexp.exe"
                else:
                    address = root + "PC_Cygwin_32bit/dskexp.exe"

            elif sys.platform == "linux":

                executable = './dskexp'
                if arch64:
                    address = root + "PC_Linux_64bit/dskexp"
                else:
                    address = root + "PC_Linux_32bit/dskexp"
                # for unix systems need to use chmod to make the download executable
                chmod = True

            elif sys.platform == 'darwin':
                executable = './dskexp'
                if arch64:
                    address = root + "MacIntel_OSX_64bit/dskexp"
                else:
                    address = root + "MacIntel_OSX_32bit/dskexp"

                # for unix systems need to use chmod to make the download executable
                chmod = True

            else:
                raise ValueError('unable to determine the proper download of dskexp.  Please see'
                                 'https://naif.jpl.nasa.gov/naif/utilities.html to pick the right one and add it'
                                 'to your path')

            print("downloading: {}".format(address), flush=True)
            # we made our own url name so no risk here
            urllib.request.urlretrieve(address, executable)  # nosec
            if chmod:

                # this is safe because we made the executable name ourselves
                run(["chmod",  "+x",  executable])

            # now run dskexp to make the obj file
            # put the filename in quotes to ensure that it doesn't include anything naughty
            # there is some risk here because we have no way to verify the dskexp executable from naif, but what are you
            # going to do about that?
            run([executable, '-dsk', str(dsk_file), '-text', tmp_file, '-format', 'obj'])  # nosec

        else:
            raise FileNotFoundError('dskexp is not available on the path')

    # use read_obj to import the obj file
    tris = read_obj(tmp_file, conv=conv, me=me)

    # delete the temp file
    os.remove(tmp_file)

    return tris


def _get_parser():

    warning = "WARNING: This script loads/saves some results from/to python pickle files.  " \
              "Pickle files can be used to execute arbitrary code, " \
              "so you should never open one from an untrusted source."
    parser = ArgumentParser(description='Convert a shape model file to a GIANT KDTree', epilog=warning)

    parser.add_argument('shape', help='The shape file to convert', type=str)
    parser.add_argument('name', help='The name to give the shape', type=str)
    parser.add_argument('-p', '--pole', help='The tpc file defining the pole for this shape', default='NA', type=str)
    parser.add_argument('-o', '--output', help='The name of the output file to save the tree to',
                        default='kdtree.pickle', type=str)
    parser.add_argument('-m', '--max_depth', help='The maximum depth to go to when building the tree',
                        default=18, type=int)
    parser.add_argument('-c', '--conv', help='scale the vertices by this amount (ie to convert m to km)',
                        default=1, type=float)
    parser.add_argument('-t', '--type', help='the type of the file to convert, defaults to being specified by the '
                                             'file extension', default=None)
    parser.add_argument('-g', '--gm', help='the gm of the body',
                        default=4.89e-9, type=float)
    parser.add_argument('-f', '--fix_offset', help='Correct the com to cof offset', action='store_true',
                        default=False)
    parser.add_argument('-e', '--memory_efficient', help='Use memory efficient triangles', action='store_true',
                        default=False)
    parser.add_argument('-s', '--compute_statistics', help='Compute the statistics for the shape', action='store_true',
                        default=False)

    return parser


def main():
    """
    This represents the command line routine used to convert a shape into a GIANT KDTree.
    """

    # get the cli
    parser = _get_parser()

    args = parser.parse_args()

    # get the extension of the shape model
    _, ext = os.path.splitext(args.shape)

    start = time.time()
    print('loading', flush=True)

    if args.type is None:
        if ext.lower() == '.obj':
            in_type = 'obj'
        elif ext.lower() == ".tab":
            in_type = 'tab'
        elif ext.lower() == '.txt':
            in_type = 'icq'

        elif ext.lower() == ".bds":
            in_type = 'dsk'

        else:
            raise ValueError("Don't know how to handle extension {}".format(ext))
    else:
        in_type = args.type.lower()

    if in_type == 'obj':
        tris = read_obj(args.shape, conv=args.conv, me=args.memory_efficient)
    elif in_type == "tab":
        tris = read_tab(args.shape, conv=args.conv, me=args.memory_efficient)
    elif in_type == 'icq':
        sobj = ShapeModel(args.shape)

        tris = sobj.get_triangles(me=args.memory_efficient)

    elif in_type == "dsk":
        tris = process_dsk(args.shape, conv=args.conv, me=args.memory_efficient)

    else:
        raise ValueError("Don't know how to handle type {}".format(in_type))

    print('loaded in {:.3f} seconds'.format(time.time() - start), flush=True)

    # recenter the shape at the COM assuming constant density if the user requested it
    if args.fix_offset:

        print('com offset (m):')
        offset = compute_com(tris)
        print(offset*1000)

        tris.translate(-offset)

        print('corrected offset (m):')
        print(compute_com(tris)*1000)

    start = time.time()
    # build the kd tree to the maximum depth specified by the user
    kd = KDTree(tris, max_depth=args.max_depth)

    print('building')

    kd.build(force=False, print_progress=False)
    print('built in {:.3f} seconds'.format(time.time() - start), flush=True)

    describe_tree(kd)

    if args.compute_statistics:
        print('computing statistics')
        mass = args.gm / GCOEF

        with Tee('shape_info.txt', mode='w'):
            describe_shape(kd, mass, args.name, args.pole)

    else:
        print('saving')
        with open('./shape_info.txt', 'w') as ifile:
            ifile.write(args.name + '\n')
            ifile.write('Pole: {}\n'.format(os.path.realpath(args.pole)))

    with open(args.output, 'wb') as pickfile:

        pickle.dump(kd, pickfile, protocol=pickle.HIGHEST_PROTOCOL)

    print('saved')


if __name__ == "__main__":

    main()
