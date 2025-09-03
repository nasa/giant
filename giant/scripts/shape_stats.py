"""
Compute statistics on a shape contained in a giant kdtree saved to a pickle file.

The statistics computed include the center of mass, volume, surface area, Inertia matrix, center of mass relative
inertia matrix, moments of inertia, rotation matrix to the inertia frame, and the center of mass offset in the inertial
frame.  They are printed to stdout, and optionally to a shape_info.txt file.

.. warning::

    This script can take a long time and use a lot of resources for very high resolution shapes so be cautious.

.. warning::

    This script loads some results from python pickle files.  Pickle files can be used to execute arbitrary
    code, so you should never open one from an untrusted source.
"""

# added warning to documentation
import pickle  # nosec
from typing import Optional
from argparse import ArgumentParser


from giant.ray_tracer.kdtree import KDTree
from giant.ray_tracer.shapes import Triangle32, Triangle64
from giant.ray_tracer.utilities import compute_stats
from giant.utilities.tee import Tee


GCOEF = 6.67408e-20
"""
The universal gravity coefficient in km**3/kg/s**2
"""


def _get_parser() -> ArgumentParser:
    """
    Helper function for the argparse extension

    :return: A setup argument parser
    """
    warning = "WARNING: This script loads some results from python pickle files.  " \
              "Pickle files can be used to execute arbitrary code, " \
              "so you should never open one from an untrusted source."
    parser = ArgumentParser(description='Provide statistics about a giant shape model',
                            epilog=warning)

    parser.add_argument('shape', help='The shape file to describe', type=str)
    parser.add_argument('-g', '--gm', help='the gm of the body in km**3/s**2',
                        default=3.986004418e14/1e9, type=float)
    parser.add_argument('-n', '--name', help='The name to give the shape', type=str)
    parser.add_argument('-p', '--pole', help='The tpc file defining the pole for this shape')
    parser.add_argument('-i', '--shape_info', help='Create a shape info file for this shape. '
                                                   'The name and pole options must be specified if this is used',
                        action='store_true')

    return parser


def describe_shape(tree: KDTree, mass: float, name: Optional[str] = None, pole: Optional[str] = None):
    """
    Describe the statistics of a tessellated shape to std out.

    :param tree: The KDTree containing the tesselated shapes
    :param mass: The mass of the object, typically computed from GM
    :param name: The optional name of the target
    :param pole: The optional pole file for the object
    """

    assert isinstance(tree.surface, (Triangle64, Triangle32)), "can only compute stats for Tree on Triangles."
    com, volume, surface_area, inertia, com_inertia, moments, rotation_matrix = compute_stats(tree.surface, mass)

    if name is not None:
        print(name)
    if pole is not None:
        print('Pole: {}'.format(pole))

    print('COM (km): {}'.format(com.ravel()))
    print('Volume (km3): {}'.format(volume))
    print('Surface Area (km2): {}'.format(surface_area))
    print('Inertia Matrix:')
    print(inertia)
    print('COM Relative Inertia Matrix:')
    print(com_inertia)
    print('moments of inertia:')
    print(moments)
    print('rotation to inertia frame')
    print(rotation_matrix)
    print('com in inertia frame: {}'.format(rotation_matrix @ com.ravel()))


def main():

    parser = _get_parser()

    args = parser.parse_args()

    with open(args.shape, 'rb') as pfile:

        # added warning to documentation
        shape = pickle.load(pfile)  # nosec

    mass = args.gm / GCOEF

    if args.shape_info:
        with Tee('shape_info.txt', mode='w'):
            describe_shape(shape, mass, args.name, args.pole)

    else:
        describe_shape(shape, mass, args.name, args.pole)


if __name__ == "__main__":

    main()
