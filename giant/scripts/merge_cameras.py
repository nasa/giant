# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
Merge camera files together into a single file.

Camera files are assumed to be stored in dill or pickle files and are store in a dill file.  The camera files should
contain the same :class:`.Camera` class with different images.  This script does not enforce this though so be careful
as you may experience unexpected results if you try to merge files containing different :class:`.Camera` classes.

.. warning::

    This script loads/saves some results from/to python pickle files.  Pickle files can be used to execute arbitrary
    code, so you should never open one from an untrusted source.

"""
from argparse import ArgumentParser

# added a warning to the documentation
import dill  # nosec

from giant._typing import PATH
from giant.camera import Camera


def _load_camera(cfile: PATH) -> Camera:
    """
    Loads and returns a camera file using dill, which can read both dill and pickle files.

    :param cfile: the pickle or dill file containing the camera object
    :return: The loaded camera object
    """

    with open(cfile, 'rb') as dfile:
        # added a warning to the documentation
        cam = dill.load(dfile)  # nosec

    return cam


def _get_parser() -> ArgumentParser:
    """
    Helper function for the argparse extension

    :return: A setup argument parser
    """

    warning = "WARNING: This script loads/saves some results from/to python pickle files.  " \
              "Pickle files can be used to execute arbitrary code, " \
              "so you should never open one from an untrusted source."
    parser = ArgumentParser(description='Merge images between two camera files. '
                                        'NOTE: The cameras must be the same as this will only merge the images.',
                            epilog=warning)
    parser.add_argument('cameras', help='The camera files to merge', type=_load_camera, nargs='+')

    parser.add_argument('-o', '--output', help='The file to store the merged camera object', default='./camera.dill')

    return parser


def main():

    parser = _get_parser()

    args = parser.parse_args()

    # get the first camera object
    camera1 = args.cameras.pop(0)

    # merge in the other camera data
    for camera2 in args.cameras:
        camera1.add_images(camera2.images, preprocessor=False, parse_data=False)

    # sort the images
    camera1.sort_by_date()

    # dump the merged object to a file
    with open(args.output, 'wb') as ofile:

        dill.dump(camera1, ofile)


if __name__ == '__main__':

    main()

