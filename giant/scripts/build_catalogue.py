# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
Build a giant star catalogue file from the UCAC4 star catalogue.

This can be run if for some reason the default catalogue file delivered with GIANT doesn't meet your needs (i.e. if it
doesn't contain high enough magnitude stars, it doesn't blend enough stars, or similar).  This script does take a while
to run so it is usually recommended to use ``nohup`` to run in the background.
"""

from giant.catalogues.giant_catalogue import build_catalogue

from argparse import ArgumentParser


def _get_parser() -> ArgumentParser:
    """
    Helper function for the argparse extension

    :return: A setup argument parser
    """

    parser = ArgumentParser('Build the giant catalogue from the UCAC4 catalogue')

    parser.add_argument('-f', '--file', help='The file to save the catalogue to', default=None,
                        type=str)
    parser.add_argument('-u', '--ucac_path', help='The path to the UCAC catalogue if it is not at the default location',
                        default=None, type=str)

    parser.add_argument('-m', '--limiting_magnitude', help='The maximum magnitude to include in the catalogue',
                        default=12, type=int)
    parser.add_argument('-b', '--blending_magnitude', help='The magnitude to try and blend dimmer stars to',
                        default=8, type=int)
    parser.add_argument('-n', '--number_of_stars', help='The number of stars that can be blended to reach the blending '
                                                        'magnitude.  Set to 0 to turn off blending',
                        default=4, type=int)

    parser.add_argument('-s', '--limiting_separation', help='The maximum distance allowed '
                                                            'for stars to be blended in degrees',
                        default=0.04, type=float)

    return parser


def main():
    """
    Parser the command line arguments and then build the giant catalogue.
    """

    parser = _get_parser()

    args = parser.parse_args()

    build_catalogue(database_file=args.file, limiting_magnitude=args.limiting_magnitude,
                    limiting_separation=args.limiting_separation, blending_magnitude=args.blending_magnitude,
                    ucac_dir=args.ucac_path)


if __name__ == '__main__':
    main()
