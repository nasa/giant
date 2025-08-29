"""
Build a star catalog file from the Gaia star catalog.

This can be run if for some reason the default catalog file delivered with GIANT doesn't meet your needs (i.e. if it
doesn't contain high enough magnitude stars, it doesn't blend enough stars, or similar).  This script does take a while
to run so it is usually recommended to use ``nohup`` to run in the background.
"""

from giant.catalogs.gaia import build_catalog

from argparse import ArgumentParser


def _get_parser() -> ArgumentParser:
    """
    Helper function for the argparse extension

    :return: A setup argument parser
    """

    parser = ArgumentParser('Build the GIANT catalog from the Gaia catalog or an already existing GIANT catalog')

    parser.add_argument('-f', '--file', help='The file to save the catalog to', default=None,
                        type=str)
    parser.add_argument('-g', '--giant_catalog_file', help='The path to the GIANT catalog file if rebuilding '
                                                             'catalog with smaller magnitude and/or different '
                                                             'star blending options',
                        default=None, type=str)
    parser.add_argument('-m', '--limiting_magnitude', help='The maximum magnitude to include in the catalog',
                        default=14., type=float)
    parser.add_argument('-b', '--blending_magnitude', help='The magnitude to try and blend dimmer stars to',
                        default=8., type=float)
    parser.add_argument('-s', '--limiting_separation', help='The maximum distance allowed '
                                                            'for stars to be blended in degrees',
                        default=0.04, type=float)
    parser.add_argument('-n', '--number_of_stars', help='The number of stars that can be blended to reach the blending '
                                                        'magnitude.  Set to 0 to turn off blending',
                        default=0, type=int)

    return parser


def main():
    """
    Parser the command line arguments and then build the GIANT catalog.
    """

    parser = _get_parser()

    args = parser.parse_args()

    build_catalog(catalog_file=args.file, giant_catalog_file=args.giant_catalog_file,
                    limiting_magnitude=args.limiting_magnitude, blending_magnitude=args.blending_magnitude,
                    limiting_separation=args.limiting_separation, number_of_stars=args.number_of_stars)


if __name__ == '__main__':
    main()
