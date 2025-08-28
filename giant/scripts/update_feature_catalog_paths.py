# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
Update the paths for features in a feature catalog

By default, feature catalog pickle files do not include the actual geometry for tracing to enable "lazy loading" and preseve memory.
This makes it difficult to transfer feature catalogs from one directory to another, or across machines.
This script can be used to update the feature catalog paths after moving a feature catalog directory structure.
"""

from pathlib import Path
from argparse import ArgumentParser

# Added warning to documentation
import pickle  # nosec

from giant.relative_opnav.estimators.sfn import FeatureCatalog

_EPILOG = """

Note that this script will not update the file names, only the path to the files.
The individual pickle files for each feature must still exist and have the same name.

WARNING: This script loads/saves some results from/to python pickle files.  
            Pickle files can be used to execute arbitrary code, 
            so you should never open one from an untrusted source.
            
"""

def _get_parser():


    parser = ArgumentParser(description='Update the paths to individual features in a feature catalog.', 
                            epilog=_EPILOG)

    parser.add_argument('feature_catalog', help='path to the shape file directory')
    parser.add_argument('new_path', help='The new directory containing the individual pickle files with' 
                        ' the geometry for each feature')
    parser.add_argument('-o', '--output', help='the new feature catalog file. '
                        'If not specified, will overwrite the original file.', default=None)

    return parser



def main():
    parser = _get_parser()

    args = parser.parse_args()

    # make the feature directory
    fc_file = Path(args.feature_catalog).resolve()
    with fc_file.open('rb') as ifile:
        fc: FeatureCatalog = pickle.load(fc_file)
        
    # get the new path
    new_feature_dir = Path(args.new_path).resolve()
    
    # do the update
    fc.update_feature_paths(new_feature_dir)
    
    # write the results
    if args.output is None:
        out = fc_file
    else:
        out = Path(args.output)
    
    with out.open('wb') as ofile:
        pickle.dump(fc, ofile)
        
    print(f'Path for feature catalog in {fc_file} updated to {new_feature_dir}')


if __name__ == '__main__':
    main()
