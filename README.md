Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.

# NOTE
We are working a new release for GIANT hopefully before April of 2025 which will have new features and a revamped and much more performant ray tracer that fully works on windows too!
In the mean time, there is a known issue with one of the dependencies (we're not certain which but we're guessing numpy or cython) GIANT makes use of slowing down the ray tracer by about a factor of ten in our internal testing.  
With preparations for the upcoming release which will completely fix this issue, we do not have the ability to investigate further at this point in time.

# GIANT

The Goddard Image Analysis and Navigation Tool (GIANT) is a collection of utilities and
scripts that aims to make all aspects of Optical Navigation (OpNav) and Camera Calibration easy.

To install it is recommended that you use the
[mamba-forge python distribution/package manager](https://github.com/conda-forge/miniforge#mambaforge).

More detailed instructions can be found in the GIANT documentation (documentation/html/index.html or https://aliounis.github.io/giant_documentation/installation.html)

The first thing to do is to download the GIANT git repository:

    git clone git@github.com:nasa/giant.git

Once the download is complete, cd into the new giant directory

    cd giant

If your are on Windows, you will need to manually install Visual Studio 2017 using the instructions at 
https://conda-forge.org/docs/maintainer/knowledge_base.html#local-testing and imagemagick (https://imagemagick.org/script/download.php)

GIANT is designed to work with python 3.7+ so it is likely that we will need to create a new environment in mamba for
our work.  To create an environment from the terminal (or mamba prompt for windows) simply type

    mamba create -n giant_env python=3 opencv matplotlib scipy pandas numpy cython pyqt astropy lxml sphinx spiceypy c-compiler openmp astroquery dill psutil


which will create a virtual environment with python and all of the usual python tools (setuptools, pip,
etc).  Once the virtual environment has been created you can run

    conda activate giant_env

to activate the environment.  Activating the environment basically adjusts your paths/aliases so that "python", "pip",
etc now point to the virtual environment executables.  

If you are on OSX you will also likely need to set the environment variable CPATH

    export CPATH=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include

or

    export CPATH=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include

to set where to find the system headers.

If you are on Linux or OSX you can also include "imagemagick" at the end of the mamba create command to install imagemagick, or install after 
activating the environment with 

    mamba install imagemagick.

With our GIANT environment activated we are good to install GIANT.  GIANT is installed like any normal python
package.  Simply navigate to the root directory of GIANT and type (note that we use the link option here because
GIANT is still under development and this allows you to update with a simple git pull.

    cd giant
    pip install -e

This will install the GIANT package to your virtual environment along with all of its dependencies.  Once GIANT is
installed you can run some basic tests to make sure everything is working by entering the unittests directory and
typing
   
    cd unittests
    python -Wignore -m unittest discover

This might return some warnings  but at the end of it all it should say "ok" and something about all of the
tests passing.

GIANT is now installed and you can begin navigating.

See the GIANT documentation (documentation/html/index.html or https://aliounis.github.io/giant_documentation/) for more information.
