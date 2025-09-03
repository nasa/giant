## Notices:
Copyright © 2024 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
 
## Disclaimer:
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."
Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.

# GIANT

The Goddard Image Analysis and Navigation Tool (GIANT) is a collection of utilities and
scripts that aims to make all aspects of Optical Navigation (OpNav) and Camera Calibration easy.

To install it is recommended that you use the
[mini-forge python distribution/package manager](https://conda-forge.org/download/).

More detailed instructions can be found in the GIANT documentation (documentation/html/index.html or https://aliounis.github.io/giant_documentation/installation.html)

The first thing to do is to download the GIANT git repository:

    git clone git@github.com:nasa/giant.git

Once the download is complete, cd into the new giant directory

    cd giant

GIANT is designed to work with python 3.11+ so it is likely that we will need to create a new environment in mamba for
our work.  To create an environment from the terminal (or mamba prompt for windows) simply type

    mamba env create -f environment.yml


which will create a virtual environment with python called giant and all of the usual python tools (setuptools, pip,
etc).  Once the virtual environment has been created you can run

    conda activate giant

to activate the environment.  Activating the environment basically adjusts your paths/aliases so that "python", "pip",
etc now point to the virtual environment executables.  

With our GIANT environment activated we are good to install GIANT.  GIANT is installed like any normal python
package.  Simply navigate to the root directory of GIANT and type (note that we use the link option here because
GIANT is still under development and this allows you to update with a simple git pull.

    cd giant
    pip install -e .

This will install the GIANT package to your virtual environment along with all of its dependencies.  Once GIANT is
installed you can run some basic tests to make sure everything is working by entering the unittests directory and
typing
   
    cd unittests
    python -Wignore -m unittest discover

This might return some warnings but at the end of it all it should say "ok" and something about all of the
tests passing.

GIANT is now installed and you can begin navigating.

See the GIANT documentation (documentation/html/index.html or https://aliounis.github.io/giant_documentation/) for more information.
