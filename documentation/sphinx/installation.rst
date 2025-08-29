.. _installation:

Installing GIANT
================

Setting up the environment
--------------------------

GIANT is primarily a python package with some modules written in Cython for speed improvements.  This makes installation
fairly straight forward in most Unix-like environments.  It should be possible to install GIANT on Windows using similar
steps to those that follow, but this is untested.  The following subsections give step-by-step instructions for
installing GIANT.

The first step to installing GIANT is to install the mini-forge python environment/package manager. This helps to
ensure that the GIANT requirements don't interfere with other python code you have on your system (and that other python
environments you have don't interfere with GIANT).  Mini-forge can be downloaded from
https://conda-forge.org/download/.
Once you have downloaded mini-forge, install it following the instructions provided (note that you can install it to your
home directory without needing admin privileges, which is a big benefit).

With conda installed, we can now set up an environment in which to install giant

#. Before we begin there are a couple OS specific instructions
    - macOS
        - You may need to set the environment variable ``CPATH`` to something like
          ``/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include``
          or ``/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include`` after installing xcode or xcode command
          line tools to ensure system headers are available
    - Windows
        - If you want to use some of the optional utilities which produce GIFs, you will need to manually install imagemagick from https://imagemagick.org/script/download.php
#. Download the GIANT source code from git
    - :code:`git clone git@aetd-git.gsfc.nasa.gov:giant/giant.git`
#. Create a new environment to install GIANT to and install the dependencies
    - :code:`cd giant`
    - :code:`mamba env create -f environment.yml` 
        - This will create a new environment called giant
    - Note that creating a new environment is not required, but is strongly recommended.
    - Note that if you did not install mini-forge, you need to specify to use the conda-forge repository instead of the default conda repository by adding :code:`-c conda-forge` to the :code:`mamba env create` command.
    - Note that if you are on windows, you will unfortunately need to manually install visual studio tools according to the instructions https://conda-forge.org/docs/maintainer/knowledge_base.html#local-testing 
#. Activate the new environment
    - :code:`conda activate giant`
    - You will need to do this step each time you want to use GIANT.
#. Install optional dependencies if desired
    - These can be installed using :code:`mamba install ...` 
        - `gdal` -- used for the feature catalog tools in the examples 
        - `plotly`-- used for the feature catalog tools in the examples
        - `dash` -- used for the feature catalog tools in the examples
        - Requirements for RoMa installed through mini-forge (RoMa is an optional feature matcher.  RoMa must be installed from https://github.com/Parskatt/RoMa)
          - `pytorch`
          - `einops`
          - `torchvision`
          - `kornia`
          - `albumentations`
          - `loguru`
          - `wandb`
          - `timm`
          - `poselib` 

Installing and testing GIANT
----------------------------
With the anaconda environment set up and activated it is now easy to install giant.

#. Install the GIANT package
    - :code:`cd giant`
    - :code:`pip install -e .`
    - You can also omit the "-e" to avoid using links, but using links makes it easier to update GIANT in the future.
    - You will likely see a lot of warnings from the c compiler that can largely be ignored.  If you have an error
      though that is a problem
#. Check the installation (optional)
    - :code:`cd unittests`
    - :code:`python -Wignore -m unittest discover`
    - This should complete with no errors (skips/warnings are OK).

GIANT should now be installed and ready for use.

