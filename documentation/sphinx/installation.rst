.. _installation:

Installing GIANT
================

Setting up the environment
--------------------------

GIANT is primarily a python package with some modules written in Cython for speed improvements.  This makes installation
fairly straight forward in most Unix-like environments.  It should be possible to install GIANT on windows using similar
steps to those that follow, but this is untested.  The following subsections give step by step instructions for
installing GIANT.

The first step to installing GIANT is to install the mamba-forge python environment/package manager. This helps to
ensure that the GIANT requirements don't interfere with other python code you have on your system (and that other python
environments you have don't interfere with GIANT).  Mamba-forge can be downloaded fom
https://github.com/conda-forge/miniforge#mambaforge
Once you have downloaded anaconda, install it following the instructions provided (note that you can install it to your
home directory without needing admin privileges, which is a big benefit).

With mamba installed, we can now set up an environment in which to install giant

#. Before we begin there are a couple OS specific instructions
    - macOS
        - You may need to set the environment variable ``CPATH`` to something like
          ``/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include``
          or ``/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include`` after installing xcode or xcode command
          line tools to ensure system headers are available
    - Windows
        - You will need to install the Visual Studio 2017 developer tools according to the instructions at https://conda-forge.org/docs/maintainer/knowledge_base.html#local-testing 
            - Note that you must do this before activating the environment as described below
            - At this time it looks like Visual Studio 2017 is required (more recent will not work)
        - You will need to manually install imagemagick from https://imagemagick.org/script/download.php
#. Create a new environment to install GIANT to and install the dependencies
    - :code:`mamba create -n giant_env python=3 opencv matplotlib scipy pandas numpy cython pyqt astropy lxml sphinx spiceypy c-compiler openmp astroquery dill psutil`
        - If you are on Mac or Linux you can install imagemagick with the same command by including ``imagemagick`` at the end of the above command.
    - Note that creating a new environment is not required, but is strongly recommended.
    - Note that if you did not install mamba-forge, you need to specify to use the conda-forge repository instead of the default conda repository by adding :code:`-c conda-forge` to the :code:`mamba create` command.
    - Note that if you are on windows, you will unfortunately need to manually install vs2017 according to the instructions https://conda-forge.org/docs/maintainer/knowledge_base.html#local-testing 
#. Activate the new environment
    - :code:`conda activate giant_env`
    - If you did not create a new environment then this step is not necessary (as long as the mamba python is first
      in your path).
    - You will need to do this step each time you want to use GIANT.

Installing and testing GIANT
----------------------------
With the anaconda environment set up and activated it is now easy to install giant.

#. Download the GIANT source code from git
    - :code:`git clone git@aetd-git.gsfc.nasa.gov:giant/giant.git`
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

