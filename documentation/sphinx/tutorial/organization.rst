Organization
============

GIANT is split into a number of submodules and subpackages. Understanding this layout will be key to quickly grasping
how GIANT works so we'll explain a few things here.  The top level outline of GIANT is shown below (submodules are
shown in blue and subpackages are shown in yellow).

GIANT makes extensive use of object oriented programing.  This means most things are stored in classes, which allows
data and the functions that operate on that data to be closely intertwined.  It also means that many of the user
interface classes behave very similarly to each other and simply add new functionality.

.. graphviz::

    digraph giant {

        rankdir=UD;

        node [shape="box", style="filled", fillcolor="gray"];

        "giant" [href="../giant.html", target="_top"];

        node [fillcolor="lightblue", style="filled"];

        "image" [href="../giant.image.html", target="_top"];
        "camera" [href="../giant.camera.html", target="_top"];
        "camera_models" [href="../giant.camera_models.html", target="_top"];
        "opnav_class" [href="../giant.opnav_class.html", target="_top"];
        "rotations" [href="../giant.rotations.html", target="_top"];
        "image_processing" [href="../giant.image_processing.html", target="_top"];

        node [fillcolor="lightyellow", style="filled"];

        "stellar_opnav" [href="../giant.stellar_opnav.html", target="_top"];
        "calibration" [href="../giant.calibration.html", target="_top"];
        "catalogues" [href="../giant.catalogues.html", target="_top"];
        "relative_opnav" [href="../giant.relative_opnav.html", target="_top"];
        "ray_tracer" [href="../giant.ray_tracer.html", target="_top"];
        "utilities" [href="../giant.utilities.html", target="_top"];


        "giant" -> {"image", "camera", "camera_models", "opnav_class", "rotations", "image_processing", "utilities",
                    "stellar_opnav", "calibration", "catalogues", "relative_opnav", "ray_tracer"};

    }

The first submodule in GIANT is the :mod:`.image` module, which defines the :class:`.OpNavImage` class that is
the primary way that image data and metadata is communicated to the various GIANT routines.
The next submodule is the :mod:`.camera` module, which defines the :class:`.Camera` class that conveys
both the images and details about the camera to the GIANT routines.
Then we have the :mod:`.camera_models` modules, which defines a number of classes that represent 3D-2D mappings of
points from a world location to a location in an image and vice-versa.
Next is the :mod:`.opnav_class` module, which provides an abstract base class (:class:`.OpNav`) that provides an outline
and basic functionality for most of the high-level OpNav techniques that are cooked into GIANT.
The :mod:`.rotations` module follows which provides an :class:`.Rotation` class to represent rotations and attitude data
as well as a number of functions to manipulate and convert this data.
The :mod:`.image_processing` module provides the majority of the functions and classes that operate directly on the
image data in GIANT.
Finally, we have the :mod:`~.giant.utilities` module which defines a number of helper functions for interfacing GIANT
with the `NAIF Spice toolbox <https://naif.jpl.nasa.gov/naif/toolkit.html>`_ and SPC, among other things.

Now we can discuss the packages in GIANT.  First up is the :mod:`.stellar_opnav` package, which provides the required
tools and a nice user interface (:class:`.StellarOpNav`) to estimate the attitude of an image based off of the observed
stars in the image.
Then there is the :mod:`.calibration` package which adds the ability to do geometric camera calibration based off of
images of stars to the :mod:`.stellar_opnav` package (:class:`.Calibration`).
Next is the :mod:`.catalogues` package which provides interfaces to star catalogues for the :mod:`.stellar_opnav` and
:mod:`.calibration` packages.
The :mod:`.relative_opnav` package follows which provides the ability to perform a number of center finding and surface
feature OpNav techniques.
Finally, the :mod:`.ray_tracer` package provides the :mod:`.relative_opnav` package the ability to track the a priori
scene knowledge and render templates of the observed bodies for cross-correlation among other uses.

Having this basic knowledge of how GIANT is designed should help you to figure out where to look for things when you
need them.

