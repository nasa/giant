# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This subpackage provides the requisite classes and functions for performing surface feature navigation in GIANT.

Description
-----------

Surface Feature Navigation (SFN) is a relative OpNav technique where we extract bearing measurements of individual
surface features from an image.  Typically, we do this for many features in a single image, which in turn allows us to
essentially triangulate the camera from a single image.  As such it is a very powerful technique for navigation and is
generally preferred once targets grow large enough in the navigation images that we can reliably recognize the features
on the surface.

There are a number of different ways to do surface feature navigation, but perhaps one of the oldest (from a space use
perspective) involves defining features as small patches of surface instead of actual features (like craters/rocks/etc).
For each of these patches of surface, we can then predict what they should look like in an image and use cross
correlation to locate the actual location in the image.  This technique was popularized by the Stereophotoclinometry
software suite developed by Dr. Robert Gaskell and conceptually defines how we do SFN in GIANT.

From this package you can import everything you need for doing SFN in GIANT.  The most important things that you will
need from here are the :class:`.SurfaceFeatureNavigation` class which does the actual navigation, the
:class:`.FeatureCatalogue` class which is used to interface a list of features we want to process with the
:class:`.Scene` in GIANT, and the :class:`.VisibleFeatureFinder` which is used to determine which features in the
catalogue should actually be visible in the images.

For more detailed description (and hints for tuning for SFN) refer to the following module and class documentation for
the objects defined in this package.

Use
---

In general you won't interact too much with the classes/functions defined in this package, with the exception of the
:class:`.VisibleFeatureFinder`, which you will need to use to specify the settings you want when identifying possibly
visible features.  Beyond that, most of the other tools are automatically handled for you in the :class:`.RelativeOpNav`
class. If you need to build a feature catalogue for use with SFN, you should take a look at the
:mod:`.spc_to_feature_catalogue` and :mod:`.tile_shape` scripts which may be able to do it for you.
"""

from giant.relative_opnav.estimators.sfn.sfn_correlators import sfn_correlator
from giant.relative_opnav.estimators.sfn.surface_features import (FeatureCatalogue, SurfaceFeature,
                                                                  VisibleFeatureFinder, VisibleFeatureFinderOptions)
from giant.relative_opnav.estimators.sfn.sfn_class import SurfaceFeatureNavigation, SurfaceFeatureNavigationOptions

__all__ = ['sfn_correlator', 'FeatureCatalogue', 'SurfaceFeatureNavigation', 'SurfaceFeatureNavigationOptions',
           'SurfaceFeature', 'VisibleFeatureFinder', 'VisibleFeatureFinderOptions']
