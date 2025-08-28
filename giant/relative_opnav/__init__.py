# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This package provides the required routines and objects to extract observables to targets in an image.

Description
-----------

In GIANT, Relative OpNav refers to the process of extracting observables from monocular images of non-star targets.
These observables can take many different forms, and are usually fed to some type of filter or estimator to use in
refining the knowledge of the relative state between the camera and the target.  Some of the most common forms of RelNav
measurements are bearing observables to the center of figure of an object, bearing observables to a surface feature on a
target, bearing measurements to points on the illuminated limb of a target, full 3DOF relative position measurements
between the camera and the center of figure of the target, and constraint measurements (or bearing measurements to the
same, possibly unknown, feature in multiple images.  In addition to the different types of observables that can be
extracted from images, there are also many different techniques for extracting these observables.  Taken together, all
of these choices can make doing RelNav and writing code to do RelNav confusing and difficult.  Therefore, as with the
:class:`.StellarOpNav` class, in GIANT we have made a single user interface to handle most of these techniques and make
doing RelNav significantly easier in the form of the :class:`.RelativeOpNav` class.

The :class:`.RelativeOpNav` class is generally the only interface a user will require when performing relative OpNav.
Indeed, in most cases, doing RelNav using this class is as simple as initializing the class with some basic settings
and then calling :meth:`~.RelativeOpNav.auto_estimate`, which will attempt to automatically deduce what technique is
best suited for extracting observables from the images based on what is expected to be seen in the images and apply
those techniques to the images.  This means it is largely possible to do RelNav without having a super in-depth
understanding of what exactly is going on, though at least a basic understanding certainly helps. For those with an
in-depth understanding, the :class:`.RelativeOpNav` class also provides easy access to many of the sub-steps required,
allowing for more fine-grained control and more advanced analysis.

This package level documentation only focuses on the use of the class to do RelNav for typical cases.  If you need a
deeper understanding of what is going on, or you need to do some advanced analysis (or possibly even create your own
RelNav technique) then we encourage you to read the sub-module/package documentation from this package
(:mod:`.relative_opnav.estimators`, :mod:`.relnav_class`, and :mod:`.relative_opnav.visualizer`)

Tuning for Successful Relative OpNav
------------------------------------

Tuning is generally simple for most relative OpNav techniques, especially when compared to stellar OpNav.  In most
cases, the only tuning parameter you really need to worry about is the :attr:`.extended_body_cutoff`.   This knob tells
the :meth:`.auto_estimate` method when to switch from using the unresolved technique, where the target in the image is
assumed to be dominated by the point spread function of the camera, to using resolved techniques based on the shape
representation of the target.  Typically you want this set around 5-10 pixels, though if you have a particularly large
PSF for your camera then you may want to increase this slightly.  Beyond that, at least for method
:meth:`.auto_estimate`, most of the default tuning parameters should be sufficient to get pretty good RelNav results.
If you need more control over the tuning for individual techniques you should refer to their module documentation which
you can get to from :mod:`.relative_opnav.estimators`.
"""

from giant.relative_opnav.relnav_class import RelativeOpNav

from giant.relative_opnav.estimators.cross_correlation import XCorrCenterFinding, XCorrCenterFindingOptions
from giant.relative_opnav.estimators.unresolved import UnresolvedCenterFinding, UnresolvedCenterFindingOptions
from giant.relative_opnav.estimators.ellipse_matching import EllipseMatching, EllipseMatchingOptions
from giant.relative_opnav.estimators.constraint_matching import ConstraintMatching

__all__ = ['RelativeOpNav', 'XCorrCenterFinding', 'XCorrCenterFindingOptions', 'EllipseMatching', 'EllipseMatchingOptions' ,'UnresolvedCenterFinding', 'UnresolvedCenterFindingOptions', 'ConstraintMatching']
