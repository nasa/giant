# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This subpackage defines all shapes that are used throughout GIANT.

In GIANT, shapes are used to represent targets in a scene for rendering as well as for other analysis.  Therefore,
shapes enable ray tracing, finding limbs for a target, tracking the location and orientation of a target, among other
things.

There are 2 primary shapes that are typically used to represent targets in GIANT.  These are :class:`.Ellipsoid` and
:class:`.Triangle64` / :class:`.Triangle32`.  :class:`.Ellipsoid` are typically used for "regular" bodies (bodies that
are well modeled by a sphere/triaxial ellipsoid like planets and moons, while :class:`.Triangle64` /
:class:`.Triangle32` are typically used for more general terrain, as well as small patches of terrain from targets that
otherwise are globally ellipsoids.  There are a few other shapes, and the following documentation discusses the typical
use case for each, but generally knowing these 2 is sufficient.

Generally a user probably won't have much direct interaction with this subpackage, as there are many higher level
scripts and modules that handle things for you, like :mod:`.ingest_shape`, :mod:`.tile_shape`,
:mod:`.spc_to_feature_catalogue`, and :mod:`.scene`, but understanding what is going on under the hood can be beneficial
so we do recommend at least skimming through this documentation.
"""


import numpy as np

from giant.ray_tracer.shapes.shape import Shape
from giant.ray_tracer.shapes.point import Point
from giant.ray_tracer.shapes.axis_aligned_bounding_box import AxisAlignedBoundingBox
from giant.ray_tracer.shapes.surface import Surface, Surface64, Surface32
from giant.ray_tracer.shapes.triangle import Triangle64, Triangle32
from giant.ray_tracer.shapes.solid import Solid
from giant.ray_tracer.shapes.ellipsoid import Ellipsoid

# turn off annoying divide by 0 errors
np.seterr(divide='ignore', invalid='ignore')


__all__ = ['Shape', 'Point', 'AxisAlignedBoundingBox', 'Surface', 'Surface32', 'Surface64', 'Triangle64', 'Triangle32',
           'Solid', 'Ellipsoid']
