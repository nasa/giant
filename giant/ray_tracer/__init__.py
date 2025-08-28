


"""
This subpackage provides the ray tracing and rendering capabilities for GIANT.

Description
-----------

In GIANT rendering is primarily done using single bounce ray tracing, where a ray is traced from the camera, to a
surface and then bounced to the light source.  This tells us a number of things.  First, is the geometry of the ray
trace, including the exidence, incidence, and normal vectors and the albedo at the intersect location.  It also tells us
if the surface is shadowed where we initially intersected.  Based on this information, we can then compute the intensity
for each ray and then use those intensities to render an image.  Because space imagery typically has a single collimated
illumination source (the sun) and most bodies we are doing OpNav with respect to are airless (and thus there is no
atmospheric scattering) the single bounce ray trace is a very accurate way to render synthetic images.

There are 2 primary ways to represent a surface in GIANT.  The first is as a triaxial :class:`.Ellipsoid`.  This is
useful for many larger celestial bodies (planets, moons, and large asteroids/comets) and is very efficient for ray u
tracing since only a single object needs to be intersected.  The second primary way is as a :class:`.Surface` object.
Using this we represent the surface as a tesselation of small planar geometry primitives (usually triangles,
:class:`.Triangle32` and :class:`.Triangle64`) where we then have to check our rays intersections against every geometry
primitive in the surface.  This allows us to represent arbitrary topography with arbitrary resolution, but because it
normally takes many many geometry primitives for a single surface tracing can be very slow.  Therefore, we also provide
an acceleration structure in the form of a :class:`.KDTree` which limits the number of triangles we need to check each
ray against using :class:`.AxisAlignedBoundingBox`.

Once a surface is represented in GIANT it is usually wrapped in a :class:`.SceneObject` and added to a :class:`.Scene`.
The :class:`.Scene` in GIANT is used to define the locations and orientations of multiple objects with respect to each
other.  It also provides functionality for automatically updating these locations and orientations for a new time and
for doing the single bounce ray trace for rendering.  Once the ray trace is complete, the subclasses of
:class:`.IlluminationModel` are used to convert the ray trace geometry into intensity values for each ray (typically)
the :class:`.McEwenIllumination` class).

When creating a surface in GIANT, you will usually use the :mod:`.ingest_shape` script which will create the surface and
build the acceleration structure automatically for you.

For more details, please refer to the following module documentation, which provides much more detail.
"""

import giant.ray_tracer.shapes as shapes
import giant.ray_tracer.kdtree as kdtree

import giant.ray_tracer.rays as rays
import giant.ray_tracer.scene as scene

import giant.ray_tracer.illumination as illumination

from giant.ray_tracer.rays import Rays, compute_rays, INTERSECT_DTYPE
from giant.ray_tracer.scene import SceneObject, Scene, CorrectionsType
from giant.ray_tracer.illumination import IlluminationModel, AshikhminShirleyDiffuseIllumination, GaskellIllumination, \
    McEwenIllumination, LambertianIllumination, LommelSeeligerIllumination, ILLUM_DTYPE

from giant.ray_tracer.shapes.triangle import Triangle64, Triangle32
from giant.ray_tracer.shapes.ellipsoid import Ellipsoid
from giant.ray_tracer.shapes.surface import Surface, Surface32, Surface64
from giant.ray_tracer.shapes.solid import Solid
from giant.ray_tracer.shapes.shape import Shape
from giant.ray_tracer.shapes.point import Point
from giant.ray_tracer.shapes.axis_aligned_bounding_box import AxisAlignedBoundingBox
from giant.ray_tracer.kdtree import KDTree

__all__ = ["Rays", "compute_rays", "INTERSECT_DTYPE", "Scene", "SceneObject", "IlluminationModel",
           "AshikhminShirleyDiffuseIllumination", "McEwenIllumination", "LambertianIllumination", "GaskellIllumination",
           "LommelSeeligerIllumination", "ILLUM_DTYPE", "Triangle32", "Triangle64", "Ellipsoid", "Surface", "Surface64",
           "Surface32", "Solid", "Shape", "Point", "AxisAlignedBoundingBox", "KDTree", "CorrectionsType",
           "shapes", "kdtree", "illumination", "rays", "scene"]
