KDTree
======

.. currentmodule:: giant.ray_tracer.kdtree

:mod:`giant.ray_tracer.kdtree`\:

.. autoclass:: KDTree
    :no-members:
    :members: order, position, rotation

    .. attribute:: bounding_box
        :type: AxisAlignedBoundingBox

        The :class:`.AxisAlignedBoundingBox` that fully contains the surface represented by this acceleration structure.

    .. attribute:: reference_ellipsoid
        :type: Ellipsoid

        The :class:`.Ellipsoid` that best represents the surface represented by this acceleration structure.

    .. attribute:: root
        :type: KDNode

        The :class:`.KDNode` that serves as the root (initial branch node) of the tree.  This is where all tracing
        begins

    .. attribute:: surface
        :type: RawSurface

        The :class:`.Surface` that this acceleration structure was built for as initially provided.

        Note that this does not get rotated/translated with the tree, so do not expect the :attr:`.RawSurface.vertices`
        or :attr:`.RawSurface.normals` attributes to be in the current frame.

    .. attribute:: max_depth
        :type: int

        The maximum depth to which to build the tree.

        The maximum depth puts a limit on how many branch nodes can end up in the tree as
        :math:`\sum_{i=0}^{\text{max_depth}}2^i`.  In many cases however, you will end up with less than
        this maximum limit since we typically don't split nodes with less than 10 primitives in them.

.. rubric:: Summary of Methods

.. autosummary::
   :nosignatures:
   :toctree:

   ~KDTree.build
   ~KDTree.compute_intersect
   ~KDTree.trace
   ~KDTree.find_limbs
   ~KDTree.compute_limb_jacobian
   ~KDTree.rotate
   ~KDTree.translate
   ~KDTree.save
   ~KDTree.load



|
