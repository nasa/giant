KDNode
======

.. currentmodule:: giant.ray_tracer.kdtree

:mod:`giant.ray_tracer.kdtree`\:

.. autoclass:: KDNode
    :no-members:
    :members: order

    .. attribute:: left
        :type: Optional[KDNode]

        The left child of this node (containing the geometry that is less than the median)
        If there are no children this will be ``None`` and this is probably a leaf node with geometry in
        :attr:`surface`.

    .. attribute:: right
        :type: Optional[KDNode]

        The right child of this node (containing the geometry that is greater than the median)
        If there are no children this will be ``None`` and this is probably a leaf node with geometry in
        :attr:`surface`.

    .. attribute:: bounding_box
        :type: AxisAlignedBoundingBox

        The :class:`.AxisAlignedBoundingBox` specifying the extent of the space covered by this node and its children.

        The bounding box contains both any geometry primitives contained in the node, as well as the bounding boxes of
        all children nodes, therefore if a ray does not intersect the bounding box of the node, it will not intersect
        any of its dependents (children or geometry)

    .. attribute:: has_surface
        :type: bool

        A flag specifying whether this is a leaf node (``True``, contains geometry) or is a branch node (``False``, only
        contains children nodes).

    .. attribute:: surface
        :type: Optional[Surface]

        The surface object specifying the geometry for this node if it is a leaf node or ``None``.

        If this is a leaf node then there should be no children nodes (:attr:`left` and :attr:`right` should both be
        ``None``).

    .. attribute:: id
        :type: int

        The unique identifier for this node as an integer.

        This is guaranteed to be unique for every node in a tree (it is not guaranteed to be unique across nodes in
        different trees).

    .. attribute:: id_order
        :type: int

        The maximum number of digits required to represent the largest ID number for an node in any given tree.  See the
        class description for more details of what this means.

.. rubric:: Summary of Methods

.. autosummary::
   :nosignatures:
   :toctree:

   ~KDNode.compute_bounding_box
   ~KDNode.rotate
   ~KDNode.split
   ~KDNode.translate
   

|
