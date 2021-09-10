# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


from typing import Optional, Tuple, Union, Type

import numpy as np

from giant.ray_tracer.shapes import Surface, AxisAlignedBoundingBox, Ellipsoid
from giant.ray_tracer.shapes.surface import RawSurface
from giant.rotations import Rotation
from giant._typing import ARRAY_LIKE, PATH

class KDNode:

    left: KDNode
    right: KDNode
    bounding_box: AxisAlignedBoundingBox
    has_surface: bool
    surface: Optional[RawSurface]
    id: int
    id_order: int

    def __init__(self,
                 surface: Optional[RawSurface] = None,
                 _left: Optional[KDNode] = None,
                 _right: Optional[KDNode] = None,
                 _bounding_box: Optional[AxisAlignedBoundingBox] = None,
                 _order: int = 0,
                 _id: Optional[int] = None,
                 _id_order: Optional[int] = None,
                 _centers: Optional[np.ndarray] = None): ...

    def __reduce__(self) -> Tuple[Type['KDNode'], Tuple[Optional[RawSurface], Optional[KDNode], Optional[KDNode],
                                                        AxisAlignedBoundingBox, int, int, int]]: ...

    @property
    def order(self) -> int: ...

    @order.setter
    def order(self, val: int): ...

    def compute_bounding_box(self): ...

    def split(self,
              force: bool = False,
              flip: bool = False,
              print_progress: bool = True) -> Union[Tuple[KDNode, KDNode], Tuple[None, None]]: ...

    def __eq__(self, other: KDNode) -> bool: ...

    def translate(self, translation: ARRAY_LIKE): ...

    def rotate(self, rotation: Rotation): ...


class KDTree(Surface):

    root: KDNode
    surface: RawSurface
    bounding_box: AxisAlignedBoundingBox
    reference_ellipsoid: Ellipsoid
    max_depth: int

    def __init__(self,
                 surface: Optional[RawSurface],
                 max_depth: int = 10,
                 _root: Optional[KDNode] = None,
                 _rotation: Optional[Rotation] = None,
                 _position: Optional[np.ndarray] = None,
                 _bounding_box: Optional[AxisAlignedBoundingBox] = None,
                 _reference_ellipsoid: Optional[Ellipsoid] = None): ...

    def __reduce__(self) -> Tuple[Type['KDTree'], Tuple[RawSurface, int, KDNode, Optional[Rotation],
                                                        Optional[np.ndarray], AxisAlignedBoundingBox,
                                                        Ellipsoid]]: ...

    @property
    def position(self) -> Optional[np.ndarray]: ...

    @property
    def rotation(self) -> Optional[Rotation]: ...

    @property
    def order(self) -> int: ...

    def build(self,
              force: bool = True,
              print_progress: bool = True): ...

    def translate(self,
                  translation: ARRAY_LIKE): ...

    def rotate(self,
               rotation: Union[Rotation, ARRAY_LIKE]): ...

    def save(self,
             filename: PATH): ...

    def load(self,
             filename: PATH): ...


def get_ignore_inds(node: KDNode,
                    vertex_id: int) -> np.ndarray: ...

def get_facet_vertices(node: KDNode,
                       facet_id: int) -> np.ndarray: ...

def describe_tree(tree: KDTree): ...
