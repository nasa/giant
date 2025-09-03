from typing import runtime_checkable, Protocol, Self, Iterator, SupportsIndex

from numpy.typing import NDArray
import numpy as np

from giant.rotations import Rotation

from giant._typing import ARRAY_LIKE, DOUBLE_ARRAY


    
@runtime_checkable
class Moveable(Protocol):
    def rotate(self, rotation: Rotation) -> Self: ...
    
    def translate(self, translation: ARRAY_LIKE) -> Self: ...
    
@runtime_checkable
class RaysLike(Moveable, Protocol):
    def __iter__(self) -> Iterator[Self]: ...
    
    def __getitem__(self, key: int | NDArray | list[int], /) -> Self: ...
    
    def __len__(self) -> int: ...
    
    @property
    def start(self) -> DOUBLE_ARRAY: ...
    @property
    def direction(self) -> DOUBLE_ARRAY: ...
    @property
    def inv_direction(self) -> DOUBLE_ARRAY: ...
    @property
    def ignore(self) -> NDArray[np.integer] | int | None: ...
    
    
@runtime_checkable
class Traceable(Protocol):
    def trace(self, rays: RaysLike) -> NDArray: ...
    
    def compute_intersect(self, ray: RaysLike) -> NDArray: ...
    
@runtime_checkable
class ShapeLike(Moveable, Traceable, Protocol): 
    pass

@runtime_checkable
class BoundingBoxLike(Moveable, Protocol):
    min_sides: DOUBLE_ARRAY
    max_sides: DOUBLE_ARRAY
    vertices: DOUBLE_ARRAY
    
    def check_intersect(self, ray: RaysLike, return_distances: bool) -> bool | tuple[bool, DOUBLE_ARRAY]: ...

@runtime_checkable
class HasBoundingBox(Protocol):
    bounding_box: BoundingBoxLike
    
@runtime_checkable
class HasFindableLimbs(Protocol):
    def find_limbs(self, scan_center: DOUBLE_ARRAY, scan_directions: DOUBLE_ARRAY, observer_position: None | ARRAY_LIKE = None) -> DOUBLE_ARRAY: ...
    
@runtime_checkable
class EllipsoidLike(HasFindableLimbs, Protocol):
    principal_axes: DOUBLE_ARRAY
    orientation: DOUBLE_ARRAY
    center: DOUBLE_ARRAY
    
@runtime_checkable
class HasCircumscribingSphere(Protocol):
    circumscribing_sphere: HasFindableLimbs
    
@runtime_checkable
class HasReferenceEllipsoid(Protocol):
    reference_ellipsoid: EllipsoidLike
    