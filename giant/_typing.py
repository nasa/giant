


from typing import Union, Protocol, runtime_checkable,  Any, SupportsIndex, Iterator, Literal, Sequence
from datetime import timedelta, datetime
from pandas import Timedelta, Timestamp
from pathlib import Path

import numpy as np
import numpy.typing as npt

DOUBLE_ARRAY = np.typing.NDArray[np.float64]
ARRAY_LIKE = npt.ArrayLike
ARRAY_LIKE_2D = npt.ArrayLike
SCALAR_OR_ARRAY = Union[float, npt.ArrayLike]
F_SCALAR_OR_ARRAY = Union[float, DOUBLE_ARRAY]

PATH = Union[Path, str]

NONENUM = Union[float, None]
NONEARRAY = Union[npt.NDArray, None]

TimedeltaLike = Union[timedelta, Timedelta]
DatetimeLike = Union[datetime, Timestamp]

@runtime_checkable
class WriteableTarget(Protocol):
    
    def flush(self) -> None: ...
    def close(self) -> None: ...
    def write(self, s: Any, /) -> int: ...
    
class BasicSequenceProtocol(Protocol):
    
    def __getitem__(self, key: SupportsIndex, /) -> Any: ...
    
    def __iter__(self) -> Iterator[Any]: ...
    
    def __len__(self) -> int: ...
    
    


EULER_ORDERS = Literal['xyz', 'xzy', 'xyx', 'xzx', 'yxz', 'yzx', 'yxy', 'yzy', 'zxy', 'zyx', 'zyz', 'zxz']

F_ARRAY_LIKE = Sequence[float] | DOUBLE_ARRAY
    