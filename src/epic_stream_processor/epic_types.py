from typing import Tuple
from typing import Union
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray


Numeric_t = Union[np.int64, np.float64]
NDArrayNum_t = Union[NDArray[np.int64], NDArray[np.float64]]
Patch_t = Union[int, str]
WatchMode_t = str
PixCoord2d_t = Tuple[Numeric_t, Numeric_t]
List_t = Union[NDArrayNum_t, list]

