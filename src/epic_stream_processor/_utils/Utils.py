from functools import lru_cache
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from ..epic_types import NDArrayNum_t
from ..epic_types import Numeric_t
from ..epic_types import Patch_t
from ..epic_types import PixCoord2d_t


class PatchMan:
    @staticmethod
    @lru_cache(maxsize=None)
    def get_patch_indices(patch_type: Patch_t = "3x3") -> List[NDArray[np.int64]]:
        patch: int = PatchMan.get_patch_size(patch_type)

        return np.meshgrid(
            np.arange(patch) - int(patch / 2), np.arange(patch) - int(patch / 2)
        )

    @staticmethod
    def get_patch_size(patch_type: Patch_t = "3x3") -> int:
        return int(str(patch_type).split("x")[0])

    @staticmethod
    def get_patch_pixels(
        pixel: Optional[PixCoord2d_t] = None,
        x: Optional[Numeric_t] = None,
        y: Optional[Numeric_t] = None,
        patch_type: Patch_t = "3x3",
    ) -> Tuple[NDArrayNum_t, NDArrayNum_t]:
        xgrid, ygrid = PatchMan.get_patch_indices(patch_type)

        if pixel is not None:
            return (xgrid + pixel[0]).flatten(), (ygrid + pixel[1]).flatten()
        elif x is not None and y is not None:
            return (xgrid + x).flatten(), (ygrid + y).flatten()
        else:
            raise Exception("Either pixel or x and y must be specified")


@lru_cache
def get_lmn_grid(xsize: int, ysize: int) -> NDArrayNum_t:
    lm_matrix = np.zeros(shape=(xsize, ysize, 3))
    m_step = 2.0 / ysize
    l_step = 2.0 / xsize
    i, j = np.meshgrid(np.arange(xsize), np.arange(ysize))
    # this builds a 3 x 64 x 64 matrix, need to transpose axes to [2, 1, 0]
    #  to getcorrect 64 x 64 x 3 shape
    lm_matrix = np.asarray([i * l_step - 1.0, j * m_step - 1.0, np.zeros_like(j)])

    return lm_matrix
