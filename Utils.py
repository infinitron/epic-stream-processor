from functools import lru_cache
import numpy as np


class PatchMan:
    @ staticmethod
    @ lru_cache(maxsize=None)
    def get_patch_indices(patch_type='3x3'):
        patch = patch_type
        if type(patch_type) == str:
            patch = int(patch_type.split('x')[0])

        return np.meshgrid(np.arange(patch)-int(patch/2),
                           np.arange(patch)-int(patch/2))

    @ staticmethod
    def get_patch_pixels(pixel, x=None, y=None, patch_type='3x3'):
        xgrid, ygrid = PatchMan.get_patch_indices(patch_type)

        if pixel is not None:
            return (xgrid+pixel[0]).flatten(), (ygrid+pixel[1]).flatten()
        else:
            return (xgrid+x).flatten(), (ygrid+y).flatten()


@lru_cache
def get_lmn_grid(xsize: int, ysize: int):
    lm_matrix = np.zeros(shape=(xsize, ysize, 3))
    m_step = 2.0 / ysize
    l_step = 2.0/xsize
    i, j = np.meshgrid(np.arange(xsize), np.arange(ysize))
    # this builds a 3 x 64 x 64 matrix, need to transpose axes to [2, 1, 0]
    #  to getcorrect 64 x 64 x 3 shape
    lm_matrix = np.asarray(
        [i * l_step - 1.0, j * m_step - 1.0, np.zeros_like(j)])

    return lm_matrix
