import numpy as np

CUPY_IMPORTED = True
try:
    import cupy as cp
except ImportError:
    CUPY_IMPORTED = False


def assure_numpy(arr):
    if type(arr) is np.ndarray:
        return arr
    elif CUPY_IMPORTED and type(arr) is cp.ndarray:
        return arr.get()
    else:
        raise ValueError("arr must be either numpy or cupy array.")
