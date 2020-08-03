import numpy

try:
    import cupy as cp
except ImportError:
    _has_cupy = False
    cp = None
else:
    _has_cupy = True


def requires_cupy(func):
    def inner(*args, **kwargs):
        if not _has_cupy:
            raise ImportError(f"func.__name__ requires cupy")
        return func(*args, **kwargs)
    return inner


def get_array_module(arr):
    if _has_cupy:
        return cp.get_array_module(arr)
    else:
        return numpy
