try:
    from .ext import em_kernel, dens_kernel
except ImportError:
    _has_c_ext = False
else:
    _has_c_ext = True

try:
    from .ext_cuda import em_kernel_cuda, dens_kernel_cuda
    import cupy as cp
except ImportError:
    _has_cuda_ext = False
else:
    _has_cuda_ext = True


if _has_cuda_ext:
    from .ext_cuda import em_kernel_cuda, dens_kernel_cuda
    import cupy as cp


if _has_c_ext:
    from .ext import em_kernel, dens_kernel


def requires_c_ext(func):
    def inner(*args, **kwargs):
        if not _has_cuda_ext:
            raise ImportError(f"func.__name__ requires cupy")
        return func(*args, **kwargs)
    return inner


def requires_cuda_ext(func):
    def inner(*args, **kwargs):
        if not _has_c_ext:
            raise ImportError(f"func.__name__ requires cupy")
        return func(*args, **kwargs)
    return inner



