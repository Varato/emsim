from enum import Enum


class BackendType(Enum):
    NUMPY = 1
    CUDA = 2
    FFTW = 3


class Backend(object):
    def __init__(self, backend_type: BackendType):
        self.backend_type = BackendType
        self._slice_builder_module = None
        self._wave_propagator_module = None

        if backend_type is BackendType.NUMPY:
            from .backend import slice_builder_numpy, wave_propagator_numpy
            self._slice_builder_module = slice_builder_numpy
            self._wave_propagator_module = wave_propagator_numpy
        elif backend_type is BackendType.CUDA:
            from .backend import slice_builder_cuda, wave_propagator_cuda
            import cupy as cp
            cp.get_default_memory_pool().set_limit(size=3*1024**3)
            self._slice_builder_module = slice_builder_cuda
            self._wave_propagator_module = wave_propagator_cuda
        elif backend_type is BackendType.FFTW:
            from .backend import slice_builder_fftw, wave_propagator_fftw
            self._slice_builder_module = slice_builder_fftw
            self._wave_propagator_module = wave_propagator_fftw

    @property
    def one_slice_builder(self):
        return self._slice_builder_module.OneSliceBuilder
    
    @property
    def multi_slices_builder(self):
        return self._slice_builder_module.MultiSlicesBuilder

    @property
    def wave_propagator(self):
        return self._wave_propagator_module.WavePropagator

    def __repr__(self):
        return f"<emsim backend {str(self.backend_type)}>"


_current_backend = Backend(BackendType.NUMPY)


def get_current_backend():
    return _current_backend


def set_backend(backend="numpy"):
    global _current_backend

    type_dict = {"numpy": BackendType.NUMPY,
                 "cuda": BackendType.CUDA,
                 "fftw": BackendType.FFTW}
    if type(backend) is str:
        backend_type = type_dict.get(backend)
        if backend_type is None:
            raise ValueError(f"emsim: unknown backend: {backend}")
    elif type(backend) is BackendType:
        backend_type = backend
    else:
        raise ValueError(f"pass str or BackendType as argumen to set_backend")

    _current_backend = Backend(backend_type)

