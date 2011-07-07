from libcuda cimport *

cdef extern from "pyerrors.h":
    ctypedef class __builtin__.Exception [object PyBaseExceptionObject]: pass

cdef class TexRefBase(object):
    cdef CUtexref _tex


cimport numpy as np

cdef class ModuleTexRef(TexRefBase)
cdef class Function(object)
cdef class Context(object)

cdef class CudaError(Exception): pass

cdef class Device(object):
    cdef CUdevice _dev
    cdef Context _ctx_create(self,  CUresult (*allocator)(CUcontext *, unsigned int, CUdevice), CUctx_flags flags)

cdef class Context(object):
    cdef CUcontext _ctx


cdef class CuHostBuffer(object):
    cdef Context ctx
    cdef size_t nbytes
    cdef void * data

cdef class CuBuffer(object):
    cdef Context ctx
    cdef size_t nbytes
    cdef CUdeviceptr buf

cdef class CuTypedBuffer(CuBuffer):
    cdef np.dtype dtype
    cdef tuple shape

cdef class Stream(object):
    cdef CUstream _stream
    cpdef bint query(self)
    cpdef synchronize(self)

cdef class Event(object):
    cdef Context ctx
    cdef CUevent _evt

    cpdef record(self, Stream stream)
    cpdef bint query(self)

cdef class Module(object):
    cdef Context ctx
    cdef CUmodule _mod


cdef class Function(object):
    cdef Module mod
    cdef CUfunction _fun
    cdef object _pstruct
    cdef unsigned int _pstruct_size

    cpdef launch_grid(self, int xgrid, int ygrid)
    cpdef setBlockShape(self, int xblock, int yblock, int zblock)
    cdef void __launchGridAsync__(self, int xgrid, int ygrid, Stream s)



cdef class ModuleTexRef(TexRefBase):
    cdef Module mod

cdef class TexRef(TexRefBase):
    cdef Context ctx

cdef CudaError translateError(CUresult error)

