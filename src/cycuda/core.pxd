from libcuda cimport *

cdef extern from "pyerrors.h":
    ctypedef class __builtin__.Exception [object PyBaseExceptionObject]: pass

cdef class TexRefBase(object):
    cdef CUtexref _tex



cdef class ModuleTexRef(TexRefBase)
cdef class Function(object)
cdef class Context(object)

cdef class CudaError(Exception): pass


cdef class LinearAllocator : pass
cdef class PitchedAllocator : pass
cdef class PinnedAllocator : pass



cdef class Device(object):
    cdef CUdevice _dev
    cdef Context _ctxCreate(self,  CUresult (*allocator)(CUcontext *, unsigned int, CUdevice), CUctx_flags flags)

cdef class Context(object):
    cdef CUcontext _ctx
    cpdef pushCurrent(self)


cdef class Buffer(object):
    cdef Context ctx
    cdef CUdeviceptr buffer_handle
    cdef size_t size

cdef class PitchedBuffer(Buffer):
    cdef size_t pitch
        

cdef class HBuffer(Buffer):
    cdef void * host


cdef class CudaBuffer(object):
    cdef Context ctx
    cdef CUdeviceptr _deviceBuf
    cdef unsigned int _pitch
    cdef unsigned int _size

cdef class DeviceBuffer(CudaBuffer): pass

cdef class HostBuffer(CudaBuffer):
    cdef void * _hostBuf

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

    cpdef Function getFunction(self, char *name)
    cpdef ModuleTexRef getTexref(self, char *name)

cdef class Function(object):
    cdef Module mod
    cdef CUfunction _fun
    cdef object _pstruct
    cdef unsigned int _pstruct_size

    cpdef launchGrid(self, int xgrid, int ygrid)
    cpdef setBlockShape(self, int xblock, int yblock, int zblock)
    cdef void __launchGridAsync__(self, int xgrid, int ygrid, Stream s)



cdef class ModuleTexRef(TexRefBase):
    cdef Module mod

cdef class TexRef(TexRefBase):
    cdef Context ctx

cdef CudaError translateError(CUresult error)

