from libcuda cimport *

cdef extern from "pyerrors.h":
    ctypedef class __builtin__.Exception [object PyBaseExceptionObject]: pass

cdef class CudaError(Exception): pass

cdef class CudaObject: pass
cdef class LinearAllocator
cdef class PitchedAllocator
cdef class PinnedAllocator

cdef class Device
cdef class Context(CudaObject):
    cdef CUcontext _ctx
    cpdef pushCurrent(self)


cdef class CudaBuffer(CudaObject):
    cdef Context ctx
    cdef FCUdeviceptr _deviceBuf
    cdef unsigned int _pitch
    cdef unsigned int _size

cdef class DeviceBuffer(CudaBuffer): pass

cdef class HostBuffer(CudaBuffer):
    cdef void * _hostBuf

cdef class Stream(CudaObject):
    cdef CUstream _stream
    cpdef bint query(self)
    cpdef synchronize(self)

cdef class Event

cdef class Module

cdef class Function


cdef class TexRefBase(CudaObject):
    cdef CUtexref _tex

cdef class TexRef(TexRefBase):
    cdef Context ctx

cdef class ModuleTexRef(TexRefBase):
    cdef Module mod


cdef CudaError translateError(CUresult error)

