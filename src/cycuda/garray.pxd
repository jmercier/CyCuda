from core cimport *
from libcuda cimport *

cdef class gndarray(object):
    cdef CudaBuffer _buf
    cdef tuple _shape
    cdef tuple _strides
    cdef tuple _offset
    cdef bint _contiguous
    cdef unsigned int _ndim

cdef extern from "stdlib.h":
    cdef void free(void *ptr)
    cdef void *malloc(size_t size)

ctypedef struct gimage_st:
    CUdeviceptr data
    unsigned int shape[2]
    unsigned int strides[2]

ctypedef struct garray_st:
    CUdeviceptr data
    unsigned int size
