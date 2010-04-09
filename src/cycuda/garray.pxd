from core cimport *
from libcuda cimport *

cdef class gndarray
cdef class g2darray

cdef extern from "stdlib.h":
    cdef void free(void *ptr)
    cdef void *malloc(size_t size)

ctypedef struct gimage_st:
    FCUdeviceptr data
    unsigned int shape[2]
    unsigned int strides[2]

ctypedef struct garray_st:
    FCUdeviceptr data
    unsigned int size
