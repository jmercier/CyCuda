from core cimport *
from libcuda cimport *

cdef extern from "stdlib.h":
    cdef void free(void *ptr)
    cdef void *malloc(size_t size)

