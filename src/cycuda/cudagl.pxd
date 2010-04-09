from libcuda cimport *

cdef extern from "GL/gl.h": pass

cdef extern from "cudaGL.h":
        cdef CUresult cuGLCtxCreate(CUcontext *, unsigned int, CUdevice) nogil
