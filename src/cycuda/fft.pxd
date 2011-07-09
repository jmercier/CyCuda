from libcuda cimport *

cdef extern from "cuComplex.h":
    pass

cdef extern from "cufft.h":
    ctypedef enum:
        CUFFT_FORWARD
        CUFFT_INVERSE

    ctypedef enum cufftCompatibility:
        CUFFT_COMPATIBILITY_NATIVE
        CUFFT_COMPATIBILITY_FFTW_PADDING
        CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC
        CUFFT_COMPATIBILITY_FFTW_ALL

    ctypedef enum cufftType:
        CUFFT_R2C
        CUFFT_C2R
        CUFFT_C2C
        CUFFT_D2Z
        CUFFT_Z2D
        CUFFT_Z2Z

    ctypedef enum cufftResult:
        CUFFT_SUCCESS
        CUFFT_INVALID_PLAN
        CUFFT_ALLOC_FAILED
        CUFFT_INVALID_TYPE
        CUFFT_INVALID_VALUE
        CUFFT_INTERNAL_ERROR
        CUFFT_EXEC_FAILED
        CUFFT_SETUP_FAILED
        CUFFT_INVALID_SIZE
        CUFFT_UNALIGNED_DATA

    ctypedef struct cufftReal:
        pass
    ctypedef struct cufftDoubleReal:
        pass
    ctypedef struct cufftComplex:
        pass
    ctypedef struct cufftDoubleComplex:
        pass
    ctypedef struct cufftHandle:
        pass

    cdef cufftResult cufftPlan1d(cufftHandle *, int, cufftType, int)
    cdef cufftResult cufftPlan2d(cufftHandle *, int, int, cufftType)
    cdef cufftResult cufftPlan3d(cufftHandle *, int, int, int, cufftType)
    cdef cufftResult cufftPlanMany(cufftHandle *, int, int *, int *, int, int, int *, int, int, cufftType, int)

    cdef cufftResult cufftDestroy(cufftHandle)

    cdef cufftResult cufftExecC2C(cufftHandle plan,
                                  cufftComplex *idata,
                                  cufftComplex *odata,
                                  int direction)

    cdef cufftResult cufftExecR2C(cufftHandle plan,
                                  cufftReal * idata,
                                  cufftComplex* odata)

    cdef cufftResult cufftExecC2R(cufftHandle plan,
                                  cufftComplex * idata,
                                  cufftReal * odata)

    cdef cufftResult cufftExecZ2Z(cufftHandle plan,
                                  cufftDoubleComplex * idata,
                                  cufftDoubleComplex * odata,
                                  int direction)

    cdef cufftResult cufftExecD2Z(cufftHandle plan,
                                  cufftDoubleReal * idata,
                                  cufftDoubleComplex *odata)

    cdef cufftResult cufftExecZ2D(cufftHandle plan,
                                  cufftDoubleComplex * idata,
                                  cufftDoubleReal * odata)

    #cdef cufftResult cufftSetStream(cufftHandle plan,
    #                                cudaStream_t stream)

    cdef cufftResult cufftSetCompatibilityMode(cufftHandle plan,
                                               cufftCompatibility mode)

    cdef cufftResult cufftGetVersion(int *version)
