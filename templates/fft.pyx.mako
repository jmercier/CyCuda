cimport core

cdef extern from "pyerrors.h":
    ctypedef class __builtin__.Exception [object PyBaseExceptionObject]: pass

cdef class CuFFTError(Exception):
    pass

cdef dict error_translation_table     = \
        { CUFFT_INVALID_PLAN        : "INVALID PLAN",
          CUFFT_ALLOC_FAILED        : "ALLOC FAILED",
          CUFFT_INVALID_TYPE        : "INVALID TYPE",
          CUFFT_INVALID_VALUE       : "INVALID VALUE",
          CUFFT_INTERNAL_ERROR      : "INTERNAL ERROR",
          CUFFT_EXEC_FAILED         : "EXEC FAILED",
          CUFFT_SETUP_FAILED        : "SETUP FAILED",
          CUFFT_INVALID_SIZE        : "INVALID SIZE",
          CUFFT_UNALIGNED_DATA      : "UNALIGNED DATA"}

cdef int CuFFTSafeCall(cufftResult res) except -1:
    if res != CUFFT_SUCCESS:
        raise CuFFTError(error_translation_table[res])
        return -1
    return 0

def version():
    cdef int version
    cufftGetVersion(&version)
    return version

FORWARD = CUFFT_FORWARD
BACKWARD = CUFFT_FORWARD

cdef class FFTPlan(object):
    cdef cufftHandle plan
    cdef bint init
    cdef size_t __ndim
    cdef tuple __shape

    def __init__(self, tuple shape, cufftType ftype):
        self.init = False
        if len(shape) == 1:
            self.__ndim = 1
            CuFFTSafeCall(cufftPlan1d(&self.plan, shape[0], ftype, 1))
        elif len(shape) == 2:
            self.__ndim = 2
            CuFFTSafeCall(cufftPlan2d(&self.plan, shape[1], shape[0], ftype))
        elif len(shape) == 3:
            self.__ndim = 3
            CuFFTSafeCall(cufftPlan3d(&self.plan, shape[2], shape[1], shape[0], ftype))
        else:
            raise RuntimeError("Incorrect shape : Only handle 1D/2D or 3D Array")

        self.__shape = shape
        self.init = True

    def dealloc(self):
        if self.init:
            cufftDestroy(self.plan)

    property compatibility_mode:
        def __set__(self, cufftCompatibility mode):
            CuFFTSafeCall(cufftSetCompatibilityMode(self.plan, mode))

    def __repr__(self):
        return "<FFTPLAN shape=%s/>" % (str(self.__shape))


cdef class FFTPlanC2R(FFTPlan):
    def __init__(self, tuple shape):
        FFTPlan.__init__(self, shape, CUFFT_C2R)

    def execute(self, core.CuBuffer inp, core.CuBuffer out):
        CuFFTSafeCall(cufftExecC2R(self.plan, <cufftComplex *>inp.buf, <cufftReal *>out.buf))


cdef class FFTPlanC2C(FFTPlan):
    def __init__(self, tuple shape):
        FFTPlan.__init__(self, shape, CUFFT_C2C)

    def execute(self, core.CuBuffer inp, core.CuBuffer out, int direction = CUFFT_FORWARD):
        CuFFTSafeCall(cufftExecC2C(self.plan, <cufftComplex *>inp.buf, <cufftComplex *>out.buf, direction))

cdef class FFTPlanR2C(FFTPlan):
    def __init__(self, tuple shape):
        FFTPlan.__init__(self, shape, CUFFT_R2C)

    def execute(self, core.CuBuffer inp, core.CuBuffer out):
        CuFFTSafeCall(cufftExecR2C(self.plan, <cufftReal *>inp.buf, <cufftComplex *>out.buf))


cdef class FFTPlanD2Z(FFTPlan):
    def __init__(self, tuple shape):
        FFTPlan.__init__(self, shape, CUFFT_D2Z)

    def execute(self, core.CuBuffer inp, core.CuBuffer out):
        CuFFTSafeCall(cufftExecD2Z(self.plan, <cufftDoubleReal *>inp.buf, <cufftDoubleComplex *>out.buf))

cdef class FFTPlanZ2D(FFTPlan):
    def __init__(self, tuple shape):
        FFTPlan.__init__(self, shape, CUFFT_Z2D)

    def execute(self, core.CuBuffer inp, core.CuBuffer out):
        CuFFTSafeCall(cufftExecZ2D(self.plan, <cufftDoubleComplex *>inp.buf, <cufftDoubleReal *>out.buf))

cdef class FFTPlanZ2Z(FFTPlan):
    def __init__(self, tuple shape):
        FFTPlan.__init__(self, shape, CUFFT_Z2Z)

    def execute(self, core.CuBuffer inp, core.CuBuffer out, int direction = CUFFT_FORWARD):
        CuFFTSafeCall(cufftExecZ2Z(self.plan,<cufftDoubleComplex *>inp.buf, <cufftDoubleComplex *>out.buf, direction))
#
# vim: filetype=pyrex
#
#

