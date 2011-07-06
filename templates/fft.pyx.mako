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

cdef class FFTPlan(object):
    cdef cufftHandle plan
    cdef bint init

    def __init__(self, tuple shape, cufftType ftype):
        self.init = False
        if len(shape) == 1:
            CuFFTSafeCall(cufftPlan1d(&self.plan, shape[0], ftype, 1))
        elif len(shape) == 2:
            CuFFTSafeCall(cufftPlan2d(&self.plan, shape[1], shape[0], ftype))
        elif len(shape) == 3:
            CuFFTSafeCall(cufftPlan3d(&self.plan, shape[2], shape[1], shape[0], ftype))
        else:
            raise RuntimeError("Incorrect shape : Only handle 1D/2D or 3D Array")

        self.init = True

    def dealloc(self):
        if self.init:
            cufftDestroy(self.plan)

    property compatibility_mode:
        def __set__(self, cufftCompatibility mode):
            CuFFTSafeCall(cufftSetCompatibilityMode(self.plan, mode))


cdef class FFTPlanC2R(FFTPlan):
    def __init__(self, tuple shape):
        FFTPlan.__init__(self, shape, CUFFT_C2R)

    def execute(self, core.DeviceBuffer inp, core.DeviceBuffer out):
        CuFFTSafeCall(cufftExecC2R(self.plan, inp.buffer_handle, out.buffer_handle))


cdef class FFTPlanC2C(FFTPlan):
    def __init__(self, tuple shape):
        FFTPlan.__init__(self, shape, CUFFT_C2C)

    def execute(self, core.DeviceBuffer inp, core.DeviceBuffer out, int direction = 1):
        CuFFTSafeCall(cufftExecC2C(self.plan, inp.buffer_handle, out.buffer_handle, direction))

cdef class FFTPlanR2C(FFTPlan):
    def __init__(self, tuple shape):
        FFTPlan.__init__(self, shape, CUFFT_R2C)

    def execute(self, core.DeviceBuffer inp, core.DeviceBuffer out):
        CuFFTSafeCall(cufftExecR2C(self.plan, inp.buffer_handle, out.buffer_handle))


cdef class FFTPlanD2Z(FFTPlan):
    def __init__(self, tuple shape):
        FFTPlan.__init__(self, shape, CUFFT_D2Z)

    def execute(self, core.DeviceBuffer inp, core.DeviceBuffer out):
        CuFFTSafeCall(cufftExecD2Z(self.plan, inp.buffer_handle, out.buffer_handle))

cdef class FFTPlanZ2D(FFTPlan):
    def __init__(self, tuple shape):
        FFTPlan.__init__(self, shape, CUFFT_Z2D)

    def execute(self, core.DeviceBuffer inp, core.DeviceBuffer out):
        CuFFTSafeCall(cufftExecZ2D(self.plan, inp.buffer_handle, out.buffer_handle))

cdef class FFTPlanZ2Z(FFTPlan):
    def __init__(self, tuple shape):
        FFTPlan.__init__(self, shape, CUFFT_Z2Z)

    def execute(self, core.DeviceBuffer inp, core.DeviceBuffer out, int direction = 1):
        CuFFTSafeCall(cufftExecZ2Z(self.plan, inp.buffer_handle, out.buffer_handle, direction))
#
# vim: filetype=pyrex
#
#

