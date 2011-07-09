from libcuda cimport *

cdef extern from "cudaProfiler.h":

    cdef enum CUoutput_mode:
        CU_OUT_KEY_VALUE_PAIR
        CU_OUT_CSV

    cdef:
        CUresult cuProfilerStart()
        CUresult cuProfilerStop()
        CUresult cuProfilerInitialize(char *configFile, char *outputFile, CUoutput_mode outputMode)

