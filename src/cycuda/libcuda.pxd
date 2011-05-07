ctypedef unsigned long long FCUdeviceptr

cdef extern from "cuda.h":
    ############################
    #
    #   Enums
    #
    ############################

    cdef enum CUresult "cudaError_enum":
        CUDA_SUCCESS                             = 0
        CUDA_ERROR_INVALID_VALUE                 = 1
        CUDA_ERROR_OUT_OF_MEMORY                 = 2
        CUDA_ERROR_NOT_INITIALIZED               = 3
        CUDA_ERROR_DEINITIALIZED                 = 4
        CUDA_ERROR_NO_DEVICE                     = 100
        CUDA_ERROR_INVALID_DEVICE                = 101
        CUDA_ERROR_INVALID_IMAGE                 = 200
        CUDA_ERROR_INVALID_CONTEXT               = 201
        CUDA_ERROR_CONTEXT_ALREADY_CURRENT       = 202
        CUDA_ERROR_MAP_FAILED                    = 205
        CUDA_ERROR_UNMAP_FAILED                  = 206
        CUDA_ERROR_ARRAY_IS_MAPPED               = 207
        CUDA_ERROR_ALREADY_MAPPED                = 208
        CUDA_ERROR_NO_BINARY_FOR_GPU             = 209
        CUDA_ERROR_ALREADY_ACQUIRED              = 210
        CUDA_ERROR_NOT_MAPPED                    = 211
        CUDA_ERROR_NOT_MAPPED_AS_ARRAY           = 212
        CUDA_ERROR_NOT_MAPPED_AS_POINTER         = 213
        CUDA_ERROR_INVALID_SOURCE                = 300
        CUDA_ERROR_FILE_NOT_FOUND                = 301
        CUDA_ERROR_INVALID_HANDLE                = 400
        CUDA_ERROR_NOT_FOUND                     = 500
        CUDA_ERROR_NOT_READY                     = 600
        CUDA_ERROR_LAUNCH_FAILED                 = 700
        CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES       = 701
        CUDA_ERROR_LAUNCH_TIMEOUT                = 702
        CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703
        #CUDA_ERROR_POINTER_IS_64BIT              = 800
        #CUDA_ERROR_SIZE_IS_64BIT                 = 801
        CUDA_ERROR_UNKNOWN                       = 999

    cdef enum CUdevice_attribute "CUdevice_attribute_enum":
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK             = 1
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X                   = 2
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y                   = 3
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z                   = 4
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X                    = 5
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y                    = 6
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z                    = 7
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK       = 8
        CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK           = 8
        CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY             = 9
        CU_DEVICE_ATTRIBUTE_WARP_SIZE                         = 10
        CU_DEVICE_ATTRIBUTE_MAX_PITCH                         = 11
        CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK           = 12
        CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK               = 12
        CU_DEVICE_ATTRIBUTE_CLOCK_RATE                        = 13
        CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT                 = 14
        CU_DEVICE_ATTRIBUTE_GPU_OVERLAP                       = 15
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT              = 16
        CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT               = 17
        CU_DEVICE_ATTRIBUTE_INTEGRATED                        = 18
        CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY               = 19
        CU_DEVICE_ATTRIBUTE_COMPUTE_MODE                      = 20
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH           = 21
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH           = 22
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT          = 23
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH           = 24
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT          = 25
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH           = 26
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH     = 27
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT    = 28
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29

    cdef enum CUfunction_attribute "CUfunction_attribute_enum":
        CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
        CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES     = 2
        CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES      = 3
        CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES      = 4
        CU_FUNC_ATTRIBUTE_NUM_REGS              = 5

    cdef enum CUctx_flags "CUctx_flags_enum":
        CU_CTX_SCHED_AUTO         = 0
        CU_CTX_SCHED_SPIN         = 1
        CU_CTX_SCHED_YIELD        = 2
        CU_CTX_SCHED_MASK         = 0x3
        CU_CTX_BLOCKING_SYNC      = 4
        CU_CTX_MAP_HOST           = 8
        CU_CTX_LMEM_RESIZE_TO_MAX = 16
        CU_CTX_FLAGS_MASK         = 0x1f

    cdef enum CUarray_format "CUarray_format_enum":
        CU_AD_FORMAT_UNSIGNED_INT8  = 0x01
        CU_AD_FORMAT_UNSIGNED_INT16 = 0x02
        CU_AD_FORMAT_UNSIGNED_INT32 = 0x03
        CU_AD_FORMAT_SIGNED_INT8    = 0x08
        CU_AD_FORMAT_SIGNED_INT16   = 0x09
        CU_AD_FORMAT_SIGNED_INT32   = 0x0a
        CU_AD_FORMAT_HALF           = 0x10
        CU_AD_FORMAT_FLOAT          = 0x20

    cdef enum CUaddress_mode "CUaddress_mode_enum":
        CU_TR_ADDRESS_MODE_WRAP   = 0
        CU_TR_ADDRESS_MODE_CLAMP  = 1
        CU_TR_ADDRESS_MODE_MIRROR = 2

    cdef enum CUfilter_mode "CUfilter_mode_enum":
        CU_TR_FILTER_MODE_POINT  = 0
        CU_TR_FILTER_MODE_LINEAR = 1

    cdef enum CUmemorytype "CUmemorytype_enum":
        CU_MEMORYTYPE_HOST   = 0x01
        CU_MEMORYTYPE_DEVICE = 0x02
        CU_MEMORYTYPE_ARRAY  = 0x03

    cdef enum CUevent_flags "CUevent_flags_enum":
        CU_EVENT_DEFAULT       = 0
        CU_EVENT_BLOCKING_SYNC = 1

    cdef enum CUjit_option "CUjit_option_enum":
        CU_JIT_MAX_REGISTER = 0
        CU_JIT_THREADS_PER_BLOCK
        CU_JIT_WALL_TIME
        CU_JIT_INFO_LOG_BUFFER
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES
        CU_JIT_ERROR_LOG_BUFFER
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
        CU_JIT_OPTIMIZATION_LEVEL
        CU_JIT_TARGET_FROM_CUCONTEXT
        CU_JIT_TARGET
        CU_JIT_FALLBACK_STRATEGY

    cdef enum CUaddress_mode "CUaddress_mode_enum":
        CU_TR_ADDRESS_MODE_WRAP   = 0
        CU_TR_ADDRESS_MODE_CLAMP  = 1
        CU_TR_ADDRESS_MODE_MIRROR = 2

    ############################
    #
    #   Structs
    #
    ############################
    ctypedef void * CUcontext
    ctypedef void * CUdevice
    ctypedef void * CUarray
    ctypedef void * CUstream
    ctypedef void * CUevent
    ctypedef void * CUmodule
    ctypedef void * CUfunction
    ctypedef void * CUtexref
    ctypedef void * CUdeviceptr

    ctypedef struct CUDA_MEMCPY2D:
        unsigned int srcXInBytes, srcY
        CUmemorytype srcMemoryType
        void *srcHost
        FCUdeviceptr srcDevice
        CUarray srcArray
        unsigned int srcPitch
        unsigned int dstXInBytes, dstY
        CUmemorytype dstMemoryType
        void *dstHost
        FCUdeviceptr dstDevice
        CUarray dstArray
        unsigned int dstPitch
        unsigned int WidthInBytes
        unsigned int Height

    ctypedef struct CUDA_ARRAY_DESCRIPTOR:
        unsigned int Height
        unsigned int NumChannels
        CUarray_format Format
        unsigned int Width

    ############################
    #
    #   Generals
    #
    ############################
    cdef CUresult cuInit(unsigned int) nogil
    cdef CUresult cuDriverGetVersion(int *) nogil

    ############################
    #
    #   Contexts
    #
    ############################
    cdef CUresult cuCtxCreate(CUcontext *, unsigned int, CUdevice) nogil
    cdef CUresult cuCtxDestroy(CUcontext) nogil
    cdef CUresult cuCtxPopCurrent(CUcontext *) nogil
    cdef CUresult cuCtxPushCurrent(CUcontext) nogil
    cdef CUresult cuCtxDetach(CUcontext) nogil
    cdef CUresult cuCtxSynchronize() nogil

    ############################
    #
    #   Devices
    #
    ############################
    cdef CUresult cuDeviceGet(CUdevice *, int) nogil
    cdef CUresult cuDeviceComputeCapability(int *, int *, CUdevice) nogil
    cdef CUresult cuDeviceGetCount(int *) nogil
    cdef CUresult cuDeviceGetName(char *, int, CUdevice) nogil
    cdef CUresult cuDeviceTotalMem(size_t *, CUdevice) nogil
    cdef CUresult cuDeviceGetAttribute(int *, CUdevice_attribute, CUdevice) nogil

    ############################
    #
    #   Memory
    #
    ############################
    cdef CUresult cuMemFree(FCUdeviceptr) nogil
    cdef CUresult cuMemAlloc(FCUdeviceptr *, unsigned int) nogil
    cdef CUresult cuMemAllocPitch(FCUdeviceptr *, size_t *, size_t, size_t, unsigned int) nogil
    cdef CUresult cuMemAllocHost(void **, unsigned int) nogil
    cdef CUresult cuMemFreeHost(void *) nogil
    cdef CUresult cuMemHostGetDevicePointer (FCUdeviceptr *,void *, unsigned int) nogil
    cdef CUresult cuMemGetInfo(size_t *, size_t *) nogil
    cdef CUresult cuMemcpy2D(CUDA_MEMCPY2D *pCopy)
    cdef CUresult cuMemcpy2DAsync(CUDA_MEMCPY2D *pCopy, CUstream)
    cdef CUresult cuMemcpyDtoD (FCUdeviceptr, FCUdeviceptr, unsigned int)
    cdef CUresult cuMemcpyDtoDAsync(FCUdeviceptr, FCUdeviceptr, unsigned int, CUstream)
    cdef CUresult cuMemcpyDtoH(void *, FCUdeviceptr, unsigned int)
    cdef CUresult cuMemcpyDtoHAsync(void *, FCUdeviceptr, unsigned int, CUstream)
    cdef CUresult cuMemcpyHtoD(FCUdeviceptr,void *, unsigned int)
    cdef CUresult cuMemcpyHtoDAsync(FCUdeviceptr,void *, unsigned int, CUstream)

    ############################
    #
    #   Events
    #
    ############################
    cdef CUresult cuEventElapsedTime(float *, CUevent, CUevent) nogil
    cdef CUresult cuEventCreate(CUevent *, unsigned int) nogil
    cdef CUresult cuEventQuery(CUevent) nogil
    cdef CUresult cuEventRecord(CUevent, CUstream) nogil
    cdef CUresult cuEventDestroy(CUevent) nogil

    ############################
    #
    #   Streams
    #
    ############################
    cdef CUresult cuStreamCreate(CUstream *, unsigned int) nogil
    cdef CUresult cuStreamQuery(CUstream hStream) nogil
    cdef CUresult cuStreamSynchronize(CUstream hStream) nogil
    cdef CUresult cuStreamDestroy(CUstream hStream) nogil

    ############################
    #
    #   Execution Control
    #
    ############################
    cdef CUresult cuFuncGetAttribute(int *, CUfunction_attribute, CUfunction) nogil
    cdef CUresult cuParamSetv(CUfunction, int, void *, unsigned int) nogil
    cdef CUresult cuFuncSetSharedSize(CUfunction, unsigned int) nogil
    cdef CUresult cuParamSetSize(CUfunction, unsigned int) nogil
    cdef CUresult cuLaunchGrid(CUfunction, int, int) nogil
    cdef CUresult cuFuncSetBlockShape(CUfunction, int, int, int) nogil
    cdef CUresult cuLaunchGridAsync(CUfunction, int, int, CUstream) nogil

    cdef CUresult cuModuleLoad(CUmodule *, char *) nogil
    cdef CUresult cuModuleLoadDataEx(CUmodule *, void *, unsigned int, CUjit_option *, void **)
    cdef CUresult cuModuleUnload(CUmodule) nogil
    cdef CUresult cuModuleGetFunction(CUfunction *, CUmodule, char *) nogil
    cdef CUresult cuModuleGetTexRef (CUtexref *, CUmodule, char *name) nogil

    ############################
    #
    # Texrefs
    #
    ############################
    cdef CUresult cuTexRefSetAddress2D (CUtexref, CUDA_ARRAY_DESCRIPTOR *, FCUdeviceptr, unsigned int) nogil
    cdef CUresult cuTexRefDestroy (CUtexref) nogil
    cdef CUresult cuTexRefCreate (CUtexref *) nogil
    cdef CUresult cuTexRefSetFlags (CUtexref, unsigned int) nogil
    cdef CUresult cuTexRefGetFlags (unsigned int *, CUtexref) nogil
    cdef CUresult cuTexRefGetFilterMode (CUfilter_mode *, CUtexref) nogil
    cdef CUresult cuTexRefSetFilterMode (CUtexref, CUfilter_mode) nogil
    cdef CUresult cuTexRefGetAddressMode (CUaddress_mode *, CUtexref, int) nogil
    cdef CUresult cuTexRefSetAddressMode (CUtexref, int, CUaddress_mode) nogil









