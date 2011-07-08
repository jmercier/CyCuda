cdef extern from "cuda.h":
    ############################
    #
    #   Enums
    #
    ############################

    ctypedef enum:
        CU_MEMHOSTALLOC_PORTABLE
        CU_MEMHOSTALLOC_DEVICEMAP
        CU_MEMHOSTALLOC_WRITECOMBINED

        CU_TRSF_READ_AS_INTEGER
        CU_TRSF_NORMALIZED_COORDINATES

    ctypedef enum CUresult:
        CUDA_SUCCESS
        CUDA_ERROR_INVALID_VALUE
        CUDA_ERROR_OUT_OF_MEMORY
        CUDA_ERROR_NOT_INITIALIZED
        CUDA_ERROR_DEINITIALIZED
        CUDA_ERROR_PROFILER_DISABLED
        CUDA_ERROR_PROFILER_NOT_INITIALIZED
        CUDA_ERROR_PROFILER_ALREADY_STARTED
        CUDA_ERROR_PROFILER_ALREADY_STOPPED
        CUDA_ERROR_NO_DEVICE
        CUDA_ERROR_INVALID_DEVICE
        CUDA_ERROR_INVALID_IMAGE
        CUDA_ERROR_INVALID_CONTEXT
        CUDA_ERROR_CONTEXT_ALREADY_CURRENT
        CUDA_ERROR_MAP_FAILED
        CUDA_ERROR_UNMAP_FAILED
        CUDA_ERROR_ARRAY_IS_MAPPED
        CUDA_ERROR_ALREADY_MAPPED
        CUDA_ERROR_NO_BINARY_FOR_GPU
        CUDA_ERROR_ALREADY_ACQUIRED
        CUDA_ERROR_NOT_MAPPED
        CUDA_ERROR_NOT_MAPPED_AS_ARRAY
        CUDA_ERROR_NOT_MAPPED_AS_POINTER
        CUDA_ERROR_ECC_UNCORRECTABLE
        CUDA_ERROR_UNSUPPORTED_LIMIT
        CUDA_ERROR_CONTEXT_ALREADY_IN_USE
        CUDA_ERROR_INVALID_SOURCE
        CUDA_ERROR_FILE_NOT_FOUND
        CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND
        CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
        CUDA_ERROR_OPERATING_SYSTEM
        CUDA_ERROR_INVALID_HANDLE
        CUDA_ERROR_NOT_FOUND
        CUDA_ERROR_NOT_READY
        CUDA_ERROR_LAUNCH_FAILED
        CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
        CUDA_ERROR_LAUNCH_TIMEOUT
        CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING
        CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
        CUDA_ERROR_PEER_ACCESS_NOT_ENABLED
        CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE
        CUDA_ERROR_CONTEXT_IS_DESTROYED
        CUDA_ERROR_UNKNOWN

    ctypedef enum CUdevice_attribute:
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
        CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK
        CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY
        CU_DEVICE_ATTRIBUTE_WARP_SIZE
        CU_DEVICE_ATTRIBUTE_MAX_PITCH
        CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
        CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK
        CU_DEVICE_ATTRIBUTE_CLOCK_RATE
        CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT
        CU_DEVICE_ATTRIBUTE_GPU_OVERLAP
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
        CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT
        CU_DEVICE_ATTRIBUTE_INTEGRATED
        CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY
        CU_DEVICE_ATTRIBUTE_COMPUTE_MODE
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES
        CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT
        CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS
        CU_DEVICE_ATTRIBUTE_ECC_ENABLED
        CU_DEVICE_ATTRIBUTE_PCI_BUS_ID
        CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID
        CU_DEVICE_ATTRIBUTE_TCC_DRIVER
        CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE
        CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH
        CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR
        CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT
        CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS
        CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID

    ctypedef enum CUfunction_attribute:
        CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
        CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
        CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
        CU_FUNC_ATTRIBUTE_NUM_REGS
        CU_FUNC_ATTRIBUTE_PTX_VERSION
        CU_FUNC_ATTRIBUTE_BINARY_VERSION
        CU_FUNC_ATTRIBUTE_MAX

    ctypedef enum CUctx_flags:
        CU_CTX_SCHED_AUTO
        CU_CTX_SCHED_SPIN
        CU_CTX_SCHED_YIELD
        CU_CTX_SCHED_BLOCKING_SYNC
        CU_CTX_BLOCKING_SYNC
        CU_CTX_SCHED_MASK
        CU_CTX_MAP_HOST
        CU_CTX_LMEM_RESIZE_TO_MAX
        CU_CTX_FLAGS_MASK

    ctypedef enum CUarray_format:
        CU_AD_FORMAT_UNSIGNED_INT8
        CU_AD_FORMAT_UNSIGNED_INT16
        CU_AD_FORMAT_UNSIGNED_INT32
        CU_AD_FORMAT_SIGNED_INT8
        CU_AD_FORMAT_SIGNED_INT16
        CU_AD_FORMAT_SIGNED_INT32
        CU_AD_FORMAT_HALF
        CU_AD_FORMAT_FLOAT

    ctypedef enum CUaddress_mode:
        CU_TR_ADDRESS_MODE_WRAP
        CU_TR_ADDRESS_MODE_CLAMP
        CU_TR_ADDRESS_MODE_MIRROR
        CU_TR_ADDRESS_MODE_BORDER

    ctypedef enum CUfilter_mode:
        CU_TR_FILTER_MODE_POINT
        CU_TR_FILTER_MODE_LINEAR

    ctypedef enum CUmemorytype:
        CU_MEMORYTYPE_HOST
        CU_MEMORYTYPE_DEVICE
        CU_MEMORYTYPE_ARRAY
        CU_MEMORYTYPE_UNIFIED

    ctypedef enum CUevent_flags:
        CU_EVENT_DEFAULT
        CU_EVENT_BLOCKING_SYNC
        CU_EVENT_DISABLE_TIMING

    ctypedef enum CUjit_option:
        CU_JIT_MAX_REGISTER
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

    ctypedef enum CUaddress_mode:
        CU_TR_ADDRESS_MODE_WRAP
        CU_TR_ADDRESS_MODE_CLAMP
        CU_TR_ADDRESS_MODE_MIRROR
        CU_TR_ADDRESS_MODE_BORDER

    ############################
    #
    #   Structs
    #
    ############################
    ctypedef unsigned long long CUdeviceptr
    ctypedef int CUdevice
    ctypedef struct CUcontext:
        pass
    ctypedef struct CUarray:
        pass
    ctypedef struct CUstream:
        pass
    ctypedef struct CUevent:
        pass
    ctypedef struct CUmodule:
        pass
    ctypedef struct CUfunction:
        pass
    ctypedef struct CUtexref:
        pass

    ctypedef struct CUDA_MEMCPY2D:
        unsigned int srcXInBytes
        unsigned int srcY
        CUmemorytype srcMemoryType
        void *srcHost
        CUdeviceptr srcDevice
        CUarray srcArray
        unsigned int srcPitch
        unsigned int dstXInBytes
        unsigned int dstY
        CUmemorytype dstMemoryType
        void *dstHost
        CUdeviceptr dstDevice
        CUarray dstArray
        unsigned int dstPitch
        unsigned int WidthInBytes
        unsigned int Height

    ctypedef struct CUDA_ARRAY_DESCRIPTOR:
        size_t Width
        size_t Height
        CUarray_format Format
        size_t NumChannels

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
    cdef:
        CUresult cuCtxCreate(CUcontext *, unsigned int, CUdevice) nogil
        CUresult cuCtxDestroy(CUcontext) nogil
        CUresult cuCtxPopCurrent(CUcontext *) nogil
        CUresult cuCtxPushCurrent(CUcontext) nogil
        CUresult cuCtxDetach(CUcontext) nogil
        CUresult cuCtxSynchronize() nogil
        CUresult cuCtxEnablePeerAccess(CUcontext, unsigned int)
        CUresult cuCtxDisablePeerAccess(CUcontext)

    ############################
    #
    #   Devices
    #
    ############################
    cdef:
        CUresult cuDeviceGet(CUdevice *, int) nogil
        CUresult cuDeviceComputeCapability(int *, int *, CUdevice) nogil
        CUresult cuDeviceGetCount(int *) nogil
        CUresult cuDeviceGetName(char *, int, CUdevice) nogil
        CUresult cuDeviceTotalMem(size_t *, CUdevice) nogil
        CUresult cuDeviceGetAttribute(int *, CUdevice_attribute, CUdevice) nogil
        CUresult cuDeviceCanAccessPeer(int *, CUdevice , CUdevice )

    ############################
    #
    #   Memory
    #
    ############################
    cdef:
        CUresult cuMemFree(CUdeviceptr) nogil
        CUresult cuMemAlloc(CUdeviceptr *, unsigned int) nogil
        CUresult cuMemAllocPitch(CUdeviceptr *, size_t *, size_t, size_t, unsigned int) nogil
        CUresult cuMemAllocHost(void **, unsigned int) nogil
        CUresult cuMemFreeHost(void *) nogil
        CUresult cuMemGetInfo(size_t *, size_t *) nogil
        CUresult cuMemcpy2D(CUDA_MEMCPY2D *pCopy)
        CUresult cuMemcpy2DAsync(CUDA_MEMCPY2D *pCopy, CUstream)
        CUresult cuMemcpyDtoD (CUdeviceptr, CUdeviceptr, unsigned int)
        CUresult cuMemcpyDtoDAsync(CUdeviceptr, CUdeviceptr, unsigned int, CUstream)
        CUresult cuMemcpyDtoH(void *, CUdeviceptr, unsigned int)
        CUresult cuMemcpyDtoHAsync(void *, CUdeviceptr, unsigned int, CUstream)
        CUresult cuMemcpyHtoD(CUdeviceptr,void *, unsigned int)
        CUresult cuMemcpyHtoDAsync(CUdeviceptr,void *, unsigned int, CUstream)
        CUresult cuMemHostGetFlags(unsigned int *, void *)
        CUresult cuMemcpy(CUdeviceptr, CUdeviceptr, size_t)
        CUresult cuMemcpyAsync(CUdeviceptr , CUdeviceptr , size_t , CUstream )
        CUresult cuMemHostGetDevicePointer(CUdeviceptr *, void *, unsigned int)


    ############################
    #
    #   Events
    #
    ############################
    cdef:
        CUresult cuEventElapsedTime(float *, CUevent, CUevent) nogil
        CUresult cuEventCreate(CUevent *, unsigned int) nogil
        CUresult cuEventQuery(CUevent) nogil
        CUresult cuEventRecord(CUevent, CUstream) nogil
        CUresult cuEventDestroy(CUevent) nogil
        CUresult cuEventSynchronize(CUevent)

    ############################
    #
    #   Streams
    #
    ############################
    cdef:
        CUresult cuStreamCreate(CUstream *, unsigned int) nogil
        CUresult cuStreamQuery(CUstream hStream) nogil
        CUresult cuStreamSynchronize(CUstream hStream) nogil
        CUresult cuStreamDestroy(CUstream hStream) nogil
        CUresult cuStreamWaitEvent(CUstream, CUevent, unsigned int )

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
    cdef CUresult cuTexRefSetAddress2D (CUtexref, CUDA_ARRAY_DESCRIPTOR *, CUdeviceptr, unsigned int) nogil
    cdef CUresult cuTexRefDestroy (CUtexref) nogil
    cdef CUresult cuTexRefCreate (CUtexref *) nogil
    cdef CUresult cuTexRefSetFlags (CUtexref, unsigned int) nogil
    cdef CUresult cuTexRefGetFlags (unsigned int *, CUtexref) nogil
    cdef CUresult cuTexRefGetFilterMode (CUfilter_mode *, CUtexref) nogil
    cdef CUresult cuTexRefSetFilterMode (CUtexref, CUfilter_mode) nogil
    cdef CUresult cuTexRefGetAddressMode (CUaddress_mode *, CUtexref, int) nogil
    cdef CUresult cuTexRefSetAddressMode (CUtexref, int, CUaddress_mode) nogil









