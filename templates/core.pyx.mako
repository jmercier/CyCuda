<%namespace file="functions.mako" import="*"/>\
${license}

<%
device_properties = { "max_grid_dim" : ['CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X',
                                        'CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y',
                                        'CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z'],
                      "max_block_dim" : ['CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X',
                                         'CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y',
                                         'CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z'],
                     "max_threads_per_block" : ['CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK'],
                     "max_shared_memory_per_block" : ['CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK'],
                     "shared_memory_per_block" : ["CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK"],
                     "total_constant_memory" : ["CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY"],
                     "warp_size" : ['CU_DEVICE_ATTRIBUTE_WARP_SIZE'],
                     "max_pitch" : ['CU_DEVICE_ATTRIBUTE_MAX_PITCH'],
                     "max_registers_per_block" : ["CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK"],
                     "registers_per_block" : ["CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK"],
                     "clock_rate" : ['CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK'],
                     "texture_alignment" : ["CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT"],
                     "gpu_overlap" : ['CU_DEVICE_ATTRIBUTE_GPU_OVERLAP'],
                     "multiprocessor_count" : ['CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT'],
                     "kernel_exec_timeout" : ['CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT'],
                     "integrated" : ['CU_DEVICE_ATTRIBUTE_INTEGRATED'],
                     "can_map_host_memory" : ['CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY'],
                     "compute_mode" : ['CU_DEVICE_ATTRIBUTE_COMPUTE_MODE'],
                     "maximum_texture_1d" : ['CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH'],
                     "maximum_texture_2d" : ['CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH',
                                             'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT'],
                     "maximum_texture_3d" : ['CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH',
                                             'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT',
                                             'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH'],
                     "maximum_texture_2d_layered" : ['CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH',
                                                     'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT',
                                                     'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS'],
                     "maximum_texture_2d_array" : ['CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH',
                                                   'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT',
                                                   'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES'],
                     "surface_alignment" : ['CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT'],
                     "concurent_kernels" : ['CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS'],
                     "ECC" : ['CU_DEVICE_ATTRIBUTE_ECC_ENABLED'],
                     "pci_bus" : ['CU_DEVICE_ATTRIBUTE_PCI_BUS_ID',
                                  'CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID',
                                  'CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID'],
                     "tcc_driver" : ['CU_DEVICE_ATTRIBUTE_TCC_DRIVER'],
                     "memory_clock_rate" : ['CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE'],
                     "global_memory_bus_width" : ['CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH'],
                     "l2_cache_size" : ['CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE'],
                     "max_threads_per_multiprocessor" : ['CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR'],
                     "async_engine_count" : ['CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT'],
                     "unified_addressing" : ['CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING'],
                     "maximum_texture_1d_layered" : ['CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH',
                                                     'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS'], }


%>
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
#from cycuda cimport cudagl

import threading
import struct

cimport numpy as np
import numpy as np

import operator

cimport profiler

def ProfilerStart():
    CudaSafeCall(profiler.cuProfilerStart())

def ProfilerStop():
    CudaSafeCall(profiler.cuProfilerStop())

def ProfilerInitialize(char * config, char * output):
    CudaSafeCall(profiler.cuProfilerInitialize(config, output, profiler.CU_OUT_KEY_VALUE_PAIR))

cdef dict error_translation_table     = \
    { CUDA_ERROR_INVALID_VALUE                  : "INVALID_VALUE",
      CUDA_ERROR_OUT_OF_MEMORY                  : "OUT_OF_MEMORY",
      CUDA_ERROR_NOT_INITIALIZED                : "NOT_INITIALIZED",
      CUDA_ERROR_DEINITIALIZED                  : "DEINITIALIZED",
      CUDA_ERROR_PROFILER_DISABLED              : "PROFILER_DISABLED",
      CUDA_ERROR_PROFILER_NOT_INITIALIZED       : "PROFILER_NOT_INITIALIZED",
      CUDA_ERROR_PROFILER_ALREADY_STARTED       : "PROFILER_ALREADY_STARTED",
      CUDA_ERROR_PROFILER_ALREADY_STOPPED       : "PROFILER_ALREADY_STOPPED",
      CUDA_ERROR_NO_DEVICE                      : "NO_DEVICE",
      CUDA_ERROR_INVALID_DEVICE                 : "INVALID_DEVICE",
      CUDA_ERROR_INVALID_IMAGE                  : "INVALID_IMAGE",
      CUDA_ERROR_INVALID_CONTEXT                : "INVALID_CONTEXT",
      CUDA_ERROR_CONTEXT_ALREADY_CURRENT        : "CONTEXT_ALREADY_CURRENT",
      CUDA_ERROR_MAP_FAILED                     : "MAP_FAILED",
      CUDA_ERROR_UNMAP_FAILED                   : "UNMAP_FAILED",
      CUDA_ERROR_ARRAY_IS_MAPPED                : "ARRAY_IS_MAPPED",
      CUDA_ERROR_ALREADY_MAPPED                 : "ALREADY_MAPPED",
      CUDA_ERROR_NO_BINARY_FOR_GPU              : "NO_BINARY_FOR_GPU",
      CUDA_ERROR_ALREADY_ACQUIRED               : "ALREADY_ACQUIRED",
      CUDA_ERROR_NOT_MAPPED                     : "NOT_MAPPED",
      CUDA_ERROR_NOT_MAPPED_AS_ARRAY            : "NOT_MAPPED_AS_ARRAY",
      CUDA_ERROR_NOT_MAPPED_AS_POINTER          : "NOT_MAPPED_AS_POINTER",
      CUDA_ERROR_ECC_UNCORRECTABLE              : "ECC_UNCORRECTABLE",
      CUDA_ERROR_UNSUPPORTED_LIMIT              : "UNSUPPORTED_LIMIT",
      CUDA_ERROR_CONTEXT_ALREADY_IN_USE         : "CONTEXT_ALREADLY_IN_USE",
      CUDA_ERROR_INVALID_SOURCE                 : "INVALID_SOURCE",
      CUDA_ERROR_FILE_NOT_FOUND                 : "FILE_NOT_FOUND",
      CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      : "SHARED_OBJECT_INIT_FAILED",
      CUDA_ERROR_OPERATING_SYSTEM               : "OPERATING_SYSTEM",
      CUDA_ERROR_INVALID_HANDLE                 : "INVALID_HANDLE",
      CUDA_ERROR_NOT_FOUND                      : "NOT_FOUND",
      CUDA_ERROR_NOT_READY                      : "NOT_READY",
      CUDA_ERROR_LAUNCH_FAILED                  : "LAUNCH_FAILED",
      CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        : "LAUNCH_OUT_OF_RESOURCES",
      CUDA_ERROR_LAUNCH_TIMEOUT                 : "LAUNCH_TIMEOUT",
      CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  : "LAUNCH_INCOMPATIBLE_TEXTURING",
      CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED    : "PEER_ACCESS_ALREADY_ENABLED",
      CUDA_ERROR_PEER_ACCESS_NOT_ENABLED        : "PEER_ACCESS_NOT_ENABLED",
      CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         : "PRIMARY_CONTEXT_ACTIVE",
      CUDA_ERROR_CONTEXT_IS_DESTROYED           : "CONTEXT_IS_DESCROYED",
      CUDA_ERROR_UNKNOWN                        : "ERROR_UNKNOWN" }


cdef CudaError translateError(CUresult error):
    return error_translation_table[error]

cdef inline int CudaSafeCall(CUresult result) except -1:
    if result != CUDA_SUCCESS:
        raise CudaError(error_translation_table[result])
        return -1
    return 0


#######################################
#
# Generic FCT
#
#######################################

cdef int _getDeviceAttribute(CUdevice_attribute att, CUdevice dev) except *:
    cdef int val
    CudaSafeCall(cuDeviceGetAttribute(&val, att, dev))
    return val

cdef int _getFunctionAttribute(CUfunction_attribute att, CUfunction hfunc) except *:
    cdef int val
    CudaSafeCall(cuFuncGetAttribute(&val, att, hfunc))
    return val

cdef inline CUaddress_mode _getTexRefAddressMode(CUtexref tex, int dim) except *:
    cdef CUaddress_mode mode
    CudaSafeCall(cuTexRefGetAddressMode(&mode, tex, dim))
    return mode

cdef inline void _setTexRefAddressMode(CUtexref tex, int dim, CUaddress_mode mode) except *:
    CudaSafeCall(cuTexRefSetAddressMode(tex, dim, mode))



DEF MAXNAMELENGTH = 100

#######################################
#
# API
#
#######################################

version = None
device_count = None
memory_info = None

cdef void set_global_attributes():
    global version, device_count, memory_info
    cdef int ver
    CudaSafeCall(cuDriverGetVersion(&ver))
    version = ver

    CudaSafeCall(cuDeviceGetCount(&ver))
    device_count = ver


cpdef init(unsigned int flags = 0):
    CudaSafeCall(cuInit(flags))
    set_global_attributes()


cpdef Device get_device(unsigned int id = 0):
    cdef CUdevice ldev
    CudaSafeCall(cuDeviceGet(&ldev, id))
    cdef Device instance = Device.__new__(Device)
    instance._dev = ldev
    return instance



def __ctx_synchronize():
    CudaSafeCall(cuCtxSynchronize())


cpdef tuple ctxMemInfo():
    cdef size_t free, total
    CudaSafeCall(cuMemGetInfo(&free, &total))
    return (free, total)


cpdef Context __ctxCurrent():
    cdef list cList = threading.current_thread().cudaCtx
    return <Context>cList[-1]


def __ctx_pop_current():
    cdef CUcontext ctx
    CudaSafeCall(cuCtxPopCurrent(&ctx))
    cdef list cList = threading.current_thread().cudaCtx
    print cList
    return <Context>cList.pop()



cpdef Event evt_create(bint blocking = False):
    cdef unsigned int flags = CU_EVENT_DEFAULT if not blocking else CU_EVENT_BLOCKING_SYNC
    cdef CUevent evt
    CudaSafeCall(cuEventCreate(&evt, flags))

    cdef Event instance = Event.__new__(Event)
    instance._evt   = evt
    instance.ctx    = __ctxCurrent()
    return instance


cpdef Stream stream_create(unsigned int flags = 0):
    cdef CUstream stream
    CudaSafeCall(cuStreamCreate(&stream, flags))

    cdef Stream instance = Stream.__new__(Stream)
    instance._stream    = stream
    instance.ctx        = __ctxCurrent()
    return instance

cpdef Module load_module(char *filename):
    cdef CUmodule mod
    CudaSafeCall(cuModuleLoad(&mod, filename))

    cdef Module instance = Module.__new__(Module)
    instance._mod       = mod
    instance.ctx        = __ctxCurrent()
    return instance

cpdef Module load_module_ex(char *data):
    cdef CUmodule mod
    CudaSafeCall(cuModuleLoadDataEx(&mod, data, 0, NULL, NULL))

    cdef Module instance = Module.__new__(Module)
    instance._mod       = mod
    instance.ctx        = __ctxCurrent()
    return instance

cpdef TexRef tex_ref_create():
    cdef CUtexref tex
    CudaSafeCall(cuTexRefCreate(&tex))
    cdef TexRef instance = TexRef.__new__(TexRef)

    instance._tex       = tex
    instance.ctx        = __ctxCurrent()
    return instance


cdef class Device(object):
    property capability:
        def __get__(self):
            cdef int cMajor, cMinor
            CudaSafeCall(cuDeviceComputeCapability(&cMajor, &cMinor, self._dev))
            return (cMajor, cMinor)

    property name:
        def __get__(self):
            cdef char name[MAXNAMELENGTH]
            CudaSafeCall(cuDeviceGetName(name, MAXNAMELENGTH, self._dev))
            return name

    def can_access_peer(self, Device dev):
        cdef int value
        CudaSafeCall(cuDeviceCanAccessPeer(&value, self.dev, dev.dev))

% for pname in device_properties:
    property ${pname}:
        def __get__(self):
<% vars = [] %>\
            % for i, p in enumerate(device_properties[pname]):
<% vars.append("c" + str(i)) %>\
            cdef int c${i} = _getDeviceAttribute(${p}, self._dev)
            % endfor
            return (${', '.join(vars)})

% endfor


    cdef Context _ctx_create(self,  CUresult (*allocator)(CUcontext *, unsigned int, CUdevice), CUctx_flags flags):
        cdef CUcontext lctx
        CudaSafeCall(allocator(&lctx, flags, self._dev))

        cdef Context instance = Context.__new__(Context)
        instance._ctx = lctx

        cThread = threading.current_thread()
        if not hasattr(cThread, "cudaCtx"):
            cThread.cudaCtx = []
        cdef list tList = cThread.cudaCtx
        tList.append(instance)
        return instance

    def ctx_create(self, CUctx_flags flags = CU_CTX_SCHED_AUTO):
        return self._ctx_create(cuCtxCreate, flags)

    def __repr__(self):
        return "%s name=%s capability=%s" % (self.__class__.__name__, self.name, str(self.capability))

"""
    def GLCtxCreate(self, CUctx_flags flags = CU_CTX_SCHED_AUTO):
        return self._ctxCreate(cudagl.cuGLCtxCreate, flags)
"""

cdef class CuDeviceBuffer(CuBuffer):
    ${cuda_dealloc("cuMemFree(self.buf)")}

    def __repr__(self):
        return "%s size=%d" % (self.__class__.__name__, self.nbytes)

cdef class CuHostBuffer(object):
    ${cuda_dealloc("cuMemFreeHost(self.data)")}
    def __repr__(self):
        return "%s size=%d" % (self.__class__.__name__, self.nbytes)

    property flag:
        def __get__(self):
            cdef unsigned int flags
            CudaSafeCall(cuMemHostGetFlags(&flags, self.data))
            return flags

    def get_device(self, unsigned int flags = 0):
        cdef CUdeviceptr buf
        CudaSafeCall(cuMemHostGetDevicePointer(&buf, self.data, flags))
        cdef CuBuffer instance = CuBuffer.__new__(CuBuffer)

        instance.base   = self
        instance.buf    = buf
        instance.nbytes = self.nbytes
        instance.ctx    = self.ctx

        return instance




cdef class CuTypedBuffer(CuDeviceBuffer):
    def __repr__(self):
        return "%s size=%d, shape=%s type=%s" % (self.__class__.__name__, self.nbytes, str(self.shape), str(self.dtype))

def __allocate_raw(size_t size):
    cdef CUdeviceptr data
    CudaSafeCall(cuMemAlloc(&data, size))

    cdef CuDeviceBuffer instance = CuDeviceBuffer.__new__(CuDeviceBuffer)

    instance.ctx        = __ctxCurrent()
    instance.nbytes     = size
    instance.buf        = data
    return instance

def __allocate(tuple shape, object dtype = 'float'):
    cdef np.dtype dt        = np.dtype(dtype)
    cdef size_t size        = reduce(operator.mul, shape) * dt.itemsize

    cdef CUdeviceptr data
    CudaSafeCall(cuMemAlloc(&data, size))

    cdef CuTypedBuffer instance = CuTypedBuffer.__new__(CuTypedBuffer)

    instance.ctx        = __ctxCurrent()
    instance.nbytes     = size
    instance.buf        = data
    instance.shape      = shape
    instance.dtype      = dt
    return instance

def __allocate_host(size_t size):
    cdef void * data
    CudaSafeCall(cuMemAllocHost(&data, size))

    cdef CuHostBuffer instance = CuHostBuffer.__new__(CuHostBuffer)

    instance.ctx    = __ctxCurrent()
    instance.nbytes = size
    instance.data   = data
    return instance

cdef class Context(object):
    ${cuda_dealloc("cuCtxDestroy(self._ctx)")}

    def push_current(self):
        CudaSafeCall(cuCtxPushCurrent(self._ctx))
        cdef list tList  = threading.current_thread().cudaCtx
        tList.append(self)

    def enable_peer(self, unsigned int flags = 0):
        CudaSafeCall(cuCtxEnablePeerAccess(self._ctx, flags))

    def disable_peer(self):
        CudaSafeCall(cuCtxDisablePeerAccess(self._ctx))

    allocate_raw    = staticmethod(__allocate_raw)
    allocate_host   = staticmethod(__allocate_host)
    allocate        = staticmethod(__allocate)

    current         = staticmethod(__ctxCurrent)
    pop_current     = staticmethod(__ctx_pop_current)
    synchronize     = staticmethod(__ctx_synchronize)






cdef class Stream(object):
    ${cuda_dealloc("cuStreamDestroy(self._stream)")}

    def query(self):
        return CUDA_SUCCESS == cuStreamQuery(self._stream)

    def synchronize(self):
        CudaSafeCall(cuStreamSynchronize(self._stream))

    def wait(self, Event evt, unsigned int flags):
        CudaSafeCall(cuStreamWaitEvent(self._stream, evt._evt, flags))





cdef class Event(object):
    ${cuda_dealloc("cuEventDestroy(self._evt)")}

    cpdef record(self, Stream stream):
        cdef CUstream cstream = stream._stream if stream is not None else <CUstream>0
        CudaSafeCall(cuEventRecord(self._evt, cstream))

    cpdef bint query(self):
        return CUDA_SUCCESS == cuEventQuery(self._evt)

    def synchronize(self):
        cuEventSynchronize(self._evt)

    def __sub__(Event self not None, Event evt2 not None):
        cdef float pMilliseconds
        CudaSafeCall(cuEventElapsedTime(&pMilliseconds, evt2._evt, self._evt))
        return pMilliseconds


cdef class Module(object):
    ${cuda_dealloc("cuModuleUnload(self._mod)")}

    def get_function(self, char *name):
        cdef CUfunction fun
        CudaSafeCall(cuModuleGetFunction(&fun, self._mod, name))

        cdef Function instance = Function.__new__(Function)
        instance.mod    = self
        instance._fun   = fun
        return instance

    def get_texref(self, char *name):
        cdef CUtexref tex
        CudaSafeCall(cuModuleGetTexRef(&tex, self._mod, name))

        cdef ModuleTexRef instance = ModuleTexRef.__new__(ModuleTexRef)
        instance.mod    = self
        instance._tex   = tex
        return instance


cdef class Function(object):
    property params:
        def __get__(self):
            return self._pstruct.format

        def __set__(self, char *s):
            self._pstruct = struct.Struct(s)
            self._pstruct_size = self._pstruct.size
            CudaSafeCall(cuParamSetSize(self._fun, self._pstruct_size))

    property maxThreadPerBlock:
        def __get__(self):
            return _getFunctionAttribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, self._fun)

    property shared_size:
        def __get__(self):
            return _getFunctionAttribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, self._fun)
        def __set__(self, unsigned int bytes):
            CudaSafeCall(cuFuncSetSharedSize(self._fun, bytes))

    property const_size:
        def __get__(self):
            return _getFunctionAttribute(CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, self._fun)

    property local_size:
        def __get__(self):
            return _getFunctionAttribute(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, self._fun)

    property num_regs:
        def __get__(self):
            return _getFunctionAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS, self._fun)

    def prepare_call(self, *args):
        s = self._pstruct.pack(*args)
        cdef char *packed = s
        CudaSafeCall(cuParamSetv(self._fun, 0, <void *>packed, self._pstruct_size))
        return s


    def __call__(self, *args, tuple block = (1, 1, 1), tuple grid = (1, 1)):
        self.prepareCall(*args)
        self.setBlockShape(block[0], block[1], block[2])
        self.launchGrid(grid[0], grid[1])

    cpdef launch_grid(self, int xgrid, int ygrid):
        CudaSafeCall(cuLaunchGrid(self._fun, xgrid, ygrid))

    cpdef setBlockShape(self, int xblock, int yblock, int zblock):
        CudaSafeCall(cuFuncSetBlockShape(self._fun, xblock, yblock, zblock))

    cdef void __launchGridAsync__(self, int xgrid, int ygrid, Stream s):
        cdef CUstream stream = <CUstream>0 if s is None else s._stream
        CudaSafeCall(cuLaunchGridAsync(self._fun, xgrid, ygrid, stream))


    def launchGridAsync(self, int xgrid, int ygrid, Stream s = None):
        self.__launchGridAsync__(xgrid, ygrid, s)



cdef class TexRefBase(object):
    property addressMode:
        def __get__(self):
            return _getTexRefAddressMode(self._tex, 0), _getTexRefAddressMode(self._tex, 1)
        def __set__(self, tuple value):
            for i, mode in enumerate(value):
                _setTexRefAddressMode(self._tex, i, mode)

    property filterMode:
        def __get__(self):
            cdef CUfilter_mode mode
            CudaSafeCall(cuTexRefGetFilterMode(&mode, self._tex))
            return mode
        def __set__(self, CUfilter_mode mode):
            CudaSafeCall(cuTexRefSetFilterMode(self._tex, mode))

    property normalized:
        def __get__(self):
            cdef unsigned int flags
            CudaSafeCall(cuTexRefGetFlags(&flags, self._tex))
            return flags == CU_TRSF_NORMALIZED_COORDINATES
        def __set__(self, bint value):
            cdef unsigned int flags = CU_TRSF_NORMALIZED_COORDINATES if value else CU_TRSF_READ_AS_INTEGER
            CudaSafeCall(cuTexRefSetFlags(self._tex, flags))


cdef class TexRef(TexRefBase):
    ${cuda_dealloc("cuTexRefDestroy(self._tex)")}



#
# vim: filetype=pyrex
#
#
