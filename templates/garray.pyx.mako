cimport core as cuda
cimport numpy as np




"""

#################################################
#
#  Helper code (C only)
#
#################################################
cdef void setHostSrc(CUDA_MEMCPY2D *st, np.ndarray ary):
        st.srcPitch = ary.strides[0]
        st.srcXInBytes = st.srcY = 0
        st.srcHost = ary.data
        st.srcMemoryType = CU_MEMORYTYPE_HOST
        st.WidthInBytes = ary.shape[1] * ary.dtype.itemsize
        st.Height = ary.shape[0]

cdef void setHostDst(CUDA_MEMCPY2D *st, np.ndarray ary):
        st.dstPitch = ary.strides[0]
        st.dstXInBytes = st.dstY = 0
        st.dstHost = ary.data
        st.dstMemoryType = CU_MEMORYTYPE_HOST

cdef void setDeviceDst(CUDA_MEMCPY2D *st, gndarray ary):
        st.dstPitch = ary._strides[0]
        st.dstXInBytes = ary._offset[1] * ary._dtype.itemsize
        st.dstY = ary._offset[0] 
        st.dstDevice = ary._buf._deviceBuf
        st.dstMemoryType = CU_MEMORYTYPE_DEVICE

cdef void setDeviceSrc(CUDA_MEMCPY2D *st, gndarray ary):
        st.srcPitch = ary._strides[0]
        st.srcXInBytes = ary._offset[1] * ary._dtype.itemsize
        st.srcY = ary._offset[0] 
        st.srcDevice = ary._buf._deviceBuf
        st.srcMemoryType = CU_MEMORYTYPE_DEVICE
        st.WidthInBytes = ary._shape[1] * ary._dtype.itemsize
        st.Height = ary._shape[0]

cdef inline void buildArray(gndarray array,
                            tuple shape,
                            tuple strides,
                            np.dtype dtype,
                            tuple offset,
                            unsigned int ndim,
                            cuda.CudaBuffer buffer,
                            bint contiguous = True):
        array._buf = buffer
        array._shape = shape
        array._strides = strides
        array._offset = offset
        array._dtype = dtype
        array._contiguous = contiguous
        array._ndim = ndim
        array._arrayDescription.size = buffer._size
        array._arrayDescription.data = buffer._deviceBuf

cdef dict dtype_to_texref_format = {
    np.dtype("uint8") : CU_AD_FORMAT_UNSIGNED_INT8,
    np.dtype("uint16") : CU_AD_FORMAT_UNSIGNED_INT16,
    np.dtype("uint32") : CU_AD_FORMAT_UNSIGNED_INT32,
    np.dtype("int8") : CU_AD_FORMAT_SIGNED_INT8,
    np.dtype("int16") : CU_AD_FORMAT_SIGNED_INT16,
    np.dtype("int32") : CU_AD_FORMAT_SIGNED_INT32,
    #np.dtype("float16") : CU_AD_FORMAT_HALF,
    np.dtype("float32") : CU_AD_FORMAT_FLOAT
}

#################################################
#
# Garray allocation code
#
#################################################

cpdef gndarray garray(tuple shape, np.dtype dtype = np.dtype("float32")):
    cdef cuda.CudaBuffer buffer = cuda.allocate1D(shape, dtype.itemsize)
    cdef gndarray instance = gndarray.__new__(gndarray)
    buildArray(instance, shape, (0, ), dtype, (0,), len(shape), buffer, False)
    return instance

cpdef g2darray gimage(tuple shape, np.dtype dtype = np.dtype("float32")):
    cdef cuda.CudaBuffer buffer = cuda.allocate2D(shape, dtype.itemsize)
    cdef g2darray instance = g2darray.__new__(g2darray)
    buildArray(instance, shape, (buffer._pitch, dtype.itemsize), dtype, (0, 0), 2, buffer, buffer._pitch == shape[0] * dtype.itemsize)
    instance._imageDescription.shape[0] = <unsigned int>shape[0]
    instance._imageDescription.shape[1] = <unsigned int>shape[1]
    instance._imageDescription.strides[0] = buffer._pitch
    instance._imageDescription.strides[1] = dtype.itemsize
    instance._imageDescription.data = <void *>instance.getDevicePtr()
    return instance





cpdef g2darray lena(np.dtype dtype = np.dtype("float32")):
    import scipy as sc
    cdef np.ndarray lcpu = sc.lena().astype(dtype)
    cdef g2darray lgpu = gimage((lcpu.shape[0], lcpu.shape[1]), dtype = lcpu.dtype)
    lgpu.set(lcpu)
    return lgpu

#################################################
#
#  gndarray Code
#
#################################################

cdef class gndarray(object):
    cdef bint checkDeviceCompatibility(self, gndarray ary):
        if self._ndim != ary._ndim:
            return False
        for i in xrange(self._ndim):
            if self._shape[i] != self._shape[i]:
                return False
        if self._dtype != ary._dtype:
            return False
        return True

    cdef bint checkHostCompatibility(self, np.ndarray ary):
        if self._ndim != ary.ndim:
            return False
        for i in xrange(self._ndim):
            if self._shape[i] != ary.shape[i]:
                return False
        if self._dtype != ary.dtype:
            return False
        return True

    cdef inline int getSize(self):
        cdef int size = 1
        cdef int i
        for i in xrange(self._ndim):
            size *= self._shape[i]
        return size

    cdef inline CUdeviceptr getDevicePtr(self):
        return <char *>self._buf._deviceBuf + <int>self._offset[0] * self._dtype.itemsize

    property shape:
        def __get__(self):
            return self._shape

    property dtype:
        def __get__(self):
            return self._dtype

    property strides:
        def __get__(self):
            return self._strides

    property size:
        def __get__(self):
            return self.getSize()

    property contiguous:
        def __get__(self):
            return self._contiguous

    property ndim:
        def __get__(self):
            return self._ndim

    property offset:
        def __get__(self):
            return self._offset

    property gpudata:
        def __get__(self):
            return <np.npy_intp>self.getDevicePtr()

    cpdef gndarray ravel(self):
        if not self._contiguous:
            raise AttributeError("garray not contiguous")
        cdef gndarray instance = gndarray.__new__(gndarray)
        buildArray(instance, (self.size,), (self._dtype.itemsize,), self._dtype, 1, (self._offset[0],), self._buf)
        return instance

    cpdef gndarray set(self, np.ndarray ary):
        if not self.checkHostCompatibility(ary):
            raise ValueError("Incompatible array")
        cdef int size = self.getSize() * self._dtype.itemsize
        cuMemcpyHtoD(self.getDevicePtr(), ary.data, size)
        return self

    cpdef gndarray set_async(self, np.ndarray ary, cuda.Stream stream = None):
        cdef CUstream st = <CUstream>0 if stream is None else stream._stream
        if not self.checkHostCompatibility(ary):
            raise ValueError("Incompatible array")
        cdef int size = self.getSize() * self._dtype.itemsize
        cuMemcpyHtoDAsync(self.getDevicePtr(), ary.data, size, st)
        return self

    cpdef np.ndarray get(self, np.ndarray ary):
        if not self.checkHostCompatibility(ary):
            raise ValueError("Incompatible array")
        cdef int size = self.getSize() * self._dtype.itemsize
        cuMemcpyDtoH(ary.data, self.getDevicePtr(), size)
        return ary

    cpdef np.ndarray get_async(self, np.ndarray ary, cuda.Stream stream = None):
        cdef CUstream st = <CUstream>0 if stream is None else stream._stream
        if not self.checkHostCompatibility(ary):
            raise ValueError("Incompatible array")
        cdef int size = self.getSize() * self._dtype.itemsize
        cuMemcpyDtoHAsync(ary.data, self.getDevicePtr(), size, st)
        return ary

    cpdef gndarray copy(self, gndarray ary):
        if not self.checkDeviceCompatibility(ary):
            raise ValueError("Incompatible array")
        cdef int size = self.getSize() * self._dtype.itemsize
        cuMemcpyDtoD(ary.getDevicePtr(), self.getDevicePtr(), size)
        return ary

    cpdef gndarray copy_async(self, gndarray ary, cuda.Stream stream = None):
        cdef CUstream st = <CUstream>0 if stream is None else stream._stream
        if not self.checkDeviceCompatibility(ary):
            raise ValueError("Incompatible array")
        cdef int size = self.getSize() * self._dtype.itemsize
        cuMemcpyDtoDAsync(ary.getDevicePtr(), self.getDevicePtr(), size, st)
        return ary

    cdef inline bint bound_checking(self, int *start, int *stop, unsigned int axis):
        start[0] = start[0] if start[0] >= 0 else self._shape[0] + start[0]
        stop[0] = stop[0] if stop[0] > 0 else self._shape[0] + stop[0]
        if stop[0] > self._shape[axis]:
            return False
        if stop[0] <= start[0]:
            return False
        return True

    def __getitem__(self, slice sli):
        cdef int start = sli.start if sli.start is not None else 0
        cdef int stop = sli.stop if sli.stop is not None else self._shape[0]
        if not self.bound_checking(&start, &stop, 0):
            raise IndexError("Out of bound")
        cdef gndarray instance = gndarray.__new__(gndarray)
        cdef tuple newshape = (stop - start,) + self._shape[1:]
        buildArray(instance, newshape, self._strides, self.dtype, (self._offset[0] + start,),self._ndim, self._buf)
        return instance



cdef class g2darray(gndarray):
    cdef gimage_st _imageDescription
    cpdef cuda.TexRef bind(self, cuda.TexRef ref):
        cdef CUDA_ARRAY_DESCRIPTOR arrdesc
        arrdesc.Height = self._imageDescription.shape[0]
        arrdesc.Width = self._imageDescription.shape[1]
        arrdesc.NumChannels = 1
        arrdesc.Format = dtype_to_texref_format[self._dtype]
        cdef CUdeviceptr address = <int>self._offset[1] * self._dtype.itemsize + self._buf._deviceBuf
        cuTexRefSetAddress2D(ref._tex, &arrdesc, address, self._desc.strides[0])
        return ref

    cpdef cuda.TexRef makeTexRef(self):
        return self.bind(cuda.texRefCreate())

    cpdef gndarray copy(self, gndarray ary):
        if not self.checkDeviceCompatibility(ary):
            raise ValueError("Incompatible array")
        cdef CUDA_MEMCPY2D st
        setDeviceSrc(&st, self)
        setDeviceDst(&st, ary)
        cuMemcpy2D(&st)
        return ary


    cpdef gndarray copy_async(self, gndarray ary, cuda.Stream stream = None):
        cdef CUstream s = <CUstream>0 if stream is None else stream._stream
        if not self.checkDeviceCompatibility(ary):
            raise ValueError("Incompatible array")
        cdef CUDA_MEMCPY2D st
        setDeviceSrc(&st, self)
        setDeviceDst(&st, ary)
        cuMemcpy2DAsync(&st, s)
        return ary


    cpdef np.ndarray get(self, np.ndarray ary):
        if not self.checkHostCompatibility(ary):
            raise ValueError("Incompatible array")
        cdef CUDA_MEMCPY2D st
        setDeviceSrc(&st, self)
        setHostDst(&st, ary)
        cuMemcpy2D(&st)
        return ary

    cpdef np.ndarray get_async(self, np.ndarray ary, cuda.Stream stream = None):
        cdef CUstream s = <CUstream>0 if stream is None else stream._stream
        if not self.checkHostCompatibility(ary):
            raise ValueError("Incompatible array")
        cdef CUDA_MEMCPY2D st
        setDeviceSrc(&st, self)
        setHostDst(&st, ary)
        cuMemcpy2DAsync(&st, s)
        return ary

    cpdef gndarray set(self, np.ndarray ary):
        if not self.checkHostCompatibility(ary):
            raise ValueError("Incompatible array")
        cdef CUDA_MEMCPY2D st
        setHostSrc(&st, ary)
        setDeviceDst(&st, self)
        cuMemcpy2D(&st)
        return self

    cpdef gndarray set_async(self, np.ndarray ary, cuda.Stream stream = None):
        cdef CUstream s = <CUstream>0 if stream is None else stream._stream
        if not self.checkHostCompatibility(ary):
            raise ValueError("Incompatible array")
        cdef CUDA_MEMCPY2D st
        setHostSrc(&st, ary)
        setDeviceDst(&st, self)
        cuMemcpy2DAsync(&st, s)
        return self

    cdef object getRawImageDescription(self):
        cdef gimage_st desc = self._imageDescription
        cdef char *s = <char *>&desc
        return s[:sizeof(desc)]

    cdef object getRawArrayDescription(self):
        cdef garray_st desc = self._arrayDescription
        cdef char *s = <char *>&desc
        return s[:sizeof(desc)]

    property rawImageDescription:
        def __get__(self):
            return self.getRawImageDescription()

    property rawArrayDescription:
        def __get__(self):
            return self.getRawArrayDescription()


"""

cimport numpy as np
import numpy as np

cpdef object page_locked(tuple shape, object dtype = 'float32'):
    return None


#
# vim: filetype=pyrex
#
#





