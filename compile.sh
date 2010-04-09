FILES="cuda garray"

for file in ${FILES}; 
do
    sed -e 's/\(.*\)\(CUDA_SAFE_CALL \)\(.*$\)/\1cdef CUresult res = \3\n\1if res != CUDA_SUCCESS: raise translateError(res)/' \
    -e 's/\(.*\)\(CUDA_SAFE_CALL_NO_INIT \)\(.*$\)/\1res = \3\n\1if res != CUDA_SUCCESS: cuda.translateError(res)/' \
    -e 's/\(.*\)\(CUDA_SAFE_CALL_EXT \)\(.*$\)/\1res = \3\n\1if res != CUDA_SUCCESS: raise cuda.translateError(res)/' \
    -e 's/\(.*\)\(CUDA_DEALLOC \)\(.*$\)/\1def __dealloc__(self):\n\1    cdef CUresult res = \3\n\1    if res != CUDA_SUCCESS: print("Deallocation failure < %s >" % self.__class__.__name__)/' \
    ${file}.pyx.in > ${file}.pyx
    cython ${file}.pyx
    wc -l ${file}.c
done

gcc cuda.c -shared -fPIC -o cuda.so -I/usr/local/cuda/include -I/usr/include/python2.6 -lpython2.6  -O3 -L/usr/local/cuda/lib64 -lcuda
gcc garray.c -shared -fPIC -o garray.so -I/usr/local/cuda/include -I/usr/include/python2.6 -lpython2.6 -O3 -L/usr/local/cuda/lib64 -lcuda
