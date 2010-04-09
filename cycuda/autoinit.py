import cuda

cuda.init()
dev = cuda.getDevice(id = 0)
ctx = dev.ctxCreate()

