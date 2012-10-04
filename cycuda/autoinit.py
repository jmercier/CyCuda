import core as cuda

cuda.init()
dev = cuda.get_device(id = 0)
ctx = dev.ctxCreate()

