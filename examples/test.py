import numpy as np
import cuda
import autoinit
import garray

im = garray.gimage((512, 512))
him = np.zeros((512, 512), dtype = 'float32')
im.set(him)

m = cuda.loadModuleEx(open("kernel/testkern.ptx").read())
f = m.getFunction("testfun")
f.params = "24si"
f.prepareCall(im.rawImageDescription, 10)
f.setBlockShape(512,1,1)
f.launchGrid(1,512)
f(im.rawImageDescription, 49, block = (512, 1, 1), grid = (1, 512))

im.get(him)
print him
