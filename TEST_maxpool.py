from complexModules import *
from torch.nn import MaxUnpool2d

real  = torch.randint(0, 11, (2, 1 , 12, 12)).float()
imag  = torch.randint(0, 11, (2, 1 , 12, 12)).float()
# print(real)
# print(imag)

bn_func = InvariantBatchNorm()
real_pool, imag_pool  = invmaxpool2d(real, imag, bn_func, 3, stride=3, padding=0,
                                dilation=1, ceil_mode=False)
# print(real_pool)
