from complexModules import *
from torch.nn import MaxUnpool2d
import numpy as np
import torch

# real  = np.array([1,2,3,4,5],[6,7,8,9,10],[-1,2,3,-4,5],[-5,5,5,5,2],[-1,-2, 3, 4 ,5]).randint
# imag  = 

real  = torch.randint(0, 11, (10, 1 , 5, 5)).float()
imag  = torch.randint(0, 11, (10, 1 , 5, 5)).float()
# print(real)


# print(imag)

bn_func = InvariantBatchNorm()
real_pool, imag_pool  = invmaxpool2d(real, imag, bn_func, 2, stride=2, padding=0,
                                dilation=1, ceil_mode=False)

print(np.asmatrix(real[7,0,:,:]))
print('\n')
print(np.asmatrix(imag[7,0,:,:]))
print('\n')
norm = torch.sqrt(real.pow(2)+ imag.pow(2))
print(np.asmatrix(norm[7,0,:,:]))
print('\n')

print(np.asmatrix(real_pool[7,0,:,:]))
# print(real_pool)
