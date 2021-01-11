from complexModules import *
import torch

# relu = InvariantReLU()
# real  = 0.5 * torch.rand((1,3)) * torch.ones(1,3)
# imag  = 0.3 *  torch.ones(1,3)

# print('real', real)
# print('imag', imag)
# print(relu(real, imag))


bn = InvariantBatchNorm()
real  = 1 * torch.randint(0,99, (1,3)) * torch.ones(1,3)
imag  = 2 * torch.randint(0,99, (1,3))

print('real', real)
print('imag', imag)
print(bn(real, imag))

