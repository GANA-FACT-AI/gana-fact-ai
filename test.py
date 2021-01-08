import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexFunctions import complex_relu, complex_max_pool2d

real = torch.tensor([1, 2], dtype=torch.float32)
imag = torch.tensor([3, 4], dtype=torch.float32)
z = torch.view_as_complex(torch.stack((real,imag), 1))
print(torch.stack((real,imag), 1))
print(z)
theta = 0.5 * math.pi
i = torch.view_as_complex(torch.tensor([0, 1], dtype=torch.float32))
print(i)
rotated = torch.exp(theta*i) * z
print(rotated)
wow = torch.matmul(z, z.T)
print(wow)