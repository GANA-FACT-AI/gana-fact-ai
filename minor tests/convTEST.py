import torch
import numpy as np
import math

from complexModules import *
from complexLayers import  ComplexLinear, ComplexConv2d



def rotate(xr,xi,theta):
    rotated_r = torch.cos(theta)*xr - torch.sin(theta)*xi
    rotated_i = torch.sin(theta)*xr + torch.cos(theta)*xi
    return rotated_r, rotated_i

for j in range(50):
	x  = torch.rand([20,3,28,28])
	theta = torch.from_numpy(np.array(np.random.uniform(0, 2*math.pi)))
	bn = InvariantBatchNorm()
	rl = InvariantReLU()
	cv = ComplexConv2d(3, 20, 5, 5, bias = False)

	xr = x 
	xi = torch.zeros(xr.shape, dtype = xr.dtype, device = xr.device)  

	# 1 test maxpool
	x1r , _ = cv(xr, xi)

	xr_temp, xi_temp = rotate(xr,xi, theta)
	xr_temp, xi_temp =  cv(xr_temp, xi_temp)

	x2r, _ = rotate(xr_temp,xi_temp, -theta)

	print(torch.mean(x1r - x2r) < 1e-7)