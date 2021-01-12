import torch
import numpy as np
import math

from complexModules import *
from complexLayers import  ComplexLinear



def rotate(xr,xi,theta):
    rotated_r = torch.cos(theta)*xr - torch.sin(theta)*xi
    rotated_i = torch.sin(theta)*xr + torch.cos(theta)*xi
    return rotated_r, rotated_i

for j in range(50):

	x  = torch.rand([20,3,28,28])
	theta = torch.from_numpy(np.array(np.random.uniform(0, 2*math.pi)))
	bn = InvariantBatchNorm()

	xr = x 
	xi = torch.zeros(xr.shape, dtype = xr.dtype, device = xr.device)  

	# 1 test maxpool
	x1r , _ = invmaxpool2d(xr, xi, bn, 3, stride=None, padding=0,
	                                dilation=1, ceil_mode=False)

	xr_temp, xi_temp = rotate(xr,xi, theta)
	xr_temp, xi_temp = invmaxpool2d(xr_temp, xi_temp, bn, 3, stride=None, padding=0,
	                                dilation=1, ceil_mode=False)
	x2r, _ = rotate(xr_temp,xi_temp, -theta)

	print(torch.mean(x1r - x2r)<1e-7, torch.mean(x1r), torch.mean(x2r))