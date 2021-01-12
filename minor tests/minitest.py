import torch
import numpy as np
import math

from complexModules import *
from complexLayers import  ComplexLinear, ComplexConv2d


def rotate(xr,xi,theta):
    rotated_r = torch.cos(theta)*xr - torch.sin(theta)*xi
    rotated_i = torch.sin(theta)*xr + torch.cos(theta)*xi
    return rotated_r, rotated_i


for j in range(35):
	x  = torch.rand([3,3,13,13])
	theta = torch.from_numpy(np.array(np.random.uniform(0, 2*math.pi)))
	bn = InvariantBatchNorm()
	rl = InvariantReLU()
	complex_cv = ComplexConv2d(3, 20, 5, 5, bias = False)
	# cv  = Conv2d(3, 20, 5, 5, bias = False)
	# cv2 = Conv2d(3, 20, 5, 5, bias = False)
	fc1 = ComplexLinear(80, 20, bias  = False)
	# fc2 

	xr = x 
	xi = torch.zeros(xr.shape, dtype = xr.dtype, device = xr.device)  

	xr_temp, xi_temp = bn(xr,xi)
	xr_temp, xi_temp = rl(xr_temp,xi_temp)
	# xr_temp, xi_temp = cv(xr_temp), cv(xi_temp)
	xr_temp, xi_temp = complex_cv(xr_temp,xi_temp)
	x1r = xr_temp.view(-1, xr_temp.shape[1]*xr_temp.shape[2]*xr_temp.shape[3])
	x1i = xi_temp.view(-1, xi_temp.shape[1]*xi_temp.shape[2]*xi_temp.shape[3])
	x1r, x1i = fc1(x1r, x1i)
	# x1r , x1i = invmaxpool2d(xr_temp, xi_temp, bn, 3, stride=None, padding=0,
	#                                 dilation=1, ceil_mode=False)

	# rotation 
	xr_temp, xi_temp = rotate(xr,xi, theta)
	xr_temp, xi_temp = bn(xr_temp, xi_temp)
	xr_temp, xi_temp = rl(xr_temp, xi_temp)
	xr_temp, xi_temp = complex_cv(xr_temp,xi_temp)

	x2r = xr_temp.view(-1, xr_temp.shape[1]*xr_temp.shape[2]*xr_temp.shape[3])
	x2i = xi_temp.view(-1, xi_temp.shape[1]*xi_temp.shape[2]*xi_temp.shape[3])
	x2r, x2i = fc1(x2r, x2i)
	# x2r, x2i = invmaxpool2d(xr_temp, xi_temp, bn, 3, stride=None, padding=0,
	#                                 dilation=1, ceil_mode=False)

	x2r,_ = rotate(x2r, x2i, - theta)
	print(torch.mean(x1r-x2r)<1e-7, torch.mean(x1r), torch.mean(x2r)) 


