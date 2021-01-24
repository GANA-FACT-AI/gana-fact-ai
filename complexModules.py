import torch
from torch.nn import Module
import torch.nn as nn
from torch.nn import Module, Parameter, init, Sequential
from torch.nn import Conv2d, Linear, BatchNorm1d, BatchNorm2d
from torch.nn import ConvTranspose2d

from torch.nn.functional import  max_pool2d


eps = 1e-5


class InvariantReLU(Module):
	"""
	Rotation invariant ReLU.

	parameter c:
		c is used in denominator as in the original paper, default: c = 1  
	"""
	def __init__(self):
		super(InvariantReLU, self).__init__()
	def forward(self,input_r,input_i):
		norm = torch.sqrt(input_r.pow(2) + input_i.pow(2))
		c = torch.mean(norm, dim=(0, 2, 3)).view(1, 256, 1, 1)
		maxpick = torch.max(norm, c)
		# print('maxpick', maxpick)
		return norm/maxpick  * input_r , norm/maxpick * input_i

class InvariantBatchNorm(Module):
	def __init__(self,):
		super(InvariantBatchNorm, self).__init__()

	def forward(self, input_r, input_i):
		b_avg = torch.sqrt(torch.mean(input_r.pow(2) + input_i.pow(2), axis=0) + eps)
		return input_r/b_avg, input_i/b_avg


def invmaxpool2d(input_r, input_i, kernel_size, stride=None, padding=0,
                                dilation=1, ceil_mode=False):
	"""
	Rotation indifferent maxpool on 2d grid
	"""
	x = torch.sqrt(input_r.pow(2) + input_i.pow(2) + eps)

	o, i = max_pool2d(x, kernel_size, stride, padding, dilation,
					  ceil_mode, return_indices=True)

	size_num = 2
	x_real = torch.flatten(input_r, size_num)
	real_o2 = torch.gather(x_real, size_num, torch.flatten(i, size_num)).view(o.size())

	x_imag = torch.flatten(input_i, size_num)
	imag_o2 = torch.gather(x_imag, size_num, torch.flatten(i, size_num)).view(o.size())
	# 
	return real_o2, imag_o2

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)




    








        
