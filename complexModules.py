import torch
from torch.nn import Module, Parameter, init, Sequential
from torch.nn import Conv2d, Linear, BatchNorm1d, BatchNorm2d
from torch.nn import ConvTranspose2d


from torch.nn.functional import  max_pool2d


class InvariantReLU(Module):
	def __init__(self, c = 1):
		super(InvariantReLU, self).__init__()
		self.c  = c
	def forward(self,input_r,input_i):
		norm = torch.sqrt(input_r.pow(2) + input_i.pow(2))
		maxpick = torch.max(norm, self.c * torch.ones(norm.shape))
		# print('maxpick', maxpick)
		return norm/maxpick  * input_r , norm/maxpick * input_i

class InvariantBatchNorm(Module):
	def __init__(self,):
		super(InvariantBatchNorm, self).__init__()
	def forward(self,input_r,input_i):
		b_avg = torch.sqrt(torch.mean(input_r.pow(2) + input_i.pow(2), axis = 0))+1e-8
		return input_r/b_avg , input_i/b_avg

def invmaxpool2d(input_r, input_i, bn_func, kernel_size, stride=None, padding=0,
                                dilation=1, ceil_mode=False):
	"""
	Rotation indifferent maxpool on 2d grid
	"""
	# 
	real_norm, imag_norm = bn_func(input_r, input_i)

	# real_norm, imag_norm = input_r, input_i

	real_o, real_i =  max_pool2d(real_norm, kernel_size, stride, padding, dilation,
                      ceil_mode, return_indices = True)
	imag_o, imag_i =  max_pool2d(imag_norm, kernel_size, stride, padding, dilation,
                      ceil_mode, return_indices = True)

	x_real  = torch.flatten(input_r, 2)
	real_o2 = torch.gather(x_real, 2, torch.flatten(real_i, 2)).view(real_o.size())

	x_imag = torch.flatten(input_i, 2)
	imag_o2 = torch.gather(x_imag, 2, torch.flatten(imag_i, 2)).view(imag_o.size())
	# print(real_norm)

	return real_o2, imag_o2
	# print(o)
	#

	# return real * b_avg[real == real_norm], imag * b_avg[real == real_norm]




    








        