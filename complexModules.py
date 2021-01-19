import torch
from torch.nn import Module, Parameter, init, Sequential
from torch.nn import Conv2d, Linear, BatchNorm1d, BatchNorm2d
from torch.nn import ConvTranspose2d



from torch.nn.functional import  max_pool2d


class InvariantReLU(Module):
	"""
	Rotation invariant ReLU.

	parameter c:
		c is used in denominator as in the original paper, default: c = 1  
	"""
	def __init__(self):
		super(InvariantReLU, self).__init__()
	def forward(self,input_r,input_i, c = 1):
		norm = torch.sqrt(input_r.pow(2) + input_i.pow(2))
		print(norm.shape)
		maxpick = torch.max(norm, c * torch.ones(norm.shape).to(norm.device))
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
	x  = torch.sqrt(input_r.pow(2) + input_i.pow(2))

	o, i =  max_pool2d(x, kernel_size, stride, padding, dilation,
	                  ceil_mode, return_indices = True)

	size_num = 2
	x_real  = torch.flatten(input_r, size_num)
	real_o2 = torch.gather(x_real, size_num, torch.flatten(i, size_num)).view(o.size())

	x_imag = torch.flatten(input_i, size_num)
	imag_o2 = torch.gather(x_imag, size_num, torch.flatten(i, size_num)).view(o.size())
	# 
	return real_o2, imag_o2





    








        