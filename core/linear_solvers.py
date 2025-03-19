import torch 
import numpy as np

from utils.helpers import get_gpu_usage


class LinearSolverAlg(object):
	def __call__(self,res_op,init):
		raise NotImplementedError
class GD(LinearSolverAlg):
	## performs gd/sgd on quadratic loss 0.5 xAx+bx 
	def __init__(self,lr=0.1,n_iter=1):
		super(GD,self).__init__()
		self.n_iter= n_iter
		self.lr= lr
	def __call__(self,linear_op,b_vector,init,apply_cross_derivatives=True):
		out_lower = init
		for i in range(self.n_iter):
			out_upper, update = linear_op(out_lower)
			out_lower = tuple([ x - self.lr*(ax+b) if ax is not None else x - self.lr*b for x,ax,b in zip(out_lower,update,b_vector)])
		
		if apply_cross_derivatives: 
			out_upper,_ = linear_op(out_lower,retain_graph=False, which='upper')
		return out_upper,out_lower

class Normal_GD(LinearSolverAlg):
	## performs gd/sgd on normal loss 0.5 || Ax+b||^2
	def __init__(self,lr=0.1,n_iter=1):
		super(Normal_GD,self).__init__()
		self.n_iter= n_iter
		self.lr= lr
	def __call__(self,linear_op,b_vector,init,apply_cross_derivatives=True):
		out_lower = init
		if linear_op.stochastic:
			retain_graph = False
		else:
			retain_graph = True
		for i in range(self.n_iter):
			out_upper, update = linear_op(out_lower,retain_graph=retain_graph)
			update = tuple([ax+b if ax is not None else b for ax,b in zip(update,b_vector)])
			if i == self.n_iter-1 and not apply_cross_derivatives:
				retain_graph = False
			out_upper, update = linear_op(update,retain_graph=retain_graph)
			out_lower = tuple([ x - self.lr*ax if ax is not None else x  for x,ax in zip(out_lower,update)])
		if apply_cross_derivatives:
			out_upper,_ = linear_op(out_lower,retain_graph=False, which='upper')

		return out_upper,out_lower


def dot(a,b):
	return torch.sum(torch.cat([torch.einsum('i,i->',u,v) for u,v in zip(a,b)],axis=0))

def norm(a):
	return torch.norm(torch.cat([torch.norm(u) for u in a ],axis=0))


