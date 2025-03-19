import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import helpers as hp
from core import utils 
import numpy as np
class Identity(nn.Module):
	def __init__(self,dim):
		super(Identity,self).__init__()
		self.param_x = torch.nn.parameter.Parameter(.00000001*torch.ones([dim]))
	def forward(self,input):
		return inputs

class LowerLoss(nn.Module):
	"""
		Computes a quadratic function of two variables of the form:	
			1/2*y^tAy + 1/2*x^tQx + y^tBx +Cx +D

	"""

	def __init__(self,x,y,device=None,cond=.1):
		super(LowerLoss,self).__init__()
		self.device = device
		self.cond = cond 
		
		self.upper_var = utils.AttrList(self,list(x),'upper_var_')
		self.lower_var = utils.AttrList(self,list(y),'lower_var_')

		self.x = self.upper_var[0]
		self.y = self.lower_var[0]
		
		N = self.x.shape[0]
		d = self.y.shape[0]
		self.d = d
		torch.manual_seed(0)

		self.AA = torch.randn(d,d, dtype=self.y.dtype, device=self.y.device)
		A = self.AA
		self.A = A@A.t()
		U,S,V = torch.svd(self.A)
		S[-1] = 0.

		lmbda = S[0]/(self.cond-1)
		
		#lmbda = max(0,lmbda)
		S = S+lmbda
		S = S/S[0]
		S[-100:-1] = 0.
		self.A = torch.einsum('ij,j,kj->ik',U,S,V)
		self.U = torch.randn([d,N],dtype=self.y.dtype, device=self.y.device)
		self.B = self.A.t()@self.U
		self.C = torch.zeros([N],dtype=self.y.dtype, device=self.y.device)
		self.D = torch.zeros([1],dtype=self.y.dtype, device=self.y.device)
		self.U = torch.linalg.lstsq(self.A,self.B).solution
		self.Q = self.B.t()@self.U #+ 1.
		

	def get_param_system(self):
		return self.A, self.B, self.C, self.D
	def forward(self,inputs):
		
		loss = 0.*torch.mean(inputs)  \
				+.5*self.y.t()@self.A@self.y \
				+ self.y.t()@self.B@self.x \
				+ self.C@self.x +self.D\
				+ 0.5*self.x.t()@self.Q@self.x 
		loss = loss.mean()
		return loss 
	def func(self,data,x,y,with_acc):
		
		loss = .5*self.y.t()@self.A@self.y \
				+ self.y.t()@self.B@self.x \
				+ self.C@self.x +self.D\
				+ 0.5*self.x.t()@self.Q@self.x
		return loss,torch.sum(torch.zeros([0])) 




class UpperLoss(nn.Module):
	"""
		Computes a quadratic function of two variables of the form:	
			1/2*x^tAx + y^tC + D

	"""

	def __init__(self,x,y,device=None,cond=.1, inner_loss_system=None):
		super(UpperLoss,self).__init__()
		self.device = device
		self.cond = cond 
		
		self.upper_var = utils.AttrList(self,list(x),'upper_var_')
		self.lower_var = utils.AttrList(self,list(y),'lower_var_')

		self.x = self.upper_var[0]
		self.y = self.lower_var[0]


		N = self.x.shape[0]
		d = self.y.shape[0]
		self.d = d
		torch.manual_seed(1)

		self.AA = torch.randn(N,N, dtype=self.y.dtype, device=self.y.device)
		A = self.AA
		self.A = A@A.t()
		U,S,V = torch.svd(self.A)
		S[-1] = 0.

		lmbda = S[0]/(self.cond-1)
		
		#lmbda = max(0,lmbda)
		S = S+lmbda
		S = S/S[0]
		self.A = torch.einsum('ij,j,kj->ik',U,S,V)

		self.C = torch.randn([d],dtype=self.y.dtype, device=self.y.device)
		self.A_y,self.B_y,_,_ = inner_loss_system
		v = self.C@torch.linalg.lstsq(self.A_y,self.B_y).solution
		self.x_star = torch.inverse(self.A)@v
		self.D = 0.5*v.t()@self.x_star


	def get_param_system(self):
		return self.A, self.B, self.C, self.D
	
	def forward(self,inputs):
		
		loss = 0.*torch.mean(inputs)  \
				+.5*self.x.t()@self.A@self.x \
				+ self.C@self.y +self.D

		loss = loss.mean()
		return loss 
	def func(self,data,x,y,with_acc):
		vv = y-self.x_star
		loss = 0.5*vv.T@self.A@vv
		return loss,torch.sum(torch.zeros([0])) 


