import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import helpers as hp
from core import utils 
import torchvision

class Identity(nn.Module):
	def __init__(self,dim):
		super(Identity,self).__init__()
		self.param_x = torch.nn.parameter.Parameter(torch.zeros([dim]))
	def forward(self,input):
		return inputs



class Linear(nn.Module):
	def __init__(self,n_features,n_classes,with_bias=False):
		super(Linear,self).__init__()
		#data = torch.distributions.normal.Normal(loc=0., scale=1.).sample([n_features,n_classes])
		self.weight = torch.nn.parameter.Parameter(torch.zeros([n_features,n_classes]))
		if with_bias:
			self.bias = torch.nn.parameter.Parameter(torch.zeros([n_classes]))
		else:
			self. bias = 0.
	def forward(self,inputs):
		inputs = inputs.view(inputs.shape[0], -1)
		return inputs @ self.weight + self.bias


class Logistic(nn.Module):
	def __init__(self,upper_model,lower_model, reg= 1., device=None):
		super(Logistic,self).__init__()
		self.device = device
		self.upper_model = upper_model
		self.lower_model = lower_model


		upper_var = list(self.upper_model.parameters())
		lower_var = list(self.lower_model.parameters())
		self.upper_var = utils.AttrList(self,upper_var,'upper_var_')
		self.lower_var = utils.AttrList(self,lower_var,'lower_var_')


		self.reg = reg
	
	def forward(self,data,with_acc=False):
		x,y = data
			
			#targets = torch.nn.functional.one_hot(y, num_classes=self.upper_model.y.shape[0]).float()
		#x,y = data
		y = y.long()
		out_x = self.lower_model(x)
		out = F.cross_entropy(out_x, y)
		#out = F.cross_entropy(out_x, targets)

		if self.reg>0.:
			out =  out +  self.reg*self.reg_term()
		if with_acc:
			pred = out_x.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			acc = pred.eq(y.view_as(pred)).sum() / len(y)
			return out,acc
		else:
			return out

	def reg_term(self):
		ones_dxc = torch.ones(self.lower_var[0].size()).to(self.lower_var[0].device)
		out =  0.5*((self.lower_var[0]**2)*torch.exp(self.upper_var[0].unsqueeze(1)*ones_dxc)).mean()
		if len(self.upper_var)>1:
			out = out+ (self.lower_var[0].abs() * torch.exp(self.upper_var[1].unsqueeze(1) * ones_dxc)).mean()
		return out




