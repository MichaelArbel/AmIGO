import  torch 
import torch.nn as nn
from torch import autograd
from core.utils import grad_with_none, config_to_instance
import core.utils as utils
from core.utils import RingGenerator

#import jax ## need to import jax before torchopt otherwise get an error
import torchopt
from itertools import cycle
import copy
from functools import partial
import torch.optim as optim
import os

from utils.helpers import get_gpu_usage

def make_selection(func,
					init_lower_var,
					loader,
					options,
					device,
					dtype):

	generator = RingGenerator(loader, device, dtype)
	
	if options.implicit_diff:
		linear_solver = config_to_instance(**options.linear_solver)
		linear_op = config_to_instance(**options.linear_op,func=func,generator=generator)
		if options.linear_op.stochastic:
			options.linear_op.compute_new_grad=True
		if isinstance(linear_op,FiniteDiff) or options.linear_op.compute_new_grad:
			track_grad_for_backward=False
		else:
			track_grad_for_backward=True
	else:
		linear_solver = None
		linear_op = None
		track_grad_for_backward = False


	optimizer = DiffOpt(func,generator,
							init_lower_var,options.optimizer,
							options.opt_iter,options.unrolled_iter,
							track_grad_for_backward=track_grad_for_backward)
	use_scheduler = options.scheduler.pop("use_scheduler", None)
	if use_scheduler:
		dummy_opt = optim.SGD(init_lower_var, lr = optimizer.lr)
		scheduler = config_to_instance(**options.scheduler, optimizer = dummy_opt)
	else:
		scheduler = None
	dual_var_warm_start = options.dual_var_warm_start
	selection = Selection(func,
						  init_lower_var,
						  generator,
						  linear_solver,
						  linear_op,
						  optimizer,
						  scheduler,
						  implicit_diff = options.implicit_diff,
						  dual_var_warm_start=dual_var_warm_start,
						  track_grad_for_backward = track_grad_for_backward
						  )
	return selection

class Selection(nn.Module):
	def __init__(self,
				func,
				init_lower_var,
				generator,
				linear_solver,
				linear_op,
				optimizer,
				scheduler,
				implicit_diff = True,
				dual_var_warm_start=True,
				track_grad_for_backward=False):
		super(Selection,self).__init__()
		self.func = func
		self.generator = generator
		
		self.lower_var = tuple(init_lower_var)
		self.linear_solver = linear_solver
		self.linear_op = linear_op
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.implicit_diff = implicit_diff
		self.track_grad_for_backward = track_grad_for_backward
		self.dual_var_warm_start = dual_var_warm_start
		if self.implicit_diff:
			self.dual_var = tuple([torch.zeros_like(p) for p in init_lower_var])
		else:
			self.dual_var = None
	def update_lr(self):
		if self.scheduler is not None:
			self.scheduler.step()
			#print('LR is ')
			#print(self.scheduler.get_last_lr()[0])
			self.optimizer.update_lr(self.scheduler.get_last_lr()[0])
		
	def update_dual(self,dual_var):
		if self.dual_var_warm_start:
			self.dual_var = dual_var

	def forward(self,*all_params,with_val=True):
		len_lower = len(self.lower_var)
		with  torch.enable_grad():
			opt_lower_var =  ArgMinOp.apply(self,len_lower,*all_params)
			val  =opt_lower_var[-1]
			opt_lower_var = opt_lower_var[:-1]
		return opt_lower_var, val

class ArgMinOp(torch.autograd.Function):

	@staticmethod
	def forward(ctx,selection, len_lower,*all_params):
		lower_var = all_params[:len_lower]
		upper_var = all_params[len_lower:]
		ctx.selection = selection
		ctx.len_lower = len_lower
		with  torch.enable_grad():
			iterates, val,grad, inputs = selection.optimizer.run(upper_var,lower_var)
		ctx.iterates = iterates
		ctx.grad = grad
		ctx.inputs = inputs
		ctx.save_for_backward(*all_params)
		
		return tuple( p.detach() for p in iterates[-1])+(val.detach(),)

	@staticmethod
	def backward(ctx, *grad_output):
		selection = ctx.selection
		iterates = ctx.iterates
		len_lower = ctx.len_lower
		upper_var = ctx.saved_tensors[len_lower:]
		lower_var = ctx.saved_tensors[:len_lower]
		grad = ctx.grad
		inputs = ctx.inputs
		with  torch.enable_grad():
			
			if len(iterates)>1:
				
				val = [ torch.einsum('...i,...i->',y,g) for y,g in zip(  iterates[-1], grad_output)]
				val = torch.sum(torch.stack(val))
				all_params = iterates[0]+upper_var
				len_lower = len(lower_var)
				grad_selection = autograd.grad(
									outputs=val, 
									inputs=all_params, 
									grad_outputs=None, 
									retain_graph=selection.track_grad_for_backward,
									create_graph=False, 
									only_inputs=True,
									allow_unused=True)
				grad_selection_lower =  grad_selection[:len_lower]
				grad_selection_upper =  grad_selection[len_lower:]
				#del iterates
				
			else:
				grad_selection_lower =  grad_output
				grad_selection_upper =  tuple([torch.zeros_like(var) for var in upper_var])
		## Solve a system Ax+b=0, 
		# A: the hessian of the lower objective, 
		# b:   grad_selection_lower
		if selection.implicit_diff:
			with  torch.enable_grad():	
				lower_var = tuple([ param.detach() for param in  iterates[-1]])
				
				for param in lower_var:
					param.requires_grad = True
				selection.linear_op.set_param_values(grad,upper_var,lower_var,inputs)
				implicit_grad,dual_var = selection.linear_solver(linear_op = selection.linear_op,
																b_vector=grad_selection_lower,
																init=selection.dual_var)			
			## summing the contributions of partial_x phi and implicit gradient term
			for g_upper,g_lin in zip(grad_selection_upper,implicit_grad):
				if g_lin is not None:
					g_upper.data.add_(g_lin.detach())
			
			## update the dual variable for next iteration 
			selection.update_dual(dual_var)
		#print(hello)

		return (None,)*(len(lower_var)+2) + grad_selection_upper






class DiffOpt(object):
	def __init__(self,func,generator,params,config_dict, 
						opt_iter,unrolled_iter,
						track_grad_for_backward=False):
		self.func = func
		self.generator = generator

		#scheduler = config_to_instance(**config_dict.scheduler)
		self.lr = config_dict.pop("lr", None)
		self.config_opt = config_dict
		self.optimizer = config_to_instance(**self.config_opt, lr = self.lr)
		self.opt_state = self.optimizer.init(params)		
		self.unrolled_iter = unrolled_iter
		assert (opt_iter >= self.unrolled_iter) 
		self.warm_start_iter = opt_iter-self.unrolled_iter
		self.track_grad_for_backward = track_grad_for_backward

		
	
	def update_lr(self,lr):
		self.optimizer = config_to_instance(**self.config_opt,lr=lr)
		
	def init_state_opt(self):
		# Detach all tensors to avoid backbropagating twice. 
		self.opt_state = tuple([utils.detach_states(state) for state in self.opt_state])


		
	def run(self,upper_var,lower_var):
		avg_val = 0.		
		cur_lower_var = lower_var
		all_lower_var = [cur_lower_var]
		track_grad = False
		total_iter = self.warm_start_iter + self.unrolled_iter
		
		self.init_state_opt()		

		for i in range(total_iter):
			inputs = next(self.generator)
			value = self.func(inputs,upper_var,cur_lower_var)
			if i>=self.warm_start_iter:
				track_grad = True

			if i == total_iter-1:
				all_grad = torch.autograd.grad(
											outputs=value, 
											inputs=upper_var+cur_lower_var, 
											retain_graph=self.track_grad_for_backward or track_grad,
											create_graph=self.track_grad_for_backward or track_grad,
											only_inputs=True,
											allow_unused=True)
				grad = all_grad[len(upper_var):]
			else:
				grad = torch.autograd.grad(
							outputs=value, 
							inputs=cur_lower_var, 
							retain_graph=track_grad,
							create_graph=track_grad,
							only_inputs=True,
							allow_unused=True)


			if  i<self.warm_start_iter:
				track_grad=False

			updates, self.opt_state = self.optimizer.update(grad, self.opt_state, inplace=not track_grad)
			cur_lower_var = torchopt.apply_updates(cur_lower_var, updates, inplace=not track_grad)
			
			if i>=self.warm_start_iter:
				all_lower_var.append(cur_lower_var)
			
			avg_val +=value.detach()

		avg_val = avg_val/total_iter

		return all_lower_var, avg_val,all_grad,inputs


class HessianOp(object):
	def __init__(self,func,generator, stochastic=False,use_new_input=False,compute_new_grad=False):
		self.func= func
		self.generator = generator
		self.cur_generator = generator
		self.upper_var = None
		self.lower_var = None
		self.grad_loss = None
		self.inputs = None
		self.use_new_input = use_new_input
		self.compute_new_grad = compute_new_grad
		self.stochastic = stochastic
		self.new_setting = False
		if self.stochastic:
			self.compute_new_grad=True

	def set_param_values(self,grad_loss, upper_var,lower_var,inputs):
		self.upper_var = upper_var
		self.lower_var = lower_var
		self.grad_loss = grad_loss
		self.new_settings = True
		self.update_generator(inputs)
		

	def update_generator(self,inputs):
		if self.stochastic:
			self.cur_generator = self.generator
		else:
			if self.use_new_input:
				self.inputs = next(self.generator)
			else: 
				self.inputs = inputs
			self.cur_generator = cycle([self.inputs])


	def eval_grad(self):
		if self.compute_new_grad:
			if self.stochastic or self.new_settings:


				self.inputs = next(self.cur_generator)
				params = self.upper_var+self.lower_var
				self.new_settings = False
				self.grad_loss = utils.grad_lower(self.func,self.upper_var,self.lower_var,self.inputs,params, 
						retain_graph=True,
						create_graph=True)



				return self.grad_loss


		return self.grad_loss


	def __call__(self,iterate,which='both',retain_graph=True):
		if which=='both':
			params = self.upper_var+self.lower_var
		elif which=='upper':
			params = self.upper_var
		grad = self.eval_grad()
		grad_lower = grad[len(self.upper_var):]
		
		hvp = utils.jvp(grad_lower,params,iterate,retain_graph=retain_graph)
		out_lower = hvp[len(self.upper_var):]
		out_upper = hvp[:len(self.upper_var)]
		return out_upper, out_lower

class FiniteDiff(HessianOp):
	def __init__(self,func,generator,epsilon=0.01,**kwargs):
		super(FiniteDiff,self).__init__(func,generator,**kwargs)	
		self.epsilon = epsilon

	def __call__(self,iterate,params= None):
		# ## y 
		if params is None:
			params = self.upper_var+self.lower_var
		grad_minus = self.eval_grad()
		
		norm = torch.cat([w.view(-1) for w in iterate]).norm()
		eps = self.epsilon / (norm.detach()+self.epsilon)

		## y + epsilon d
		lower_var_plus = tuple([p+eps* d if d is not None else p  for p,d in zip(self.lower_var,iterate)])

		grad_plus= util.grad_lower(self.func,self.upper_var,lower_var_plus,self.inputs,params, 
						retain_graph=False,
						create_graph=False)
		
		hvp  = [(p-m)/(eps) for p,m in zip(grad_plus,grad_minus)]
		out_lower = hvp[len(self.upper_var):]
		out_upper = hvp[:len(self.upper_var)]
		return out_upper, out_lower
