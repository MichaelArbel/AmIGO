import torch
import time
from core.selection import make_selection
from core.utils import Functional
import utils.helpers as hp
from core import utils
import core

from utils.metrics import Metrics
from examples.datasetdistillation.loaders import make_loaders


from utils.helpers import get_gpu_usage

class Trainer:
	def __init__(self, args, logger):
		self.args = args
		self.logger = logger
		self.device = hp.assign_device(args.system.device)
		self.dtype = hp.get_dtype(args.system.dtype)
		
		self.mode = 'train'
		self.counter = 0
		self.epoch = 0
		self.avg_upper_loss = 0. 
		self.avg_lower_loss = 0.
		self.amortized_grad = None
		self.upper_grad = None
		self.alg_time = 0.

		self.build_model()

	# def __getstate__(self):
	# 	return {x: self.__dict__[x] for x in self.__dict__ if x not in {'loaders'}}


	def build_model(self):

		# create data loaders for lower and upper problems
		
		self.loaders, self.meta_data = make_loaders(self.args.training.loader,
									num_workers=self.args.system.num_workers,
									dtype=self.dtype,
									device=self.device)
		self.lower_loader = self.loaders['lower_loader'] 
		self.upper_loader = self.loaders['upper_loader']
		
		# create either a pytorch Module or a list of parameters


		
		training_arg = self.args.training
		lower_model_path = training_arg.lower.model.pop("path", None)
		
		self.lower_model = hp.config_to_instance(**training_arg.lower.model)
		self.lower_model = hp.init_model(self.lower_model,lower_model_path,
										self.dtype, 
										self.device
										)

		upper_model_path = training_arg.upper.model.pop("path", None)
		x,y = next(iter(self.upper_loader))
		self.upper_model = hp.config_to_instance(**training_arg.upper.model,
												shape=self.meta_data['shape'],
												x=None,y=None)
		
		self.upper_model = hp.init_model(self.upper_model,upper_model_path,
										self.dtype, 
										self.device
										)

		self.lower_var = tuple(self.lower_model.parameters())
		self.upper_var = tuple(self.upper_model.parameters())

		# create a pytorch Modules whose output is a scalar

		self.lower_loss_module = hp.config_to_instance(**training_arg.lower.objective,
								upper_model=self.upper_model, 
								lower_model=self.lower_model, 
								device=self.device)


		self.upper_loss_module = hp.config_to_instance(**training_arg.upper.objective,
								upper_model=self.upper_model, 
								lower_model=self.lower_model, 
								device=self.device)
		
		## Make the loss modules functional
		self.lower_loss = Functional(self.lower_loss_module)
		self.upper_loss = Functional(self.upper_loss_module)



		self.upper_optimizer = hp.config_to_instance(params=self.upper_var, **training_arg.upper.optimizer)
		self.use_upper_scheduler = training_arg.upper.scheduler.pop("use_scheduler", None)
		if self.use_upper_scheduler:
			self.upper_scheduler = hp.config_to_instance(optimizer=self.upper_optimizer, **training_arg.upper.scheduler)
		
		#Construct the selection
		self.selection = make_selection(self.lower_loss,
									self.lower_var,
									self.lower_loader,
									self.args.algorithm,
									self.device,
									self.dtype)


		self.count_max, self.total_batches = self.set_count_max()


		self.metrics = Metrics(training_arg.metrics,self.device,self.dtype)
		name = training_arg.metrics.name
		condition = lambda counter : counter%self.total_batches==0
		self.metrics.register_metric(self.upper_loss,
									self.loaders['test_upper_loader'],
									0,
									'test_upper',
									func_args={'upper_var':self.upper_var,
												'lower_var':self.lower_var,
												'train_mode': False},
									metric=name,
									condition=condition)

		self.best_loss = None

	def main(self):
		print(f'==> Mode: {self.mode}')
		if self.mode == 'train':
			self.train()

	def train(self):
		self.upper_optimizer.zero_grad()
		if self.counter==0:
			self.alg_time = 0.
		while self.counter<=self.count_max:

			for batch_idx, data in enumerate(self.upper_loader):
				#print(batch_idx)
				if self.counter>self.count_max:
					break					
				self.counter +=1
				metrics= self.iteration(data)
				self.metrics.eval_metrics(self.counter,metrics)
			self.update_schedule(metrics['train_upper_loss'])
			metrics = self.disp_metrics()

			self.epoch += 1
			weights = self.upper_model.x.data.cpu().numpy()
			#self.logger.log_artifacts({'weights':weights}, f"weights/{self.epoch}", artifact_type="numpy")


	def zero_grad(self):
		self.upper_optimizer.zero_grad()
		for p in self.lower_var:
			p.grad = None
	def update_lower_var(self,opt_lower_var):
		for p,new_p in zip(self.lower_var,opt_lower_var):
			p.data.copy_(new_p.data)
	def iteration(self,data):
		start_time_iter = time.time()
		data = utils.set_device_and_type(data,self.device,self.dtype)
		self.zero_grad()
		params = self.lower_var + self.upper_var
		opt_lower_var,lower_loss = self.selection(*params)
		loss,acc = self.upper_loss(data,self.upper_var,opt_lower_var, with_acc=True)		
		loss.backward()
		if self.args.training.upper.clip:
			torch.nn.utils.clip_grad_norm_(self.upper_var, self.args.training.upper.max_norm)
		self.upper_optimizer.step()
		self.update_lower_var(opt_lower_var)
		end_time_iter = time.time()
		self.alg_time += end_time_iter-start_time_iter
		metrics = self.iteration_metrics(loss,lower_loss,acc)
		return  metrics

	def iteration_metrics(self,loss,lower_loss,acc ):
		loss, acc, lower_loss = loss.detach().item(),acc.detach().item(),lower_loss.detach().item()
		
		upper_grad_norm = torch.norm(torch.stack([torch.norm(var.grad) if var.grad is not None else torch.norm(torch.zeros_like(var)) for var in  self.upper_var],axis=0)).detach().item()

		if self.selection.dual_var:
			dual_var_norm = torch.norm(torch.stack([torch.norm(b) for b in  self.selection.dual_var],axis=0)).detach().item()
		else:
			dual_var_norm = 0.
		metrics = { 'train_upper_loss': loss,
			'train_lower_loss': lower_loss,
			'train_upper_acc': 100.*acc,
			'upper_grad_norm':upper_grad_norm,
			'dual_var_norm':dual_var_norm}
		return metrics

	def update_schedule(self,loss):
		if self.use_upper_scheduler:
			self.upper_scheduler.step(loss)
		self.selection.update_lr()

	def disp_metrics(self):

		metrics = self.metrics.avg_metrics()
		metrics.update({'iter': self.counter, 'time': self.alg_time, 'epoch':self.epoch})
		#metrics.update(self.get_grad_counts())
		self.logger.log_metrics(metrics, log_name="metrics")		
		disp_keys = ['epoch','iter','time','train_upper_loss','train_lower_loss','train_upper_acc', 'test_upper_loss', 'test_upper_acc','upper_grad_norm','dual_var_norm' ]
		
		try:
			if self.epoch%self.args.training.metrics.disp_freq==0:
				print(metrics)
		except:
			pass
		return metrics

	def set_count_max(self):
		total_batches = int(self.meta_data['total_samples']/self.meta_data['b_size'])
		if total_batches==0:
			total_batches=1
		return self.args.training.total_epoch*total_batches, total_batches
