import torch

import importlib
import os
from torchvision import transforms
import pickle as pkl

import numpy as np
import signal

import torchvision

from core import utils
from itertools import cycle
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.model_selection import train_test_split


class CustomDataset(torch.utils.data.Dataset):

  def __init__(self, x,y):
    # Your code

    self.x = x
    self.y = y
    self.length = len(x)
  def __getitem__(self, idx):
    return self.x[idx].to_dense() ,self.y[idx] # In case you stored your data on a list called instances

  def __len__(self):
    return self.length


class ListIterator:
	def __init__(self, loader, device,dtype):
		self.device= device
		self.dtype = dtype
		self.loader = loader
		self.tensor_list = None
		self.iterator = None
	def make_tensor_list(self):
		self.tensor_list = []
		for i, data in enumerate(self.loader):
			data= utils.set_device_and_type(data,self.device,self.dtype)
			self.tensor_list.append(data)			
		self.iterator = iter(self.tensor_list)

	def __next__(self, *args):
		try:
			idx = next(self.iterator)
		except:
			self.make_tensor_list()
			return next(self.iterator)
	def __iter__(self):
		try:
		
			return iter(self.tensor_list)
		except:
			self.make_tensor_list()
			return iter(self.tensor_list)
		
	def __getitem__(self,i):
		return self.tensor_list[i]
	def __getstate__(self):
		return {'tensor_list': self.tensor_list,
				'iterator': None}
	def __setstate__(self, d ):
		self.tensor_list = d['tensor_list']
		self.iterator = None



def from_sparse(x):
    x = x.tocoo()
    values = x.data
    indices = np.vstack((x.row, x.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = x.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def make_loaders(args,num_workers,dtype,device):
	b_size = args.b_size
	val_size_ratio = args.val_size_ratio
	data_path= args.data_path
	work_dir = os.getcwd()
	path = os.path.join(work_dir,data_path,'20newsgroups_'+str(b_size)+'.pkl')
	# try:
	# 	with open(path,'rb') as f:
			
	# 		data = pkl.load(f)
	# 		loaders = data['loaders']
	# 		meta_data = data['meta_data']
	# except:
		
	X, y = fetch_20newsgroups_vectorized(subset='train', return_X_y=True)
	x_test, y_test = fetch_20newsgroups_vectorized(subset='test', return_X_y=True)
	x_train, x_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=val_size_ratio)

	train_samples, n_features = x_train.shape
	test_samples, n_features = x_test.shape
	val_samples, n_features = x_val.shape
	n_classes = np.unique(y_train).shape[0]

	y_test = torch.from_numpy(y_test).long()
	y_train = torch.from_numpy(y_train).long()
	y_val = torch.from_numpy(y_val).long()
	
	x_train = from_sparse(x_train)
	x_test = from_sparse(x_test)
	x_val = from_sparse(x_val)

	dataset = [(x_train,y_train),(x_val,y_val),(x_test,y_test)]

	iterators = []
	cuda = (device== 'cuda')
	cuda = False
	kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


	lower_data = CustomDataset(dataset[0][0],dataset[0][1])
	upper_data = CustomDataset(dataset[1][0],dataset[1][1])
	test_data = CustomDataset(dataset[2][0],dataset[2][1])



	lower_loader = torch.utils.data.DataLoader(lower_data,
									  batch_size=b_size,
									  shuffle=True,
									  num_workers=1)

	upper_loader = torch.utils.data.DataLoader(upper_data,
										  batch_size=b_size,
										  shuffle=True,
										  num_workers=1)

	test_loader = torch.utils.data.DataLoader(test_data,
										  batch_size=b_size,
										  shuffle=True,
										  num_workers=1)


	lower_loader = ListIterator(lower_loader,device, dtype)
	upper_loader = ListIterator(upper_loader,device,dtype)
	test_loader = ListIterator(test_loader,device,dtype)

	x,y= next(iter(lower_loader))
	shape = list(x.shape)
	shape[0] = n_classes
	n_features = np.prod(np.array(shape[1:]))


	loaders = {'lower_loader':lower_loader,
		 'upper_loader':upper_loader,
		 'test_upper_loader': test_loader,
		 'test_lower_loader': None,
		}
	meta_data = {'n_features':n_features, 
				 'n_classes': n_classes, 
				 'shape':x_train[0].shape,
				 'total_samples': x_train.shape[0],
				 'b_size': b_size }


		
		# with open(path,'wb') as f:
		# 	pkl.dump({'loaders':loaders, 'meta_data':meta_data},f)	
	
	return loaders, meta_data
