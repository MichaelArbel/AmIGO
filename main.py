
from __future__ import print_function
import sys
print(sys.version)
import scipy.stats._qmc
import mlxp

import argparse


import yaml

import torch
import os
import importlib
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import hydra
import dill as pkl

print("Check if cuda available: ")
print(torch.cuda.is_available())

print("Working dir: ")
print(os.getcwd())



#print(torch.cuda.current_device())





# check whether we want to load a pretrained model depending on the given parameters


work_dir = os.getcwd()

@mlxp.launch(config_path="configs")
def run(ctx: mlxp.Context):
	os.chdir(work_dir)
	module, attr = os.path.splitext(ctx.config.training.trainer_name) 
	module = importlib.import_module(module)
	Trainer = getattr(module, attr[1:])
	trainer = Trainer(ctx.config, ctx.logger)

	if ctx.config.training.resume:
		try:
			checkpoint_name = os.path.join(trainer.logger.dir,'checkpoints', 'last_0.pkl' )
			with open(checkpoint_name,'rb') as f:
				trainer = pkl.load(f)
		except:
			pass
		
	trainer.main()
	print('Finished!')

if __name__ == "__main__":

    run()

