#!/bin/bash


parent_log_dir="data/outputs/debug"


HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py\
		training='hyperparameter_opt' \
		training.upper.optimizer.lr=100.\
		algorithm='amigo'\
		algorithm.optimizer.lr=100.\
		algorithm.linear_solver.lr=0.5\
		algorithm.linear_solver.name='core.linear_solvers.GD'\
        +mlxp.logger.parent_log_dir=$parent_log_dir\



