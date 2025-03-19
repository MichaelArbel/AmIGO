#!/bin/bash


parent_log_dir="data/outputs/debug"

HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py\
		algorithm='amigo'\
		training='quadratic_toy'\
        +mlxp.logger.parent_log_dir=$parent_log_dir\


