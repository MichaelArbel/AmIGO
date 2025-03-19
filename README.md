# Code for Reproducing Results from AmIGO and Bilevel-Games with Selection

This repository contains code for reproducing the experimental results from the papers:

- **Amortized Implicit Differentiation for Stochastic Bilevel Optimization** ([Paper Link](https://arxiv.org/abs/2111.14580))
- **Non-Convex Bilevel Games with Critical Point Selection Maps** ([Paper Link](https://arxiv.org/abs/2207.04888))

## Installation

To run this code, first install the required dependencies, including the MLXP experiment manager,  `torchopt` and  `functorch` :

```sh
pip install MLXP
pip install torchopt
pip install functorch
```

## Reproducing Experimental Results

You can reproduce the results from the papers by running the following scripts with the parameters specified in the papers:

### 1. Toy Experiment using Quadratic Objectives

```sh
.scripts/quadratic_toy.sh
```

### 2. Hyperparameter Optimization on the 20 Newsgroups Dataset

```sh
.scripts/hyperparameter_opt.sh
```

### 3. Dataset Distillation on CIFAR-10

```sh
.scripts/distillation_cifar10.sh
```

For more details on experimental settings and parameters, please refer to the respective papers.

---

If you encounter any issues, feel free to open an issue or reach out to the authors.


## Attribution

If you find this work useful, please cite our papers:

```bibtex

@inproceedings{Arbel:2022a,
	author = {Arbel, Michael and Mairal, Julien},
	booktitle = {International Conference on Learning Representations (ICLR)},
	title = {{Amortized implicit differentiation for stochastic bilevel optimization}},
	year = {2022}}



@article{Arbel:2022,
	author = {Arbel, Michael and Mairal, Julien},
	journal = {Advances in Neural Information Processing Systems (NeurIPS) 2022},
	title = {Non-Convex Bilevel Games with Critical Point Selection Maps},
	year = {2022}}

```
