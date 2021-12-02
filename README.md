# A Pytorch Implementation for DeepXDE
## Introduction

DeepXDE is a library for scientific machine learning. You can see more details [here](https://github.com/lululxvi/deepxde).

This is a simplified pythoch implementation that inherits the deep library. We deleted the part about TensorFlow to make the file more lightweight. At the same time, some functions of DeepXDE have been optimized for personalized needs.

This job contains a huge amount of work. The work to be done is as follows. If you have other suggestions, please feel free to leave a message to contact me: 

**To do list**
- [x] Test and verify that the DeepXDE part we need can work normally. 
- [x] Add the realization of Helmholtz Equation, refer to [here](https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks/tree/main/TensorFlow/Helmholtz%20Equation) 
- [ ] Support more optimization methods to get better predictions. Currently included in the plan: Linear-search.
- [x] Increase the visualization part to make it easier to observe the difference between the model and the method.
- [ ] Implement a wrapper for scipy.optimize to make it a PyTorch Optimizer. This will make subsequent migration of TensorFlow to pytorch easier  
- [ ] To be continue...












**Papers on algorithms**

- Solving PDEs and IDEs via PINN [[SIAM Rev.](https://doi.org/10.1137/19M1274067)], gradient-enhanced PINN (gPINN) [[arXiv](https://arxiv.org/abs/2111.02801)]
- Solving fPDEs via fPINN [[SIAM J. Sci. Comput.](https://epubs.siam.org/doi/abs/10.1137/18M1229845)]
- Solving stochastic PDEs via NN-arbitrary polynomial chaos (NN-aPC) [[J. Comput. Phys.](https://www.sciencedirect.com/science/article/pii/S0021999119305340)]
- Solving inverse design/topology optimization via PINN with hard constraints (hPINN) [[SIAM J. Sci. Comput.](https://doi.org/10.1137/21M1397908)]
- Learning nonlinear operators via DeepONet [[Nat. Mach. Intell.](https://doi.org/10.1038/s42256-021-00302-5), [arXiv](https://arxiv.org/abs/2111.05512)], DeepM&Mnet [[J. Comput. Phys.](https://doi.org/10.1016/j.jcp.2021.110296), [J. Comput. Phys.](https://doi.org/10.1016/j.jcp.2021.110698)]
- Learning from multi-fidelity data via MFNN [[J. Comput. Phys.](https://doi.org/10.1016/j.jcp.2019.109020), [PNAS](https://www.pnas.org/content/117/13/7052)]

## Features

DeepXDE has implemented many algorithms as shown above and supports many features:

- complex domain geometries without tyranny mesh generation. The primitive geometries are interval, triangle, rectangle, polygon, disk, cuboid, and sphere. Other geometries can be constructed as constructive solid geometry (CSG) using three boolean operations: union, difference, and intersection.
- multi-physics, i.e., (time-dependent) coupled PDEs.
- 5 types of boundary conditions (BCs): Dirichlet, Neumann, Robin, periodic, and a general BC, which can be defined on an arbitrary domain or on a point set.
- different neural networks, such as (stacked/unstacked) fully connected neural network, residual neural network, and (spatio-temporal) multi-scale fourier feature networks.
- 6 sampling methods: uniform, pseudorandom, Latin hypercube sampling, Halton sequence, Hammersley sequence, and Sobol sequence. The training points can keep the same during training or be resampled every certain iterations.
- conveniently save the model during training, and load a trained model.
- uncertainty quantification using dropout.
- many different (weighted) losses, optimizers, learning rate schedules, metrics, etc.
- callbacks to monitor the internal states and statistics of the model during training, such as early stopping.
- enables the user code to be compact, resembling closely the mathematical formulation.

All the components of DeepXDE are loosely coupled, and thus DeepXDE is well-structured and highly configurable. It is easy to customize DeepXDE to meet new demands.

## Installation

DeepXDE requires one of the following backend-specific dependencies to be installed:

- PyTorch: [PyTorch](https://pytorch.org/)


- For developers, you should clone the folder to your local machine and put it along with your project scripts.

```
$ git clone [git http url]
```

- Other dependencies

  - [Matplotlib](https://matplotlib.org/)
  - [NumPy](http://www.numpy.org/)
  - [scikit-learn](https://scikit-learn.org)
  - [scikit-optimize](https://scikit-optimize.github.io)
  - [SciPy](https://www.scipy.org/)


## Cite DeepXDE

If you use DeepXDE for academic research, you are encouraged to cite the following paper:

```
@article{lu2021deepxde,
  author  = {Lu, Lu and Meng, Xuhui and Mao, Zhiping and Karniadakis, George Em},
  title   = {{DeepXDE}: A deep learning library for solving differential equations},
  journal = {SIAM Review},
  volume  = {63},
  number  = {1},
  pages   = {208-228},
  year    = {2021},
  doi     = {10.1137/19M1274067}
}
```

Also, if you would like your paper to appear [here](https://deepxde.readthedocs.io/en/latest/user/research.html), open an issue in the GitHub "Issues" section.

## License

[Apache license 2.0](https://github.com/lululxvi/deepxde/blob/master/LICENSE)
