# connFinder
Code for the connectivity estimation of neural populations presented in the paper 'From Correlation to Causation: Estimating Effective Connectivity from Zero-Lag Covariances of Brain Signals' by Jonathan Schiefer, Alexander Niederbühl, Volker Pernice, Carolin Lennartz, Jürgen Hennig, Pierre LeVan and Stefan Rotter.


# Organization of the code
gradient_methods.py: code for the estimation itself,
simulations_functions.py: code for simulation Ornstein-Uhlenbeck processes as used in the paper
helper_functions.py: code for data processesing and evaluation of the estimation.

# How to use:
The connfinder notebook contains an example of how the estimation procedure can be applied, including the simulations used in the paper.

The code should run under both, Python2 and Python3. If you use conda, the corresponding requirement files should install all package necessary for the code and the jupyter notebook.
