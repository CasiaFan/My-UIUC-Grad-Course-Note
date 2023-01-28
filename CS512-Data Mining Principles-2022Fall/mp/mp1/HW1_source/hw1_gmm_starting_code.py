import numpy as np

"""
Generate data and initalize
We assume that there are two clusters
"""

x1 = np.random.normal(-10, 10, 2000)
x2 = np.random.normal(5, 5, 2000)
x = np.concatenate((x1, x2), axis=0)
w_1 = 0.5
w_2 = 0.5
mu_1 = -2
mu_2 = 2
sigma_1 = 2
sigma_2 = 2

"""
Apply EM algorithm to update the GMM model
"""
