from tkinter import N
import numpy as np
import math
"""
Generate data and initalize
We assume that there are two clusters
"""

np.random.seed(1)
x1 = np.random.normal(-10, 10, 2000)
np.random.seed(1)
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
def gaussian_prob(x, mu, sigma):
    var = sigma ** 2
    return 1/math.sqrt(2*math.pi*var)*math.exp(-(x-mu)**2/(2*var))


def EM(iter, x, w_1, w_2, mu_1, mu_2, sigma_1, sigma_2):
    # E-step
    # wi1, wi2 = np.zeros_like(x)
    n = len(x)
    for i in range(iter):
        w1p = np.array([w_1 * gaussian_prob(xi, mu_1, sigma_1) for xi in x])
        w2p = np.array([w_2 * gaussian_prob(xi, mu_2, sigma_2) for xi in x])
        wi1 = w1p / (w1p+w2p)
        wi2 = w2p / (w1p+w2p)
        # M-step
        mu_1 = np.sum(wi1*x)/np.sum(wi1)
        mu_2 = np.sum(wi2*x)/np.sum(wi2)
        sigma_1 = math.sqrt(np.sum(wi1*(x-mu_1)**2)/np.sum(wi1))
        sigma_2 = math.sqrt(np.sum(wi2*(x-mu_2)**2)/np.sum(wi2))
        w_1 = np.sum(wi1)/n
        w_2 = np.sum(wi2)/n
        print("Iteration {}: w1: {}; w2: {}, mu1: {}, mu2: {}, sigma1: {}, sigma2: {}".format(i, w_1, w_2, mu_1, mu_2, sigma_1, sigma_2))
    pred = np.zeros_like(w1p)
    pred[w1p>w2p] = 0
    pred[w1p<=w2p] = 1
    gt = np.zeros_like(w1p)
    gt[2000:] = 1
    print("ACC: {}".format(np.sum(pred==gt)/len(gt)))


if __name__ == "__main__":
    iter = 10
    EM(iter, x, w_1, w_2, mu_1, mu_2, sigma_1, sigma_2) 






