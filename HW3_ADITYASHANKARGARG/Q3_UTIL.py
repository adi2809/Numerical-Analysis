import numpy as np
import scipy as sp

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.gridspec as gridspec
import seaborn as sns


sns.set_style("darkgrid")
np.random.seed(42)

def rbf_kernel(xa, xb, s2=1, l=1):
    """
    exponentiated quadratic kernel with 
    sigma = 1, l = 1

    """

    sq_norm = -0.5 * sp.spatial.distance.cdist(xa, xb, 'sqeuclidean')/(l**2)
    return s2 * np.exp(sq_norm)


def gp(x_train, y_train, x_test, kernel = rbf_kernel):
    """
    calculate the posterior mean and covariance matrix for
    y_test, based on the corresponding input x_test, the 
    observed data (x_train, y_train) and prior kernel func.

    """
    k1 = kernel(x_train, x_train) # kernel of observed data

    k12 = kernel(x_train, x_test) # kernel of observed data and test data

    solved = sp.linalg.solve(k1, k12, assume_a='pos').T

    mu2 = solved@y_train # posterior mean
    k22 = kernel(x_test, x_test) # kernel of test data

    k2 = k22 - (solved@k12)

    return mu2, k2

def main():

    f = lambda x : (2*x**0.2-np.sin(x)).flatten()

    n1 = 5 # number of training points
    n2 = 35 # number of test points

    ny = 5 # number of sampled functions
    domain = (0, 4.5) # domain of the function

    x_train = np.array([1.00, 1.25, 3.40, 3.60, 4.00]).reshape(n1, 1)
    y_train = np.array([1.1990, 1.0971,  2.8547, 3.0333, 3.4158])

    x_test = np.linspace(domain[0], domain[1], n2).reshape(n2, 1)

    mu2, k2 = gp(x_train, y_train, x_test)

    s2 = np.sqrt(np.diag(k2))

    y2 = np.random.multivariate_normal(mu2, cov=k2, size = ny)


    # Plot the postior distribution and some samples
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))

    # Plot the distribution of the function (mean, covariance)
    ax1.plot(x_test, f(x_test), 'b--', label='$f(x)$')
    ax1.fill_between(x_test.flat, mu2-2*s2, mu2+2*s2, color='red', alpha=0.15, label='$2 \sigma_{2|1}$')

    ax1.plot(x_test, mu2, 'r-', lw=2, label='$\mu_{2|1}$')
    ax1.plot(x_train, y_train, 'ko', linewidth=2, label='$(x_1, y_1)$')

    ax1.set_xlabel('$x$', fontsize=13)
    ax1.set_ylabel('$y$', fontsize=13)

    ax1.set_title('Distribution of posterior and prior data.')

    ax1.axis([domain[0], domain[1], 0, 6])
    ax1.legend()

    # Plot some samples from this function
    ax2.plot(x_test, y2.T, '-')
    ax2.set_xlabel('$x$', fontsize=13)
    ax2.set_ylabel('$y$', fontsize=13)

    ax2.set_title('5 different function realizations from posterior')
    ax1.axis([domain[0], domain[1], 0, 6])
    ax2.set_xlim([0, 4.5])
    plt.tight_layout()
    plt.show()




    

