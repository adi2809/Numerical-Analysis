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


def plot_func():

    num_samples = 50 # number of points for each function 
    num_functions = 5 # number of functions for each sample

    X = np.expand_dims(np.linspace(0, 0.50, num_samples), axis=1)
    S = rbf_kernel(X, X) # kernel of the data points in the range

    # draw samples from the prior at our data points. 
    # assume the mean as 0 just for the sake of simplicity. 

    ys = np.random.multivariate_normal(np.zeros(num_samples), S, num_functions)

    # plot the sampled functions : 

    plt.figure(figsize=(20, 10))
    for i in range(num_functions):
        plt.plot(X, ys[i], linestyle='-', marker='o', markersize=3)

    plt.xlabel('$x$', fontsize=13)
    plt.ylabel('$y = f(x)$', fontsize=13)
    plt.title('sampled functions')

    plt.xlim([0, 0.50])
    plt.show()


if __name__ == "__main__":
    plot_func()