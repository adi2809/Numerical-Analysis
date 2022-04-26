import numpy as np
from scipy.optimize import minimize
import time



def Kernel(squared_distance, L):
    '''  Kernel function (squared exponential, with length scale L)
    '''

    return np.exp(-1/(2*L**2)*squared_distance)


def FindGramMatrix(X, L, Sigma, N):
    ''' Function that creates and inverts gram matrix for a squared
        exponential kernel with length-scale L.
    '''

    squared_distances = FindSquaredDistances(X)
    K = Kernel(squared_distances, L)
    C = K + Sigma**2*np.identity(N)
    InvC = np.linalg.inv(C)
    return K, C, InvC


def FindSquaredDistances(X):
    ''' Function that finds the squared distances between
        inputs points (efficiently).
    '''

    if np.size(X[0]) == 1:
        Xsq = X**2
        squared_distances = -2*np.outer(X, X) + (Xsq[:, None] + Xsq[None, :])
    else:
        Xsq = np.sum(X**2, 1)
        squared_distances = -2.*np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    return squared_distances


def NegLogLikelihoodFun(Theta, a):
    ''' Returns negative log-likelihood function in a form
        suitable for scipy.optimize.fmin_bfgs.
    '''

    X = a[0]
    Y = a[1]
    N = a[2]
    L = Theta[0]
    Sigma = Theta[1]
    K, C, InvC = FindGramMatrix(X, L, Sigma, N)
    (Sign, LogDetC) = np.linalg.slogdet(C)
    LogDetC = Sign*LogDetC
    return 0.5*LogDetC + 0.5*np.dot(Y, np.dot(InvC, Y))


def dC_dL_Fun(X, L, K, N):
    ''' Derivative of matrix C w.r.t L (length scale).
    '''

    squared_distances = FindSquaredDistances(X)
    return np.power(L, -3) * np.multiply(squared_distances, K)


def dC_dSigma_Fun(Sigma, K, N):
    ''' Derivative of matrix C w.r.t sigma (noise std).
    '''
    return 2*Sigma*np.eye(N)


def dNLL_dTheta(Theta, a):
    ''' Derivative of the negative log-likelihood w.r.t hyperparameters in a
        form suitable for scipy.optimize.fmin_bfgs.
    '''

    X = a[0]
    Y = a[1]
    N = a[2]
    L = Theta[0]
    Sigma = Theta[1]
    K, C, InvC = FindGramMatrix(X, L, Sigma, N)

    # Find dC / dL
    dC_dL = dC_dL_Fun(X, L, K, N)

    # Find dC / dSigma
    dC_dSigma = dC_dSigma_Fun(Sigma, K, N)

    # Find dlogp / dL
    dLogL_dL = (0.5 * np.trace(np.dot(InvC, dC_dL)) -
                0.5 * np.dot(Y, np.dot(InvC, np.dot(dC_dL, np.dot(InvC,
                                                                  Y)))))

    # Find dlogp / dSigma
    dLogL_dSigma = (0.5*np.trace(np.dot(InvC, dC_dSigma)) -
                    0.5*np.dot(Y, np.dot(InvC, np.dot(dC_dSigma, np.dot(InvC,
                                                                        Y)))))

    # Gradient vector
    gradient = np.array([dLogL_dL, dLogL_dSigma])

    return gradient


def Train(L0, Sigma0, X, Y, N):
    ''' Update hyperparameters using scipy.minimize
    '''

    Theta0 = [L0, Sigma0]
    a = (X, Y, N)
    b1 = (1e-6, 4)
    b2 = (1e-6, 1)
    bnds = (b1, b2)
    start_time = time.time()

    sol = minimize(NegLogLikelihoodFun, x0=Theta0, args=(a, ),
                   method='SLSQP', jac=dNLL_dTheta, bounds=bnds)

    elapsed_time = time.time() - start_time
    ThetaOpt = sol.x
    L = ThetaOpt[0]
    Sigma = ThetaOpt[1]
    K, C, InvC = FindGramMatrix(X, L, Sigma, N)
    return L, Sigma, K, C, InvC, elapsed_time


def callbackF(Theta):
    ''' Callback function for the fmin_bfgs optimisation routine
    '''

    print(Theta[0], Theta[1])


def Predict(X, xStar, L, Sigma, Y, K, C, InvC, N):
    ''' GP prediction
    '''

    if np.size(X[0]) == 1:
        squared_distances = (X-xStar)**2
    else:
        squared_distances = np.sum((X-xStar)**2, 1)
    k = Kernel(squared_distances, L)
    c = 1 + Sigma**2   # Always true for this particular kernel
    yStarMean = np.dot(k, np.dot(InvC, Y))
    yStarStd = np.sqrt(c - np.dot(k, np.dot(InvC, k)))
    return yStarMean, yStarStd