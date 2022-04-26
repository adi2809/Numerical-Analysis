from numpy.linalg import * 
from scipy.linalg import *
from autograd import jacobian, grad
from scipy  import * 

from math import * 
def euler(f, x0, y0, xn, h):
    """
    this is the implementation of the above pseudo code.
    ----------------------------------------------------
    
    input : 
    1)f  = function f(x,y) used in the iteration scheme.
    2)x0 = the initial value for the independent variable.
    3)y0 = the initial condition for the differential eqn.
    4)xn = the final point at which we have to stop (x<xn)
    5)h  = the step size that has to be used in the prob. 
    
    returns: 
    
    x, y
    """
    x, y = [x0], [y0]
    n = int((xn-x0)//h)
    yn = y0
    for i in range(1,n+1):
        
        
        yn = yn + h*f(x0+(i*h), yn)
        
        y.append(yn)
        x.append(x0+(i*h))
    
    x.append(x0+((n+1)*h))
    y.append(yn)
    return x, y

def ERK4(f, x0, y0, xn, h):
    """
    this is the implementation of the above pseudo code.
    ----------------------------------------------------
    
    input : 
    1)f  = function f(x,y) used in the iteration scheme.
    2)x0 = the initial value for the independent variable.
    3)y0 = the initial condition for the differential eqn.
    4)xn = the final point at which we have to stop (x<xn)
    5)h  = the step size that has to be used in the prob. 
    
    returns: 
    
    x, y
    """
    X, Y = [], []
    n = int(((xn - x0)/h))
    y = y0
    
    for i in range(1,n+1):
        Y.append(y)
        "Apply Runge Kutta Formulas to find next value of y"
        k1 = h * f(x0, y)
        k2 = h * f(x0 + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(x0 + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(x0 + h, y + k3)
 
        # Update next value of y
        y = y + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
 
        # Update next value of x
        x0 = x0+h
        X.append(x0)
        
        
     
    Y.append(y)
    
    return X, Y

class Onestepmethod (object):
    def __init__(self,f,y0,t0,te,N, tol):
       self.f = f
       self.y0 = y0.astype(float)
       self.t0 = t0
       self.interval = [t0 , te]
       self.grid = linspace(t0,te,N+2) # N interior points
       self.h = (te-t0)/(N+1)
       self.N = N
       self.tol = tol
       self.m = len(y0)
       self.s = len(self.b)
    def step(self):
       ti , yi = self.grid[0], self.y0 # initial condition
       tim1 = ti
       yield ti , yi

       for ti in self.grid[1:]:
           yi = yi + self.h*self.phi(tim1, yi)
           tim1 = ti
           yield ti , yi

    def solve(self):
       self.solution = list(self.step())

class RungeKutta_implicit(Onestepmethod):

    def phi_solve(self, t0, y0, initVal, J, M):
        """
        This function solves the sm x sm system
        F(Y_i)=0
        by Newton’s method with an initial guess initVal.
        Parameters:
        -------------
        t0 = float, current timestep
        y0 = 1 x m vector, the last solution y_n. Where m is the
           length
        of the initial condition y_0 of the IVP.
        initVal = initial guess for the Newton iteration
        J = m x m matrix, the Jacobian matrix of f() evaluated in y_i
        M = maximal number of Newton iterations
        Returns:
        -------------
        The stage derivative Y’_i
        """
        JJ = eye(self.s*self.m)-self.h*kron(self.A, J)
        luFactor = linalg.lu_factor(JJ)
        for i in range(M):
            initVal, norm_d = self.phi_newtonstep(t0, y0, initVal,luFactor)
            if norm_d < self.tol:

                print("Newton converged in {} steps".format(i))
                break
            elif i == M-1:
                raise ValueError("The Newton iteration did not converge")

        return initVal

    def phi_newtonstep(self, t0, y0, initVal, luFactor):
        """
        Takes one Newton step by solvning
            G’(Y_i)(Y^(n+1)_i-Y^(n)_i)=-G(Y_i)
        where
        G(Y_i) = Y_i - y_n - h*sum(a_{ij}*Y’_j) for j=1,...,s
        Parameters:
        -------------
        t0 = float, current timestep
        y0 = 1 x m vector, the last solution y_n. Where m is the
            length
        of the initial condition y_0 of the IVP.
        initVal = initial guess for the Newton iteration
        luFactor = (lu, piv) see documentation for linalg.lu_factor
        Returns:
        The difference Y^(n+1)_i-Y^(n)_i
        """
        d = linalg.lu_solve(luFactor, - self.F(initVal.flatten(),t0, y0))
        return initVal.flatten() + d, norm(d)
    
    def F(self, stageDer, t0, y0):
        """
        Returns the subtraction Y’_{i}-f(t_{n}+c_{i}*h, Y_{i}),
            where Y are
        the stage values, Y’ the stage derivatives and f the
            function of
        the IVP y’=f(t,y) that should be solved by the RK-method.
        Parameters:
        -------------
        stageDer = initial guess of the stage derivatives Y’
        t0 = float, current timestep
        y0 = 1 x m vector, the last solution y_n. Where m is the
            length
        of the initial condition y_0 of the IVP.
        """
        stageDer_new = empty((self.s,self.m)) # the i:th stageDer is on the i:th row
        for i in range(self.s): #iterate over all stageDer
            stageVal = y0 + array([self.h*dot(self.A[i,:],
                stageDer.reshape(self.s,self.m)[:, j]) for j in range(self.m)])

            stageDer_new[i, :] = self.f(t0 + self.c[i] * self.h,
                stageVal) #the ith stageDer is set on the ith row
        return stageDer - stageDer_new.reshape(-1)

    

    def phi(self, t0, y0):
       """
       Calculates the summation of b_j*Y_j in one step of the
           RungeKutta
       method with
       y_{n+1} = y_{n} + h*sum_{j=1}^{s} b_{j}*Y
       where j=1,2,...,s, and s is the number of stages, b the
           nodes, and Y the
       stage values of the method.
       Parameters:
       -------------
       t0 = float, current timestep
       y0 = 1 x m vector, the last solution y_n. Where m is the
           length
       of the initial condition y_0 of the IVP.
       """
       M = 10 # max number of newton iterations
       stageDer = array(self.s*[self.f(t0,y0)]) # initial value: Y’_0

       J = jacobian(self.f, t0, y0)
       stageVal = self.phi_solve(t0, y0, stageDer, J, M)
       return array([dot(self.b, stageVal.reshape(self.s,self.m)[:,j]) for j in range(self.m)])


class SDIRK(RungeKutta_implicit):
    def phi_solve(self, t0, y0, initVal, J, M):
        """
        This function solves F(Y_i)=0 by solving s systems of size m
        x m each.
        Newton’s method is used with an initial guess initVal.
        Parameters:
        -------------
        t0 = float, current timestep
        y0 = 1 x m vector, the last solution y_n. Where m is the length of the initial condition y_0 of the IVP.
        initVal = initial guess for the Newton iteration
        J = m x m matrix, the Jacobian matrix of f() evaluated in y_i
        M = maximal number of Newton iterations
        Returns:
        -------------
        The stage derivative Y’_i
        """
        JJ = eye(self.m) - self.h*self.A[0,0]*J
        luFactor = linalg.lu_factor(JJ)
        for i in range(M):
            initVal, norm_d = self.phi_newtonstep(t0, y0, initVal, J, luFactor)
            if norm_d < self.tol:
                print ("Newton converged in {} steps".format(i))
                break
            elif i == M-1:
              raise ValueError("The Newton iteration did not converge.")
        return initVal

    def phi_newtonstep(self, t0, y0, initVal, J, luFactor):
        """
        Takes one Newton step by solvning
            G’(Y_i)(Y^(n+1)_i-Y^(n)_i)=-G(Y_i)
        where
        G(Y_i) = Y_i - haY’_i - y_n - h*sum(a_{ij}*Y’_j) for j=1,...,i-1
        Parameters:
        -------------
        t0 = float, current timestep
        y0 = 1 x m vector, the last solution y_n. Where m is the
            length
        of the initial condition y_0 of the IVP.
        initVal = initial guess for the Newton iteration
        luFactor = (lu, piv) see documentation for linalg.lu_factor
        Returns:
        The difference Y^(n+1)_i-Y^(n)_i
        """
        x = []
        for i in range(self.s): # solving the s mxm systems
            rhs = - self.F(initVal.flatten(), t0,
                y0)[i*self.m:(i+1)*self.m] + sum([self.h*self.A[i,j]*dot(J,x[j]) for j in range(i)], axis = 0)
            d = linalg.lu_solve(luFactor, rhs)
            x.append(d)
        return initVal + x, norm(x)

class Gauss(RungeKutta_implicit): #order 6
    A=array([[5/36, 2/9 - sqrt(15)/15, 5/36 - sqrt(15)/30],[ 5/36 +
       sqrt(15)/24, 2/9, 5/36 - sqrt(15)/24],[ 5/36 + sqrt(15)/30,
       2/9 + sqrt(15)/15, 5/36]])
    b=[5/18,4/9,5/18]
    c=[1/2-sqrt(15)/10,1/2,1/2+sqrt(15)/10]


class SDIRK_tableau2s(SDIRK): # order 3
    p = (3 - sqrt(3))/6
    A = array([[p, 0], [1 - 2*p, p]])
    b = array([1/2, 1/2])
    c = array([p, 1 - p])

    