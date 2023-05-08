""" 
Newton's method implementation with line search
"""

import scipy.linalg as sl
import numpy as np


__all__ = ["newton_method_ls", "simple_line_search"]

def simple_line_search(x, p, f, initial_slope, alpha=1e-4, max_search_iter=100):
    """Simple line search that halves lambda until the Armijo rule is satisfied
    
    x - initial point (at lambda=0)
    p - search direction, search is along x+lambda*p
    f - (callback) function to minimize
    initial_slope - derivative of f in x in p-direction"""
    lamda = 1.
    if initial_slope==0.0:
        return x, 1.0, 0
    for j in range(max_search_iter):
        xt = x + lamda*p
        # check Armijo rule:
        if f(xt) - f(x) <= alpha*lamda*initial_slope:
            break
        lamda /= 2.
    else:
        raise Exception("Line search did not converge")
    return xt, lamda, j

def newton_method_ls(F, jac, f, line_search, x_0, atol = 1.e-5, maxiter=100, verbose=False):
    "Newton method with line search"
    x_n = []
    y_n = []
    x = x_0
    
    # iterate until we hit break either as we hit tolerance or maximum number iterations
    # since we include the initial guess, the max. number of entries is maxiter+1
    for i in range(maxiter+1):
        x_n.append(x)
        Fx = F(x)

        y_n.append(Fx)
        # Newton update:
        if isinstance(Fx, float) or len(Fx)==1:
            p = -Fx/jac(x)
        else:
            p = sl.solve(jac(x), -Fx)
                
        x, lamda, reductions = line_search(x, p, f, np.dot(F(x), p))
        if verbose:
            print("In iteration {}, x={}; {} reductions were needed (lambda={})".format(i, x_n[-1], reductions, lamda))
        if sl.norm(x - x_n[-1]) < atol:
            break
    
    return x_n, y_n