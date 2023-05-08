import numpy as np


__all__ = ["conjugate_gradient"]

def conjugate_gradient(A, b, x0):
    r0 = b - A @ x0
    p = r0
    x = x0
    num_iterations = 0
    
    print(f"Iteration {num_iterations}: x = {x0}\n")
    
    for i in range(A.shape[0]):
        Ap = A @ p
        alpha = np.dot(r0, r0) / np.dot(p, Ap)
        x = x + alpha * p
        num_iterations += 1
        r1 = r0 - alpha * Ap
        
        beta = np.dot(r1, r1)/np.dot(r0, r0)
        p = r1 + beta * p
        r0 = r1
        
        print(f"Iteration {num_iterations}: x = {x}\n")

        # Break when the solution is close to the exact solution.
        if np.allclose(A @ x, b):
            break
            
    print(f"Exact solution is reached in {num_iterations} iterations with x = {x}. A @ x: {A @ x}")
    return x
