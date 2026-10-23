import numpy as np
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
import matplotlib.pyplot as plt

def problem_1(): 

    f = lambda x: 1.0 / (1.0 + (10*x)**2)
    
    a = -1.0
    b = 1.0
    
    for N in [40, 50]:
        
        xint = np.linspace(a, b, N)
        yint = f(xint)
        
        V = Vandermonde(xint, N)
        
        Vinv = inv(V)
        coef = Vinv @ yint
        
        Neval = 1000
        xeval = np.linspace(a, b, Neval + 1)
        
        yeval = eval_monomial(xeval, coef, N - 1, Neval)
        
        yex = f(xeval)
        
        plt.figure(figsize=(8, 5))
        plt.plot(xint, yint, 'ro', label=f'N={N}', zorder=5)
        plt.plot(xeval, yeval, 'b-', label='p(x)', linewidth=2)
        plt.plot(xeval, yex, 'k--', label='Exact f(x)', alpha=0.7)
        plt.ylim(-5, 5)
        plt.xlim(-1.1, 1.1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f"Runge's Phenomenon with Monomial Basis (N = {N})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

def problem_2():
    f = lambda x: 1.0 / (1.0 + (10*x)**2)
    
    a = -1.0
    b = 1.0

    for N in [40, 50]:
        xint = np.linspace(a, b, N)
        yint = f(xint)

        w = compute_barycentric_weights(xint)

        Neval = 1000
        xeval = np.linspace(a, b, Neval + 1)
        yeval = eval_barycentric(xeval, xint, yint, w)

        yex = f(xeval)

        plt.figure(figsize=(8, 5))
        plt.plot(xint, yint, 'ro', label=f'N={N}', zorder=5)
        plt.plot(xeval, yeval, 'b-', label='p(x)', linewidth=2)
        plt.plot(xeval, yex, 'k--', label='Exact f(x)', alpha=0.7)

        plt.ylim(-5, 5)
        plt.xlim(-1.1, 1.1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f"Barycentric Lagrange Interpolation (N = {N})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

def problem_3():
    f = lambda x: 1.0 / (1.0 + (10*x)**2)
    
    a = -1.0
    b = 1.0

    for N in [40, 50]:
        j = np.arange(1, N + 1)
        xint = np.cos((2 * j - 1) * np.pi / (2 * N))
        yint = f(xint)

        # Uncomment for monomial basis interpolation
        # V = Vandermonde(xint, N)
        
        # Vinv = inv(V)
        # coef = Vinv @ yint
        
        # Neval = 1000
        # xeval = np.linspace(a, b, Neval + 1)
        
        # yeval = eval_monomial(xeval, coef, N - 1, Neval)

        # Uncomment for barycentric Lagrange interpolation
        w = compute_barycentric_weights(xint)

        Neval = 1000
        xeval = np.linspace(a, b, Neval + 1)
        yeval = eval_barycentric(xeval, xint, yint, w)

        yex = f(xeval)

        plt.figure(figsize=(8, 5))
        plt.plot(xint, yint, 'ro', label=f'N={N}', zorder=5)
        plt.plot(xeval, yeval, 'b-', label='p(x)', linewidth=2)
        plt.plot(xeval, yex, 'k--', label='Exact f(x)', alpha=0.7)

        plt.ylim(-5, 5)
        plt.xlim(-1.1, 1.1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f"Barycentric Lagrange Interpolation w/ Chebyshev Points (N = {N})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def eval_monomial(xeval,coef,N,Neval):

    yeval = coef[0]*np.ones(Neval+1)
    
    for j in range(1,N+1):
      for i in range(Neval+1):
        yeval[i] = yeval[i] + coef[j]*xeval[i]**j

    return yeval

   
def Vandermonde(xint,N):

    V = np.zeros((N,N))
    
    for j in range(N):
        V[j][0] = 1.0
        for i in range(1, N):
            V[j][i] = xint[j]**i

    return V     

def compute_barycentric_weights(xint):
    N = len(xint)
    w = np.zeros(N)
    for j in range(N):
        # w_j = 1 / product_{i != j} (x_j - x_i)
        diffs = xint[j] - xint[np.arange(N) != j]
        w[j] = 1.0 / np.prod(diffs)
    return w

def eval_barycentric(xeval, xint, yint, w):
    yeval = np.zeros_like(xeval)
    
    for idx, x in enumerate(xeval):
        match = np.where(np.isclose(x, xint))[0]
        if len(match) > 0:
            yeval[idx] = yint[match[0]]
        else:
            numer = np.sum(w * yint / (x - xint))
            denom = np.sum(w / (x - xint))
            yeval[idx] = numer / denom
            
    return yeval

# problem_1()
# problem_2()
problem_3()