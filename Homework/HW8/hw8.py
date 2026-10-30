import numpy as np
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
import matplotlib.pyplot as plt

def problem_2():
    f = lambda x: 1./(1. + x**2)
    fp = lambda x: -2*x/(1.+x**2)**2

    a = -5.0
    b = 5.0

    for N in [5, 10, 15, 20]:

        j = np.arange(N + 1)
        xint = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(
            (2 * j + 1) / (2 * (N + 1)) * np.pi
        )
        xint = np.sort(xint)

        ''' create equispaced interpolation nodes'''
        # xint = np.linspace(a,b,N+1)

        ''' create interpolation data'''
        yint = f(xint)
        ypint = fp(xint)

        ''' create points for evaluating the Lagrange interpolating polynomial'''
        Neval_g = 1000
        xeval = np.linspace(a,b,Neval_g+1)
        yeval_l= np.zeros(Neval_g+1) # lagrange
        yeval_h = np.zeros(Neval_g+1) # hermite
        
        ''' evaluate lagrange poly '''
        for kk in range(Neval_g+1):
            yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
            yeval_h[kk] = eval_hermite(xeval[kk], xint, yint, ypint, N)

        (M_n, C_n, D_n) = create_natural_spline(yint, xint, N) 
        yeval_cs = eval_cubic_spline(xeval, Neval_g, xint, N, M_n, C_n, D_n)
        (M_c, C_c, D_c) = create_clamped_spline(yint, ypint, xint, N) 
        yeval_ccs = eval_cubic_spline(xeval, Neval_g, xint, N, M_c, C_c, D_c)

        ''' create vector with exact values'''
        fex = f(xeval)

        # Plot 1: Interpolation vs Exact Function
        plt.figure(figsize=(8, 4))
        plt.plot(xeval, fex, 'r-', label='Exact f(x)', linewidth=2)
        plt.plot(xeval, yeval_l, 'b--', label=f'Lagrange (N={N})', linewidth=1.5)
        plt.plot(xeval, yeval_h, 'c--', label=f'Hermite (N={N})', linewidth=1.5)
        plt.plot(xeval, yeval_cs, 'm--', label=f'Cubic Spline (N={N})', linewidth=1.5)
        plt.plot(xeval, yeval_ccs, 'y--', label=f'Clamped Cubic Spline (N={N})', linewidth=1.5)
        plt.plot(xint, yint, 'ko', label='Interpolation Nodes')
        plt.title(f'Lagrange vs. Hermite vs. Cubic Spline vs. Clamped Cubic Spline Interpolation with N = {N}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)

        # Plot 2: Absolute Error on Log Scale
        plt.figure(figsize=(8, 4))
        err_l = np.abs(yeval_l - fex)
        err_h = np.abs(yeval_h - fex)
        err_cs = np.abs(yeval_cs - fex)
        err_ccs = np.abs(yeval_ccs - fex)
        plt.semilogy(xeval, err_l, 'b--', label=f'Lagrange Error (N={N})')
        plt.semilogy(xeval, err_h, 'c--', label=f'Hermite Error (N={N})')
        plt.semilogy(xeval, err_cs, 'm--', label=f'Cubic Spline Error (N={N})')
        plt.semilogy(xeval, err_ccs, 'y--', label=f'Clamped Cubic Spline Error (N={N})')
        plt.title(f'Absolute Error (Semilog) for N = {N}')
        plt.xlabel('x')
        plt.ylabel('|Error|')
        plt.legend()
        plt.grid(True)
        plt.show()

def problem_4():
    # Example for a periodic function on [0, 2*pi]
    f = lambda x: np.sin(10 * x) + 1
    fp = lambda x: 10 * np.cos(10 * x)

    a = 0.0
    b = 2 * np.pi
    for N in [20, 30, 40, 50]:

        xint = np.linspace(a, b, N + 1)
        yint = f(xint)

        # Neval_g evaluation points
        Neval_g = 1000
        xeval = np.linspace(a,b,Neval_g+1)

        # Create and evaluate the periodic cubic spline
        (M_p, C_p, D_p) = create_periodic_spline(yint, xint, N)
        yeval_ps = eval_cubic_spline(xeval, Neval_g, xint, N, M_p, C_p, D_p)

        fex = f(xeval)

        # Plot 1: Interpolation vs Exact Function
        plt.figure(figsize=(8, 4))
        plt.plot(xeval, fex, 'r-', label='Exact f(x)', linewidth=2)
        plt.plot(xeval, yeval_ps, 'b--', label=f'Periodic Spline (N={N})', linewidth=1.5)
        plt.plot(xint, yint, 'ko', label='Interpolation Nodes')
        plt.title(f'Periodic Spline Interpolation with N = {N}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)

        # Plot 2: Absolute Error on Log Scale
        plt.figure(figsize=(8, 4))
        err_ps = np.abs(yeval_ps - fex)
        plt.semilogy(xeval, err_ps, 'b--', label=f'Periodic Spline Error (N={N})')
        plt.title(f'Absolute Error (Semilog) for N = {N}')
        plt.xlabel('x')
        plt.ylabel('|Error|')
        plt.legend()
        plt.grid(True)
        plt.show()

def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)

    for count in range(N+1):
        for jj in range(N+1):
            if (jj != count):
                lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.

    for jj in range(N+1):
        yeval = yeval + yint[jj]*lj[jj]

    return(yeval)

def eval_hermite(xeval,xint,yint,ypint,N):

    ''' Evaluate all Lagrange polynomials'''

    lj = np.ones(N+1)
    for count in range(N+1):
        for jj in range(N+1):
            if (jj != count):
                lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    ''' Construct the l_j'(x_j)'''
    lpj = np.zeros(N+1)
    # lpj2 = np.ones(N+1)
    for count in range(N+1):
        for jj in range(N+1):
            if (jj != count):
                # lpj2[count] = lpj2[count]*(xint[count] - xint[jj])
                lpj[count] = lpj[count]+ 1./(xint[count] - xint[jj])

    yeval = 0.

    for jj in range(N+1):
        Qj = (1.-2.*(xeval-xint[jj])*lpj[jj])*lj[jj]**2
        Rj = (xeval-xint[jj])*lj[jj]**2
        # if (jj == 0):
        # print(Qj)
        # print(Rj)
        # print(Qj)
        # print(xeval)
        # return

        yeval = yeval + yint[jj]*Qj+ypint[jj]*Rj

    return(yeval)

def create_natural_spline(yint,xint,N):
    # create the right hand side for the linear system
    b = np.zeros(N+1)
    # vector values
    h = np.zeros(N+1)
    for i in range(1,N):
        hi = xint[i]-xint[i-1]
        hip = xint[i+1] - xint[i]
        b[i] = (yint[i+1]-yint[i])/hip - (yint[i]-yint[i-1])/hi
        h[i-1] = hi
        h[i] = hip

    # create matrix so you can solve for the M values
    # This is made by filling one row at a time
    A = np.zeros((N+1,N+1))
    A[0][0] = 1.0
    for j in range(1,N):
        A[j][j-1] = h[j-1]/6
        A[j][j] = (h[j]+h[j-1])/3
        A[j][j+1] = h[j]/6
    A[N][N] = 1

    Ainv = inv(A)

    M = Ainv.dot(b)

    # Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
        C[j] = yint[j]/h[j]-h[j]*M[j]/6
        D[j] = yint[j+1]/h[j]-h[j]*M[j+1]/6

    return(M,C,D)

def create_clamped_spline(yint, ypint, xint, N):
    # Create the right hand side for the linear system (size N+1)
    b = np.zeros(N+1)
    h = np.zeros(N)
    
    for i in range(N):
        h[i] = xint[i+1] - xint[i]

    # Clamped boundary condition at x_0
    b[0] = (3.0 / h[0]) * ((yint[1] - yint[0]) / h[0] - ypint[0])
    
    # Interior nodes
    for i in range(1, N):
        b[i] = (3.0 / h[i]) * (yint[i+1] - yint[i]) / h[i] - (3.0 / h[i-1]) * (yint[i] - yint[i-1]) / h[i-1]

    # Clamped boundary condition at x_N
    b[N] = (3.0 / h[N-1]) * (ypint[N] - (yint[N] - yint[N-1]) / h[N-1])

    # Create tridiagonal matrix A for M values
    A = np.zeros((N+1, N+1))
    
    # Row 0: Clamped condition for M[0] and M[1]
    A[0, 0] = 2.0 * h[0]
    A[0, 1] = h[0]

    # Interior rows
    for j in range(1, N):
        A[j, j-1] = h[j-1]
        A[j, j] = 2.0 * (h[j-1] + h[j])
        A[j, j+1] = h[j]

    # Row N: Clamped condition for M[N-1] and M[N]
    A[N, N-1] = h[N-1]
    A[N, N] = 2.0 * h[N-1]

    # Solve for M coefficients
    Ainv = inv(A)
    M = Ainv.dot(b)

    # Create the linear coefficients C and D
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
        C[j] = yint[j]/h[j] - h[j]*M[j]/6.0
        D[j] = yint[j+1]/h[j] - h[j]*M[j+1]/6.0

    return (M, C, D)

def eval_local_spline(xeval,xi,xip,Mi,Mip,C,D):
# Evaluates the local spline as defined in class
# xip = x_{i+1}; xi = x_i
# Mip = M_{i+1}; Mi = M_i

    hi = xip-xi
    yeval = (Mi*(xip-xeval)**3 +(xeval-xi)**3*Mip)/(6*hi) \
            + C*(xip-xeval) + D*(xeval-xi)
    
    return yeval

def eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D):

    yeval = np.zeros(Neval+1)

    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j]
        btmp= xint[j+1]

# find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]

# evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,M[j],M[j+1],C[j],D[j])
# print('yloc = ', yloc)
# copy into yeval
        yeval[ind] = yloc

    return(yeval)

def create_periodic_spline(yint, xint, N):
  # Step sizes
  h = np.diff(xint)

  # Construct the cyclic tridiagonal matrix A and right-hand side vector d
  # Size of system is N (since M_0 = M_N, we solve for M_0 through M_{N-1})
  A = np.zeros((N, N))
  d = np.zeros(N)

  for i in range(N):
    im1 = (i - 1) % N
    ip1 = (i + 1) % N

    h_im1 = h[im1]
    h_i = h[i]

    A[i, im1] = h_im1 / 6.0
    A[i, i] = (h_im1 + h_i) / 3.0
    A[i, ip1] = h_i / 6.0

    d[i] = (yint[ip1] - yint[i]) / h_i - (yint[i] - yint[im1]) / h_im1

  # Solve for moments (M_0 to M_{N-1})
  M_reduced = np.linalg.solve(A, d)

  # Enforce periodic boundary condition: M_N = M_0
  M = np.append(M_reduced, M_reduced[0])

  # Compute C and D matching `eval_local_spline` notation:
  # yeval = (Mi*(xip-xeval)**3 +(xeval-xi)**3*Mip)/(6*hi) + C*(xip-xeval) + D*(xeval-xi)
  # Therefore: C_j = yint[j]/h[j] - h[j]*M[j]/6, D_j = yint[j+1]/h[j] - h[j]*M[j+1]/6
  C = np.zeros(N)
  D = np.zeros(N)
  for j in range(N):
    C[j] = yint[j] / h[j] - h[j] * M[j] / 6.0
    D[j] = yint[j + 1] / h[j] - h[j] * M[j + 1] / 6.0

  return M, C, D

# problem_2()
problem_4()