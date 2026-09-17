from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt

x_bar = 2.5
x = np.linspace(0, x_bar, 1000)
t = 60 * 24 * 3600
alpha = 1.38e-7
f_2 = lambda x: erf(x / (2 * np.sqrt(alpha * t))) - 3/7
f_2_prime = lambda x: (1/np.sqrt(np.pi * alpha * t)) * np.exp(-(x**2)/(4 * alpha * t))

def bisection(f,a,b,tol):
# Inputs:
# f,a,b - function and endpoints of initial interval
# tol - bisection stops when interval length < tol
# Returns:
# astar - approximation of root
# ier - error message
# - ier = 1 => Failed
# - ier = 0 == success

# first verify there is a root we can find in the interval
    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
        ier = 1
        astar = a
        return [astar, ier]

    if (fa == 0):
        astar = a
        ier =0
        return [astar, ier]

    if (fb ==0):
        astar = b
        ier = 0
        return [astar, ier]

    count = 0
    d = 0.5*(a+b)

    while (abs(d-a)> tol):
        fd = f(d)
        if (fd ==0):
            astar = d
            ier = 0
            return [astar, ier]
        
        if (fa*fd<0):
            b = d
        else:
            a = d
            fa = fd

        d = 0.5*(a+b)
        count = count +1
# print('abs(d-a) = ', abs(d-a))
    astar = d
    ier = 0
    print('count = ', count)
    return [astar, ier]

def newton(f,fp,p0,tol,Nmax):
    """
    Newton iteration.
    Inputs:
    f,fp - function and derivative
    p0 - initial guess for root
    tol - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
    Returns:
    p - an array of the iterates
    pstar - the last iterate
    info - success message
    - 0 if we met tol
    - 1 if we hit Nmax iterations (fail)
    """
    p = np.zeros(Nmax+1);
    p[0] = p0
    for it in range(Nmax):
        p1 = p0-f(p0)/fp(p0)
        p[it+1] = p1
        if (abs(p1-p0) < tol):
            pstar = p1
            info = 0
            return [p,pstar,info,it]
        p0 = p1
    pstar = p1
    info = 1
    return [p,pstar,info,it]

def p_2_a():
    plt.plot(x, f_2(x))
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

def p_2_b():
    a0 = 0
    b0 = x_bar
    tol = 1e-13

    [a_star, ier] = bisection(f_2, a0, b0, tol)
    print('the approximate root is: ', a_star)
    print('the error message reads: ', ier)
    print('f(a_star) = ', f_2(a_star))

def p_2_c():
    x0 = 0.01
    tol = 1e-13
    Nmax = 100

    [x_iterate, x_star, ier, count] = newton(f_2, f_2_prime, x0, tol, Nmax)
    print('number of iterations: ', count)
    print('sequence of iterates: ', x_iterate)
    print('approximate root: ', x_star)
    print('error message reads: ', ier)

# p_2_a()
# p_2_b()
p_2_c()