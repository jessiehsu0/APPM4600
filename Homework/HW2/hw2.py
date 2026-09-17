import math
import numpy as np
import matplotlib.pyplot as plt

# y = lambda x: math.e ** x
# print(y(9.999999995 * (10 ** -10)))
# print(y(9.999999995 * (10 ** -10)) - 1)

def driver():
    f = lambda x: x**3 + x - 4
    a = 1
    b = 4
    tol = 1e-3

    [astar,ier] = bisection(f,a,b,tol)

    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))

# define routines
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

def fixedpt(f,df,x0,tol,Nmax):
    ''' x0 = initial guess'''
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    count = 0
    while (count <Nmax):
        count = count +1
        x1 = f(x0)
        if ((abs(x1-x0)/abs(1 - df(x1))) <tol):
            xstar = x1
            ier = 0
            return [xstar,ier]
        x0 = x1
    xstar = x1
    ier = 1
    return [xstar, ier]

def problem_6_a():
    x = np.linspace(-1.5, 7.5, 1000)

    f = lambda x: x - 4 * np.sin(2 * x) - 3

    plt.plot(x, f(x))
    plt.axhline(0, color='black', linestyle='--')

    plt.xlabel('x')
    plt.ylabel('f(x)')

    plt.show()

def problem_6_b():
    f = lambda x: -np.sin(2 * x) + 5 * x / 4 - 0.75
    df = lambda x: -2*np.cos(2*x) + 5/4
    Nmax = 100
    tol = 1e-10

    x0 = 0.0
    [xstar, ier] = fixedpt(f, df, x0, tol, Nmax)
    print('the approximate fixed point is:',xstar)
    print('f(xstar):',f(xstar))
    print('Error message reads:',ier)

driver()
# problem_6_a()
problem_6_b()