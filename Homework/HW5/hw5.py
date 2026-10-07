import numpy as np
import math
import time
from numpy.linalg import inv
from numpy.linalg import norm

def driver():
    x0 = np.array([1, 1])

    Nmax = 100
    tol = 1e-10

    t = time.time()
    for j in range(50):
        [xstar,ier,its] = fixedSys(x0, tol, Nmax)
    elapsed = time.time() - t

    print(xstar)
    print('Lazy Newton: the error message reads:',ier)
    print('Lazy Newton: took this many seconds:',elapsed/50)
    print('Lazy Newton: number of iterations is:',its)

    x0_surface = np.array([1, 1, 1])

    t = time.time()
    for j in range(50):
        [xstar,ier,its] = Newton(x0, tol, Nmax)
    elapsed = time.time() - t

    print(xstar)
    print('Newton: the error message reads:',ier)
    print('Newton: took this many seconds:',elapsed/50)
    print('Newton: number of iterations is:',its)


    print(f"{'Iter':<5} | {'x':<18} | {'y':<18} | {'z':<18} | {'f(x,y,z)':<20}")
    t = time.time()
    for j in range(50):
        [xstar,ier,its] = projectSurface(x0_surface, tol, Nmax)
    elapsed = time.time() - t

    print(xstar)
    print('Surface Projection: the error message reads:',ier)
    print('Surface Projection: took this many seconds:',elapsed/50)
    print('Surface Projection: number of iterations is:',its)

def evalF(x):
    F = np.zeros(2)

    F[0] = 3*x[0]**2 - x[1]**2
    F[1] = 3*x[0]*(x[1]**2) - x[0]**3 - 1

    return F

def evalJ(x):
    J = np.array([[6*x[0], -2*x[1]],
        [3*x[1]**2 - 3*x[0]**2, 6*x[0]*x[1]]])

    return J

def fixedSys(x0, tol, Nmax):
    M = np.array([[1/6, 1/18],
                  [0,   1/6]])

    for its in range(Nmax):
        x1 = x0 - M.dot(evalF(x0))
        if (norm(x1 - x0) < tol):
            xstar = x1
            ier = 0
            return[xstar, ier, its]
        x0 = x1

    xstar = x1
    ier = 1
    return[xstar, ier, its]

def Newton(x0, tol, Nmax):
    for its in range(Nmax):
        J = evalJ(x0)
        Jinv = inv(J)
        F = evalF(x0)

        x1 = x0 - Jinv.dot(F)

        if (norm(x1 - x0) < tol):
            xstar = x1
            ier = 0
            return[xstar, ier, its]

        x0 = x1

    xstar = x1
    ier = 1
    return[xstar, ier, its]

def evalSurfaceF(x):
    return x[0]**2 + 4*x[1]**2 + 4*x[2]**2 - 16.0

def evalGradF(x):
    return np.array([2*x[0], 8*x[1], 8*x[2]])

def projectSurface(x0, tol, Nmax):
    for its in range(Nmax):
        f = evalSurfaceF(x0)
        grad = evalGradF(x0)

        print(f"{its:<5} | {x0[0]:<18.12f} | {x0[1]:<18.12f} | {x0[2]:<18.12f} | {f:<20.10e}")

        d = f / np.dot(grad, grad)

        x1 = x0 - d * grad

        if (norm(x1-x0) < tol):
            xstar = x1
            ier = 0
            return [xstar, ier, its]

        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, its]

driver()