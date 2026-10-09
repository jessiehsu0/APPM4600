import numpy as np
import math
import time
from numpy.linalg import inv
from numpy.linalg import norm

def problem_1():
    tol = 1e-4
    Nmax = 100

    x0_i = np.array([-0.5, 0.25])
    x0_ii = np.array([-1.5, 1.25])
    x0_iii = np.array([-3, 3])

    t = time.time()
    for j in range(50):
        [xstar, ier, its] = fixedSys_1(x0_i, tol, Nmax)
    elapsed = time.time() - t

    print("Fixed Point: took this many seconds: ", elapsed)
    # print("Fixed Point: approximation is: ", xstar)
    # print("Fixed Point: error message reads: ", ier)
    # print("Fixed Point: number of iterations is: ", its)

    t = time.time()
    for j in range(50):
        [xstar,ier,its] = Newton_1(x0_i, tol, Nmax)
    elapsed = time.time() - t

    print("Newton's: took this many seconds: ", elapsed)
    # print("Newton's: approximation is: ", xstar)
    # print("Newton's: error message reads: ", ier)
    # print("Newton's: number of iterations is: ", its)

def problem_2():
    tol = 1e-10
    Nmax = 100

    x0_i = np.array([1, 1])
    x0_ii = np.array([1, -1])
    x0_iii = np.array([0, 0])

    t = time.time()
    for j in range(50):
        [xstar, ier, its] = Newton_2(x0_iii, tol, Nmax)
    elapsed = time.time() - t

    print("Newton's: took this many seconds: ", elapsed)
    print("Newton's: approximation is: ", xstar)
    print("Newton's: error message reads: ", ier)
    print("Newton's: number of iterations is: ", its)

    t = time.time()
    for j in range(50):
        [xstar, ier, its] = LazyNewton_2(x0_iii, tol, Nmax)
    elapsed = time.time() - t

    print("Lazy Newton's: took this many seconds: ", elapsed)
    print("Lazy Newton's: approximation is: ", xstar)
    print("Lazy Newton's: error message reads: ", ier)
    print("Lazy Newton's: number of iterations is: ", its)

    t = time.time()
    for j in range(50):
        [xstar, ier, its] = Broyden_2(x0_iii, tol, Nmax)
    elapsed = time.time() - t

    print("Broyden: took this many seconds: ", elapsed)
    print("Broyden: approximation is: ", xstar)
    print("Broyden: error message reads: ", ier)
    print("Broyden: number of iterations is: ", its)

def problem_3():
    tol = 1e-6
    Nmax = 100

    x0 = np.array([0.5, 0.5, 0.5])

    t = time.time()
    for j in range(50):
        [xstar, ier, its] = Newton_3(x0, tol, Nmax)
    elapsed = time.time() - t

    print("Newton's: took this many seconds: ", elapsed)
    print("Newton's: approximation is: ", xstar)
    print("Newton's: error message reads: ", ier)
    print("Newton's: number of iterations is: ", its)

    t = time.time()
    for j in range(50):
        [xstar, gval, ier, its] = SteepestDescent_3(x0, tol, Nmax)
    elapsed = time.time() - t

    print("Steepest Descent: took this many seconds: ", elapsed)
    print("Steepest Descent: found the solution: ", xstar)
    print("Steepest Descent: g evaluated at this point is: ", gval)
    print("Steepest Descent: error message reads: ", ier)
    print("Steepest Descent: number of iterations is: ", its)

    t = time.time()
    for j in range(50):
        [xstar, ier, its] = SteepestDescent_Newton_3(x0, tol, Nmax)
    elapsed = time.time() - t

    print("Hybrid: took this many seconds: ", elapsed)
    print("Hybrid: approximation is: ", xstar)
    print("Hybrid: error message reads: ", ier)
    print("Hybrid: number of iterations is: ", its)

def evalF_1(x):
    F = np.zeros(2)

    F[0] = 3*x[0]**2 + 4*x[1]**2 - 1 # 3x^2 + 4y^2 - 1
    F[1] = x[1]**3 - 8*x[0]**3 - 1 # y^3 - 8x^3 - 1

    return F

def evalJ_1(x):
    J = np.array([[6*x[0], 8*x[1]],
        [-24*x[0]**2, 3*x[1]**2]])

    return J

def evalF_2(x):
    F = np.zeros(2)

    F[0] = x[0]**2 + x[1]**2 - 4 # x^2 + y^2 - 4
    F[1] = np.exp(x[0]) + x[1] - 1 # e^x + y - 1

    return F

def evalJ_2(x):
    J = np.array([[2*x[0], 2*x[1]],
        [np.exp(x[0]), 1]])

    return J

def evalF_3(x):
    F = np.zeros(3)

    F[0] = x[0] + np.cos(x[0]*x[1]*x[2]) - 1 # x + cos(xyz) - 1
    F[1] = (1-x[0])**(0.25) + x[1] + 0.05*x[2]**2 - 0.15*x[2] - 1 # (1-x)^1/4 + y + 0.05z^2 - 0.15z - 1
    F[2] = -x[0]**2 - 0.1*x[1]**2 + 0.01*x[1] + x[2] - 1 # -x^2 - 0.1y^2 + 0.01y + z - 1

    return F

def evalJ_3(x):
    J = np.array([[1 - x[1]*x[2]*np.sin(x[0]*x[1]*x[2]), -x[0]*x[2]*np.sin(x[0]*x[1]*x[2]), -x[0]*x[1]*np.sin(x[0]*x[1]*x[2])],
        [-0.25*(1-x[0])**(-0.75), 1, 0.1*x[2] - 0.15],
        [-2*x[0], -0.2*x[1] + 0.01, 1]])

    return J

def evalG_3(x):
    F = evalF_3(x)
    g = F[0]**2 + F[1]**2 + F[2]**2
    return g

def evalGradG_3(x):
    F = evalF_3(x)
    J = evalJ_3(x)

    gradg = np.transpose(J).dot(F)
    return gradg

def fixedSys_1(x0, tol, Nmax):
    M = np.array([[0.016, -0.17],
                  [0.52, -0.26]])

    # print('Iteration | x')
    # print(0, ' | ', x0)
    for its in range(Nmax):
        x1 = x0 - M.dot(evalF_1(x0))
        # print(its+1, ' | ', x1)

        if np.any(np.isinf(x1)) or np.any(np.isnan(x1)):
            xstar = x1
            ier = 1 
            return [xstar, ier, its+1]

        if (norm(x1 - x0) < tol):
            xstar = x1
            ier = 0
            return[xstar, ier, its+1]
        
        x0 = x1

    xstar = x1
    ier = 1
    return[xstar, ier, its]

def Newton_1(x0, tol, Nmax):
    # print('Iteration | x')
    # print(0, ' | ', x0)

    for its in range(Nmax):
        J = evalJ_1(x0)
        Jinv = inv(J)
        F = evalF_1(x0)

        x1 = x0 - Jinv.dot(F)
        # print(its+1, ' | ', x1)

        if (norm(x1 - x0) < tol):
            xstar = x1
            ier = 0
            return[xstar, ier, its+1]

        x0 = x1

    xstar = x1
    ier = 1
    return[xstar, ier, its]

def Newton_2(x0, tol, Nmax):

    for its in range(Nmax):
        J = evalJ_2(x0)
        Jinv = inv(J)
        F = evalF_2(x0)

        x1 = x0 - Jinv.dot(F)

        if (norm(x1 - x0) < tol):
            xstar = x1
            ier = 0
            return[xstar, ier, its+1]

        x0 = x1

    xstar = x1
    ier = 1
    return[xstar, ier, its]

def LazyNewton_2(x0,tol,Nmax):
    J = evalJ_2(x0)
    Jinv = inv(J)

    for its in range(Nmax):
        F = evalF_2(x0)
        x1 = x0 - Jinv.dot(F)

        if (norm(x1-x0) < tol):
            xstar = x1
            ier =0
            return[xstar, ier,its+1]
        
        x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar, ier, its]

def Broyden_2(x0,tol,Nmax):
    A0 = evalJ_2(x0)

    v = evalF_2(x0)
    A = np.linalg.inv(A0)

    s = -A.dot(v)
    xk = x0+s
    for its in range(Nmax):
        w = v
        v = evalF_2(xk)
        y = v-w
        z = -A.dot(y)
        p = -np.dot(s,z)
        u = np.dot(s,A)
        tmp = s+z
        tmp2 = np.outer(tmp,u)
        A = A+1./p*tmp2
        s = -A.dot(v)
        xk = xk+s
        if (norm(s)<tol):
            alpha = xk
            ier = 0
            return[alpha,ier,its+1]
        alpha = xk
        ier = 1
    return[alpha, ier, its]

def Newton_3(x0, tol, Nmax):
    for its in range(Nmax):
        J = evalJ_3(x0)
        Jinv = inv(J)
        F = evalF_3(x0)

        x1 = x0 - Jinv.dot(F)

        if (norm(x1 - x0) < tol):
            xstar = x1
            ier = 0
            return[xstar, ier, its+1]

        x0 = x1

    xstar = x1
    ier = 1
    return[xstar, ier, its]

def SteepestDescent_3(x,tol,Nmax):
    for its in range(Nmax):
        g1 = evalG_3(x)
        z = evalGradG_3(x)
        z0 = norm(z)

        if z0 == 0:
            print("zero gradient")
        z = z/z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3*z
        g3 = evalG_3(dif_vec)

        while g3>=g1:
            alpha3 = alpha3/2
            dif_vec = x - alpha3*z
            g3 = evalG_3(dif_vec)

        if alpha3<tol:
            print("no likely improvement")
            ier = 0
            return [x,g1,ier, its]

        alpha2 = alpha3/2
        dif_vec = x - alpha2*z
        g2 = evalG_3(dif_vec)

        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3

        alpha0 = 0.5*(alpha2 - h1/h3)
        dif_vec = x - alpha0*z
        g0 = evalG_3(dif_vec)

        if g0<=g3:
            alpha = alpha0
            gval = g0
        else:
            alpha = alpha3
            gval =g3

        x = x - alpha*z

        if abs(gval - g1)<tol:
            ier = 0
            return [x,gval,ier, its+1]

    print('max iterations exceeded')
    ier = 1
    return [x,g1,ier, its]

def SteepestDescent_Newton_3(x,tol,Nmax):
    for its in range(Nmax):
        g1 = evalG_3(x)
        z = evalGradG_3(x)
        z0 = norm(z)

        if z0 == 0:
            print("zero gradient")
        z = z/z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3*z
        g3 = evalG_3(dif_vec)

        while g3>=g1:
            alpha3 = alpha3/2
            dif_vec = x - alpha3*z
            g3 = evalG_3(dif_vec)

        if alpha3<tol:
            print("no likely improvement")
            ier = 0
            return [x,g1,ier]

        alpha2 = alpha3/2
        dif_vec = x - alpha2*z
        g2 = evalG_3(dif_vec)

        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3

        alpha0 = 0.5*(alpha2 - h1/h3)
        dif_vec = x - alpha0*z
        g0 = evalG_3(dif_vec)

        if g0<=g3:
            alpha = alpha0
            gval = g0
        else:
            alpha = alpha3
            gval =g3

        x = x - alpha*z

        if abs(gval - g1)<5e-2:
            [xstar, ier, newton_its] = Newton_3(x, tol, Nmax)
            return [xstar, ier, (its+1) + newton_its]

    print('max iterations exceeded')
    ier = 1
    return [x,g1,ier]

# problem_1()
# problem_2()
problem_3()