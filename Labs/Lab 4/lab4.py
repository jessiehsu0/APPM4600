import numpy as np

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

def modified_bisection(f,f_prime, f_2_prime, a,b,Nmax, tol):
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

    while (((f(d) * f_2_prime(d))/(f_prime(d)**2)) >= 1):
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
    # return [astar, ier]

    p0 = d
    p = np.zeros(Nmax+1)
    p[0] = p0
    for it in range(Nmax):
        p1 = p0-f(p0)/f_prime(p0)
        p[it+1] = p1
        if (abs(p1-p0) < tol):
            pstar = p1
            info = 0
            return [p,pstar,info,it]
        p0 = p1
    pstar = p1
    info = 1
    return [p,pstar,info,it]

def driver():
    f = lambda x: np.exp(x**2 + 7*x - 30) - 1
    f_prime = lambda x: (2*x + 7) * np.exp(x**2 + 7*x - 30)
    f_double_prime = lambda x: (((2*x + 7) ** 2) * np.exp(x**2 + 7*x - 30)) + (2 * np.exp(x**2 + 7*x - 30))
    a = 2
    b = 4.5
    x0 = 4.5
    Nmax = 100
    tol = 1e-10

    [x_iterate, x_star, ier, count] = modified_bisection(f,f_prime, f_double_prime, a, b, Nmax, tol)
    print('================BISECTION================')
    [astar, ier_b] = bisection(f, a, b, tol)
    print('approximate root: ', astar)
    print('error message reads: ', ier_b)
    print('f(astar): ', f(astar))

    print('================NEWTON\'S================')
    [p, p_star, ier_n, count_n] = newton(f, f_prime, x0, tol, Nmax)
    print('number of iterations: ', count_n)
    print('sequence of iterates: ', p)
    print('approximate root: ', p_star)
    print('error message reads: ', ier_n)
    print('f(xstar): ', f(x_star))


    print ('================MODIFIED BISECTION================')
    print('number of iterations: ', count)
    print('sequence of iterates: ', x_iterate)
    print('approximate root: ', x_star)
    print('error message reads: ', ier)
    print('f(xstar): ', f(x_star))

    # print('the approximate root is',astar)
    # print('the error message reads:',ier)
    # print('f(astar) =', f(astar))

driver()