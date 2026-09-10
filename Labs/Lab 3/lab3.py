import numpy as np

def driver():
    # test functions
    f1 = lambda x: 1+0.5*np.sin(x)
    # fixed point is alpha1 = 1.4987....
    f2 = lambda x: 3+2*np.sin(x)
    #fixed point is alpha2 = 3.09...

    Nmax = 100
    tol = 1e-6

    # test f1 '''
    x0 = 0.0
    [xstar1,ier] = fixedpt(f1,x0,tol,Nmax)
    print('the sequence of approximated fixed points is: ', xstar1)
    print('Error message reads: ', ier)

    [papprox1, ier] = aitkens(xstar1, tol, Nmax)
    print('the aitkens sequence of approximated fixed points is: ', papprox1)
    print('Error message reads: ', ier)

    #test f2 '''
    x0 = 0.0
    [xstar2,ier] = fixedpt(f2, x0, tol, Nmax)
    print('the sequence of approximated fixed points is: ', xstar2)
    print('Error message reads:', ier)

    [papprox2, ier] = aitkens(xstar2, tol, Nmax)
    print('the aitkens sequence of approximated fixed points is: ', papprox2)
    print('Error message reads: ', ier)

def fixedpt(f, x0, tol, Nmax):
    x = np.zeros((Nmax + 1, 1))
    x[0] = x0

    count = 0

    while (count < Nmax):
        count += 1
        x1 = f(x0)
        x[count] = x1

        if (abs(x1 - x0) < tol):
            ier = 0
            return [x, ier]

        x0 = x1

    ier = 1

    return [x, ier]

def aitkens(pn, tol, Nmax):
    p_approx = np.zeros((Nmax + 1, 1))
    p_approx[0] = pn[0]

    i = 0

    while (i < Nmax):
        i += 1
        pn_approx = pn[i] - ((pn[i + 1] - pn[i]) ** 2 / (pn[i + 2] - 2 * pn[i + 1] + pn[i]))
        p_approx[i] = pn_approx

        if (abs(pn_approx - p_approx[i - 1]) < tol):
            ier = 0
            return [p_approx, ier]

    ier = 1

    return [p_approx, ier]

driver()