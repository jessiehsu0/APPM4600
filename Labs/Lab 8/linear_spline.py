"""
    This script provides you the opprotunity to insert your own line evaluator.
    You will practice some python skills while getting a better understanding of
    how splines work.
    NOTE: this code will not compile until you add the line evaluator in the
    appropriate location.
"""
#############################################
"""
Copyright (C) 2025 Adrianna M. Gillman
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
#############################################

import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv

def driver():
    f = lambda x: 1 / (1 + (10*x)**2)
    a = -1
    b = 1

    # create points you want to evaluate at
    Neval = 100
    xeval = np.linspace(a,b,Neval)

    # number of intervals
    Nint = 10

    # evaluate the linear spline
    yeval = eval_lin_spline(xeval, Neval, a,b,f,Nint)

    # evaluate f at the evaluation points
    fex = f(xeval)

    plt.plot(xeval,fex,'ro-',label='exact function')
    plt.plot(xeval,yeval,'bs--',label='natural spline')
    plt.legend()
    plt.show()

    err = abs(yeval-fex)
    plt.figure()
    plt.semilogy(xeval,err,'ro--',label='absolute error')
    plt.legend()
    plt.show()

def eval_line(x0, x1, fx0, fx1, alpha):
    slope = (fx1 - fx0) / (x1 - x0)
    return fx0 + slope * (alpha - x0)

def eval_lin_spline(xeval, Neval, a,b,f,Nint):
    # create the intervals for piecewise approximations
    xint = np.linspace(a,b,Nint+1)

    # create vector to store the evaluation of the linear splines
    yeval = np.zeros(Neval)

    for jint in range(Nint):
        # temporarily store your info for creating a line in the interval of interest
        a1= xint[jint]
        fa1 = f(a1)
        b1 = xint[jint+1]
        fb1 = f(b1)

        # find indices of xeval in interval (xint[jint], xint[jint + 1])
        # include the right boundary on the final interval to capture the endpoint x = b
        if jint == Nint - 1:
            ind = np.where((xeval >= a1) & (xeval <= b1))[0]
        else:
            ind = np.where((xeval >= a1) & (xeval < b1))[0]

        n = len(ind)
        print(n)

        for kk in range(n):
            alpha = xeval[ind[kk]]
            yeval[ind[kk]] = eval_line(a1, b1, fa1, fb1, alpha)

    return yeval

if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver()