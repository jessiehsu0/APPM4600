import numpy as np
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
import matplotlib.pyplot as plt
import scipy.integrate as integrate

"""
This script tests the use of different composite quadrature methods
for approximating integrals. There is an option to adjust the commented
lines to illustrate convergence when the number of panels is increased.
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
def driver():
    f = lambda x: np.exp(x) * np.cos(x)
    a = 0
    b = np.pi

    # exact integral
    I_ex = -0.5 * (np.exp(np.pi) + 1)
    tol = 1e-10

    n_trap_list = []
    errorT_list = []
    n = 2

    while True:
        I_trap = CompTrap(a, b, n, f)
        abs_err = abs(I_ex - I_trap)
        rel_err = abs_err / abs(I_ex)
        
        n_trap_list.append(n)
        errorT_list.append(abs_err)
        
        if rel_err < tol:
            break
        n *= 2

    n_simp_list = []
    errorS_list = []
    n = 2
    while True:
        I_simp = CompSimp(a, b, n, f)
        abs_err = abs(I_ex - I_simp)
        rel_err = abs_err / abs(I_ex)
        
        n_simp_list.append(n)
        errorS_list.append(abs_err)
        
        if rel_err < tol:
            break
        n *= 2

# Convert lists to arrays and calculate h for each method
    n_trap = np.array(n_trap_list)
    errorT = np.array(errorT_list)
    h_trap = (b - a) / n_trap

    n_simp = np.array(n_simp_list)
    errorS = np.array(errorS_list)
    h_simp = (b - a) / n_simp

    # Print Part (a) results
    print(f"Trapezoidal Rule reached rel error < 1e-10 at n = {n_trap[-1]} (h = {h_trap[-1]:.5e})")
    print(f"  Final Absolute Error: {errorT[-1]:.5e}")
    print(f"  Final Relative Error: {errorT[-1]/abs(I_ex):.5e}\n")

    print(f"Simpson's Rule reached rel error < 1e-10 at n = {n_simp[-1]} (h = {h_simp[-1]:.5e})")
    print(f"  Final Absolute Error: {errorS[-1]:.5e}")
    print(f"  Final Relative Error: {errorS[-1]/abs(I_ex):.5e}\n")

    # --- Part (b): Slopes on log-log scale ---
    # Slope formula between last two steps: log(E_k / E_{k-1}) / log(h_k / h_{k-1})
    slopeT = np.log(errorT[-1] / errorT[-2]) / np.log(h_trap[-1] / h_trap[-2])
    slopeS = np.log(errorS[-1] / errorS[-2]) / np.log(h_simp[-1] / h_simp[-2])

    print("Observed slope (Trapezoidal):", round(slopeT, 4))
    print("Observed slope (Simpson):    ", round(slopeS, 4))

    # Log-Log Plot: Absolute Error vs h
    plt.figure(figsize=(8, 6))
    plt.loglog(h_trap, errorT, 'o-', label='Composite Trapezoidal')
    plt.loglog(h_simp, errorS, 's-', label='Composite Simpson')
    
    # Reference lines for O(h^2) and O(h^4)
    plt.loglog(h_trap, errorT[0] * (h_trap / h_trap[0])**2, 'k--', label=r'Reference $O(h^2)$')
    plt.loglog(h_simp, errorS[0] * (h_simp / h_simp[0])**4, 'k:', label=r'Reference $O(h^4)$')

    plt.xlabel('Subinterval length $h$')
    plt.ylabel('Absolute Error')
    plt.title('Convergence Analysis: Absolute Error vs. $h$')
    plt.gca().invert_xaxis()  # Invert x-axis so h decreases to the right
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.show()

def prob_2():
    f = lambda s: 1.0 / (1.0 + s**2)
    a = -5.0
    b = 5.0
    exact_val = 2.0 * np.arctan(5.0)

    n_trap = 1291
    n_simp = 110

    T_val = CompTrap(a, b, n_trap, f)
    feval_trap = n_trap + 1

    S_val = CompSimp(a, b, n_simp, f)
    feval_simp = n_simp + 1

    res_q4, err_q4, info_q4 = integrate.quad(f, a, b, epsabs=1e-4, epsrel=1e-4, full_output=True)
    neval_q4 = info_q4['neval']

    res_q6, err_q6, info_q6 = integrate.quad(f, a, b, full_output=True)
    neval_q6 = info_q6['neval']

    print(f"Exact Integral Value       : {exact_val:.10f}\n")
    
    print(f"Trapezoidal Rule (n = {n_trap}):")
    print(f"  Computed Value           : {T_val:.10f}")
    print(f"  Absolute Error           : {abs(exact_val - T_val):.2e}")
    print(f"  Function Evaluations     : {feval_trap}\n")

    print(f"Simpson's Rule (n = {n_simp})    :")
    print(f"  Computed Value           : {S_val:.10f}")
    print(f"  Absolute Error           : {abs(exact_val - S_val):.2e}")
    print(f"  Function Evaluations     : {feval_simp}\n")

    print(f"SciPy quad (tol = 1e-4)    :")
    print(f"  Computed Value           : {res_q4:.10f}")
    print(f"  Function Evaluations     : {neval_q4}\n")

    print(f"SciPy quad (default tol)   :")
    print(f"  Computed Value           : {res_q6:.10f}")
    print(f"  Function Evaluations     : {neval_q6}")

def CompTrap(a,b,n,f):
    h = (b-a)/n
    xnode = a+np.arange(0,n+1)*h

    I_trap = h * (0.5 * f(xnode[0]) + np.sum(f(xnode[1:n])) + 0.5 * f(xnode[n]))

    return I_trap

def CompSimp(a,b,n,f):
    h = (b-a)/n
    xnode = a+np.arange(0,n+1)*h
    I_simp = f(xnode[0])

    nhalf = n/2
    for j in range(1,int(nhalf)+1):
        # odd part
        I_simp += 4*f(xnode[2*j-1])
        # even part (interior only; x_n is added below)
        if j < nhalf:
            I_simp = I_simp+2*f(xnode[2*j])

    I_simp += f(xnode[n])
    return (h/3) * I_simp

# driver()
prob_2()