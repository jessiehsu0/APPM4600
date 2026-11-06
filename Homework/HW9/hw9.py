import numpy as np
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
import matplotlib.pyplot as plt

def driver():
    maclaurin = lambda x: x - x**3 / 6 + x**5 / 120
    f1 = lambda x: (60*x - 7*x**3) / (60 + 3*x**2)
    f2 = lambda x: (60*x - 7*x**3) / (60 + 3*x**2)
    f3 = lambda x: (360*x) / (360 + 60*x**2 + 7*x**4)

    x = np.linspace(0, 5, 1000)
    y_exact = np.sin(x)

    err_mac = np.abs(maclaurin(x) - y_exact)
    err_33 = np.abs(f1(x) - y_exact)
    err_42 = np.abs(f2(x) - y_exact)
    err_24 = np.abs(f3(x) - y_exact)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, err_mac, label='Maclaurin $P_6(x)$', color='black', linestyle='--')
    plt.plot(x, err_33, label='Padé [3/3]', color='blue')
    plt.plot(
      x,
      err_42,
      label='Padé [4/2] (Identical to [3/3])',
      color='green',
      linestyle=':',
      linewidth=3,
    )
    plt.plot(x, err_24, label='Padé [2/4]', color='red')

    # Logarithmic scale to clearly show orders-of-magnitude error differences
    plt.yscale('log')
    plt.title(
      'Absolute Error Comparison for $\sin(x)$ Approximations over $[0, 5]$',
      fontsize=14,
    )
    plt.xlabel('$x$', fontsize=12)
    plt.ylabel('Absolute Error (Log Scale)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

driver()