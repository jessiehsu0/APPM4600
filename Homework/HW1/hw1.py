import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def problem_1():
    n = 0.001
    x = np.arange(1.920, 2.081, n)

    p1 = lambda x: x**9 - 18*x**8 + 144*x**7 - 672*x**6 + 2016*x**5 - 4032*x**4 + 5376*x**3 - 4608*x**2 + 2304*x - 512
    p2 = lambda x: (x-2)**9

    plt.plot(x, p1(x))
    plt.plot(x, p2(x))
    plt.xlabel('x')
    plt.ylabel('p(x)')

    plt.show()

def problem_5():
    delta = 10.0 ** np.arange(-16, 1)

    x1 = np.pi
    x2 = 10**6

    f_original = lambda x: np.cos(x + delta) - np.cos(x)
    f_stable = lambda x: -2.0 * np.sin(x + delta / 2.0) * np.sin(delta / 2.0)

    diff_1 = np.abs(f_stable(x1) - f_original(x1))
    diff_2 = np.abs(f_stable(x2) - f_original(x2))

    plt.semilogx(delta, diff_1, label = "x = pi")
    plt.semilogx(delta, diff_2, label = "x = 10^6")
    plt.xlabel('x')
    plt.ylabel('original - stable')

    plt.legend()
    plt.show()

# problem_1()
problem_5()