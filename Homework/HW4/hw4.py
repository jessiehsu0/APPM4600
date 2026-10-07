import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# Define function and derivative
def f(x):
    return x**6 - x - 1

def df(x):
    return 6*x**5 - 1

# 1. Compute the exact root alpha to high precision
res = root_scalar(f, fprime=df, x0=1.1, method='newton', xtol=1e-15, rtol=1e-15)
alpha = res.root
print(f"Exact root alpha = {alpha:.15f}")

# 2. Newton's Method
x_newton = [2.0]
for _ in range(10):
    x_curr = x_newton[-1]
    x_next = x_curr - f(x_curr) / df(x_curr)
    x_newton.append(x_next)
    if abs(x_next - alpha) < 1e-15:
        break

# 3. Secant Method
x_secant = [2.0, 1.0]
for _ in range(12):
    x_p, x_c = x_secant[-2], x_secant[-1]
    x_n = x_c - f(x_c) * (x_c - x_p) / (f(x_c) - f(x_p))
    x_secant.append(x_n)
    if abs(x_n - alpha) < 1e-15:
        break

# 4. Calculate absolute errors e_k = |x_k - alpha|
e_newton = [abs(x - alpha) for x in x_newton]
e_secant = [abs(x - alpha) for x in x_secant]

# 5. Prepare error arrays e_k and e_{k+1} for plotting
e_k_N = np.array(e_newton[:-1])
e_k1_N = np.array(e_newton[1:])
# Filter out values below machine precision threshold
valid_N = (e_k_N > 1e-14) & (e_k1_N > 1e-14)

e_k_S = np.array(e_secant[:-1])
e_k1_S = np.array(e_secant[1:])
valid_S = (e_k_S > 1e-14) & (e_k1_S > 1e-14)

# Compute asymptotic slopes via linear fit on log-transformed data
slope_N, _ = np.polyfit(np.log(e_k_N[valid_N]), np.log(e_k1_N[valid_N]), 1)
slope_S, _ = np.polyfit(np.log(e_k_S[valid_S]), np.log(e_k1_S[valid_S]), 1)

# 6. Plot log-log graph
plt.figure(figsize=(8, 6))

plt.loglog(e_k_N[valid_N], e_k1_N[valid_N], 'o-', 
           label=f"Newton's Method (Slope ≈ {slope_N:.2f})", 
           color='crimson', linewidth=2, markersize=7)

plt.loglog(e_k_S[valid_S], e_k1_S[valid_S], 's--', 
           label=f"Secant Method (Slope ≈ {slope_S:.2f})", 
           color='navy', linewidth=2, markersize=7)

plt.xlabel(r'Error at step $k$: $e_k = |x_k - \alpha|$', fontsize=12)
plt.ylabel(r'Error at step $k+1$: $e_{k+1} = |x_{k+1} - \alpha|$', fontsize=12)
plt.title(r'Log-Log Plot of Error Convergence: $|x_{k+1} - \alpha|$ vs $|x_k - \alpha|$', fontsize=13)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()

# Save and display figure
plt.savefig('error_loglog_plot.png', dpi=300)
plt.show()