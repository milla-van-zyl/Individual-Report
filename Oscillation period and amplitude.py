import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

# Parameters
g = 9.81
L = 67.0

def pendulum(t, state):
    x, y, vx, vy = state
    dxdt = vx
    dydt = vy
    dvxdt = -(g/L**2)*x*np.sqrt(L**2 - x**2 - y**2)
    dvydt = -(g/L**2)*y*np.sqrt(L**2 - x**2 - y**2)
    return [dxdt, dydt, dvxdt, dvydt]

# Initial amplitudes

amplitudes = np.linspace(1, 25, 10)
periods = []

for A in amplitudes:
    sol = solve_ivp(pendulum, (0, 200), [A, 0, 0, 0], t_eval=np.linspace(0, 200, 8000), method="DOP853", rtol=1e-10, atol=1e-10)
    t = sol.t
    x = sol.y[0]

    # Find maxima
    maxima_times = []
    for i in range(1, len(x)-1):
        if x[i-1] < x[i] and x[i] > x[i+1]:
            maxima_times.append(t[i])

    maxima_times = np.array(maxima_times)

    T = np.mean(np.diff(maxima_times[:6]))
    periods.append(T)

# Linear small-angle period

T0 = 2*np.pi*np.sqrt(L/g)

# Plot
plt.figure(figsize=(8,5))
plt.plot(amplitudes, periods, color = 'b', label='Numerical period')
plt.axhline(T0, linestyle='--', color='r', label='Small-angle period')
plt.xlabel('Initial amplitude A (m)', fontsize=25)
plt.ylabel('Oscillation period T (s)', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.title("Increase of oscillation preriod with amplitude", fontsize=25)
plt.legend(fontsize=25)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
