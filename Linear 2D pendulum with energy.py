import numpy as np
import ODE_RK4
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

# Parameters
# V0 and gamma are changed for different cases

m = 28.0
g = 9.81
L = 67.0
gamma = 0.01 # friction parameter
V0 = [1,0,0,0] # the initial condition [x0,vx0,y0,vy0]
dt = 0.0002 # time step
fig_dt = 0.01 # plotting interval 
t0 = 0 # start ime
t_end = 60.0 # end time

# Function to compute total mechanical energy

def energy(x,vx,y,vy):
    # Kinetic energy
    K = 0.5 * m * (vx**2 + vy**2)
    # Gravitational potential energy
    V = 0.5 * (g/L) * m * (x**2 + y**2)
    return K + V

# Equations

def F(t, V):
    x, vx, y, vy = V
    return np.array([vx, -(g/L)*x - gamma*vx, vy, -(g/L)*y - gamma*vy])

# Solve

t, V = ODE_RK4.runge_kutta_2nd_order_system(F, V0, t0, t_end, dt, fig_dt)

x = np.array([v[0] for v in V])
vx = np.array([v[1] for v in V])
y = np.array([v[2] for v in V])
vy = np.array([v[3] for v in V])

E = energy(x, vx, y, vy)
E0 = E[0]

# Plot

plt.figure(figsize=(8,6))

if gamma == 0:
    plt.plot(t, (E - E0)/E0 * 100, 'r', linewidth=2, label=r'$\frac{E-E_0}{E_0}$')
    plt.axhline(0, linestyle='--', color='blue', label = 'Zero reference')
    plt.ylabel(r'Relative energy error (%)', fontsize = 25)
    plt.legend(fontsize=22)
    plt.title("Energy Conservation (Linear Model)", fontsize=25)
else:
    plt.plot(t, E, 'r', linewidth=2)
    plt.ylabel('Total energy $E$ (J)', fontsize = 25)
    plt.title("Energy Decay (Linear Model)", fontsize=25)

plt.xlabel('Time $t$ (s)', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()








