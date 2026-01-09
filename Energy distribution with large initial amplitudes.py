from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

# Parameters

m = 28.0
g = 9.81
L = 67.0

def pendulum(t, state):
    x, y, vx, vy = state
    dxdt = vx
    dydt = vy
    dvxdt = -x/L**2 * ((L**2*(vx**2 + vy**2) - (x*vy - y*vx)**2)/(L**2 - x**2 - y**2)) - (g/L**2)*x*np.sqrt(L**2 - x**2 - y**2)
    dvydt = -y/L**2 * ((L**2*(vx**2 + vy**2) - (x*vy - y*vx)**2)/(L**2 - x**2 - y**2)) - (g/L**2)*y*np.sqrt(L**2 - x**2 - y**2)
    return [dxdt, dydt, dvxdt, dvydt]

amplitudes = [3, 10, 20]
colors = ['blue', 'green', 'red']
labels = ['3 m (Panthéon)', '10 m', '20 m']

# Total Energy Conservation
for A, color, label in zip(amplitudes, colors, labels):
    sol = solve_ivp(pendulum, (0, 40), [A, 0, 0, 0],
                   t_eval=np.linspace(0, 40, 2000),
                   method='DOP853', rtol=1e-10, atol=1e-10)
    
    t = sol.t
    x, y, vx, vy = sol.y
    
    # Calculate energies
    z = -np.sqrt(L**2 - x**2 - y**2)
    vz = -(x*vx + y*vy)/z
    
    K = 0.5 * m * (vx**2 + vy**2 + vz**2)
    P = m * g * (z + L)
    E_total = K + P
    
    plt.plot(t, E_total, color=color, linewidth=2.5, label=label, alpha=0.8)

plt.xlabel('Time (s)', fontsize=25)
plt.ylabel('Total Energy (J)', fontsize=25)
plt.legend(fontsize=20)
plt.grid(alpha=0.3)
plt.tick_params(labelsize=20)
plt.tight_layout()
plt.show()


# Energy Distribution (Kinetic energy and Potential energy over 40 seconds)

for A, color, label in zip(amplitudes, colors, labels):
    
    sol = solve_ivp(pendulum, (0, 40), [A, 0, 0, 0],
                   t_eval=np.linspace(0, 40, 2000),
                   method='DOP853', rtol=1e-10, atol=1e-10)
    
    t = sol.t
    x, y, vx, vy = sol.y
    
    z = -np.sqrt(L**2 - x**2 - y**2)
    vz = -(x*vx + y*vy)/z
    K = 0.5 * m * (vx**2 + vy**2 + vz**2)
    P = m * g * (z + L)

    plt.plot(t, K, color=color, linewidth=2.5, 
            label=f'{label} (Kinetic energy)', alpha=0.8, linestyle='-')
    plt.plot(t, P, color=color, linewidth=2.5, 
            label=f'{label} (Potential energy)', alpha=0.6, linestyle='--')

plt.xlabel('Time (s)', fontsize=25)
plt.ylabel('Energy (J)', fontsize=25)
plt.legend(fontsize=15, ncol=2, loc='upper right')
plt.grid(alpha=0.3)
plt.tick_params(labelsize=20)
plt.xlim([0, 40])
plt.title("Energy Distribution", fontsize=25)
plt.tight_layout()
plt.show()
