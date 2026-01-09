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
gamma = 0.0

def pendulum(t, state):
    x, y, vx, vy = state
    dxdt = vx
    dydt = vy
    dvxdt = -x/L**2 * ((L**2*(vx**2 + vy**2) - (x*vy - y*vx)**2)/(L**2 - x**2 - y**2)) - (g/L**2)*x*np.sqrt(L**2 - x**2 - y**2) - (gamma/m)*vx
    dvydt = -y/L**2 * ((L**2*(vx**2 + vy**2) - (x*vy - y*vx)**2)/(L**2 - x**2 - y**2)) - (g/L**2)*y*np.sqrt(L**2 - x**2 - y**2) - (gamma/m)*vy
    return [dxdt, dydt, dvxdt, dvydt]

timespan = (0, 120)
state0 = [1.0, 0.0, 0.0, 0.0]

# Better solver with higher accuracy

sol = solve_ivp(pendulum, timespan, state0, 
                t_eval=np.linspace(0, 120, 11000),
                method='DOP853',
                rtol=1e-11, atol=1e-11)


t = sol.t
x, y, vx, vy = sol.y

z = -np.sqrt(L**2 - x**2 - y**2)
vz = np.where(np.abs(z + L) > 0.001, -(x*vx + y*vy)/z, 0.0)

# Energies
K = 0.5 * m * (vx**2 + vy**2 + vz**2)
V = m * g * (z + L)
E = K + V
E0 = E[0]

# Plot

plt.figure(figsize=(8,6))
if gamma == 0:
    plt.plot(t, (E-E0)/E0*100, 'b-', linewidth=2)  # Plot relative error instead
    plt.ylim(-0.005, 0.005)
    plt.ylabel('Relative energy error (%)', fontsize=25)
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)
else:
    plt.plot(t, E, 'r', linewidth=2, label=r'$\gamma = 0.01$')
    plt.ylabel('Total energy $E$ (J)', fontsize=25)
    plt.legend(fontsize=22)
    
plt.xlabel('Time $t$ (s)', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=23)
plt.title("Energy Conservation (Noninear Model)", fontsize=25)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Max relative error: {np.max(np.abs((E-E0)/E0))*100:.6f}%")
