import numpy as np
import math
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def sign(x):
    if x > 0:
        return 1
    else:
        if x==0:
            return 0
        return -1

print(sign(1))

def SDGL(t, Z, R, g, mu, nu):
    x, v, alpha, omega = Z
    dxdt = v
    dvdt = (math.exp(-sign(v)*nu*alpha)*mu*(x*omega**2-g*math.cos(alpha))-g)/(1+mu*math.exp(-sign(v)*nu*alpha))-R*(g/x*math.sin(alpha)-(2*v+R*omega)*omega/x)
    dalphadt = omega
    domegadt = g/x*math.sin(alpha)-(2*v+R*omega)*omega/x
    return [dxdt, dvdt, dalphadt, domegadt]

# Parameters
g = 9.81
mu = 0.01
nu = 0.1
R = 0.01
l = 10

# Initial conditions: [x, v, alpha, omega]
Z0 = [l, 0, math.pi/2, 0]

# Time range
t_span = (0, 20)  # from t=0 to t=20
t_eval = np.linspace(*t_span, 300)  # times at which to store the solution

# Solve the system
sol = solve_ivp(SDGL, t_span, Z0, args=(R, g, mu, nu), t_eval=t_eval)

# Set up the figure and axis for animation
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

# Plot for the pendulum motion
line1, = ax1.plot([], [], 'b-', label='Pendulum tip')
line2, = ax1.plot([], [], 'r-', label='Mass position')
ax1.set_xlim(-l, l)
ax1.set_ylim(-l, l)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()

# Plot for x vs t
line3, = ax2.plot([], [], 'g-', label='x vs t')
ax2.set_xlim(t_span)
ax2.set_ylim(-l, l)
ax2.set_xlabel('Time (t)')
ax2.set_ylabel('Position (x)')
ax2.legend()

# Initialize the data arrays
x_data, y1_data, x2_data, y2_data, t_data, x_vs_t_data = [], [], [], [], [], []

# Variable to store the previous alpha value
previous_alpha = Z0[2]

def animate(i):
    t = sol.t[i]
    x_pos = sol.y[0, i]
    alpha = sol.y[2, i]

    mass_x = R
    mass_y = R*alpha + x_pos - l
    
    pendulum_x = R*math.cos(alpha) - x_pos*math.sin(alpha)
    pendulum_y = R*math.sin(alpha) + x_pos*math.cos(alpha) 


    x_data.append(pendulum_x)
    y1_data.append(pendulum_y)

    x2_data.append(mass_x)  # x-coordinate of mass position is constantly R
    y2_data.append(mass_y)

    t_data.append(t)
    x_vs_t_data.append(x_pos)

    line1.set_data(x_data, y1_data)
    line2.set_data(x2_data, y2_data)
    line3.set_data(t_data, x_vs_t_data)


    return line1, line2, line3

# Create the animation object
animate.ani = FuncAnimation(fig, animate, frames=len(t_eval), interval=50, blit=True)

# Run the animation
plt.show()


