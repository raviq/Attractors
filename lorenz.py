
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy import integrate
from matplotlib.colors import cnames
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint as ODEint
import random

N_trajectories = 2
N = 1000 # number of points
M = 3 # x, y, z

def lorentz_deriv(XYZ, t0, sigma=10., beta=8./3, rho=28.0):
    # Compute the time-derivative of a Lorentz system.
    x, y, z = XYZ
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


# initialization function: plot the background of each frame
def init():
    for line, pt in zip(lines, pts):
        line.set_data([], [])
        line.set_3d_properties([])
        pt.set_data([], [])
        pt.set_3d_properties([])
    return lines + pts

# Choose random starting points, uniformly distributed from -15 to 15
#np.random.seed(1)
x0 = -15 + 30 * np.random.random((N_trajectories, M))

print ('Starting point for each trajectory (', N_trajectories,' trajectories):\n', x0, '\n')

# Solve for the trajectories
t = np.linspace(0, 4, N)
x_t = np.asarray([integrate.odeint(lorentz_deriv, x0i, t) for x0i in x0])

print ('(N_trajectories, N, M) = ', x_t.shape)

x_t[:, :, 2] = 0 # set z to 0

# traj 0 and 1
#plt.plot(x_t[0, :, 0], x_t[0, :, 1], 'b.')
#plt.plot(x_t[1, :, 0], x_t[1, :, 1], 'r.')



# Set up figure & 3D axis for animation
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.axis('off')

# choose a different color for each trajectory
colors = plt.cm.jet(np.linspace(0, 1, N_trajectories))

# set up lines and points
lines = sum([ax.plot([], [], [], '-', c=c) for c in colors], [])
pts = sum([ax.plot([], [], [], '.', c=c) for c in colors], [])

# prepare the axes limits
ax.set_xlim((-25, 25))
ax.set_ylim((-35, 35))
ax.set_zlim((5, 55))

# set point-of-view: specified by (altitude degrees, azimuth degrees)
ax.view_init(30, 0)

# animation function.  This will be called sequentially with the frame number
def animate(i):
    # we'll step two time-steps per frame.  This leads to nice results.
    i = (2 * i) % x_t.shape[1]

    for line, pt, xi in zip(lines, pts, x_t):
        x, y, z = xi[:i].T
        line.set_data(x, y)
        line.set_3d_properties(z)

        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    ax.view_init(30, 0.3 * i)
    fig.canvas.draw()
    return lines + pts

# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=500, interval=30, blit=True)

# Save as mp4. This requires mplayer or ffmpeg to be installed
#anim.save('lorentz_attractor.mp4', fps=15, extra_args=['-vcodec', 'libx264'])

plt.show()


################################################################################


exit()

y, z = 10.* np.cos(x), 10.*np.sin(x) # something simple

fig = plt.figure()
ax = fig.add_subplot(1,2,1,projection='3d')
ax.plot(x, y, z)

# now Lorentz
times = np.linspace(0, 4, 1000) 

start_pts = 30. - 15.*np.random.random((20,3))  # 20 random xyz starting values

trajectories = []
for start_pt in start_pts:
    trajectory = ODEint(lorentz_deriv, start_pt, times)
    trajectories.append(trajectory)

ax = fig.add_subplot(1,2,2, projection='3d')

for trajectory in trajectories:
    x, y, z = trajectory.T  # transpose and unpack 
    # x, y, z = zip(*trajectory)  # this also works!
    ax.plot(x, y, z)

plt.show()