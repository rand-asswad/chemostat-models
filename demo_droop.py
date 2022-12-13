import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from labellines import labelLines

Sin = 5.0
s = np.arange(0.05, 10, 0.05)
x = np.arange(0.05, 10, 0.05)
S, X = np.meshgrid(s, x)
Q = (Sin - S) / X

fig = plt.figure(figsize=(15,15))
fig.suptitle("Conservation surface $\\Sigma$")

ax = fig.add_subplot(221, projection='3d')
sigma = ax.plot_surface(S, X, Q, cmap=cm.viridis, linewidth=0, rstride=15, cstride=10)
fig.colorbar(sigma)

#ax = fig.add_subplot(122, projection='3d')
#ax.plot_wireframe(S, X, Q, rstride=15, cstride=20);
ax = fig.add_subplot(222)
ax.set_aspect('equal')
ax.pcolormesh(S, X, Q, cmap=cm.viridis)

ax = fig.add_subplot(223)
ax.set_title('$q = q_x(s) = \\frac{-1}{x}s + \\frac{Sin}{x}$')
ax.set_aspect('equal')
#for x in [0.01, 0.02, 0.05, 0.1, 0.2, 1]:
ax.set_xlabel('s')
ax.set_ylabel('q')
for xx in [0.001, 0.5, 1, 5, 100]:
    ax.plot(s, (Sin-s)/xx, label='$x={}$'.format(xx))
ax.axis([0, 10, -5, 5])
ax.legend()
#labelLines(ax.get_lines(), zorder=2.5);

ax = fig.add_subplot(224)
ax.set_title('$q = q_s(x) = \\frac{Sin-s}{x}$')
ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('q')
for ss in [0.5, 2, 4, 7, 20]:
    ax.plot(x, (Sin-ss)/x, label='$s={}$'.format(ss))
ax.plot(x, 0*x, label='$s=S_{in}$')
ax.axis([0, 10, -5, 5])
labelLines(ax.get_lines(), zorder=2.5)

plt.show()