import numpy as np
from matplotlib import pyplot as plt

from matplotlib import rcParams
#rcParams['text.usetex'] = True

def phase_portrait(xlim, ylim, dF, mesh=1.0, step=0.1,
                   ax=None, dF_params=None, **kwargs):
    """
    Plots 2D phase portrait
    """
    if not ax:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    kwargs.pop('equal_axes', True) and ax.axes.set_aspect('equal')
    xmesh, ymesh = mesh if type(mesh) is tuple else (mesh, mesh)
    xstep, ystep = step if type(step) is tuple else (step, step)
    title = kwargs.pop("title", None)
    xlabel = kwargs.pop("xlabel", None)
    ylabel = kwargs.pop("ylabel", None)
    scale = kwargs.pop("scale", 1)
    alpha = kwargs.pop("alpha", 0.2)
    density = kwargs.pop("density", 10)
    
    # plot vector field
    x = np.arange(xlim[0], xlim[1], xmesh)
    y = np.arange(ylim[0], ylim[1], ymesh)
    X, Y = np.meshgrid(x, y)
    dX, dY = dF(X, Y, **dF_params) if dF_params else dF(X, Y)
    Q = ax.quiver(X, Y, dX, dY, angles='xy', scale_units='xy', scale=scale,
        alpha=alpha, **kwargs)
    
    # plot stream plot
    x = np.arange(xlim[0], xlim[1], xstep)
    y = np.arange(ylim[0], ylim[1], ystep)
    X, Y = np.meshgrid(x, y)
    dX, dY = dF(X, Y, **dF_params) if dF_params else dF(X, Y)
    SP = ax.streamplot(X, Y, dX, dY, density=density, **kwargs)

    # legend
    if title:
        fig.suptitle(title)
    if xlabel:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    return ax

def phase_portrait_3D(xlim, ylim, dF, z_func, z_axis=2, **kwargs):
    def f_2D(x, y, **dF_params):
        if z_axis == 0:
            _, dx, dy = dF(z_func(x, y, **dF_params), x, y, **dF_params)
        elif z_axis == 1:
            dx, _, dy = dF(x, z_func(x, y, **dF_params), y, **dF_params)
        elif z_axis == 2:
            dx, dy, _ = dF(x, y, z_func(x, y, **dF_params), **dF_params)
        else:
            raise ValueError('fixed_axis should be in [0, 1, 2]')
        return dx, dy
    return phase_portrait(xlim, ylim, f_2D, **kwargs)

def plot_eigenspaces(u, v, pt=[0,0], t=[-1,1], ax=None, **kwargs):
    if ax:
        xmin, xmax, ymin, ymax = ax.axis()
    else:
        _, ax = plt.subplots()
        xmin, xmax = t[0] + pt[0], t[-1] + pt[1]
        ymin, ymax = t[0] + pt[0], t[-1] + pt[1]
    t = np.linspace(t[0], t[-1], 3)
    ax.plot(u[0]*t + pt[0], u[1]*t + pt[1], label='$\\lambda_1$')
    ax.plot(v[0]*t + pt[0], v[1]*t + pt[1], label='$\\lambda_2$')
    ax.axis([xmin, xmax, ymin, ymax])
    return ax

def set_lims(center=[0,0], size=2):
    c = np.array(center)
    d = np.array(size) if hasattr(size, '__len__') else size * np.ones(len(center))
    d /= 2
    xlim = center[0] - d[0], center[0] + d[0]
    ylim = center[1] - d[1], center[1] + d[1]
    return xlim, ylim

def plot_equilibria(eq, name=None, ax=None, **kwargs):
    if not ax: ax = plt.subplots()
    color = kwargs.pop('color', 'black')
    marker = kwargs.pop('marker', 'o')
    E = np.array(eq)
    ax.plot(E[:,0], E[:,1], color=color, marker=marker, linestyle='', **kwargs)
    if name and len(name) == len(eq):
        fontsize = kwargs.pop('fontsize', 12)
        for i in range(len(eq)):
            ax.text(E[i,0] + 0.02, E[i,1] + 0.02, name[i], fontsize=fontsize)
    return ax

def plot_equilibria_from_3D(eq, z_axis=2, **kwargs):
    eq2 = []
    indices = [[1, 2], [0, 2], [0, 1]]
    ind = indices[z_axis]
    for pt in eq:
        eq2.append([pt[ind[0]], pt[ind[1]]])
    return plot_equilibria(eq2, **kwargs)