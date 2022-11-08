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
    ax.axes.set_aspect('equal')
    xmesh, ymesh = mesh if type(mesh) is tuple else mesh, mesh
    xstep, ystep = step if type(step) is tuple else step, step
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