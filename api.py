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
        ax.axes.set_aspect('equal')
    xmesh, ymesh = mesh if type(mesh) is tuple else mesh, mesh
    xstep, ystep = step if type(mesh) is tuple else step, step
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
    if not fig:
        fig = ax.get_figure()
    if title:
        fig.suptitle(title)
    if xlabel:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    return ax