import numpy as np

params = {
    'D': 0.8,
    'Sin': 2.0,
    'μmax': 1.0,
    'ks': 0.2
}

def mu(s, μmax=1.0, ks=0.2, **kw):
    return (μmax*s) / (ks + s)

def mu_inv(r, μmax=1.0, ks=0.2, **kw):
    return r*ks/(μmax - r)

def mu_deriv(s, μmax=1.0, ks=0.2, **kw):
    return (μmax*ks) / (ks + s)**2

def growth_rate(s, x, D=1.0, Sin=2.0, μmax=1.0, ks=0.2):
    μ = mu(s, μmax=μmax, ks=ks)
    ds = -μ*x + D*(Sin - s)
    dx = (μ - D)*x
    return ds, dx

def isocline_ds(s, D=1.0, Sin=2.0, μmax=1.0, ks=0.2):
    return D*(Sin - s) / mu(s, μmax=μmax, ks=ks)

# plots
def plot_isocline_dx(ax, **kwargs):
    xmin, xmax, ymin, ymax = ax.axis()
    color = kwargs.pop('color', 'b')
    label = kwargs.pop('label', '$\\dot{x}(t)=0$')
    D = kwargs.get('D', None)

    ax.axhline(y=0, xmin=xmin, xmax=xmax, color=color, label=label)
    if D:
        ax.axvline(x=mu_inv(D, **kwargs), ymin=ymin, ymax=ymax, color=color)
    return ax


def plot_isocline_ds(ax, s=None, step=0.01, **kwargs):
    xmin, xmax, ymin, ymax = ax.axis()
    color = kwargs.pop('color', 'r')
    label = kwargs.pop('label', '$\\dot{s}(t)=0$')
    D = kwargs.get('D', None)

    if not s:
        smin = xmin if xmin > 0 else step
        s = np.arange(smin, xmax, step)
    xs = isocline_ds(s, **kwargs)
    ax.plot(s, xs, color=color, label=label)
    ax.axis([xmin, xmax, ymin, ymax])
    return ax