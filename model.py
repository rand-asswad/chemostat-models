import numpy as np

params = {
    'D': 0.8,
    'Sin': 2.0,
    'μmax': 1.0,
    'ks': 0.2,
    'Y': 1.0,
    'kd': 0,
    'km': 0,
}

def mu(s, μmax=1.0, ks=0.2, **kw):
    return (μmax*s) / (ks + s)

def mu_inv(r, μmax=1.0, ks=0.2, **kw):
    return r*ks/(μmax - r)

def mu_deriv(s, μmax=1.0, ks=0.2, **kw):
    return (μmax*ks) / (ks + s)**2

def growth_rate(s, x, D=1.0, Sin=2.0, μmax=1.0, ks=0.2,
                Y=1.0, km=0, kd=0):
    μ = mu(s, μmax=μmax, ks=ks)
    ds = -(μ/Y + km)*x + D*(Sin - s)
    dx = (μ - D - kd)*x
    return ds, dx

def isocline_ds(s, D=1.0, Sin=2.0, μmax=1.0, ks=0.2, Y=1.0, km=0, **kw):
    return D*(Sin - s) / (mu(s, μmax=μmax, ks=ks)/Y + km)

def equilibria(D=1.0, Sin=2.0, kd=0, **kw):
    E = [[Sin, 0]]
    D_kd = D + kd
    if D_kd < mu(Sin, **kw) and D_kd < kw['μmax']:
        s1 = mu_inv(D_kd, **kw)
        x1 = isocline_ds(s1, D=D, Sin=Sin, **kw)
        E.append([s1, x1])
    return E

# plots
def plot_isocline_dx(ax, **kwargs):
    xmin, xmax, ymin, ymax = ax.axis()
    color = kwargs.pop('color', 'b')
    label = kwargs.pop('label', '$\\dot{x}(t)=0$')
    D = kwargs.get('D', None)
    kd = kwargs.get('kd', 0)

    ax.axhline(y=0, xmin=xmin, xmax=xmax, color=color, label=label)
    if D and (D + kd) < kwargs['μmax']:
        ax.axvline(x=mu_inv(D + kd, **kwargs), ymin=ymin, ymax=ymax, color=color)
    return ax


def plot_isocline_ds(ax, s=None, step=0.01, **kwargs):
    xmin, xmax, ymin, ymax = ax.axis()
    color = kwargs.pop('color', 'r')
    label = kwargs.pop('label', '$\\dot{s}(t)=0$')

    if not s:
        smin = xmin if xmin > 0 else step
        s = np.arange(smin, xmax, step)
    xs = isocline_ds(s, **kwargs)
    ax.plot(s, xs, color=color, label=label)
    ax.axis([xmin, xmax, ymin, ymax])
    return ax

def plot_sigma(ax, s=None, step=0.01, **kwargs):
    xmin, xmax, ymin, ymax = ax.axis()
    color = kwargs.pop('color', 'orange')
    label = kwargs.pop('label', '$\\dot{\\sigma}(t)=0$')

    if not s:
        smin = xmin if xmin > 0 else step
        s = np.arange(smin, xmax, step)
    x = kwargs['Y'] * (kwargs['Sin'] - s)
    ax.plot(s, x, color=color, label=label)
    ax.axis([xmin, xmax, ymin, ymax])
    return ax