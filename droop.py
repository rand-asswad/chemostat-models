import numpy as np

params = {
    'D': 0.8,
    'Sin': 2.0,
    'μmax': 1.0,
    'ρmax': 1.0,
    'qmin': 0.5,
    'ks': 0.2,
    'kq': 0.5,
    'Y': 1.0,
}

def rho(s, ρmax=1.0, ks=0.2, **kw):
    return (ρmax*s) / (ks + s)

def rho_deriv(s, ρmax=1.0, ks=0.2, **kw):
    return (ρmax*ks) / (ks + s)**2

def mu(q, μmax=1.0, qmin=0.5, kq=0.5, **kw):
    Q = max(q - qmin, 0)
    return (μmax*Q) / (kq + Q)

def mu_inv(r, μmax=1.0, qmin=0.5, kq=0.5, **kw):
    if r == 0:
        return 0 # q in [0,qmin]
    return qmin + r*kq/(μmax - r)

def mu_deriv(q, μmax=1.0, qmin=0.5, kq=0.5, **kw):
    if q < qmin:
        return 0
    return (μmax*kq) / (kq + max(q - qmin, 0))**2

def f(s, x, q, D=1.0, Sin=2.0, μmax=1.0, ks=0.2,
      ρmax=1.0, qmin=0.5, kq=0.5, Y=1.0):
    ρ = rho(s, ρmax=ρmax, ks=ks)
    μ = mu(q, μmax=μmax, qmin=qmin, kq=kq)
    ds = -(ρ/Y)*x + D*(Sin - s)
    dx = (μ - D)*x
    dq = ρ - q*μ
    return ds, dx, dq

def isocline_dx(s, D=1.0, Sin=2.0, **kw):
    return (Sin - s) / mu_inv(D, **kw)

def isocline_ds(s, D=1.0, Sin=2.0, ρmax=1.0, ks=0.2, Y=1.0, **kw):
    return (Y*D)*(Sin - s) / rho(s, ρmax=ρmax, ks=ks)

def equilibria(D=1.0, Sin=2.0, kd=0, **kw):
    E = [[Sin, 0]]
    D_kd = D + kd
    if D_kd < mu(Sin, **kw) and D_kd < kw['μmax']:
        s1 = mu_inv(D_kd, **kw)
        x1 = isocline_ds(s1, D=D, Sin=Sin, **kw)
        E.append([s1, x1])
    return E

# plots
def plot_isocline_dx(ax, s=None, step=0.01, **kwargs):
    xmin, xmax, ymin, ymax = ax.axis()
    color = kwargs.pop('color', 'b')
    label = kwargs.pop('label', '$\\dot{x}(t)=0$')
    D = kwargs.get('D', None)

    if not s:
        smin = xmin if xmin > 0 else step
        s = np.arange(smin, xmax, step)

    ax.axhline(y=0, xmin=xmin, xmax=xmax, color=color, label=label)
    if D and D < kwargs['μmax']:
        #ax.axvline(x=mu_inv(D, **kwargs), ymin=ymin, ymax=ymax, color=color)
        xs = isocline_dx(s, **kwargs)
        ax.plot(s, xs, color=color)
    ax.axis([xmin, xmax, ymin, ymax])
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