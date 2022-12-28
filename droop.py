import numpy as np

params = {
    'D': 0.8,
    'Sin': 2.0,
    'μmax': 1.0,
    'ρmax': 1.0,
    'qmin': 0.5,
    'ks': 0.2,
    'kq': 0.5,
}

def rho(s, ρmax=1.0, ks=0.2, **kw):
    return (ρmax*s) / (ks + s)

def rho_inv(r, ρmax=1.0, ks=0.2, **kw):
    return r*ks/(ρmax - r)

def rho_deriv(s, ρmax=1.0, ks=0.2, **kw):
    return (ρmax*ks) / (ks + s)**2

def mu(q, μmax=1.0, qmin=0.5, kq=0.5, **kw):
    if isinstance(q, np.ndarray):
        Q = np.zeros(q.shape)
        ind = q > qmin
        Q[ind] = q[ind] - qmin
    else:
        Q = max(q - qmin, 0)
    return (μmax*Q) / (kq + Q)

def mu_inv(d, μmax=1.0, qmin=0.5, kq=0.5, **kw):
    if d == 0:
        return 0 # q in [0,qmin]
    return qmin + d*kq/(μmax - d)

def mu_deriv(q, μmax=1.0, qmin=0.5, kq=0.5, **kw):
    if q < qmin:
        return 0
    return (μmax*kq) / (kq + max(q - qmin, 0))**2

def f(s, x, q, D=1.0, Sin=2.0, μmax=1.0, ks=0.2,
      ρmax=1.0, qmin=0.5, kq=0.5):
    ρ = rho(s, ρmax=ρmax, ks=ks)
    μ = mu(q, μmax=μmax, qmin=qmin, kq=kq)
    ds = -ρ*x + D*(Sin - s)
    dx = (μ - D)*x
    dq = ρ - q*μ
    return ds, dx, dq

def isocline_ds_get_x(s, D=1.0, Sin=2.0, ρmax=1.0, ks=0.2, **kw):
    return D*(Sin - s) / rho(s, ρmax=ρmax, ks=ks)

def isocline_dq_get_q(s, **kw):
    a = kw['qmin'] + rho(s, **kw)/kw['μmax']
    b = 4 * (kw['qmin'] - kw['kq']) * rho(s, **kw)/kw['μmax']
    return (a + np.sqrt(a**2 - b)) / 2

def isocline_dq_get_s(q, **kw):
    if isinstance(q, np.ndarray):
        s = np.zeros(q.shape)
        ind = q > kw['qmin']
        qmuq = q[ind] * mu(q[ind], **kw)
        s[ind] = kw['ks'] * qmuq / (kw['ρmax'] - qmuq)
        return s
    if hasattr(q, "__len__"):
        return isocline_dq_get_s(np.array(q), **kw)
    if q <= kw['qmin']:
        return 0
    qmuq = q * mu(q, **kw)
    return kw['ks'] * qmuq / (kw['ρmax'] - qmuq)

def equilibria(**kw):
    q0 = isocline_dq_get_q(kw['Sin'], **kw)
    E = [[kw['Sin'], 0, q0]]
    D = kw['D']
    if D < kw['μmax']:
        q1 = mu_inv(D, **kw)
        if q1 > 0 and q1 < kw['ρmax'] and q1 < q0:
            s1 = rho_inv(q1*D, **kw)
            x1 = isocline_ds_get_x(s1, **kw)
            if s1 > 0 and x1 > 0:
                E.append([s1, x1, q1])
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
    xs = isocline_ds_get_x(s, **kwargs)
    ax.plot(s, xs, color=color, label=label)
    ax.axis([xmin, xmax, ymin, ymax])
    return ax