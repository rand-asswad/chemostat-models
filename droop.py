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

def f(s, x, q, D=1.0, Sin=2.0, **kw):
    ρ = rho(s, **kw)
    μ = mu(q, **kw)
    ds = -ρ*x + D*(Sin - s)
    dx = (μ - D)*x
    dq = ρ - q*μ
    return ds, dx, dq

def isocline_ds_get_x(s, D=1.0, Sin=2.0, **kw):
    return D*(Sin - s) / rho(s, **kw)

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
        if q1 > 0 and q1*D < kw['ρmax'] and q1 < q0:
            s1 = rho_inv(q1*D, **kw)
            x1 = isocline_ds_get_x(s1, **kw)
            if s1 > 0 and x1 > 0:
                E.append([s1, x1, q1])
    return E

# plots
def plot_isoclines_sx(ax, **params):
    D = params['D']
    Sin = params['Sin']
    xmin, xmax, ymin, ymax = ax.axis()

    # s-nullcline
    s = np.arange(max(xmin,0.01), Sin, 0.01)
    ax.plot(s, isocline_ds_get_x(s, **params), color='r', label='$\\dot{s}=0$')
    # x-nullcline
    ax.axhline(y=0, xmin=xmin, xmax=xmax, color='b', label='$\\dot{x}=0$')
    x = (Sin - s) / mu_inv(D, **params)
    ax.plot(s, x, color='b')
    # q-nullcline
    x = (Sin - s) / isocline_dq_get_q(s, **params)
    ax.plot(s, x, color='orange', linestyle='dashed', label='$\\dot{q}=0$')
    
    ax.axis([xmin, xmax, ymin, ymax])
    return ax

def plot_isoclines_sq(ax, **params):
    D = params['D']
    Sin = params['Sin']
    xmin, xmax, ymin, ymax = ax.axis()
    
    # s-nullcline
    s = np.arange(max(xmin,0.01), xmax, 0.1)
    ax.plot(s, rho(s, **params) / D, color='r', label='$\\dot{s}=0$')
    ax.axvline(x=Sin, ymin=ymin, ymax=ymax, color='r')
    # x-nullcline
    ax.axhline(y=mu_inv(D, **params), xmin=xmin, xmax=xmax, color='b', linestyle='dashed', label='$\\dot{x}=0$')
    ax.axvline(x=Sin, ymin=ymin, ymax=ymax, color='b', linestyle=(0, (5, 10)))
    # q-nullcline
    ax.plot(s, isocline_dq_get_q(s, **params), color='orange', label='$\\dot{q}=0$')

    ax.axis([xmin, xmax, ymin, ymax])
    return ax

def plot_isoclines_xq(ax, **params):
    D = params['D']
    Sin = params['Sin']
    xmin, xmax, ymin, ymax = ax.axis()

    # x-nullcline
    ax.axhline(y=mu_inv(D, **params), xmin=xmin, xmax=xmax, color='b', label='$\\dot{x}=0$')
    ax.axvline(x=0, ymin=ymin, ymax=ymax, color='b')
    # q-nullcline
    q = np.arange(max(ymin,0.01), ymax, 0.01)
    x = (Sin - rho_inv(q * mu(q, **params), **params)) / q
    ax.plot(x, q, color='orange', label='$\\dot{q}=0$')
    # s-nullcline
    x = (Sin - rho_inv(D*q, **params)) / q
    ax.plot(x, q, color='r', linestyle='dashed', label='$\\dot{s}=0$')
    ax.axvline(x=0, ymin=ymin, ymax=ymax, color='r', linestyle=(0, (5, 10)))

    ax.axis([xmin, xmax, ymin, ymax])
    return ax