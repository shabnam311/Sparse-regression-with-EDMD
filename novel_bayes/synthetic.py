"""Synthetic stand-in for the truck data so every script can be smoke-tested without TruckMaker.
Nonlinear single-track model with tanh tyre saturation -> degree-2 lifting is MISSPECIFIED."""
import numpy as np

P = dict(m=1500., Iz=2500., a=1.2, b=1.6, Cf=80000., Cr=80000., Fm=7000.)


def plant_step(x, u, dt=0.05, sub=5):
    vx, vy, w = x
    steer, acc = u
    h = dt / sub
    for _ in range(sub):
        vxs = max(vx, 3.0)
        af = steer - (vy + P['a'] * w) / vxs
        ar = -(vy - P['b'] * w) / vxs
        Ff = P['Fm'] * np.tanh(P['Cf'] * af / P['Fm'])
        Fr = P['Fm'] * np.tanh(P['Cr'] * ar / P['Fm'])
        dvy = (Ff + Fr) / P['m'] - vx * w
        dw = (P['a'] * Ff - P['b'] * Fr) / P['Iz']
        dvx = acc + vy * w
        vx, vy, w = vx + h * dvx, vy + h * dvy, w + h * dw
    return np.array([vx, vy, w])


def make_traj(rng, T=2000, dt=0.05):
    x = np.array([rng.uniform(10, 20), 0.0, 0.0])
    f = rng.uniform(0.05, 0.4, 3); ph = rng.uniform(0, 6.28, 3); amp = rng.uniform(0.02, 0.12)
    vt = rng.uniform(10, 20)
    X, U = [x.copy()], []
    for t in range(T - 1):
        s = amp * np.sum(np.sin(2 * np.pi * f * t * dt + ph)) / 1.5 + 0.005 * rng.standard_normal()
        a = 0.5 * (vt - x[0]) + 0.1 * rng.standard_normal()
        u = np.array([s, np.clip(a, -3, 3)])
        x = plant_step(x, u, dt)
        U.append(u); X.append(x.copy())
    X, U = np.array(X), np.array(U)
    return X, np.vstack([U, U[-1:]])        # U padded to length T


def make_dataset(n_train=30, n_val=10, n_test=10, T=2000, seed=0):
    rng = np.random.default_rng(seed)
    raw = {k: [make_traj(rng, T) for _ in range(n)] for k, n in
           [('train', n_train), ('val', n_val), ('test', n_test)]}
    allX = np.concatenate([X for X, _ in raw['train']]); allU = np.concatenate([U for _, U in raw['train']])
    mx, sx, mu, su = allX.mean(0), allX.std(0), allU.mean(0), allU.std(0)
    scale = lambda X, U: ((X - mx) / sx, (U - mu) / su)
    data = {k: [scale(X, U) for X, U in v] for k, v in raw.items()}
    data['scaler'] = (mx, sx, mu, su)
    return data
