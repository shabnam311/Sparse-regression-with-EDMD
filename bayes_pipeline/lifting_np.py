"""Pure-NumPy implementation of the degree-2 polynomial state lifting.
The column order exactly matches baseline_stlsq/lifting.py (lift_state):
    [vx, vy, omega, vx^2, vy^2, omega^2, vx*vy, vx*omega, vy*omega]
The first 3 columns are the raw states, so projection back to state space is C = [I 0]."""
import numpy as np


def lift_poly2(X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    vx, vy, omega = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    vx_sq, vy_sq, omega_sq = vx**2, vy**2, omega**2
    vx_vy, vx_omega, vy_omega = vx*vy, vx*omega, vy*omega
    return np.hstack([vx, vy, omega, vx_sq, vy_sq, omega_sq, vx_vy, vx_omega, vy_omega])


def check_against_repo(repo_lift_fn, X_sample):
    """Call once with the repo's lifting function to prove both agree."""
    a = np.asarray(repo_lift_fn(X_sample))
    b = lift_poly2(X_sample)
    assert a.shape == b.shape and np.allclose(a, b), "lifting mismatch: fix column order"
    return True
