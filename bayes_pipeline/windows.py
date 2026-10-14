"""Multi-step windows and streaming sufficient statistics.

Window at time t:  regressor  phi = [ z_t , u_t , u_{t+1} , ... , u_{t+H-1} ]   (z_t = lift(x_t))
                   targets    Y[k] = lift(x_{t+k+1}),  k = 0..H-1
Convention: U[t] is applied between X[t] and X[t+1] (same as the one-step pairs of the repo).
Prediction of step k only uses the prefix [z_t, u_t..u_{t+k}]  (causality).
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Stats:
    N: int
    G: np.ndarray        # (d, d)      Phi^T Phi
    XtY: np.ndarray      # (H, d, L)   Phi^T Y_k
    YtY: np.ndarray      # (H, L, L)   Y_k^T Y_k
    H: int
    L: int
    m: int

    def dk(self, k):     # regressor length for step k
        return self.L + self.m * (k + 1)

    def scale_for_effective_samples(self, tau: float):
        """
        Scales sufficient statistics by 1 / tau to account for autocorrelated residual
        samples, yielding an effective sample size N_eff = N / tau.
        This prevents parameter covariance from collapsing faster than true parameter uncertainty.
        """
        if tau <= 1.0:
            return self
        return Stats(
            N=max(int(round(self.N / tau)), 1),
            G=self.G / tau,
            XtY=self.XtY / tau,
            YtY=self.YtY / tau,
            H=self.H,
            L=self.L,
            m=self.m
        )


def estimate_autocorrelation_time(residuals, max_lag=50):
    """
    Estimates the residual autocorrelation time tau = 1 + 2 * sum_{k=1}^K rho_k.
    Used to adjust sufficient statistics to effective sample size N_eff = N / tau.
    """
    r = np.asarray(residuals, dtype=float).ravel()
    n = len(r)
    if n < 2 * max_lag:
        return 1.0
    r_cent = r - np.mean(r)
    c0 = np.dot(r_cent, r_cent) / n
    if c0 <= 1e-14:
        return 1.0
    tau = 1.0
    for lag in range(1, min(max_lag, n // 4)):
        ck = np.dot(r_cent[:-lag], r_cent[lag:]) / (n - lag)
        rho_k = ck / c0
        if rho_k <= 0.05:
            break
        tau += 2.0 * rho_k
    return float(max(tau, 1.0))


def build_windows(X, U, H, lift, idx):
    Z = lift(X)
    L = Z.shape[1]
    parts = [Z[idx]] + [U[idx + j] for j in range(H)]
    Phi = np.concatenate(parts, axis=1)
    Y = np.stack([Z[idx + k + 1] for k in range(H)], axis=1)   # (n, H, L)
    return Phi, Y


def count_windows(trajs, H):
    return int(sum(max(len(X) - H, 0) for X, _ in trajs))


def accumulate_stats(trajs, H, lift, keep_prob=1.0, rng=None):
    if keep_prob < 1.0 and rng is None:
        rng = np.random.default_rng(0)
    L = lift(trajs[0][0][:1]).shape[1]
    m = trajs[0][1].shape[1]
    d = L + m * H
    G = np.zeros((d, d)); XtY = np.zeros((H, d, L)); YtY = np.zeros((H, L, L)); N = 0
    for X, U in trajs:
        nw = len(X) - H
        if nw <= 0:
            continue
        idx = np.arange(nw)
        if keep_prob < 1.0:
            idx = idx[rng.random(nw) < keep_prob]
            if len(idx) == 0:
                continue
        for s in range(0, len(idx), 20000):                 # chunk for memory
            ii = idx[s:s + 20000]
            Phi, Y = build_windows(X, U, H, lift, ii)
            G += Phi.T @ Phi
            N += len(ii)
            for k in range(H):
                XtY[k] += Phi.T @ Y[:, k, :]
                YtY[k] += Y[:, k, :].T @ Y[:, k, :]
    return Stats(N, G, XtY, YtY, H, L, m)


def build_eval_windows(trajs, H, lift, stride=1):
    Ps, Ys = [], []
    for X, U in trajs:
        nw = len(X) - H
        if nw <= 0:
            continue
        idx = np.arange(0, nw, stride)
        P, Y = build_windows(X, U, H, lift, idx)
        Ps.append(P); Ys.append(Y)
    return np.concatenate(Ps), np.concatenate(Ys)
