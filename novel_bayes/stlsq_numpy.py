"""Baseline: STLSQ in numpy on the SAME data statistics (H>=1 stats, step k=0 only),
plus the recursive multi-step predictor z_{k+1} = A z_k + B u_k used by the original repo.
Set lam/alpha to what koopman_fit.py uses, and check nnz(A)=11/81, nnz(B)=3/18."""
import numpy as np


def stlsq_from_stats(st, lam=0.005, alpha=0.05, n_iter=20):
    d0 = st.dk(0)                                   # L + m
    G = st.G[:d0, :d0]; B = st.XtY[0][:d0, :]       # (d0, L)
    W = np.linalg.solve(G + alpha * np.eye(d0), B)
    for _ in range(n_iter):
        small = np.abs(W) < lam
        W[small] = 0.0
        for i in range(W.shape[1]):
            act = np.where(~small[:, i])[0]
            if len(act) == 0:
                continue
            W[act, i] = np.linalg.solve(G[np.ix_(act, act)] + alpha * np.eye(len(act)), B[act, i])
        if (np.abs(W) < lam).sum() == small.sum():
            W[np.abs(W) < lam] = 0.0
            break
    A = W[:st.L, :].T
    Bm = W[st.L:, :].T
    return A, Bm


class StlsqPredictor:
    def __init__(self, A, B, H):
        self.A, self.B, self.H = A, B, H
        self.L, self.m = A.shape[0], B.shape[1]

    def predict_x(self, Phi, n=3):
        z = Phi[:, :self.L]; out = []
        for k in range(self.H):
            u = Phi[:, self.L + self.m * k: self.L + self.m * (k + 1)]
            z = z @ self.A.T + u @ self.B.T
            out.append(z[:, :n])
        return np.stack(out, axis=1)

    def predict_y(self, Phi, c):
        z = Phi[:, :self.L]
        cL = np.zeros(self.L); cL[:len(c)] = c
        out = []
        for k in range(self.H):
            u = Phi[:, self.L + self.m * k: self.L + self.m * (k + 1)]
            z = z @ self.A.T + u @ self.B.T
            out.append(z @ cL)
        return np.stack(out, axis=1)
