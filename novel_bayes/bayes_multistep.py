"""Sparse-Bayesian (ARD / RVM-style) fit of the CONDENSED multi-step Koopman predictor.

For every horizon step k and every lifted output i:
      z_{t+k+1, i} = w_{k,i}^T phi_k + noise,    phi_k = [z_t, u_t, ..., u_{t+k}]
      w_{k,i,j} ~ N(0, 1/alpha_{k,i,j})   (automatic relevance determination)
Because the model is linear in phi, E_k, F_k of the condensed predictor are the rows w_{k,i}.
Fitting uses only sufficient statistics (G, b, y'y, N) so 1e6+ windows are cheap.
"""
import numpy as np


def ard_from_stats(G, b, yty, N, max_iter=500, tol=1e-6, prune=1e5):
    d = len(b)
    act = np.arange(d)
    alpha = np.ones(d)
    rho = N / max(yty, 1e-12)
    mu = np.zeros(d); Sig = np.zeros((d, d))
    for _ in range(max_iter):
        Ga = G[np.ix_(act, act)]
        A = rho * Ga + np.diag(alpha[act]) + 1e-10 * np.eye(len(act))
        Sa = np.linalg.inv(A)
        ma = rho * Sa @ b[act]
        gamma = 1.0 - alpha[act] * np.diag(Sa)
        new_alpha = alpha.copy()
        new_alpha[act] = gamma / np.maximum(ma ** 2, 1e-30)
        res = yty - 2 * ma @ b[act] + ma @ Ga @ ma
        rho_new = max(N - gamma.sum(), 1e-6) / max(res, 1e-12)
        delta = np.max(np.abs(np.log(new_alpha[act]) - np.log(alpha[act])))
        alpha, rho = new_alpha, rho_new
        keep = alpha[act] < prune
        if not keep.all():
            act = act[keep]
            if len(act) == 0:
                break
            continue
        if delta < tol:
            break
    mu[:] = 0; Sig[:] = 0
    if len(act):
        Ga = G[np.ix_(act, act)]
        Sa = np.linalg.inv(rho * Ga + np.diag(alpha[act]) + 1e-10 * np.eye(len(act)))
        mu[act] = rho * Sa @ b[act]
        Sig[np.ix_(act, act)] = Sa
    return mu, Sig, rho, act


class BayesMultiStep:
    def fit(self, st):
        self.H, self.L, self.m = st.H, st.L, st.m
        self.mu, self.Sig, self.rho, self.nnz = [], [], [], []
        for k in range(st.H):
            dk = st.dk(k)
            G = st.G[:dk, :dk]
            Mk = np.zeros((st.L, dk)); Sk = np.zeros((st.L, dk, dk)); rk = np.zeros(st.L); nz = 0
            for i in range(st.L):
                b = st.XtY[k][:dk, i]
                mu, Sig, rho, act = ard_from_stats(G, b, st.YtY[k][i, i], st.N)
                Mk[i], Sk[i], rk[i] = mu, Sig, rho
                nz += len(act)
            self.mu.append(Mk); self.Sig.append(Sk); self.rho.append(rk); self.nnz.append(nz)
        return self

    def dk(self, k):
        return self.L + self.m * (k + 1)

    def predict_lifted(self, Phi, k):
        return Phi[:, :self.dk(k)] @ self.mu[k].T                      # (n, L)

    def predict_x(self, Phi, n=3):
        return np.stack([self.predict_lifted(Phi, k)[:, :n] for k in range(self.H)], axis=1)   # (n, H, 3)

    def predict_y(self, Phi, c):
        cL = np.zeros(self.L); cL[:len(c)] = c
        return np.stack([self.predict_lifted(Phi, k) @ cL for k in range(self.H)], axis=1)

    def _cL(self, c):
        cL = np.zeros(self.L); cL[:len(c)] = c
        return cL

    def S_matrix(self, k, c):
        """S_k = sum_i c_i^2 Sigma_{k,i}  so that epistemic variance of y_{k+1} = phi' S_k phi."""
        cL = self._cL(c)
        return np.einsum('i,ijl->jl', cL ** 2, self.Sig[k])

    def noise_var(self, k, c):
        cL = self._cL(c)
        return float(np.sum(cL ** 2 / self.rho[k]))

    def var_epi(self, Phi, k, c):
        P = Phi[:, :self.dk(k)]
        S = self.S_matrix(k, c)
        return np.einsum('nj,jl,nl->n', P, S, P)

    def sigma_epi(self, Phi, c):
        return np.stack([np.sqrt(np.maximum(self.var_epi(Phi, k, c), 0)) for k in range(self.H)], 1)

    def sigma_pred(self, Phi, c):
        return np.stack([np.sqrt(np.maximum(self.var_epi(Phi, k, c) + self.noise_var(k, c), 0))
                         for k in range(self.H)], 1)
