"""Sparse-Bayesian (ARD / RVM-style) fit of the CONDENSED multi-step Koopman predictor.

For every horizon step k and every lifted output i:
      z_{t+k+1, i} = w_{k,i}^T phi_k + noise,    phi_k = [z_t, u_t, ..., u_{t+k}]
      w_{k,i,j} ~ N(0, 1/alpha_{k,i,j})   (automatic relevance determination)
Because the model is linear in phi, E_k, F_k of the condensed predictor are the rows w_{k,i}.
Fitting uses only sufficient statistics (G, b, y'y, N) so 1e6+ windows are cheap.

Uncertainty interpretation:
  - Epistemic covariance Sig reflects parameter uncertainty under the effective sample size N_eff = N/tau.
  - Conformal margin q_k in the hybrid scheme covers unmodeled residual misspecification bias.
"""
import numpy as np
from scipy import linalg


class ARDResult(tuple):
    """Tuple (mu, Sig, rho, act) with attributes for both named access and tuple unpacking."""
    def __new__(cls, mu, Sig, rho, act, converged):
        return super(ARDResult, cls).__new__(cls, (mu, Sig, rho, act))

    def __init__(self, mu, Sig, rho, act, converged):
        self.mu = mu
        self.Sig = Sig
        self.rho = rho
        self.act = act
        self.converged = converged


def ard_from_stats(G, b, yty, N, max_iter=500, tol=1e-6, prune=1e5, variance_floor=False):
    """
    Automatic Relevance Determination (ARD) from sufficient statistics (G, b, yty, N).
    
    Numerics:
    - Standardizes columns by sqrt(diag(G)/N) so regularization and pruning are scale-invariant.
    - Initialises noise precision rho from a ridge residual estimate rather than total output variance.
    - Uses Cholesky decomposition (with robust fallback) for matrix solves.
    - Optionally retains a prior-variance floor (1/prune) on pruned weights.
    """
    d = len(b)
    # 1. Scale standardization for scale-invariant thresholds
    diag_g = np.diag(G)
    scales = np.sqrt(np.maximum(diag_g / max(N, 1), 1e-12))
    G_s = G / (scales[:, None] * scales[None, :])
    b_s = b / scales
    yty_s = float(yty)
    
    # 2. Initialise rho from a ridge residual estimate
    w_ridge = np.linalg.solve(G_s + 0.01 * np.eye(d), b_s)
    res_init = max(yty_s - 2.0 * float(w_ridge @ b_s) + float(w_ridge @ G_s @ w_ridge), 1e-8 * max(yty_s, 1e-12))
    rho = float(max(N, 1) / res_init)
    
    act = np.arange(d)
    alpha = np.ones(d)
    converged = False
    
    for _ in range(max_iter):
        Ga = G_s[np.ix_(act, act)]
        A = rho * Ga + np.diag(alpha[act]) + 1e-10 * np.eye(len(act))
        
        try:
            c, lower = linalg.cho_factor(A, lower=True, check_finite=False)
            Sa = linalg.cho_solve((c, lower), np.eye(len(act)), check_finite=False)
            ma = rho * linalg.cho_solve((c, lower), b_s[act], check_finite=False)
        except linalg.LinAlgError:
            Sa = np.linalg.pinv(A)
            ma = rho * Sa @ b_s[act]
            
        gamma = 1.0 - alpha[act] * np.diag(Sa)
        new_alpha = alpha.copy()
        new_alpha[act] = gamma / np.maximum(ma ** 2, 1e-30)
        res = yty_s - 2.0 * float(ma @ b_s[act]) + float(ma @ Ga @ ma)
        rho_new = float(max(N - gamma.sum(), 1e-6) / max(res, 1e-12))
        
        delta = float(np.max(np.abs(np.log(np.maximum(new_alpha[act], 1e-30)) - np.log(np.maximum(alpha[act], 1e-30)))))
        alpha, rho = new_alpha, rho_new
        keep = alpha[act] < prune
        if not keep.all():
            act = act[keep]
            if len(act) == 0:
                break
            continue
        if delta < tol:
            converged = True
            break
            
    mu_s = np.zeros(d)
    Sig_s = np.zeros((d, d))
    if len(act):
        Ga = G_s[np.ix_(act, act)]
        A = rho * Ga + np.diag(alpha[act]) + 1e-10 * np.eye(len(act))
        try:
            c, lower = linalg.cho_factor(A, lower=True, check_finite=False)
            Sa = linalg.cho_solve((c, lower), np.eye(len(act)), check_finite=False)
            ma = rho * linalg.cho_solve((c, lower), b_s[act], check_finite=False)
        except linalg.LinAlgError:
            Sa = np.linalg.pinv(A)
            ma = rho * Sa @ b_s[act]
        mu_s[act] = ma
        Sig_s[np.ix_(act, act)] = Sa
        
    if variance_floor:
        pruned_idx = np.setdiff1d(np.arange(d), act)
        if len(pruned_idx):
            Sig_s[pruned_idx, pruned_idx] = 1.0 / prune
            
    # Unscale parameters and covariance to original feature dimensions
    mu = mu_s / scales
    Sig = Sig_s / (scales[:, None] * scales[None, :])
    
    return ARDResult(mu, Sig, rho, act, converged)


class BayesMultiStep:
    def fit(self, st, tau=1.0):
        """
        Fits multi-step Bayesian ARD model from statistics st.
        If tau > 1.0, scales sufficient statistics to effective sample size N_eff = N / tau.
        """
        if tau > 1.0:
            st = st.scale_for_effective_samples(tau)
        self.H, self.L, self.m = st.H, st.L, st.m
        self.tau = tau
        self.mu, self.Sig, self.rho, self.nnz, self.converged = [], [], [], [], []
        for k in range(st.H):
            dk = st.dk(k)
            G = st.G[:dk, :dk]
            Mk = np.zeros((st.L, dk)); Sk = np.zeros((st.L, dk, dk)); rk = np.zeros(st.L); nz = 0
            conv_k = []
            for i in range(st.L):
                b = st.XtY[k][:dk, i]
                res = ard_from_stats(G, b, st.YtY[k][i, i], st.N)
                Mk[i], Sk[i], rk[i] = res.mu, res.Sig, res.rho
                nz += len(res.act)
                conv_k.append(res.converged)
            self.mu.append(Mk); self.Sig.append(Sk); self.rho.append(rk); self.nnz.append(nz)
            self.converged.append(conv_k)
        return self

    def dk(self, k):
        return self.L + self.m * (k + 1)

    def predict_lifted(self, Phi, k):
        return Phi[:, :self.dk(k)] @ self.mu[k].T                      # (n, L)

    def predict_x(self, Phi, n=3):
        return np.stack([self.predict_lifted(Phi, k)[:, :n] for k in range(self.H)], axis=1)   # (n, H, 3)

    def predict_y(self, Phi, c):
        cL = self._cL(c)
        return np.stack([self.predict_lifted(Phi, k) @ cL for k in range(self.H)], axis=1)

    def _cL(self, c):
        cL = np.zeros(self.L); cL[:len(c)] = c
        return cL

    def S_matrix(self, k, c):
        """
        S_k = sum_i c_i^2 Sigma_{k,i} so that epistemic variance of y_{k+1} = phi' S_k phi.
        Note: Assumes single-output constraint (row independence).
        """
        assert np.count_nonzero(c) == 1, (
            f"Cross-output covariance is ignored in this row-independent model. "
            f"c must select exactly one channel (e.g. c=(0, 0, 1)), got c={c}."
        )
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

    def compare_with_recursive(self):
        """
        Diagnostic comparing the direct multi-step matrices mu[k] against recursive powers
        of the 1-step model mu[0] = [A, B]. Returns relative difference for each horizon step.
        """
        if len(self.mu) == 0:
            return []
        A = self.mu[0][:, :self.L]
        B = self.mu[0][:, self.L:]
        diffs = []
        for k in range(self.H):
            # True direct matrix mu[k] has shape (L, L + m*(k+1))
            M_direct = self.mu[k]
            # Construct equivalent condensed recursive matrix M_rec
            # z_{k+1} = A^{k+1} z_0 + sum_{j=0}^k A^{k-j} B u_j
            M_rec = np.zeros_like(M_direct)
            A_pow = np.linalg.matrix_power(A, k + 1)
            M_rec[:, :self.L] = A_pow
            for j in range(k + 1):
                col_start = self.L + j * self.m
                col_end = col_start + self.m
                M_rec[:, col_start:col_end] = np.linalg.matrix_power(A, k - j) @ B
            rel_diff = float(np.linalg.norm(M_direct - M_rec) / max(np.linalg.norm(M_direct), 1e-12))
            diffs.append(rel_diff)
        return diffs
