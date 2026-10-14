"""Constraint-tightening schemes (all give a margin delta[n, k] per window and horizon step).

  none               : delta = 0
  conformal_onestep  : Liang-style single scalar Q from ONE-STEP |error|, same for every k
  conformal_horizon  : split-conformal quantile q_k of |error_k| (marginal, horizon-indexed)
  conformal_joint    : Bonferroni-corrected split-conformal quantile q_k(eps/H) for joint horizon coverage
  posterior_epi      : kappa * sigma_epi(phi)         (parameter/epistemic uncertainty only)
  posterior_pred     : kappa * sqrt(var_epi + noise)  (full Bayesian predictive)
  hybrid             : kappa * sigma_epi(phi) + q_k,  q_k = conformal quantile of (|err_k| - kappa*sigma_epi)
                       -> posterior covers the epistemic part, q_k covers misspecification bias.
  hybrid_joint       : hybrid margin with Bonferroni-corrected q_k(eps/H) for joint horizon coverage.
"""
import numpy as np
from scipy.stats import norm

SCHEMES = ['none', 'conformal_onestep', 'conformal_horizon', 'conformal_joint',
           'posterior_epi', 'posterior_pred', 'hybrid', 'hybrid_joint']


def conf_quantile(scores, eps):
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - eps)) / n, 1.0)
    try:
        return float(np.quantile(scores, level, method='higher'))
    except (TypeError, ValueError):
        return float(np.quantile(scores, level, interpolation='higher'))


class Tightener:
    def __init__(self, model, c, eps, clip_nonneg=True):
        self.model, self.c, self.eps, self.clip = model, np.asarray(c, float), eps, clip_nonneg
        self.kappa = float(norm.ppf(1 - eps / 2))
        self.has_sigma = hasattr(model, 'sigma_epi')

    def true_y(self, Y):
        return Y[:, :, :len(self.c)] @ self.c

    def calibrate(self, Phi_val, Y_val):
        H = self.model.H
        r = np.abs(self.true_y(Y_val) - self.model.predict_y(Phi_val, self.c))
        # Marginal quantiles
        self.q_hor = np.array([conf_quantile(r[:, k], self.eps) for k in range(r.shape[1])])
        self.q_one = conf_quantile(r[:, 0], self.eps)
        # Joint Bonferroni quantiles (eps / H)
        eps_joint = self.eps / max(H, 1)
        self.q_hor_joint = np.array([conf_quantile(r[:, k], eps_joint) for k in range(r.shape[1])])
        if self.has_sigma:
            s = r - self.kappa * self.model.sigma_epi(Phi_val, self.c)
            q = np.array([conf_quantile(s[:, k], self.eps) for k in range(s.shape[1])])
            self.q_hyb = np.maximum(q, 0.0) if self.clip else q
            q_joint = np.array([conf_quantile(s[:, k], eps_joint) for k in range(s.shape[1])])
            self.q_hyb_joint = np.maximum(q_joint, 0.0) if self.clip else q_joint
        return self

    def delta(self, scheme, Phi):
        n, H = len(Phi), self.model.H
        if scheme == 'none':
            return np.zeros((n, H))
        if scheme == 'conformal_onestep':
            return np.full((n, H), self.q_one)
        if scheme == 'conformal_horizon':
            return np.tile(self.q_hor, (n, 1))
        if scheme == 'conformal_joint':
            return np.tile(self.q_hor_joint, (n, 1))
        if scheme == 'posterior_epi':
            return self.kappa * self.model.sigma_epi(Phi, self.c)
        if scheme == 'posterior_pred':
            return self.kappa * self.model.sigma_pred(Phi, self.c)
        if scheme == 'hybrid':
            return self.kappa * self.model.sigma_epi(Phi, self.c) + self.q_hyb[None, :]
        if scheme == 'hybrid_joint':
            return self.kappa * self.model.sigma_epi(Phi, self.c) + self.q_hyb_joint[None, :]
        raise ValueError(scheme)

    def evaluate(self, Phi, Y, schemes):
        r = np.abs(self.true_y(Y) - self.model.predict_y(Phi, self.c))
        out = {}
        for s in schemes:
            d = self.delta(s, Phi)
            is_viol = r > d
            viol_joint = float(is_viol.any(axis=1).mean())
            out[s] = dict(
                viol_k=is_viol.mean(0),
                viol=float(is_viol.mean()),
                viol_worst_k=float(is_viol.mean(0).max()),
                viol_joint=viol_joint,
                width=float(d.mean()),
                rmse=float(np.sqrt((r ** 2).mean()))
            )
        return out
