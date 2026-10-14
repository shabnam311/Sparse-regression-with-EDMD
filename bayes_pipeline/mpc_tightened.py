"""Condensed linear-Gaussian MPC with posterior/conformal-tightened output constraint |c.x| <= ymax.

Decision: U = [u_0..u_{H-1}].  Prediction (linear in U):  z_{k+1} = M_k phi_k,  phi_k=[z_0,u_0..u_k]
Constraint k:   |c' x_{k+1}| + delta_k(phi_k) <= ymax
   hybrid:   delta_k = kappa*sqrt(phi_k' S_k phi_k) + q_k      <- second-order-cone in U (exact SOCP)
   frozen=True: delta_k evaluated at the previous iterate  -> constraints become LINEAR (QP-like)
Solved with scipy SLSQP (no extra dependency, fine for H*m ~ 20). A cvxpy version is sketched at the bottom;
it is NOT tested here -- use it if you want a real conic solver (ECOS/Clarabel).
"""
import numpy as np
from scipy.optimize import minimize
from lifting_np import lift_poly2


class TightenedMPC:
    def __init__(self, model, tight, c, ymax, scheme, u_lo, u_hi, Q=(0.05, 0.0, 1.0), R=0.05, frozen=False):
        self.m, self.tg, self.c, self.ymax, self.scheme = model, tight, np.asarray(c, float), ymax, scheme
        self.H, self.L, self.mdim = model.H, model.L, model.m
        self.u_lo, self.u_hi, self.Q, self.R, self.frozen = np.asarray(u_lo), np.asarray(u_hi), np.asarray(Q), R, frozen
        self.S = [model.S_matrix(k, self.c) for k in range(self.H)]
        self.s0 = [model.noise_var(k, self.c) for k in range(self.H)]
        self.cL = np.zeros(self.L); self.cL[:len(c)] = c
        self._U_prev = None

    def _phi(self, z0, U, k):
        return np.concatenate([z0, U[:k + 1].ravel()])

    def _delta_k(self, z0, U, k):
        s, tg = self.scheme, self.tg
        if s == 'none': return 0.0
        if s == 'conformal_onestep': return tg.q_one
        if s == 'conformal_horizon': return tg.q_hor[k]
        if s == 'conformal_joint': return getattr(tg, 'q_hor_joint', tg.q_hor)[k]
        phi = self._phi(z0, U, k)
        epi = float(phi @ self.S[k] @ phi)
        if s == 'posterior_epi': return tg.kappa * np.sqrt(max(epi, 0) + 1e-12)
        if s == 'posterior_pred': return tg.kappa * np.sqrt(max(epi + self.s0[k], 0) + 1e-12)
        if s == 'hybrid': return tg.kappa * np.sqrt(max(epi, 0) + 1e-12) + tg.q_hyb[k]
        if s == 'hybrid_joint': return tg.kappa * np.sqrt(max(epi, 0) + 1e-12) + getattr(tg, 'q_hyb_joint', tg.q_hyb)[k]
        raise ValueError(s)

    def _pred(self, z0, U):
        return [self.m.mu[k] @ self._phi(z0, U, k) for k in range(self.H)]   # lifted z_{k+1}

    def solve(self, x_scaled, ref, u_prev, U0=None):
        z0 = lift_poly2(x_scaled[None, :])[0]
        H, md = self.H, self.mdim
        
        # Warm start: shift previous solution forward if available
        if U0 is None:
            if self._U_prev is not None:
                U0 = np.empty((H, md))
                U0[:-1] = self._U_prev[1:]
                U0[-1] = self._U_prev[-1]
            else:
                U0 = np.tile(u_prev, (H, 1))

        Uf = self._U_prev if (self.frozen and self._U_prev is not None) else U0
        frozen_d = [self._delta_k(z0, Uf, k) for k in range(H)] if self.frozen else None

        # Decision vector includes H controls and H output slack variables to ensure feasibility
        slack_penalty = 1e4
        linear_slack_penalty = 1e2
        v0 = np.concatenate([U0.ravel(), np.zeros(H)])

        def cost(v):
            U = v[:H * md].reshape(H, md)
            slack = v[H * md:]
            Z = self._pred(z0, U)
            J = 0.0
            for k in range(H):
                e = Z[k][:3] - ref[k]
                J += float(e @ (self.Q * e))
                du = U[k] - (U[k - 1] if k else u_prev)
                J += self.R * float(du @ du)
            # Slack penalty on constraint violations
            J += slack_penalty * float(np.sum(slack ** 2)) + linear_slack_penalty * float(np.sum(slack))
            return J

        def cons(v):
            U = v[:H * md].reshape(H, md)
            slack = v[H * md:]
            Z = self._pred(z0, U)
            g = []
            for k in range(H):
                y = float(self.cL @ Z[k])
                d = frozen_d[k] if self.frozen else self._delta_k(z0, U, k)
                # Softened constraints: ymax - d + slack - y >= 0 and ymax - d + slack + y >= 0
                g += [self.ymax - d + slack[k] - y, self.ymax - d + slack[k] + y]
            return np.array(g)

        bnds_u = [(self.u_lo[j], self.u_hi[j]) for _ in range(H) for j in range(md)]
        bnds_slack = [(0.0, 10.0) for _ in range(H)]
        bnds = bnds_u + bnds_slack

        res = minimize(cost, v0, method='SLSQP', bounds=bnds,
                       constraints=[{'type': 'ineq', 'fun': cons}], options=dict(maxiter=80, ftol=1e-6))
        
        success = bool(res.success and not np.any(np.isnan(res.x)))
        if success:
            U = res.x[:H * md].reshape(H, md)
            self._U_prev = U
            u_opt = U[0]
        else:
            # Fallback to shifted warm start on solver failure
            U = U0
            u_opt = np.clip(U0[0], self.u_lo, self.u_hi)
            self._U_prev = U
            
        return u_opt, U, success

# ---- cvxpy sketch (UNTESTED here: cvxpy was not available in the sandbox) --------------------------
# import cvxpy as cp
# U = cp.Variable((H, m))
# for k in range(H):
#     phi = cp.hstack([z0, cp.vec(U[:k+1])])
#     y   = cL @ (mu_k @ phi);   Lk = np.linalg.cholesky(S_k + 1e-12*np.eye(d_k))
#     cons += [ y + kappa*cp.norm(cp.hstack([Lk.T @ phi, 1e-6])) + q_k <= ymax,
#              -y + kappa*cp.norm(cp.hstack([Lk.T @ phi, 1e-6])) + q_k <= ymax ]
