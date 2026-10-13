import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from lifting_np import lift_poly2
from windows import accumulate_stats, build_eval_windows
from bayes_multistep import BayesMultiStep, ard_from_stats
from tightening import Tightener, conf_quantile


def test_lifting_order():
    x = np.array([[2., 3., 5.]])
    assert np.allclose(lift_poly2(x)[0], [2, 3, 5, 4, 9, 25, 6, 10, 15])


def test_ard_recovers_sparse_weights():
    rng = np.random.default_rng(0)
    Phi = rng.standard_normal((5000, 12)); w = np.zeros(12); w[[1, 4, 7]] = [1.5, -2.0, 0.7]
    t = Phi @ w + 0.1 * rng.standard_normal(5000)
    mu, Sig, rho, act = ard_from_stats(Phi.T @ Phi, Phi.T @ t, t @ t, len(t))
    assert {1, 4, 7} <= set(act) and len(act) <= 5      # ARD may keep a spurious ~0 weight
    assert np.allclose(mu, w, atol=0.05)
    assert abs(1 / rho - 0.01) < 0.003


def test_linear_system_is_identified_and_uncertainty_shrinks():
    rng = np.random.default_rng(1)
    A = np.array([[0.9, 0.1, 0], [0, 0.8, 0.05], [0, 0, 0.7]]); B = np.array([[0.1], [0.0], [0.2]])
    def traj(T):
        X = [rng.standard_normal(3)]; U = rng.standard_normal((T, 1))
        for t in range(T - 1): X.append(A @ X[-1] + B @ U[t] + 0.01 * rng.standard_normal(3))
        return np.array(X), U
    ident = lambda X: X       # linear "lifting" -> no misspecification
    sig = []
    for T in (300, 30000):
        st = accumulate_stats([traj(T)], 3, ident); bm = BayesMultiStep().fit(st)
        P, Y = build_eval_windows([traj(2000)], 3, ident, 1)
        err = np.abs(bm.predict_y(P, np.array([1., 0, 0])) - Y[:, :, 0]).mean()
        sig.append(bm.sigma_epi(P, np.array([1., 0, 0])).mean())
        assert err < 0.05
    assert sig[1] < sig[0] / 3          # epistemic sigma collapses with data


def test_conformal_quantile_coverage():
    rng = np.random.default_rng(2); cal = np.abs(rng.standard_normal(2000)); new = np.abs(rng.standard_normal(20000))
    q = conf_quantile(cal, 0.1)
    assert 0.87 < (new <= q).mean() < 0.93


if __name__ == '__main__':
    test_lifting_order()
    test_ard_recovers_sparse_weights()
    test_linear_system_is_identified_and_uncertainty_shrinks()
    test_conformal_quantile_coverage()
    print("All unit tests in test_smoke.py passed successfully.")
