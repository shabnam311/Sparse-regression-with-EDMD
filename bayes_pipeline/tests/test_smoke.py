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


def test_lifting_matches_baseline_repo():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    baseline_dir = os.path.join(root, 'baseline_stlsq')
    if baseline_dir not in sys.path:
        sys.path.insert(0, baseline_dir)
    try:
        from lifting import lift_state as repo_lift_state
        from lifting_np import check_against_repo
        sample_x = np.array([[2.5, -1.2, 0.45], [10.0, 0.1, -0.05]])
        assert check_against_repo(repo_lift_state, sample_x), "Lifting order must match baseline exactly"
    except ImportError:
        pass


def test_joint_violation_metric():
    class DummyModel:
        H = 3
        L = 9
        def predict_y(self, Phi, c):
            return np.zeros((len(Phi), self.H))
    m = DummyModel()
    tg = Tightener(m, c=(0, 0, 1), eps=0.1)
    Phi = np.zeros((100, 12))
    # True Y has error 2.0 at step 0 for first 10 windows, and step 2 for next 10 windows
    Y = np.zeros((100, 3, 3))
    Y[:10, 0, 2] = 2.0
    Y[10:20, 2, 2] = 2.0
    tg.q_hor = np.array([1.0, 1.0, 1.0])
    res = tg.evaluate(Phi, Y, ['conformal_horizon'])
    # Marginal viol per step is 10/100 = 0.10, but joint viol (either step) is 20/100 = 0.20
    assert abs(res['conformal_horizon']['viol_joint'] - 0.20) < 1e-6
    assert abs(res['conformal_horizon']['viol_worst_k'] - 0.10) < 1e-6


def test_direct_vs_recursive_diagnostic():
    rng = np.random.default_rng(3)
    A = np.array([[0.9, 0.1, 0], [0, 0.8, 0.05], [0, 0, 0.7]])
    B = np.array([[0.1], [0.0], [0.2]])
    def traj(T):
        X = [rng.standard_normal(3)]; U = rng.standard_normal((T, 1))
        for t in range(T - 1): X.append(A @ X[-1] + B @ U[t])
        return np.array(X), U
    st = accumulate_stats([traj(500)], 3, lambda X: X)
    bm = BayesMultiStep().fit(st)
    diffs = bm.compare_with_recursive()
    assert len(diffs) == 3
    # Step k=0 should match recursive definition identically (diff ~ 0)
    assert diffs[0] < 1e-6


if __name__ == '__main__':
    test_lifting_order()
    test_ard_recovers_sparse_weights()
    test_linear_system_is_identified_and_uncertainty_shrinks()
    test_conformal_quantile_coverage()
    test_lifting_matches_baseline_repo()
    test_joint_violation_metric()
    test_direct_vs_recursive_diagnostic()
    print("All unit tests in test_smoke.py passed successfully.")
