"""Closed-loop demo on the SYNTHETIC plant (the real truck needs your simulator: pass your own plant_step).
Counts true-plant violations of |omega_scaled| <= ymax for each tightening scheme."""
import numpy as np, time
from config import Cfg
from lifting_np import lift_poly2
from windows import accumulate_stats, build_eval_windows, count_windows
from bayes_multistep import BayesMultiStep
from tightening import Tightener
from mpc_tightened import TightenedMPC
import synthetic as S


def closed_loop(mpc, data, steps=120, seed=3, ymax=1.0, plant_step=S.plant_step):
    mx, sx, mu, su = data['scaler']
    rng = np.random.default_rng(seed)
    x = np.array([15.0, 0.0, 0.0]); u_prev = np.zeros(2); viol = 0; tracked = 0.0
    for t in range(steps):
        xs = (x - mx) / sx
        ref = np.zeros((mpc.H, 3)); ref[:, 0] = (15.0 - mx[0]) / sx[0]
        ref[:, 2] = 1.6 * np.sign(np.sin(2 * np.pi * (t + np.arange(mpc.H)) / 60.0))   # demand beyond ymax
        u0, _, ok = mpc.solve(xs, ref, u_prev)
        u_raw = u0 * su + mu
        x = plant_step(x, u_raw) + np.array([0, 0, 0.0])
        u_prev = u0
        w = ((x - mx) / sx)[2]
        viol += abs(w) > ymax; tracked += (w - ref[0, 2]) ** 2
    return viol / steps, tracked / steps


if __name__ == '__main__':
    cfg = Cfg(); data = S.make_dataset(seed=0); rng = np.random.default_rng(0)
    Pv, Yv = build_eval_windows(data['val'], cfg.H, lift_poly2, cfg.val_stride)
    tot = count_windows(data['train'], cfg.H)
    for N in (500, 50_000):
        st = accumulate_stats(data['train'], cfg.H, lift_poly2, keep_prob=min(1, N / tot), rng=rng)
        bm = BayesMultiStep().fit(st); tg = Tightener(bm, cfg.c, cfg.eps).calibrate(Pv, Yv)
        mu_s, su_s = data['scaler'][2], data['scaler'][3]
        ulo, uhi = (np.array([-0.15, -3.0]) - mu_s) / su_s, (np.array([0.15, 3.0]) - mu_s) / su_s
        for scheme in ['none', 'posterior_epi', 'conformal_horizon', 'hybrid']:
            mpc = TightenedMPC(bm, tg, cfg.c, 1.0, scheme, ulo, uhi)
            t0 = time.time(); v, tr = closed_loop(mpc, data)
            print(f"N={N:>6d} {scheme:18s} true-plant violation rate={v:.3f}  tracking MSE={tr:.3f}  ({time.time()-t0:.0f}s)")
