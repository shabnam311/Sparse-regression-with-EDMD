"""Fair head-to-head: baseline (STLSQ + recursive rollout) vs Bayesian (Bayesian condensed multi-step).
Same data split, same scaling, same lifting, same windows, same test set.
Run from repo root:  python compare/run_comparison.py --source synthetic"""
import os, sys, argparse, csv
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'bayes_pipeline'))
from config import Cfg
from lifting_np import lift_poly2
from windows import accumulate_stats, build_eval_windows
from bayes_multistep import BayesMultiStep
from stlsq_numpy import stlsq_from_stats, StlsqPredictor
from adapter import get_data


def main(source):
    cfg = Cfg(); data = get_data(source)
    st = accumulate_stats(data['train'], cfg.H, lift_poly2)
    bm = BayesMultiStep().fit(st)
    A, B = stlsq_from_stats(st, cfg.stlsq_lambda, cfg.stlsq_alpha)
    sp = StlsqPredictor(A, B, cfg.H)
    # fairness: also give the baseline a lambda sweep chosen on the VALIDATION set (multi-step RMSE)
    Pv, Yv = build_eval_windows(data['val'], cfg.H, lift_poly2, cfg.val_stride)
    best = None
    for lam in (0.0, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2):
        Al, Bl = stlsq_from_stats(st, lam, cfg.stlsq_alpha)
        e = float(np.sqrt(np.mean((StlsqPredictor(Al, Bl, cfg.H).predict_x(Pv) - Yv[:, :, :3]) ** 2)))
        print(f"  baseline lambda={lam:<7g} val multi-step RMSE={e:.5f}  nnz={int((Al!=0).sum()+(Bl!=0).sum())}")
        if best is None or e < best[0]: best = (e, lam, Al, Bl)
    print(f"  -> best baseline lambda on validation: {best[1]}")
    sp_tuned = StlsqPredictor(best[2], best[3], cfg.H)
    # sanity: does our numpy STLSQ reproduce the repo's matrices?
    pa = os.path.join(ROOT, 'baseline_stlsq', 'models', 'A.npy'); pb = os.path.join(ROOT, 'baseline_stlsq', 'models', 'B.npy')
    if os.path.exists(pa) and os.path.exists(pb):
        Ar, Br = np.load(pa), np.load(pb)
        print(f"repo A nnz={int((Ar!=0).sum())}/{Ar.size}  ours={int((A!=0).sum())}   repo B nnz={int((Br!=0).sum())}/{Br.size}  ours={int((B!=0).sum())}")
        print("max |A_repo - A_ours(train)| =", float(np.abs(Ar - A).max()) if Ar.shape == A.shape else 'shape mismatch')
        st_tv = accumulate_stats(data['train'] + data['val'], 1, lift_poly2)
        Atv, Btv = stlsq_from_stats(st_tv, cfg.stlsq_lambda, cfg.stlsq_alpha)
        print("max |A_repo - A_ours(train+val)| =", float(np.abs(Ar - Atv).max()) if Ar.shape == Atv.shape else 'shape mismatch')
    Pt, Yt = build_eval_windows(data['test'], cfg.H, lift_poly2, cfg.test_stride)
    names = ['vx', 'vy', 'omega']; rows = []
    for nm, mdl in [('baseline_stlsq_recursive', sp), ('baseline_stlsq_tuned_lambda', sp_tuned), ('bayes_pipeline_condensed', bm)]:
        X = mdl.predict_x(Pt)
        for k in range(cfg.H):
            for j in range(3):
                rows.append(dict(model=nm, k=k + 1, state=names[j], rmse=float(np.sqrt(np.mean((X[:, k, j] - Yt[:, k, j]) ** 2)))))
    print(f"{'k':>3} {'state':>6} {'base(lam)':>10} {'base(tuned)':>10} {'Bayesian':>10}")
    for k in range(1, cfg.H + 1):
        for s in names:
            g = lambda pre: [r['rmse'] for r in rows if r['model'] == pre and r['k'] == k and r['state'] == s][0]
            print(f"{k:>3} {s:>6} {g('baseline_stlsq_recursive'):>10.5f} {g('baseline_stlsq_tuned_lambda'):>10.5f} {g('bayes_pipeline_condensed'):>10.5f}")
    print("nnz baseline:", int((A != 0).sum() + (B != 0).sum()), " nnz Bayesian (sum over k, rows):", int(sum(bm.nnz)))
    os.makedirs(os.path.join(ROOT, 'compare'), exist_ok=True)
    with open(os.path.join(ROOT, 'compare', 'rmse_by_horizon.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'k', 'state', 'rmse']); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--source', default='synthetic'); main(ap.parse_args().source)
