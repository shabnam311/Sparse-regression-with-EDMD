"""Violation-rate vs training-set-size experiment (the key plot) for BOTH pipelines.
   python run_bayes.py --source synthetic
Output: results_bayes/sweep.csv and results_bayes/violation_vs_N.png"""
import argparse, csv, os, time
import numpy as np
from config import Cfg
from lifting_np import lift_poly2
from windows import accumulate_stats, build_eval_windows, count_windows
from bayes_multistep import BayesMultiStep
from stlsq_numpy import stlsq_from_stats, StlsqPredictor
from tightening import Tightener
from adapter import get_data

BAYES_SCHEMES = ['none', 'conformal_onestep', 'conformal_horizon', 'posterior_epi', 'posterior_pred', 'hybrid']
STLSQ_SCHEMES = ['none', 'conformal_onestep', 'conformal_horizon']


DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_bayes')

def run(data, cfg, out_dir=DEFAULT_OUT_DIR, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    total = count_windows(data['train'], cfg.H)
    Pv, Yv = build_eval_windows(data['val'], cfg.H, lift_poly2, cfg.val_stride)
    Pt, Yt = build_eval_windows(data['test'], cfg.H, lift_poly2, cfg.test_stride)
    rows = []
    for N in cfg.train_sizes:
        if N > total:
            N = total
        st = accumulate_stats(data['train'], cfg.H, lift_poly2, keep_prob=min(1.0, N / total), rng=rng)
        t0 = time.time(); bm = BayesMultiStep().fit(st); tfit = time.time() - t0
        A, B = stlsq_from_stats(st, cfg.stlsq_lambda, cfg.stlsq_alpha)
        models = {'bayes_multistep': (bm, BAYES_SCHEMES),
                  'stlsq_recursive': (StlsqPredictor(A, B, cfg.H), STLSQ_SCHEMES)}
        for name, (mdl, schemes) in models.items():
            tg = Tightener(mdl, cfg.c, cfg.eps, cfg.clip_bias_nonneg).calibrate(Pv, Yv)
            res = tg.evaluate(Pt, Yt, schemes)
            for s, r in res.items():
                rows.append(dict(N=st.N, model=name, scheme=s, viol=r['viol'], viol_worst_k=r['viol_worst_k'],
                                 width=r['width'], rmse=r['rmse'], eps=cfg.eps,
                                 nnz=(int(sum(bm.nnz)) if name == 'bayes_multistep' else int((A != 0).sum() + (B != 0).sum())),
                                 fit_s=tfit if name == 'bayes_multistep' else np.nan))
        if verbose:
            print(f"N={st.N:>8d}  nnz(A)={int((A!=0).sum())}/{A.size} nnz(B)={int((B!=0).sum())}/{B.size}  bayes fit {tfit:.1f}s")
            for r in rows[-(len(BAYES_SCHEMES) + len(STLSQ_SCHEMES)):]:
                print(f"   {r['model']:16s} {r['scheme']:18s} viol={r['viol']:.3f}  worst_k={r['viol_worst_k']:.3f}  width={r['width']:.4f}")
    with open(os.path.join(out_dir, 'sweep.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    plot(rows, cfg, out_dir)
    return rows


def plot(rows, cfg, out_dir):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for model, scheme in sorted({(r['model'], r['scheme']) for r in rows}):
        rr = sorted([r for r in rows if r['model'] == model and r['scheme'] == scheme], key=lambda r: r['N'])
        ls = '-' if model == 'bayes_multistep' else '--'
        ax[0].plot([r['N'] for r in rr], [r['viol'] for r in rr], ls, marker='o', label=f'{model}/{scheme}')
        ax[1].plot([r['N'] for r in rr], [r['width'] for r in rr], ls, marker='o')
    ax[0].axhline(cfg.eps, color='k', lw=1); ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].set_xlabel('training windows N'); ax[0].set_ylabel('violation rate (test)'); ax[0].set_title('target eps = black line')
    ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel('training windows N'); ax[1].set_ylabel('mean margin (conservativeness)')
    ax[0].legend(fontsize=6); plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'violation_vs_N.png'), dpi=150)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--source', default='synthetic'); a = ap.parse_args()
    cfg = Cfg(train_sizes=(500, 5_000, 50_000) if a.source == 'synthetic' else Cfg().train_sizes)
    run(get_data(a.source), cfg)
