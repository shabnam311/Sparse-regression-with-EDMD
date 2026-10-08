"""Glue between the ORIGINAL repo (baseline_stlsq/) and the novel pipeline.

get_data() must return {'train': [(X,U),...], 'val': [...], 'test': [...]} where
  X : (T, 3) SCALED states [vx, vy, omega],   U : (T, m) SCALED inputs  (U[t] acts between X[t], X[t+1])
Fill in `_from_repo()` using the functions of baseline_stlsq/data_loader.py (see README_novel.md, point 33).
Until then, `source='synthetic'` lets you smoke-test everything."""
import numpy as np


def _from_repo(data_dir='data/dataset', split_ratio=(60, 5, 5), seed=42):
    import sys, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    baseline_dir = os.path.join(root, 'baseline_stlsq')
    if baseline_dir not in sys.path:
        sys.path.insert(0, baseline_dir)
    from data_loader import load_and_preprocess_data

    repo_data = load_and_preprocess_data(os.path.join(root, data_dir), split_ratio=split_ratio, seed=seed)
    
    out = {}
    for split_key in ['train', 'val', 'test']:
        trajs = repo_data[split_key][3]  # list of trajectory dicts with 'x' (scaled) and 'u' (scaled)
        out[split_key] = [(t['x'], t['u']) for t in trajs]
        
    scaler_x, scaler_u = repo_data['scalers']
    mx = scaler_x.mean_ if getattr(scaler_x, 'with_mean', False) else np.zeros(3)
    sx = scaler_x.scale_
    mu = scaler_u.mean_ if getattr(scaler_u, 'with_mean', False) else np.zeros(2)
    su = scaler_u.scale_
    
    out['scaler'] = (mx, sx, mu, su)
    return out


def get_data(source='synthetic', **kw):
    if source == 'synthetic':
        from synthetic import make_dataset
        return make_dataset(**kw)
    return _from_repo(**kw)
