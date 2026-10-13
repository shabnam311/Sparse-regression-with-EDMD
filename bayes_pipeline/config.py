"""Central settings for the Bayesian (Bayesian multi-step) pipeline."""
from dataclasses import dataclass


@dataclass
class Cfg:
    H: int = 10                      # prediction horizon (steps)
    eps: float = 0.05                # target violation rate
    # training-set sizes (number of multi-step windows) for the data-size sweep
    train_sizes: tuple = (1_000, 10_000, 100_000, 1_000_000)
    val_stride: int = 10             # >= H, reduces overlap between calibration windows
    test_stride: int = 5
    # constrained output y = c . [vx, vy, omega] in SCALED units (default: yaw rate)
    c: tuple = (0.0, 0.0, 1.0)
    # STLSQ baseline (match what koopman_fit.py uses!)
    stlsq_lambda: float = 0.05
    stlsq_alpha: float = 0.05
    clip_bias_nonneg: bool = True    # hybrid bias term q_k >= 0 (more conservative)
    seed: int = 0
