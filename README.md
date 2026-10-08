# Koopman Operator Modeling Pipelines

This repository contains two data-driven Koopman operator modeling and constraint-tightening pipelines for vehicle dynamic modeling and linear Model Predictive Control (MPC):

1. **Baseline STLSQ Pipeline (`baseline_stlsq/`)**: PySINDy-based Sequential Thresholded Least Squares (STLSQ) Extended Dynamic Mode Decomposition (EDMD) with 1-step pair regression and degree-2 polynomial lifting.
2. **Novel Bayesian Pipeline (`novel_bayes/`)**: Condensed Multi-step Bayesian Koopman predictor with misspecification-aware constraint tightening (Hybrid, Conformal, Epistemic Posterior).
3. **Head-to-Head Comparison (`compare/`)**: Comparison suite evaluating multi-step prediction error (RMSE) and constraint violation rates across horizon lengths and dataset sizes.

---

## Repository Structure

```text
.
├── baseline_stlsq/      # Untouched baseline STLSQ EDMD pipeline
│   ├── data_loader.py   # Dataset loading, unit validation, standard scaling
│   ├── lifting.py       # Degree-2 polynomial lifting (9 observables)
│   ├── koopman_fit.py   # STLSQ sparse regression solver
│   ├── validate.py      # 1-step & open-loop rollout validation
│   ├── main.py          # Primary baseline orchestration script
│   ├── models/          # Trained A, B matrices and scaler files
│   └── results/         # Baseline figures, reports, and validation plots
│
├── novel_bayes/         # Novel Bayesian Multi-Step & Tightening pipeline
│   ├── adapter.py       # Data adapter wrapping baseline_stlsq/data_loader.py
│   ├── config.py        # Central configuration settings
│   ├── lifting_np.py    # Pure-NumPy degree-2 polynomial lifting
│   ├── windows.py       # Streaming sufficient statistics & multi-step window builder
│   ├── bayes_multistep.py # ARD/RVM sparse Bayesian multi-step fit
│   ├── stlsq_numpy.py   # NumPy STLSQ baseline from sufficient statistics
│   ├── tightening.py    # Conformal, Epistemic, and Hybrid tightening schemes
│   ├── mpc_tightened.py # SOCP & linear QP-tightened MPC solver
│   ├── run_novel.py     # Violation rate vs training size sweep experiment
│   ├── run_closed_loop.py # Closed-loop MPC simulation demo
│   ├── synthetic.py     # Synthetic plant simulation generator for unit testing
│   └── tests/           # Unit test suite
│
├── compare/             # Head-to-head comparison scripts
│   └── run_comparison.py # Multi-step RMSE comparison script
│
└── README.md            # Landing page
```

---

## 1. Baseline STLSQ Pipeline (`baseline_stlsq/`)

Identifies a single 1-step discrete-time linear state-space model:

$$ z_{k+1} = A z_k + B u_k $$

where $z = [v_x, v_y, \omega, v_x^2, v_y^2, \omega^2, v_x v_y, v_x \omega, v_y \omega]^T \in \mathbb{R}^9$ and $u = [F_x, \delta]^T \in \mathbb{R}^2$.

### Running the Baseline Pipeline
```bash
cd baseline_stlsq
py main.py
```

### Running Benchmark Scenario Validation
```bash
cd baseline_stlsq
py run_open_loop_scenarios.py
```

---

## 2. Novel Bayesian Pipeline (`novel_bayes/`)

Identifies a condensed multi-step predictor:

$$ z_{t+k+1} = M_k \phi_k + \epsilon_k $$

where $\phi_k = [z_t, u_t, u_{t+1}, \dots, u_{t+k}]^T$. 

### Constraint Tightening Schemes Implemented:
* `none`: Unconstrained margin ($\delta = 0$).
* `conformal_onestep`: Single 1-step split-conformal quantile.
* `conformal_horizon`: Horizon-indexed split-conformal quantile $q_k$.
* `posterior_epi`: Epistemic parameter uncertainty margin $\kappa \cdot \sigma_{\text{epi}}(\phi)$.
* `posterior_pred`: Full Bayesian predictive uncertainty margin.
* `hybrid`: Misspecification-aware hybrid margin $\kappa \cdot \sigma_{\text{epi}}(\phi) + q_k$.

### Running Unit Tests
```bash
py -m pytest novel_bayes/tests
```

### Running the Data-Size Sweep Experiment
```bash
py novel_bayes/run_novel.py --source repo
```

---

## 3. Head-to-Head Comparison (`compare/`)

To compare multi-step prediction RMSE across prediction horizons between the baseline STLSQ model and the novel Bayesian condensed model:

```bash
py compare/run_comparison.py --source repo
```

---

## Experimental Results Summary

### Multi-Step RMSE Comparison ($k=10$ steps / $0.1\text{ s}$)
| State Variable | Baseline STLSQ (Tuned $\lambda$) | Novel Bayesian Condensed |
| :--- | :--- | :--- |
| **$v_x$ (Longitudinal Velocity)** | `0.00047 m/s` | **`0.00046 m/s`** |
| **$v_y$ (Lateral Velocity)** | `0.02212 m/s` | **`0.02039 m/s`** |
| **$\omega$ (Yaw Rate)** | `0.01052 rad/s` | **`0.01040 rad/s`** |

### Key Findings
1. **Epistemic Uncertainty Collapse**: Purely epistemic posterior tightening (`posterior_epi`) becomes overly confident as data size $N$ increases (violation rates rise from $0.82 \to 0.98$), demonstrating the necessity of misspecification-aware terms.
2. **Hybrid Tightening Reliability**: The `hybrid` scheme maintains constraint violation rates strictly below target $\epsilon = 0.05$ across all dataset sizes.
3. **Multi-Step Prediction Precision**: The condensed Bayesian multi-step predictor achieves lower multi-step $v_y$ error than recursive 1-step STLSQ rollout.
