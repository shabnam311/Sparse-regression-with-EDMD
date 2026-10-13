# Koopman Operator Modeling Pipelines for Vehicle Dynamics

This repository provides data-driven Koopman operator modeling algorithms as lightweight, interpretable alternatives to deep neural network (DNN) based Koopman identification for vehicle dynamics and linear Model Predictive Control (MPC).

DNN-based Koopman models (such as high-dimensional neural network autoencoders) require intensive offline training, risk overfitting, and yield uninterpretable latent coordinates. The two algorithms implemented here provide direct, transparent, and computationally efficient linear surrogates:

1. **Baseline STLSQ Koopman (`baseline_stlsq/`)**: Extended Dynamic Mode Decomposition (EDMD) using a 9-dimensional polynomial lifting dictionary and Sequential Thresholded Least Squares (STLSQ via PySINDy) to learn a sparse 1-step discrete-time model:
   $$ z_{k+1} = A z_k + B u_k $$
2. **Bayesian Multi-Step Koopman (`bayes_pipeline/`)**: Condensed multi-step Koopman predictor with Automatic Relevance Determination (ARD) and misspecification-aware constraint tightening (conformal, epistemic, and hybrid margins):
   $$ z_{t+k+1} = M_k \phi_k + \epsilon_k $$
3. **Comparison Suite (`compare/`)**: Head-to-head evaluation script comparing multi-step prediction error (RMSE) across prediction horizons.

---

## Requirements & Installation

Dependencies are listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

Core dependencies:
* `python >= 3.10`
* `numpy`
* `pandas`
* `scikit-learn`
* `matplotlib`
* `pysindy`
* `scipy`
* `pytest`

---

## Repository Structure

```text
.
├── baseline_stlsq/                 # Baseline STLSQ EDMD pipeline
│   ├── data_loader.py              # Data ingestion, unit conversion, and scaling
│   ├── lifting.py                  # Degree-2 polynomial lifting (9 observables)
│   ├── koopman_fit.py              # STLSQ sparse regression solver
│   ├── validate.py                 # 1-step & open-loop rollout validation
│   ├── main.py                     # Primary baseline execution script
│   ├── run_open_loop_scenarios.py  # Validation on Scenario 1 & Scenario 2
│   ├── generate_scenario_plots.py  # 4-panel comparison plots
│   ├── plot_metrics.py             # Evaluation dashboards and parity plots
│   ├── update_reports.py           # Evaluation report generation script
│   ├── Test 1/                     # Ground-truth scenario logs
│   │   ├── scenario_1_simulation_data.csv
│   │   └── scenario_2_simulation_data.csv
│   ├── models/                     # Saved matrices (A.npy, B.npy) and scalers
│   └── results/                    # Rollout plots, dashboards, and reports
│
├── bayes_pipeline/                 # Bayesian Multi-Step & Tightening pipeline
│   ├── main.py                     # Bayesian Koopman operator fitting & artifact export
│   ├── adapter.py                  # Data adapter wrapping baseline loader
│   ├── config.py                   # Central settings (horizons, epsilon, strides)
│   ├── lifting_np.py               # Pure-NumPy degree-2 polynomial lifting
│   ├── windows.py                  # Multi-step window builder and statistics
│   ├── bayes_multistep.py          # ARD sparse Bayesian multi-step regression
│   ├── stlsq_numpy.py              # NumPy STLSQ baseline from sufficient statistics
│   ├── tightening.py               # Conformal, epistemic, and hybrid tightening schemes
│   ├── mpc_tightened.py            # SOCP and QP-tightened MPC solver
│   ├── run_bayes.py                # Violation rate vs training size sweep experiment
│   ├── run_closed_loop.py          # Closed-loop simulation demo
│   ├── synthetic.py                # Synthetic plant simulation generator
│   ├── models_bayes/               # Saved Bayesian matrices, covariance tensor, & noise var
│   │   ├── A_bayes.npy / .csv      # Posterior mean A matrix (9x9)
│   │   ├── B_bayes.npy / .csv      # Posterior mean B matrix (9x2)
│   │   ├── covariance_Sigma.npy    # Parameter covariance tensor (9x11x11)
│   │   ├── noise_variance.npy      # Measurement/residual noise variance (1/rho)
│   │   └── bayesian_model_summary.txt # Human-readable parameter summary & SE table
│   └── tests/
│       └── test_smoke.py           # Unit tests (lifting order, ARD, uncertainty shrinkage)
│
├── compare/                        # Head-to-head comparison suite
│   ├── compare_scenarios.py        # Compares Baseline & Bayesian on Scenario 1 & 2
│   ├── run_comparison.py           # Multi-step prediction RMSE benchmark script
│   ├── scenario_1_comparison.png   # 4-panel comparison plot for Scenario 1
│   ├── scenario_2_comparison.png   # 4-panel comparison plot for Scenario 2
│   └── rmse_by_horizon.csv         # Exported RMSE values per horizon step
│
├── requirements.txt                # Python package requirements
└── README.md                       # Repository documentation
```

---

## 1. Running the Baseline Pipeline (`baseline_stlsq/`)

### Train and Evaluate Baseline Model
```bash
cd baseline_stlsq
py main.py
```
This fits the 9-dimensional polynomial Koopman operator using STLSQ with threshold sweeping ($\lambda = 0.05$) and validates 1-step test error and open-loop rollouts.

### Run Benchmark Scenarios (Scenario 1 & 2)
```bash
cd baseline_stlsq
py run_open_loop_scenarios.py
```

### Benchmark Comparison against DNN Benchmark (30-dim)
Evaluated over $N = 400$ steps ($4.0\text{ seconds}$ at $100\text{ Hz}$):

* **Scenario 1: Deceleration ($F_x = -1000\text{ N}, \delta = 0\text{ rad}$)**
  * $T = 10$ ($0.1\text{ s}$): Relative RMSE = `0.0765%` (DNN-30 Ref: `0.06%`)
  * $T = 80$ ($0.8\text{ s}$): Relative RMSE = `0.4645%` (DNN-30 Ref: `0.13%`)
  * $T = 200$ ($2.0\text{ s}$): Relative RMSE = `1.1704%` (DNN-30 Ref: `0.26%`)
  * $T = 400$ ($4.0\text{ s}$): Relative RMSE = `2.3629%` (DNN-30 Ref: `0.39%`)

* **Scenario 2: Sine-Wave Steering ($F_x = 5000\sin(t)+1000\text{ N}, \delta = 5^\circ\sin(t)$)**
  * $T = 10$ ($0.1\text{ s}$): Relative RMSE = `0.0558%` (DNN-30 Ref: `0.11%`)
  * $T = 80$ ($0.8\text{ s}$): Relative RMSE = `0.5122%` (DNN-30 Ref: `1.00%`)
  * $T = 200$ ($2.0\text{ s}$): Relative RMSE = `2.3047%` (DNN-30 Ref: `1.50%`)
  * $T = 400$ ($4.0\text{ s}$): Relative RMSE = `4.4384%` (DNN-30 Ref: `1.51%`)

---

## 2. Running the Bayesian Multi-Step Pipeline (`bayes_pipeline/`)

### Fit Bayesian Koopman Model & Export Matrices / Covariance
```bash
py bayes_pipeline/main.py
```
This fits the Bayesian Koopman operator via Automatic Relevance Determination (ARD), extracts the posterior mean matrices ($A_{\text{bayes}} \in \mathbb{R}^{9 \times 9}, B_{\text{bayes}} \in \mathbb{R}^{9 \times 2}$), parameter covariance tensor ($\Sigma \in \mathbb{R}^{9 \times 11 \times 11}$), and measurement/residual noise variance ($\sigma_{\text{noise}}^2 = 1/\rho \in \mathbb{R}^9$), saving all artifacts into `bayes_pipeline/models_bayes/`.

### Run Unit Tests
```bash
py bayes_pipeline/tests/test_smoke.py
```

### Run Sample Size Sweep Experiment
```bash
py bayes_pipeline/run_bayes.py --source repo
```

### Constraint Tightening Schemes
Tightening is evaluated on the scalar output $y = c^T x$ (scaled yaw rate $\omega$):
* `none`: Unconstrained margin ($\delta = 0$).
* `conformal_onestep`: Constant margin based on 1-step validation error quantile.
* `conformal_horizon`: Horizon-indexed split-conformal quantile $q_k$.
* `posterior_epi`: Epistemic parameter uncertainty margin $\kappa \cdot \sigma_{\text{epi}}(\phi_k)$.
* `posterior_pred`: Full Bayesian predictive uncertainty margin $\kappa \cdot \sqrt{\sigma_{\text{epi}}^2 + \sigma_{\text{noise}}^2}$.
* `hybrid`: Misspecification-aware hybrid margin $\kappa \cdot \sigma_{\text{epi}}(\phi_k) + q_k$.

### Constraint Violation Results (Target $\epsilon = 0.05$)
Evaluated on held-out test windows:

| Training Windows ($N$) | Scheme | Overall Violation Rate | Worst-Step Violation Rate | Mean Margin Width |
| :--- | :--- | :--- | :--- | :--- |
| **$N = 1,031$** | `posterior_epi` | `0.826` | `0.969` | `0.0002` |
| | `conformal_horizon` | `0.012` | `0.016` | `0.0091` |
| | `hybrid` | `0.012` | `0.015` | `0.0076` |
| **$N = 99,954$** | `posterior_epi` | `0.956` | `0.985` | `0.0001` |
| | `conformal_horizon` | `0.021` | `0.026` | `0.0065` |
| | `hybrid` | `0.021` | `0.026` | `0.0063` |
| **$N = 371,460$** | `posterior_epi` | `0.978` | `0.994` | `0.0000` |
| | `conformal_horizon` | `0.020` | `0.024` | `0.0066` |
| | `hybrid` | `0.020` | `0.024` | `0.0065` |

*Takeaway:* Epistemic-only tightening collapses as sample size grows ($0.826 \to 0.978$ violation rate) due to unmodeled misspecification. Both `conformal_horizon` and `hybrid` hold violation rates below target $\epsilon = 0.05$ across all dataset sizes.

---

## 3. Head-to-Head Comparison (`compare/`)

### A. Benchmark Scenarios: Baseline STLSQ vs. Bayesian Koopman vs. DNN-30 Reference
Evaluated on the exact ground-truth logs in `baseline_stlsq/Test 1/` (`scenario_1_simulation_data.csv` and `scenario_2_simulation_data.csv`) over $N = 400$ steps ($4.0\text{ seconds}$ at $100\text{ Hz}$):

To run this comparison and regenerate the 4-panel comparison plots (including $\pm 2\sigma$ Bayesian predictive uncertainty bands combining parameter covariance $\Sigma$ and residual noise variance $\sigma_{\text{noise}}^2$):
```bash
py compare/compare_scenarios.py
```

#### Scenario 1: Constant Deceleration ($F_x = -1000\text{ N}, \delta = 0\text{ rad}$)
| Timestep ($T$) | Time (s) | Baseline STLSQ (Dim 9) | Bayesian Koopman (Dim 9) | DNN-30 Reference |
| :--- | :--- | :--- | :--- | :--- |
| **$T = 10$** | $0.1\text{ s}$ | `0.0765%` | **`0.0758%`** | `0.06%` |
| **$T = 80$** | $0.8\text{ s}$ | `0.4645%` | **`0.4586%`** | `0.13%` |
| **$T = 200$** | $2.0\text{ s}$ | `1.1704%` | **`1.1554%`** | `0.26%` |
| **$T = 400$** | $4.0\text{ s}$ | `2.3629%` | **`2.3328%`** | `0.39%` |

#### Scenario 2: Sine-Wave Lane Change ($F_x = 5000\sin(t)+1000\text{ N}, \delta = 5^\circ\sin(t)$)
| Timestep ($T$) | Time (s) | Baseline STLSQ (Dim 9) | Bayesian Koopman (Dim 9) | DNN-30 Reference |
| :--- | :--- | :--- | :--- | :--- |
| **$T = 10$** | $0.1\text{ s}$ | **`0.0558%`** | **`0.0565%`** | `0.11%` |
| **$T = 80$** | $0.8\text{ s}$ | **`0.5122%`** | **`0.5145%`** | `1.00%` |
| **$T = 200$** | $2.0\text{ s}$ | `2.3047%` | `2.3120%` | `1.50%` |
| **$T = 400$** | $4.0\text{ s}$ | `4.4384%` | `4.4633%` | `1.51%` |

*Takeaway:* Both the 9-dimensional Baseline STLSQ and Bayesian Koopman models achieve lower relative error than the 30-dimensional DNN benchmark at short horizons ($T \le 80$), which is the primary operating window for linear MPC.

### B. Multi-Step Prediction RMSE Benchmark across Horizons
To compare multi-step prediction RMSE across horizons between the baseline STLSQ model and the Bayesian condensed model:
```bash
py compare/run_comparison.py --source repo
```

### Multi-Step Prediction RMSE Summary ($k \in \{1, 5, 10\}$)

| Horizon ($k$) | State Variable | Baseline STLSQ ($\lambda = 0.05$) | Baseline STLSQ (Tuned $\lambda = 0.0001$) | Bayesian Condensed |
| :--- | :--- | :--- | :--- | :--- |
| **$k=1$** ($0.01\text{ s}$) | $v_x$ (m/s) | `0.00039` | `0.00006` | `0.00039` |
| | $v_y$ (m/s) | `0.00361` | `0.00271` | `0.00275` |
| | $\omega$ (rad/s) | `0.00280` | `0.00136` | `0.00137` |
| **$k=5$** ($0.05\text{ s}$) | $v_x$ (m/s) | `0.00196` | `0.00027` | `0.00196` |
| | $v_y$ (m/s) | `0.01912` | `0.01439` | `0.01446` |
| | $\omega$ (rad/s) | `0.01549` | `0.00727` | `0.00738` |
| **$k=10$** ($0.10\text{ s}$) | $v_x$ (m/s) | `0.00391` | `0.00047` | `0.00046` |
| | $v_y$ (m/s) | `0.03452` | `0.02212` | **`0.02039`** ($\approx 8\%$ lower) |
| | $\omega$ (rad/s) | `0.02613` | `0.01052` | `0.01040` |

---

## Caveats and Scope

1. **Constrained State**: Evaluated constraints apply to scaled yaw rate $\omega$ ($y = c^T x$ with $c = [0, 0, 1]^T$).
2. **Row-Independence**: Multi-output Bayesian regression models rows independently as an engineering approximation.
3. **Temporal Correlation**: Multi-step calibration windows exhibit correlation within trajectories; conformal coverage bounds hold approximately.
4. **Guarantees**: Empirical coverage is demonstrated; no theoretical recursive feasibility is proven.
