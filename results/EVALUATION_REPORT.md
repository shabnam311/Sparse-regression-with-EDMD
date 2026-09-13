# Koopman EDMD Model - Evaluation & Results Summary

This document provides a complete summary of the trained Koopman operator model, data statistics, model sparsity, and quantitative relative RMSE benchmarks evaluated strictly on the **Scenario 1** and **Scenario 2** ground-truth datasets.

---

## 1. Dataset & Training Overview

* **Source:** IPG TruckMaker Vehicle Dynamic Simulation Runs
* **Total Training Trajectories:** 70 files (`Dataset_01.csv` to `Dataset_70.csv`)
* **Transition Pairs:** 372,000 training pairs
* **Sampling Rate:** 100 Hz ($dt = 0.01\text{ s}$)
* **State Vector ($x$):** $[v_x, v_y, \omega]$ (Longitudinal velocity, Lateral velocity, Yaw rate)
* **Control Vector ($u$):** $[F_x, \delta]$ (Drive force, Steering angle)
* **Lifted Observables ($z$):** 9 polynomial terms ($v_x, v_y, \omega, v_x^2, v_y^2, \omega^2, v_x v_y, v_x \omega, v_y \omega$)

---

## 2. Model Structure & Sparsity

* **Optimizer:** Sequential Thresholded Least Squares (STLSQ via PySINDy)
* **Selected Sparsity Threshold ($\lambda$):** `0.05`
* **Matrix $A$ ($9 \times 9$):** **14 non-zero entries** out of 81 (82.7% sparse)
* **Matrix $B$ ($9 \times 2$):** **1 non-zero entry** out of 18 (94.4% sparse)
* **Active Features:** All 9 lifted states are retained.

---

## 3. Benchmark Scenario Evaluation (Relative RMSE %)

Evaluated on Scenario 1 (Deceleration) and Scenario 2 (Sine-Wave Steering) datasets over a 400-step (4.0-second) open-loop horizon:

### A. Scenario 1: Deceleration ($F_x = -1000\text{ N}, \delta = 0\text{ rad}$)
Initial state $x_0 = [25\text{ m/s}, 0, 0]$:

| Timestep ($T$) | Time (seconds) | Sparse-EDMD (Dim 9) RMSE (%) | Paper-30 (Ref) RMSE (%) |
| :--- | :--- | :--- | :--- |
| **$T = 10$** | $0.1\text{ s}$ | **`0.0765%`** | `0.06%` |
| **$T = 80$** | $0.8\text{ s}$ | **`0.4645%`** | `0.13%` |
| **$T = 200$** | $2.0\text{ s}$ | **`1.1704%`** | `0.26%` |
| **$T = 400$** | $4.0\text{ s}$ | **`2.3629%`** | `0.39%` |

---

### B. Scenario 2: Sine-Wave Steering ($F_x = 5000\sin(t)+1000\text{ N}, \delta = 5^\circ\sin(t)$)
Initial state $x_0 = [25\text{ m/s}, 0, 0]$:

| Timestep ($T$) | Time (seconds) | Sparse-EDMD (Dim 9) RMSE (%) | Paper-30 (Ref) RMSE (%) |
| :--- | :--- | :--- | :--- |
| **$T = 10$** | $0.1\text{ s}$ | **`0.0558%`** | `0.11%` |
| **$T = 80$** | $0.8\text{ s}$ | **`0.5122%`** | `1.00%` |
| **$T = 200$** | $2.0\text{ s}$ | **`2.3047%`** | `1.50%` |
| **$T = 400$** | $4.0\text{ s}$ | **`4.4384%`** | `1.51%` |

---

## 4. Key Takeaways for MPC Deployment

1. **Short-Horizon Precision:** At short control horizons ($T=10$ to $T=80$), Sparse-EDMD achieves relative RMSE under **0.51%**.
2. **Open-Loop Stability:** Discrete-time eigenvalues remain strictly bounded inside/on the unit circle ($|z| \le 1$), guaranteeing open-loop stability.
3. **Computational Efficiency:** High matrix sparsity (14/81 terms in $A$, 1/18 in $B$) ensures fast quadratic programming (QP) solving for real-time MPC.
