# Sparse Regression with EDMD (Koopman Operator Modeling)

This project builds a data-driven Koopman operator model of a truck's longitudinal and lateral dynamics using Extended Dynamic Mode Decomposition (EDMD) with sparse regression (SINDy/STLSQ). The pipeline processes simulated trajectory datasets from IPG TruckMaker (`Dataset.rar`), lifts the state into a higher-dimensional polynomial observable space, and identifies a linear discrete-time mapping:

$$ z_{k+1} = A z_k + B u_k $$

The resulting $A$ and $B$ matrices are sparse, physically interpretable, and highly stable over long horizons, making them perfectly suited for linear Model Predictive Control (MPC).

## Features
* **Modular Pipeline**: Clean separation of data loading, lifting, sparse regression (STLSQ), validation, graphical dashboard reporting, and orchestration.
* **Flexible Header & Format Support**: Dynamically parses both standard CSV files (`time, Fx, delta, vx, vy, omega`) and TruckMaker's 3-line header blocks, validating physical units and sampling rate ($dt = 0.01\text{ s}$).
* **Multi-Trajectory Splitting**: Splits trajectories cleanly (60 Train / 5 Validation / 5 Test) to prevent data leakage between simulation runs.
* **Data Cleaning**: Automatically detects and drops irregular timestamp resets (`dt <= 0`) or missing values.
* **Standardized Koopman Lifting**: Scales state variables cleanly by standard deviation (pure multiplicative scaling, preserving origin $0 \to 0$). The lifting function expands the base state $[v_x, v_y, \omega]$ into a 9-dimensional polynomial subspace ($v_x, v_y, \omega, v_x^2, v_y^2, \omega^2, v_x v_y, v_x \omega, v_y \omega$).
* **Sparse Regression via STLSQ**: Uses PySINDy's Sequential Thresholded Least Squares (STLSQ) with a 1-SE validation rule to select the sparsest stable model.
* **Multi-Step Open-Loop Validation**: Evaluates model stability by simulating forward in open-loop for thousands of steps without feedback.
* **Graphical Evaluation Dashboards**: Automatically outputs single-panel accuracy bar charts, parity correlation plots, eigenvalue stability maps, and rollout comparison figures.

## Project Structure
* `data_loader.py` - Loads `Dataset_01.csv` to `Dataset_70.csv`, parses units, splits 60/5/5, scales data, and generates 372,000 transition pairs.
* `lifting.py` - Implements the $\phi(x)$ Koopman lifting dictionary using degree-2 polynomials (9 observables).
* `koopman_fit.py` - Implements STLSQ sparse regression with threshold sweeping ($\lambda$).
* `validate.py` - Computes one-step metrics (RMSE, MAE, Accuracy %) and multi-step open-loop rollouts.
* `plot_metrics.py` - Generates visual evaluation dashboards (`results/accuracy_and_rmse_summary.png`, `results/evaluation_dashboard.png`).
* `main.py` - The primary orchestration script tying all components together.

## Performance Results (New Dataset Run)
The model was trained on **372,000 transition pairs** (from 60 training trajectories) and evaluated against 5 unseen test trajectories:

* **Sparsity ($\lambda = 0.05$)**: The $A$ matrix contains **14 active terms** out of 81 (82.7% sparse), and $B$ contains **1 active term** out of 18 (94.4% sparse).
* **1-Step Test Accuracy ($dt = 0.01\text{ s}$)**:
  * **$v_x$ (Longitudinal Velocity)**: RMSE = `0.0007 m/s` (Range: `6.85 m/s`) $\rightarrow$ **99.99% Accuracy**
  * **$v_y$ (Lateral Velocity)**: RMSE = `0.0001 m/s` (Range: `0.07 m/s`) $\rightarrow$ **99.86% Accuracy**
  * **$\omega$ (Yaw Rate)**: RMSE = `0.0000 rad/s` (Range: `0.03 rad/s`) $\rightarrow$ **99.98% Accuracy**
* **Open-Loop Stability**: Over 6,201-step (62-second) continuous open-loop rollouts, discrete-time eigenvalues remain bounded on/inside the unit circle ($|z| \le 1$), preventing mathematical explosion.

## Artifacts & Outputs
* **`models/`**: Saved matrices (`A.npy`, `B.npy`), human-readable exports (`A_matrix.csv`, `B_matrix.csv`, `matrix_summary.txt`), and Scikit-Learn scalers (`scaler_x.pkl`, `scaler_u.pkl`).
* **`results/`**: Graphical dashboards (`accuracy_and_rmse_summary.png`, `evaluation_dashboard.png`), rollout plots (`rollout_1.png`, `rollout_2.png`, `rollout_3.png`), and plain text/markdown evaluation summaries (`EVALUATION_REPORT.txt`, `EVALUATION_REPORT.md`).

## Requirements & Usage
* `numpy`
* `pandas`
* `scikit-learn`
* `matplotlib`
* `pysindy`

To run the entire pipeline:
```bash
py main.py
```

To generate updated graphical dashboard plots:
```bash
py plot_metrics.py
```
