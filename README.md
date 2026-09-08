# Sparse Regression with EDMD (Koopman Operator Modeling)

This repository provides a data-driven Koopman operator model of a heavy vehicle's longitudinal and lateral dynamics using Extended Dynamic Mode Decomposition (EDMD) with Sparse Identification of Nonlinear Dynamics (SINDy / STLSQ).

The pipeline processes simulation datasets generated from IPG TruckMaker, lifts the state vector into a higher-dimensional polynomial observable space, and identifies a sparse discrete-time linear state-space representation:

$$ z_{k+1} = A z_k + B u_k $$

Where:
* $x = [v_x, v_y, \omega]^T$ is the base state (longitudinal velocity, lateral velocity, yaw rate).
* $u = [F_x, \delta]^T$ is the control input vector (drive force, steering angle).
* $z = \phi(x) \in \mathbb{R}^9$ is the lifted polynomial observable state.

---

## Key Features
* **Modular Architecture**: Clean separation of data loading, polynomial lifting, STLSQ sparse regression, validation, MATLAB integration, and graphical dashboard reporting.
* **Format & Unit Parsing**: Dynamically parses both standard CSV files (`time, Fx, delta, vx, vy, omega`) and TruckMaker's 3-line header blocks, validating SI units ($m/s, rad/s, N, rad, s$).
* **Trajectory-Based Splitting**: Splits trajectories cleanly (60 Train / 5 Validation / 5 Test) to prevent data leakage across simulation runs.
* **Data Preprocessing & Scaling**: Automatically detects and drops irregular timestamp resets (`dt <= 0`). Scales state and control inputs using standard deviation scaling (preserving the physical origin at zero).
* **Polynomial Koopman Lifting**: Expands the base state $[v_x, v_y, \omega]$ into a 9-dimensional polynomial subspace ($v_x, v_y, \omega, v_x^2, v_y^2, \omega^2, v_x v_y, v_x \omega, v_y \omega$).
* **Sparse Identification via STLSQ**: Uses PySINDy's Sequential Thresholded Least Squares (STLSQ) optimizer with cross-validation selection to produce lightweight matrices suited for real-time linear Model Predictive Control (MPC).
* **MATLAB & Simulink Integration**: Exports trained model matrices and scaling vectors directly to `.mat` format for control system prototyping in MATLAB/Simulink.

---

## Project Structure
* `data_loader.py` - Parses trajectory files, validates units, splits datasets (60/5/5), scales data, and generates 372,000 transition pairs.
* `lifting.py` - Implements the 9-dimensional degree-2 polynomial lifting dictionary.
* `koopman_fit.py` - Executes STLSQ sparse regression with threshold sweeping ($\lambda$).
* `validate.py` - Calculates one-step prediction metrics and multi-step open-loop rollout stability.
* `plot_metrics.py` - Generates visual evaluation dashboards (`results/accuracy_and_rmse_summary.png`, `results/evaluation_dashboard.png`).
* `run_open_loop_scenarios.py` - Evaluates the model against benchmark ground-truth scenarios (Scenario 1 & Scenario 2) and calculates relative RMSE and accuracy percentages.
* `export_to_matlab.py` - Exports model parameters to `Test 1/mpc_design_data_dim9.mat` for MATLAB/Simulink.
* `main.py` - Primary orchestration script tying all pipeline components together.

---

## Dataset Overview
* **Source**: IPG TruckMaker Vehicle Dynamics Simulations (`Dataset_01.csv` to `Dataset_70.csv`).
* **Total Trajectories**: 70 files (6,201 timesteps / 62 seconds per file at $100\text{ Hz}$).
* **Dataset Splitting**:
  * **Train Set (60 files)**: 372,000 transition pairs
  * **Validation Set (5 files)**: 31,000 transition pairs
  * **Test Set (5 files)**: 31,000 transition pairs

---

## Performance Results

### Model Structure and Sparsity ($\lambda = 0.05$)
* **Matrix $A$ ($9 \times 9$)**: 14 active non-zero terms out of 81 (82.7% sparse).
* **Matrix $B$ ($9 \times 2$)**: 1 active non-zero term out of 18 (94.4% sparse).
* **Lifted Features**: All 9 lifted state observables are actively utilized.

### One-Step-Ahead Test Performance ($dt = 0.01\text{ s}$)
Evaluated across all 31,000 transition pairs in the unseen test set:

| State Variable | Operating Range | Test RMSE | Test MAE | Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- |
| **$v_x$ (Longitudinal Velocity)** | $6.85\text{ m/s}$ | $0.0007\text{ m/s}$ | $0.0005\text{ m/s}$ | **99.99%** |
| **$v_y$ (Lateral Velocity)** | $0.07\text{ m/s}$ | $0.0001\text{ m/s}$ | $0.0000\text{ m/s}$ | **99.86%** |
| **$\omega$ (Yaw Rate)** | $0.03\text{ rad/s}$ | $0.0000\text{ rad/s}$ | $0.0000\text{ rad/s}$ | **99.98%** |

---

## Benchmark Scenario Validation

The trained model was evaluated against open-loop benchmark scenarios using TruckMaker ground-truth data:

### Scenario 1: Constant Deceleration ($F_x = -1000\text{ N}, \delta = 0\text{ rad}$)
Initial state $x_0 = [25\text{ m/s}, 0, 0]$ over $N = 400$ steps ($4.0\text{ seconds}$):

| Timestep ($N$) | Elapsed Time | Relative RMSE (%) | Accuracy (%) |
| :--- | :--- | :--- | :--- |
| **$N = 10$** | $0.1\text{ s}$ | `0.0765%` | **99.92%** |
| **$N = 80$** | $0.8\text{ s}$ | `0.4645%` | **99.54%** |
| **$N = 200$** | $2.0\text{ s}$ | `1.1704%` | **98.83%** |
| **$N = 400$** | $4.0\text{ s}$ | `2.3629%` | **97.64%** |

### Scenario 2: Sine-Wave Steering ($F_x = 5000\sin(t)+1000\text{ N}, \delta = 5^\circ\sin(t)$)
Initial state $x_0 = [25\text{ m/s}, 0, 0]$ over $N = 400$ steps ($4.0\text{ seconds}$):

| Timestep ($N$) | Elapsed Time | Relative RMSE (%) | Accuracy (%) | Paper Benchmark Accuracy (30-dim) |
| :--- | :--- | :--- | :--- | :--- |
| **$N = 10$** | $0.1\text{ s}$ | `0.0558%` | **99.94%** | 99.89% |
| **$N = 80$** | $0.8\text{ s}$ | `0.5122%` | **99.49%** | 99.00% |
| **$N = 200$** | $2.0\text{ s}$ | `2.3047%` | **97.70%** | 98.50% |
| **$N = 400$** | $4.0\text{ s}$ | `4.4384%` | **95.56%** | 98.49% |

---

## Artifacts and Output Files

* **`models/`**:
  * `A.npy`, `B.npy`: Trained NumPy matrices.
  * `A_matrix.csv`, `B_matrix.csv`, `matrix_summary.txt`: Human-readable text and CSV matrix exports.
  * `scaler_x.pkl`, `scaler_u.pkl`: Saved Scikit-Learn standard scalers.
* **`Test 1/`**:
  * `mpc_design_data_dim9.mat`: Exported MATLAB model file containing $A_l, B_l, C_l, x_{mean}, x_{std}, u_{mean}, u_{std}, T_s$.
  * `run_open_loop_validation.m`: MATLAB validation script.
  * `scenario_1_simulation_data.csv`, `scenario_2_simulation_data.csv`: Ground-truth benchmark logs.
* **`results/`**:
  * `accuracy_and_rmse_summary.png`: Single-panel accuracy bar chart.
  * `evaluation_dashboard.png`: 4-panel executive dashboard (Parity plots, Pole-Zero map, Error growth).
  * `scenario_1_validation.png`, `scenario_2_validation.png`: Open-loop scenario comparison plots.
  * `EVALUATION_REPORT.txt`, `EVALUATION_REPORT.md`: Comprehensive plain text and Markdown reports.

---

## Installation & Usage

### Requirements
* `python 3.10+`
* `numpy`
* `pandas`
* `scikit-learn`
* `matplotlib`
* `scipy`
* `pysindy`

### Running the Full Pipeline
To train the model and generate all validation results:
```bash
py main.py
```

### Running Benchmark Scenario Validation
To evaluate Scenario 1 & Scenario 2 against TruckMaker ground truth:
```bash
py run_open_loop_scenarios.py
```

### Generating Visual Dashboards
To re-render summary plots:
```bash
py plot_metrics.py
```

### Exporting to MATLAB
To export model matrices to `.mat` format:
```bash
py export_to_matlab.py
```
