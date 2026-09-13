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
* **Data Preprocessing & Scaling**: Automatically detects and drops irregular timestamp resets (`dt <= 0`). Scales state and control inputs using standard deviation scaling (preserving the physical origin at zero).
* **Polynomial Koopman Lifting**: Expands the base state $[v_x, v_y, \omega]$ into a 9-dimensional polynomial subspace ($v_x, v_y, \omega, v_x^2, v_y^2, \omega^2, v_x v_y, v_x \omega, v_y \omega$).
* **Sparse Identification via STLSQ**: Uses PySINDy's Sequential Thresholded Least Squares (STLSQ) optimizer to produce lightweight matrices suited for real-time linear Model Predictive Control (MPC).
* **Benchmark Scenario Validation**: Evaluates model performance directly against Scenario 1 (Deceleration) and Scenario 2 (Sine-Wave Steering) ground-truth datasets (`scenario_1_simulation_data.csv` and `scenario_2_simulation_data.csv`).
* **MATLAB & Simulink Integration**: Exports trained model matrices and scaling vectors directly to `.mat` format for control system prototyping in MATLAB/Simulink.

---

## Project Structure
* `data_loader.py` - Parses trajectory files, validates units, scales data, and generates 372,000 transition pairs.
* `lifting.py` - Implements the 9-dimensional degree-2 polynomial lifting dictionary.
* `koopman_fit.py` - Executes STLSQ sparse regression with threshold sweeping ($\lambda$).
* `validate.py` - Calculates open-loop prediction metrics and rollout stability on benchmark scenarios.
* `generate_scenario_plots.py` - Generates combined 4-panel scenario validation figures matching the benchmark publication layout.
* `plot_metrics.py` - Generates visual evaluation dashboards (`results/accuracy_and_rmse_summary.png`, `results/evaluation_dashboard.png`).
* `run_open_loop_scenarios.py` - Evaluates the model against benchmark ground-truth scenarios (Scenario 1 & Scenario 2) and calculates relative RMSE values.
* `export_to_matlab.py` - Exports model parameters to `Test 1/mpc_design_data_dim9.mat` for MATLAB/Simulink.
* `main.py` - Primary orchestration script.

---

## Performance Results

### Model Structure and Sparsity ($\lambda = 0.05$)
* **Matrix $A$ ($9 \times 9$)**: 14 active non-zero terms out of 81 (82.7% sparse).
* **Matrix $B$ ($9 \times 2$)**: 1 active non-zero term out of 18 (94.4% sparse).
* **Lifted Features**: All 9 lifted state observables are actively utilized.

---

## Benchmark Scenario Validation (Relative RMSE %)

### Scenario 1: Constant Deceleration ($F_x = -1000\text{ N}, \delta = 0\text{ rad}$)
Initial state $x_0 = [25\text{ m/s}, 0, 0]$ over $N = 400$ steps ($4.0\text{ seconds}$):

| Timestep ($T$) | Time (seconds) | Sparse-EDMD (Dim 9) Relative RMSE (%) | Paper-30 (Ref) Relative RMSE (%) |
| :--- | :--- | :--- | :--- |
| **$T = 10$** | $0.1\text{ s}$ | **`0.0765%`** | `0.06%` |
| **$T = 80$** | $0.8\text{ s}$ | **`0.4645%`** | `0.13%` |
| **$T = 200$** | $2.0\text{ s}$ | **`1.1704%`** | `0.26%` |
| **$T = 400$** | $4.0\text{ s}$ | **`2.3629%`** | `0.39%` |

### Scenario 2: Sine-Wave Steering ($F_x = 5000\sin(t)+1000\text{ N}, \delta = 5^\circ\sin(t)$)
Initial state $x_0 = [25\text{ m/s}, 0, 0]$ over $N = 400$ steps ($4.0\text{ seconds}$):

| Timestep ($T$) | Time (seconds) | Sparse-EDMD (Dim 9) Relative RMSE (%) | Paper-30 (Ref) Relative RMSE (%) |
| :--- | :--- | :--- | :--- |
| **$T = 10$** | $0.1\text{ s}$ | **`0.0558%`** | `0.11%` |
| **$T = 80$** | $0.8\text{ s}$ | **`0.5122%`** | `1.00%` |
| **$T = 200$** | $2.0\text{ s}$ | **`2.3047%`** | `1.50%` |
| **$T = 400$** | $4.0\text{ s}$ | **`4.4384%`** | `1.51%` |

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
  * `scenario_1_combined_validation.png`: Combined 4-panel figure for Scenario 1 matching benchmark publication layout.
  * `scenario_2_combined_validation.png`: Combined 4-panel figure for Scenario 2 matching benchmark publication layout.
  * `accuracy_and_rmse_summary.png`: Relative RMSE bar chart across timesteps.
  * `evaluation_dashboard.png`: 4-panel executive dashboard (Parity plots, Pole-Zero map, Error growth).
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

### Running Benchmark Scenario Validation and Generating Combined Figures
To run validation on Scenario 1 & Scenario 2 and output the combined 4-panel plots:
```bash
py generate_scenario_plots.py
```

### Running the Full Pipeline
To retrain the model and update artifacts:
```bash
py main.py
```

### Exporting to MATLAB
To export model matrices to `.mat` format:
```bash
py export_to_matlab.py
```
