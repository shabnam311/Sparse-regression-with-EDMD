# Sparse Regression with EDMD (Koopman Operator Modeling)

This project builds a data-driven Koopman operator model of a truck's longitudinal and lateral dynamics using Extended Dynamic Mode Decomposition (EDMD) with sparse regression. The pipeline is designed to process simulated trajectory datasets (from IPG TruckMaker), lift the state into a higher-dimensional observable space, and identify a linear discrete-time mapping:

$$ z_{k+1} = A z_k + B u_k + C $$

The resulting $A$, $B$, and $C$ (intercept) matrices are sparse and predictive over long horizons, making them perfectly suited for linear Model Predictive Control (MPC).

## Features
* **Modular Pipeline**: Clean separation of data loading, lifting, sparse regression (STLSQ), validation, and orchestration.
* **Robust Header & Unit Validation**: Dynamically parses TruckMaker's 3-line header block to map columns safely by name (not just position), validates physical units (e.g. converting steering from `deg` to `rad` or time from `ms` to `s`), and calculates the nominal timestep (`dt`) dynamically using the dataset's median time-delta rather than hardcoding.
* **Variable Trajectory Lengths**: Seamlessly handles trajectories of varying durations without artificial truncation or boundary crossover.
* **Data Cleaning**: Automatically detects and drops irregular timestamp resets (`dt <= 0`) common in simulator outputs.
* **Standardized Koopman Lifting**: Normalizes state variables cleanly by standard deviation (pure scalar normalization, no mean shifting, keeping the physical origin at zero). The lifting function then operates on these dimension-consistent scaled variables (e.g., `vx_scaled`, `vy_scaled^2`) expanding the base state $[v_x, v_y, \omega]$ into a 9-dimensional subspace. 
* **Sparse Regression with Intercept**: Leverages PySINDy's Sequential Thresholded Least Squares (STLSQ) on the scaled variables, actively learning an intercept $C$ to correctly handle baseline geometric offsets.
* **Multi-Step Open-Loop Validation**: Extensively validates the identified linear map's stability by predicting up to ~29,800 steps forward in open-loop.

## Project Structure
* `data_loader.py` - Parses headers and units dynamically, splits into Train/Val/Test (60/5/5), cleans irregular time steps, scales inputs/states (pure standard deviation scaling), and prepares one-step transition pairs.
* `lifting.py` - Implements the $\phi(\hat{x})$ Koopman lifting dictionary using degree 2 polynomials.
* `koopman_fit.py` - Implements the STLSQ sparse regression. Sweeps over a grid of sparsity threshold values ($\lambda$) to find the optimal balance between accuracy and sparsity.
* `validate.py` - Runs both one-step-ahead validation metrics (RMSE, MAE) and long-horizon multi-step (open-loop) rollouts on the Test set trajectories, generating comparison plots with contextual dataset ranges.
* `main.py` - The primary orchestration script tying all components together.

## Performance Results
The model was trained on roughly **1.46 million transition pairs** (from 60 training trajectories) and validated against 5 entirely unseen test trajectories.
* **Optimal Sparsity**: With a threshold of $\lambda = 0.001$, the final $A$ matrix contains only 26 non-zero terms out of 81, $B$ contains 3 out of 18, and $C$ uses 6 terms. All 9 scaled lifted features contribute to the final system.
* **Predictive Accuracy**: Over long open-loop rollouts extending up to ~29,800 continuous steps (298 seconds), the model demonstrated incredible stability:
    * **Yaw rate ($\omega$)**: RMSE $\approx 0.0007$ rad/s (Operating range typically ~0.80 rad/s)
    * **Lateral velocity ($v_y$)**: RMSE $\approx 0.07$ m/s (Operating range typically ~5-7 m/s)
    * **Longitudinal velocity ($v_x$)**: Handles drift reasonably well over 5-minute continuous open-loop spans (Operating range ~30+ m/s).

## Artifacts
The final trained matrices (`A.npy`, `B.npy`, `C.npy`) and Scikit-Learn standard scalers (`scaler_x.pkl`, `scaler_u.pkl`) are saved inside the `models/` directory, readily accessible for deployment within an MPC controller. Visual rollout graphs are exported to the `results/` directory.

## Requirements
* `numpy`
* `pandas`
* `scikit-learn`
* `matplotlib`
* `pysindy`

To run the full pipeline, simply execute:
```bash
python main.py
```
