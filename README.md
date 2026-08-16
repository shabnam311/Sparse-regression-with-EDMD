# Sparse Regression with EDMD (Koopman Operator Modeling)

This project builds a data-driven Koopman operator model of a truck's longitudinal and lateral dynamics using Extended Dynamic Mode Decomposition (EDMD) with sparse regression. The pipeline is designed to process simulated trajectory datasets (such as those from IPG TruckMaker), lift the state into a higher-dimensional observable space, and identify a linear discrete-time mapping:

$$ z_{k+1} = A z_k + B u_k $$

The resulting $A$ and $B$ matrices are sparse and highly predictive over long horizons, making them perfectly suited for linear Model Predictive Control (MPC).

## Features
* **Modular Pipeline**: Clean separation of data loading, lifting, sparse regression (STLSQ), validation, and orchestration.
* **Variable Trajectory Lengths**: Seamlessly handles trajectories of varying durations without artificial truncation or boundary crossover.
* **Data Cleaning**: Automatically detects and drops irregular timestamp resets (`dt <= 0`) common in simulator outputs.
* **Koopman Lifting**: Utilizes a Degree 2 polynomial lifting dictionary of the physical states.
* **Sparse Regression**: Leverages PySINDy's Sequential Thresholded Least Squares (STLSQ) applied directly on the lifted discrete-time mappings.
* **Multi-Step Open-Loop Validation**: Extensively validates the identified linear map's stability by predicting up to 30,000 steps forward in open-loop.

## Project Structure
* `data_loader.py` - Parses the dataset, splits into Train/Val/Test, cleans irregular time steps, scales inputs/states on the train set, and prepares one-step transition pairs.
* `lifting.py` - Implements the $\phi(x)$ Koopman lifting dictionary. Expands the base state $[v_x, v_y, \omega]$ with polynomial terms up to degree 2 (creating 9 dimensions).
* `koopman_fit.py` - Implements the STLSQ sparse regression using PySINDy. Sweeps over a grid of sparsity threshold values ($\lambda$) to find the optimal balance between accuracy and sparsity.
* `validate.py` - Runs both one-step-ahead validation metrics (RMSE, MAE) and long-horizon multi-step (open-loop) rollouts on the Test set trajectories, generating comparison plots.
* `main.py` - The primary orchestration script tying all components together.

## Performance Results
The model was trained on roughly **1.2 million transition pairs** and validated against 11 entirely unseen test trajectories.
* **Optimal Sparsity**: With a threshold of $\lambda = 0.005$, the final $A$ matrix contains only 11 non-zero terms out of 81, and $B$ contains 3 out of 18. All 9 lifted features contribute to the final system.
* **Predictive Accuracy**: Over long open-loop rollouts extending up to 29,000 continuous steps (290 seconds), the model demonstrated incredible stability.
    * Yaw rate ($\omega$) RMSE $\approx 0.0007$ rad/s
    * Lateral velocity ($v_y$) RMSE $\approx 0.07$ m/s
    * Longitudinal velocity ($v_x$) follows physical constraints effectively with minimal drift.

## Artifacts
The final trained matrices (`A.npy`, `B.npy`) and Scikit-Learn standard scalers (`scaler_x.pkl`, `scaler_u.pkl`) are saved inside the `models/` directory, readily accessible for deployment within an MPC controller. Visual rollout graphs are exported to the `results/` directory.

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
