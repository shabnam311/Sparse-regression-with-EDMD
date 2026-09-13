import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
from lifting import lift_state

def generate_combined_scenario_plots():
    print("==========================================================================")
    print(" GENERATING COMBINED SCENARIO 1 & 2 VALIDATION PLOTS (MATCHING IMAGE LAYOUT)")
    print("==========================================================================")

    # 1. Load trained model matrices & scalers
    A = np.load('models/A.npy')
    B = np.load('models/B.npy')

    with open('models/scaler_x.pkl', 'rb') as f:
        scaler_x = pickle.load(f)
    with open('models/scaler_u.pkl', 'rb') as f:
        scaler_u = pickle.load(f)

    # 2. Setup simulation parameters
    N = 400 # 400 steps (4.0 seconds @ 100Hz)
    Ts = 0.01
    t_vec = np.arange(N) * Ts
    checkpoints = [10, 80, 200, 400]

    x0 = np.array([25.0, 0.0, 0.0]) # Initial state [vx, vy, omega]

    # Input sequences
    Fx_seq_1 = -1000.0 * np.ones(N)
    delta_seq_1 = np.zeros(N)

    Fx_seq_2 = 5000.0 * np.sin(t_vec) + 1000.0
    delta_seq_2 = 5.0 * np.sin(t_vec) * np.pi / 180.0

    # 3. Predict helper
    def simulate_koopman(x0, Fx_seq, delta_seq, N):
        x_pred = np.zeros((N + 1, 3))
        x_pred[0] = x0
        x_curr_raw = x0.copy()

        for k in range(N):
            x_curr_sc = scaler_x.transform(x_curr_raw.reshape(1, -1))
            u_phys = np.array([[Fx_seq[k], delta_seq[k]]])
            u_curr_sc = scaler_u.transform(u_phys)

            z_curr = lift_state(x_curr_sc)[0]
            z_next = A @ z_curr + B @ u_curr_sc[0]

            x_next_sc = z_next[:3].reshape(1, -1)
            x_next_raw = scaler_x.inverse_transform(x_next_sc)[0]

            x_pred[k + 1] = x_next_raw
            x_curr_raw = x_next_raw

        return x_pred

    x_pred_1 = simulate_koopman(x0, Fx_seq_1, delta_seq_1, N)
    x_pred_2 = simulate_koopman(x0, Fx_seq_2, delta_seq_2, N)

    # Load Ground Truth
    gt1 = pd.read_csv(r'Test 1/scenario_1_simulation_data.csv')
    gt2 = pd.read_csv(r'Test 1/scenario_2_simulation_data.csv')

    x_true_1 = gt1[['vx', 'vy', 'YawRate']].values[:N+1]
    x_true_2 = gt2[['vx', 'vy', 'YawRate']].values[:N+1]

    # Compute Relative RMSE (%) at checkpoints
    def compute_rmse_pct(x_pred, x_true, checkpoints):
        rmse_pcts = []
        for cp in checkpoints:
            pred_slice = x_pred[:cp+1]
            true_slice = x_true[:cp+1]

            err = pred_slice - true_slice
            num = np.sqrt(np.sum(err**2))
            den = np.sqrt(np.sum(true_slice**2))
            rmse_pcts.append((num / den) * 100.0)
        return rmse_pcts

    sc1_rmse = compute_rmse_pct(x_pred_1, x_true_1, checkpoints)
    sc2_rmse = compute_rmse_pct(x_pred_2, x_true_2, checkpoints)

    paper_sc1 = [0.06, 0.13, 0.26, 0.39]
    paper_sc2 = [0.11, 1.00, 1.50, 1.51]

    print("\n--- EXACT RMSE VALUES (%) AT TIMESTEPS [10, 80, 200, 400] ---")
    print("Scenario 1 (Deceleration):")
    print("  Our Model (Dim 9):", [round(v, 4) for v in sc1_rmse])
    print("  Paper-30 (Ref):   ", paper_sc1)

    print("Scenario 2 (Lane Change):")
    print("  Our Model (Dim 9):", [round(v, 4) for v in sc2_rmse])
    print("  Paper-30 (Ref):   ", paper_sc2)

    os.makedirs('results', exist_ok=True)
    t_axis = np.linspace(0, N*Ts, N+1)

    # Helper to plot 4-panel combined figure
    def make_combined_plot(title_prefix, sc_name, sc_rmse, paper_rmse, x_true, x_pred, filename):
        fig, axs = plt.subplots(4, 1, figsize=(8, 12), gridspec_kw={'height_ratios': [1.2, 1, 1, 1]})

        # Subplot 1: Open-Loop Model Comparison (RMSE vs Prediction Timestep T)
        axs[0].plot(checkpoints, sc_rmse, 'o-', color='#1f77b4', linewidth=2, markersize=6, label='Sparse-EDMD (Dim 9)')
        axs[0].plot(checkpoints, paper_rmse, 's--', color='black', linewidth=1.5, markersize=6, label='Paper-30 (Ref)')
        axs[0].set_title(f'{title_prefix} — Open-Loop Model Comparison', fontsize=12, fontweight='bold')
        axs[0].set_xlabel('Prediction Timestep T', fontsize=10)
        axs[0].set_ylabel('Relative RMSE (%)', fontsize=10)
        axs[0].set_xticks(checkpoints)
        axs[0].legend(fontsize=9, loc='upper left')
        axs[0].grid(True, linestyle=':', alpha=0.6)

        # Subplot 2: Longitudinal Velocity vx
        axs[1].plot(t_axis, x_true[:, 0], 'k-', label='TruckMaker Ground Truth', linewidth=1.8)
        axs[1].plot(t_axis, x_pred[:, 0], '--', color='#1f77b4', label='Sparse-EDMD (Dim 9)', linewidth=1.8)
        axs[1].set_title(f'{title_prefix} — Longitudinal Velocity $v_x$', fontsize=11, fontweight='bold')
        axs[1].set_ylabel(r'$v_x$ (m/s)', fontsize=10)
        axs[1].legend(fontsize=9, loc='lower left')
        axs[1].grid(True, linestyle=':', alpha=0.6)

        # Subplot 3: Lateral Velocity vy
        axs[2].plot(t_axis, x_true[:, 1], 'k-', label='TruckMaker Ground Truth', linewidth=1.8)
        axs[2].plot(t_axis, x_pred[:, 1], '--', color='#1f77b4', label='Sparse-EDMD (Dim 9)', linewidth=1.8)
        axs[2].set_title(f'{title_prefix} — Lateral Velocity $v_y$', fontsize=11, fontweight='bold')
        axs[2].set_ylabel(r'$v_y$ (m/s)', fontsize=10)
        axs[2].legend(fontsize=9, loc='lower left')
        axs[2].grid(True, linestyle=':', alpha=0.6)

        # Subplot 4: Yaw Rate omega
        axs[3].plot(t_axis, x_true[:, 2], 'k-', label='TruckMaker Ground Truth', linewidth=1.8)
        axs[3].plot(t_axis, x_pred[:, 2], '--', color='#1f77b4', label='Sparse-EDMD (Dim 9)', linewidth=1.8)
        axs[3].set_title(f'{title_prefix} — Yaw Rate $\omega$', fontsize=11, fontweight='bold')
        axs[3].set_xlabel('Time (s)', fontsize=10)
        axs[3].set_ylabel(r'$\omega$ (rad/s)', fontsize=10)
        axs[3].legend(fontsize=9, loc='lower left')
        axs[3].grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved combined plot: {filename}")

    make_combined_plot('Scenario 1: Deceleration', 'Scenario 1', sc1_rmse, paper_sc1, x_true_1, x_pred_1, 'results/scenario_1_combined_validation.png')
    make_combined_plot('Scenario 2: Lane Change', 'Scenario 2', sc2_rmse, paper_sc2, x_true_2, x_pred_2, 'results/scenario_2_combined_validation.png')

if __name__ == '__main__':
    generate_combined_scenario_plots()
