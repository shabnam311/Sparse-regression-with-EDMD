import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
from lifting import lift_state

def run_open_loop_scenarios():
    print("==========================================================================")
    print(" RUNNING OPEN-LOOP SCENARIO 1 & 2 VALIDATION (TRAINED KOOPMAN MODEL)")
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

    # --- Scenario 1 Input Sequence ---
    # Fx = -1000 N, delta = 0
    Fx_seq_1 = -1000.0 * np.ones(N)
    delta_seq_1 = np.zeros(N)

    # --- Scenario 2 Input Sequence ---
    # Fx = 5000*sin(t) + 1000, delta = 5*sin(t) in degrees -> rad
    Fx_seq_2 = 5000.0 * np.sin(t_vec) + 1000.0
    delta_seq_2 = 5.0 * np.sin(t_vec) * np.pi / 180.0

    # 3. Prediction simulation function
    def simulate_koopman(x0, Fx_seq, delta_seq, N):
        x_pred = np.zeros((N + 1, 3))
        x_pred[0] = x0

        # Current unnormalized physical state
        x_curr_raw = x0.copy()

        for k in range(N):
            # Scale current state and input
            x_curr_sc = scaler_x.transform(x_curr_raw.reshape(1, -1))
            u_phys = np.array([[Fx_seq[k], delta_seq[k]]])
            u_curr_sc = scaler_u.transform(u_phys)

            # Lift scaled state
            z_curr = lift_state(x_curr_sc)[0] # 9-dim

            # Koopman step: z_{k+1} = A * z_k + B * u_k
            z_next = A @ z_curr + B @ u_curr_sc[0]

            # Extract predicted scaled states and inverse transform
            x_next_sc = z_next[:3].reshape(1, -1)
            x_next_raw = scaler_x.inverse_transform(x_next_sc)[0]

            x_pred[k + 1] = x_next_raw
            x_curr_raw = x_next_raw # Open-loop roll forward

        return x_pred

    print("\nRunning Koopman open-loop prediction for Scenario 1 (Deceleration)...")
    x_pred_1 = simulate_koopman(x0, Fx_seq_1, delta_seq_1, N)

    print("Running Koopman open-loop prediction for Scenario 2 (Sine-Wave Steering)...")
    x_pred_2 = simulate_koopman(x0, Fx_seq_2, delta_seq_2, N)

    # 4. Import Ground-Truth CSVs
    sc1_path = r'Test 1/scenario_1_simulation_data.csv'
    sc2_path = r'Test 1/scenario_2_simulation_data.csv'

    gt1 = pd.read_csv(sc1_path)
    gt2 = pd.read_csv(sc2_path)

    x_true_1 = gt1[['vx', 'vy', 'YawRate']].values[:N+1]
    x_true_2 = gt2[['vx', 'vy', 'YawRate']].values[:N+1]

    # 5. Compute Relative RMSE (%) at Checkpoints
    def compute_rmse_pct(x_pred, x_true, checkpoints):
        rmse_pcts = []
        for cp in checkpoints:
            pred_slice = x_pred[:cp+1]
            true_slice = x_true[:cp+1]

            err = pred_slice - true_slice
            num = np.sqrt(np.sum(err**2))
            den = np.sqrt(np.sum(true_slice**2))
            rmse_pct = (num / den) * 100.0
            rmse_pcts.append(rmse_pct)
        return rmse_pcts

    sc1_rmse = compute_rmse_pct(x_pred_1, x_true_1, checkpoints)
    sc2_rmse = compute_rmse_pct(x_pred_2, x_true_2, checkpoints)

    # Paper comparison benchmarks (from Table I & II of the paper for Koopman models)
    paper_sc1 = [0.06, 0.13, 0.26, 0.39]
    paper_sc2 = [0.11, 1.00, 1.50, 1.51]

    print("\n==========================================================================")
    print(" SCENARIO 1 (DECELERATION) - RELATIVE RMSE (%):")
    print("==========================================================================")
    print(f"{'Timestep':<15}" + "".join([f"{cp:<12}" for cp in checkpoints]))
    print(f"{'Our Model (%)':<15}" + "".join([f"{val:<12.4f}" for val in sc1_rmse]))
    print(f"{'Paper-30 (%)':<15}" + "".join([f"{val:<12.2f}" for val in paper_sc1]))

    print("\n==========================================================================")
    print(" SCENARIO 2 (SINE-WAVE LANE CHANGE) - RELATIVE RMSE (%):")
    print("==========================================================================")
    print(f"{'Timestep':<15}" + "".join([f"{cp:<12}" for cp in checkpoints]))
    print(f"{'Our Model (%)':<15}" + "".join([f"{val:<12.4f}" for val in sc2_rmse]))
    print(f"{'Paper-30 (%)':<15}" + "".join([f"{val:<12.2f}" for val in paper_sc2]))
    print("==========================================================================\n")

    # 6. Generate Plot Figures
    t_axis = np.linspace(0, N*Ts, N+1)
    os.makedirs('results', exist_ok=True)

    # Figure Scenario 1
    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig1.suptitle('Scenario 1: Deceleration (Fx = -1000 N, delta = 0 rad)', fontsize=13, fontweight='bold')

    axs1[0].plot(t_axis, x_true_1[:, 0], 'r-', label='TruckMaker Ground Truth', linewidth=1.5)
    axs1[0].plot(t_axis, x_pred_1[:, 0], 'g--', label='Trained Koopman Model (9-dim)', linewidth=1.8)
    axs1[0].set_ylabel(r'$v_x$ (m/s)', fontsize=11)
    axs1[0].legend(loc='upper right')
    axs1[0].grid(True, linestyle='--', alpha=0.6)

    axs1[1].plot(t_axis, x_true_1[:, 1], 'r-', label='TruckMaker Ground Truth', linewidth=1.5)
    axs1[1].plot(t_axis, x_pred_1[:, 1], 'g--', label='Trained Koopman Model (9-dim)', linewidth=1.8)
    axs1[1].set_ylabel(r'$v_y$ (m/s)', fontsize=11)
    axs1[1].legend(loc='upper right')
    axs1[1].grid(True, linestyle='--', alpha=0.6)

    axs1[2].plot(t_axis, x_true_1[:, 2], 'r-', label='TruckMaker Ground Truth', linewidth=1.5)
    axs1[2].plot(t_axis, x_pred_1[:, 2], 'g--', label='Trained Koopman Model (9-dim)', linewidth=1.8)
    axs1[2].set_ylabel(r'$\omega$ (rad/s)', fontsize=11)
    axs1[2].set_xlabel('Time (seconds)', fontsize=11)
    axs1[2].legend(loc='upper right')
    axs1[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    p1 = 'results/scenario_1_validation.png'
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved: {p1}")

    # Figure Scenario 2
    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig2.suptitle(r'Scenario 2: Sine-Wave Steering ($F_x=5000\sin(t)+1000$, $\delta=5^\circ\sin(t)$)', fontsize=13, fontweight='bold')

    axs2[0].plot(t_axis, x_true_2[:, 0], 'r-', label='TruckMaker Ground Truth', linewidth=1.5)
    axs2[0].plot(t_axis, x_pred_2[:, 0], 'g--', label='Trained Koopman Model (9-dim)', linewidth=1.8)
    axs2[0].set_ylabel(r'$v_x$ (m/s)', fontsize=11)
    axs2[0].legend(loc='upper right')
    axs2[0].grid(True, linestyle='--', alpha=0.6)

    axs2[1].plot(t_axis, x_true_2[:, 1], 'r-', label='TruckMaker Ground Truth', linewidth=1.5)
    axs2[1].plot(t_axis, x_pred_2[:, 1], 'g--', label='Trained Koopman Model (9-dim)', linewidth=1.8)
    axs2[1].set_ylabel(r'$v_y$ (m/s)', fontsize=11)
    axs2[1].legend(loc='upper right')
    axs2[1].grid(True, linestyle='--', alpha=0.6)

    axs2[2].plot(t_axis, x_true_2[:, 2], 'r-', label='TruckMaker Ground Truth', linewidth=1.5)
    axs2[2].plot(t_axis, x_pred_2[:, 2], 'g--', label='Trained Koopman Model (9-dim)', linewidth=1.8)
    axs2[2].set_ylabel(r'$\omega$ (rad/s)', fontsize=11)
    axs2[2].set_xlabel('Time (seconds)', fontsize=11)
    axs2[2].legend(loc='upper right')
    axs2[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    p2 = 'results/scenario_2_validation.png'
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved: {p2}")

if __name__ == '__main__':
    run_open_loop_scenarios()
