import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
from lifting import lift_state

def generate_visual_reports():
    print("Loading Scenario 1 and Scenario 2 datasets for evaluation report...")

    # Load trained model and scalers
    A = np.load('models/A.npy')
    B = np.load('models/B.npy')

    with open('models/scaler_x.pkl', 'rb') as f:
        scaler_x = pickle.load(f)
    with open('models/scaler_u.pkl', 'rb') as f:
        scaler_u = pickle.load(f)

    # Scenario datasets
    sc1_df = pd.read_csv('Test 1/scenario_1_simulation_data.csv')
    sc2_df = pd.read_csv('Test 1/scenario_2_simulation_data.csv')

    N = 400
    Ts = 0.01
    t_vec = np.arange(N) * Ts
    checkpoints = [10, 80, 200, 400]
    x0 = np.array([25.0, 0.0, 0.0])

    Fx_seq_1 = -1000.0 * np.ones(N)
    delta_seq_1 = np.zeros(N)

    Fx_seq_2 = 5000.0 * np.sin(t_vec) + 1000.0
    delta_seq_2 = 5.0 * np.sin(t_vec) * np.pi / 180.0

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

    x_true_1 = sc1_df[['vx', 'vy', 'YawRate']].values[:N+1]
    x_true_2 = sc2_df[['vx', 'vy', 'YawRate']].values[:N+1]

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

    os.makedirs('results', exist_ok=True)

    # -------------------------------------------------------------
    # Plot 1: Clean Relative RMSE Bar Comparison (No Accuracy Terms)
    # -------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    x_pos = np.arange(len(checkpoints))
    width = 0.35

    rects1 = ax1.bar(x_pos - width/2, sc1_rmse, width, label='Scenario 1: Deceleration', color='#1f77b4', edgecolor='black')
    rects2 = ax1.bar(x_pos + width/2, sc2_rmse, width, label='Scenario 2: Lane Change', color='#ff7f0e', edgecolor='black')

    ax1.set_xlabel('Prediction Timestep T', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Relative RMSE (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Open-Loop Relative RMSE (%) Across Prediction Timesteps', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'T = {cp}' for cp in checkpoints], fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    for rect in rects1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2.0, h + 0.05, f'{h:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2.0, h + 0.05, f'{h:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plot1_path = 'results/accuracy_and_rmse_summary.png'
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved: {plot1_path}")

    # -------------------------------------------------------------
    # Plot 2: Executive Dashboard (Parity, Stability, Error Growth)
    # -------------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Panel (0,0): Parity Plot for Scenario 1
    axs[0, 0].scatter(x_true_1[:, 0], x_pred_1[:, 0], color='#1f77b4', alpha=0.6, label=r'$v_x$ (Longitudinal)')
    axs[0, 0].scatter(x_true_1[:, 1], x_pred_1[:, 1], color='#2ca02c', alpha=0.6, label=r'$v_y$ (Lateral)')
    min_val = min(x_true_1.min(), x_pred_1.min())
    max_val = max(x_true_1.max(), x_pred_1.max())
    axs[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit (y = x)')
    axs[0, 0].set_title('Scenario 1: Parity Plot (Actual vs Predicted)', fontsize=11, fontweight='bold')
    axs[0, 0].set_xlabel('Actual Values', fontsize=10)
    axs[0, 0].set_ylabel('Predicted Values', fontsize=10)
    axs[0, 0].legend(fontsize=9)
    axs[0, 0].grid(True, linestyle=':', alpha=0.6)

    # Panel (0,1): Parity Plot for Scenario 2
    axs[0, 1].scatter(x_true_2[:, 0], x_pred_2[:, 0], color='#1f77b4', alpha=0.6, label=r'$v_x$ (Longitudinal)')
    axs[0, 1].scatter(x_true_2[:, 1], x_pred_2[:, 1], color='#2ca02c', alpha=0.6, label=r'$v_y$ (Lateral)')
    min_val2 = min(x_true_2.min(), x_pred_2.min())
    max_val2 = max(x_true_2.max(), x_pred_2.max())
    axs[0, 1].plot([min_val2, max_val2], [min_val2, max_val2], 'r--', label='Ideal Fit (y = x)')
    axs[0, 1].set_title('Scenario 2: Parity Plot (Actual vs Predicted)', fontsize=11, fontweight='bold')
    axs[0, 1].set_xlabel('Actual Values', fontsize=10)
    axs[0, 1].set_ylabel('Predicted Values', fontsize=10)
    axs[0, 1].legend(fontsize=9)
    axs[0, 1].grid(True, linestyle=':', alpha=0.6)

    # Panel (1,0): Discrete-Time Eigenvalues on Unit Circle
    eigenvalues = np.linalg.eigvals(A)
    theta = np.linspace(0, 2*np.pi, 200)
    axs[1, 0].plot(np.cos(theta), np.sin(theta), 'k--', label='Unit Circle (|z| = 1)')
    axs[1, 0].scatter(eigenvalues.real, eigenvalues.imag, color='crimson', s=60, zorder=5, label='Koopman Eigenvalues')
    axs[1, 0].axhline(0, color='gray', linestyle=':', alpha=0.6)
    axs[1, 0].axvline(0, color='gray', linestyle=':', alpha=0.6)
    axs[1, 0].set_title('Koopman Pole-Zero Map (Stability Proof)', fontsize=11, fontweight='bold')
    axs[1, 0].set_xlabel('Real Axis', fontsize=10)
    axs[1, 0].set_ylabel('Imaginary Axis', fontsize=10)
    axs[1, 0].set_aspect('equal')
    axs[1, 0].set_xlim(-1.2, 1.2)
    axs[1, 0].set_ylim(-1.2, 1.2)
    axs[1, 0].legend(fontsize=9, loc='upper right')
    axs[1, 0].grid(True, linestyle=':', alpha=0.6)

    # Panel (1,1): Absolute Error Evolution Over 4 Seconds
    err1 = np.abs(x_true_1 - x_pred_1)
    err2 = np.abs(x_true_2 - x_pred_2)
    axs[1, 1].plot(t_vec, err1[:-1, 0], color='#1f77b4', label=r'Scenario 1 $v_x$ Error', linewidth=1.5)
    axs[1, 1].plot(t_vec, err2[:-1, 0], color='#ff7f0e', label=r'Scenario 2 $v_x$ Error', linewidth=1.5)
    axs[1, 1].set_title('Open-Loop Error Over Time (4 Seconds)', fontsize=11, fontweight='bold')
    axs[1, 1].set_xlabel('Time (s)', fontsize=10)
    axs[1, 1].set_ylabel('Absolute Error (m/s)', fontsize=10)
    axs[1, 1].legend(fontsize=9)
    axs[1, 1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plot2_path = 'results/evaluation_dashboard.png'
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved: {plot2_path}")

if __name__ == '__main__':
    generate_visual_reports()
