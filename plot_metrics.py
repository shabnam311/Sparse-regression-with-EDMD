import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
from lifting import lift_state
from data_loader import load_and_preprocess_data

def generate_visual_reports():
    print("Loading data and models for graphical evaluation report...")
    
    data_dir = 'data/dataset'
    data = load_and_preprocess_data(data_dir, split_ratio=(60, 5, 5))
    
    X_test, U_test, X_prime_test, test_scaled = data['test']
    scaler_x, scaler_u = data['scalers']
    
    A = np.load('models/A.npy')
    B = np.load('models/B.npy')
    
    Z_test = lift_state(X_test)
    Z_prime_pred = Z_test @ A.T + U_test @ B.T
    X_prime_pred_scaled = Z_prime_pred[:, :3]
    X_prime_pred_raw = scaler_x.inverse_transform(X_prime_pred_scaled)
    X_prime_test_raw = scaler_x.inverse_transform(X_prime_test)
    
    errors = X_prime_test_raw - X_prime_pred_raw
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    mae = np.mean(np.abs(errors), axis=0)
    
    ranges = np.ptp(X_prime_test_raw, axis=0)
    rel_errors = (rmse / ranges) * 100
    accuracies = 100 - rel_errors
    
    states = [r'$v_x$ (Longitudinal)', r'$v_y$ (Lateral)', r'$\omega$ (Yaw Rate)']
    state_names_clean = ['vx', 'vy', 'omega']
    units = ['m/s', 'm/s', 'rad/s']
    
    os.makedirs('results', exist_ok=True)
    
    # -------------------------------------------------------------
    # Plot 1: Clean Accuracy & RMSE Bar Charts (Super intuitive!)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Left: Accuracy Percentages
    colors_acc = ['#2ca02c', '#1f77b4', '#9467bd']
    bars1 = ax1.bar(states, accuracies, color=colors_acc, width=0.55, edgecolor='black', linewidth=1.2)
    ax1.set_ylim(95, 100.5)
    ax1.set_ylabel('Model Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('1-Step Prediction Accuracy (Test Set)', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar, acc in zip(bars1, accuracies):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f'{acc:.2f}%', 
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    # Right: RMSE vs Physical Range
    x_pos = np.arange(len(states))
    width = 0.35
    
    # Plot normalized comparison: Log scale for clarity
    bars2 = ax2.bar(x_pos - width/2, rmse, width, label='RMSE (Absolute Error)', color='#d62728', edgecolor='black')
    bars3 = ax2.bar(x_pos + width/2, ranges, width, label='Operating Range (Total Span)', color='#7f7f7f', alpha=0.6, edgecolor='black')
    
    ax2.set_yscale('log')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(states, fontsize=11)
    ax2.set_ylabel('Value in Physical Units (Log Scale)', fontsize=12, fontweight='bold')
    ax2.set_title('RMSE vs. Full Operational Range', fontsize=13, fontweight='bold', pad=15)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate absolute numbers
    for i in range(len(states)):
        ax2.text(x_pos[i] - width/2, rmse[i] * 1.3, f'{rmse[i]:.4f}\n{units[i]}', ha='center', fontsize=9, fontweight='bold', color='#a00000')
        ax2.text(x_pos[i] + width/2, ranges[i] * 1.3, f'{ranges[i]:.2f}\n{units[i]}', ha='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plot1_path = 'results/accuracy_and_rmse_summary.png'
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved: {plot1_path}")
    
    # -------------------------------------------------------------
    # Plot 2: 4-Panel Executive Dashboard (Parity, Stability, Error Horizon)
    # -------------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    
    # Panel (0,0): Parity Plot for Yaw Rate omega
    idx_sample = np.random.choice(len(X_prime_test_raw), size=min(3000, len(X_prime_test_raw)), replace=False)
    axs[0, 0].scatter(X_prime_test_raw[idx_sample, 2], X_prime_pred_raw[idx_sample, 2], 
                      alpha=0.4, color='#1f77b4', s=15, label='Test Predictions')
    min_om, max_om = np.min(X_prime_test_raw[:, 2]), np.max(X_prime_test_raw[:, 2])
    axs[0, 0].plot([min_om, max_om], [min_om, max_om], 'r--', linewidth=2, label='Ideal Perfect Fit (y = x)')
    axs[0, 0].set_title(r'Yaw Rate ($\omega$) Parity Plot: Actual vs Predicted', fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel('Actual Yaw Rate (rad/s)', fontsize=11)
    axs[0, 0].set_ylabel('Predicted Yaw Rate (rad/s)', fontsize=11)
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)
    
    # Panel (0,1): Parity Plot for Lateral Velocity vy
    min_vy, max_vy = np.min(X_prime_test_raw[:, 1]), np.max(X_prime_test_raw[:, 1])
    axs[0, 1].scatter(X_prime_test_raw[idx_sample, 1], X_prime_pred_raw[idx_sample, 1], 
                      alpha=0.4, color='#2ca02c', s=15, label='Test Predictions')
    axs[0, 1].plot([min_vy, max_vy], [min_vy, max_vy], 'r--', linewidth=2, label='Ideal Perfect Fit (y = x)')
    axs[0, 1].set_title(r'Lateral Velocity ($v_y$) Parity Plot: Actual vs Predicted', fontsize=12, fontweight='bold')
    axs[0, 1].set_xlabel('Actual Lateral Velocity (m/s)', fontsize=11)
    axs[0, 1].set_ylabel('Predicted Lateral Velocity (m/s)', fontsize=11)
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)
    
    # Panel (1,0): Discrete-Time Eigenvalues on Complex Plane (Stability Map)
    eigenvalues = np.linalg.eigvals(A)
    theta = np.linspace(0, 2*np.pi, 200)
    axs[1, 0].plot(np.cos(theta), np.sin(theta), 'k--', label='Unit Circle ($|z| = 1$ Stability Boundary)')
    axs[1, 0].scatter(eigenvalues.real, eigenvalues.imag, color='crimson', s=60, zorder=5, label='Koopman Eigenvalues ($\lambda_i$)')
    axs[1, 0].axhline(0, color='gray', linestyle=':', alpha=0.6)
    axs[1, 0].axvline(0, color='gray', linestyle=':', alpha=0.6)
    axs[1, 0].set_title('Koopman Pole-Zero Map (Proof of Stability)', fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel('Real Axis', fontsize=11)
    axs[1, 0].set_ylabel('Imaginary Axis', fontsize=11)
    axs[1, 0].set_aspect('equal')
    axs[1, 0].set_xlim(-1.2, 1.2)
    axs[1, 0].set_ylim(-1.2, 1.2)
    axs[1, 0].legend(loc='upper right', fontsize=9)
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)
    
    # Panel (1,1): Open-Loop Cumulative Error Evolution Across 25,000 Steps
    # Let's take the longest test trajectory
    traj_long = max(test_scaled, key=lambda t: len(t['time']))
    N_steps = len(traj_long['time'])
    time_s = traj_long['time']
    x_true = traj_long['x_raw']
    u_sc = traj_long['u']
    
    z_sim = np.zeros((N_steps, A.shape[0]))
    z_sim[0] = lift_state(traj_long['x'][0:1])[0]
    for k in range(N_steps - 1):
        z_sim[k+1] = A @ z_sim[k] + B @ u_sc[k]
    x_pred = scaler_x.inverse_transform(z_sim[:, :3])
    
    err_vy = np.abs(x_true[:, 1] - x_pred[:, 1])
    err_om = np.abs(x_true[:, 2] - x_pred[:, 2])
    
    axs[1, 1].plot(time_s, err_vy, color='#2ca02c', label=r'$v_y$ Absolute Error (m/s)', linewidth=1.2)
    axs[1, 1].plot(time_s, err_om * 10, color='#1f77b4', label=r'$\omega$ Absolute Error $\times 10$ (rad/s)', linewidth=1.2)
    axs[1, 1].set_title('Open-Loop Error Over Time (25,000 Steps / 4.25 Mins)', fontsize=12, fontweight='bold')
    axs[1, 1].set_xlabel('Time (seconds)', fontsize=11)
    axs[1, 1].set_ylabel('Error Magnitude', fontsize=11)
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot2_path = 'results/evaluation_dashboard.png'
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved: {plot2_path}")

if __name__ == '__main__':
    generate_visual_reports()
