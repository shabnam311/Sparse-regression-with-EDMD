import numpy as np
import matplotlib.pyplot as plt
import os
from lifting import lift_state

def validate_one_step(A, B, Z_test, U_test, X_prime_raw_test, scaler_x):
    """
    Evaluates one-step ahead prediction on the test set.
    """
    # Predict lifted state: Z' = Z A^T + U B^T
    Z_prime_pred = Z_test @ A.T + U_test @ B.T
    
    # Extract predicted physical states (first 3 columns)
    X_prime_pred_scaled = Z_prime_pred[:, :3]
    
    # Unscale to original units
    X_prime_pred_raw = scaler_x.inverse_transform(X_prime_pred_scaled)
    
    # Calculate errors per state
    errors = X_prime_raw_test - X_prime_pred_raw
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    mae = np.mean(np.abs(errors), axis=0)
    
    print("\n--- One-Step Ahead Prediction (Test Set) ---")
    states = ['vx (m/s)', 'vy (m/s)', 'omega (rad/s)']
    for i, state in enumerate(states):
        print(f"  {state:15s} | RMSE: {rmse[i]:.4f} | MAE: {mae[i]:.4f}")
        
    return rmse, mae

def validate_rollout(A, B, test_trajectories, scaler_x, num_plots=3):
    """
    Evaluates multi-step (open-loop) rollout on full trajectories.
    Plots predicted vs actual states.
    """
    print("\n--- Multi-Step Open-Loop Rollout (Test Set) ---")
    
    os.makedirs('results', exist_ok=True)
    
    # Pick a few trajectories (including the longest and shortest for variety)
    lengths = [len(traj['time']) for traj in test_trajectories]
    sorted_idx = np.argsort(lengths)
    
    # Select shortest, median, and longest
    if len(test_trajectories) >= 3:
        plot_indices = [sorted_idx[0], sorted_idx[len(sorted_idx)//2], sorted_idx[-1]]
    else:
        plot_indices = list(range(len(test_trajectories)))
        
    for idx_count, idx in enumerate(plot_indices):
        traj = test_trajectories[idx]
        N = len(traj['time'])
        
        # True values
        time = traj['time']
        x_raw_true = traj['x_raw']
        u_scaled = traj['u']
        x_scaled = traj['x']
        
        # Initialize prediction array
        z_dim = A.shape[0]
        z_pred = np.zeros((N, z_dim))
        
        # Lift the initial state
        z_pred[0] = lift_state(x_scaled[0:1])[0]
        
        # Rollout
        for k in range(N - 1):
            z_pred[k+1] = A @ z_pred[k] + B @ u_scaled[k]
            
        # Extract states and unscale
        x_pred_scaled = z_pred[:, :3]
        x_pred_raw = scaler_x.inverse_transform(x_pred_scaled)
        
        # Calculate rollout RMSE for this trajectory
        errors = x_raw_true - x_pred_raw
        rmse = np.sqrt(np.mean(errors**2, axis=0))
        
        print(f"Trajectory {os.path.basename(traj['file'])} (len={N}): vx RMSE={rmse[0]:.4f}, vy RMSE={rmse[1]:.4f}, omega RMSE={rmse[2]:.4f}")
        
        if idx_count < num_plots:
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            states = ['vx (m/s)', 'vy (m/s)', 'omega (rad/s)']
            
            for i, ax in enumerate(axs):
                ax.plot(time, x_raw_true[:, i], 'k-', label='True')
                ax.plot(time, x_pred_raw[:, i], 'r--', label='Koopman Predict')
                ax.set_ylabel(states[i])
                ax.legend(loc='upper right')
                ax.grid(True)
                
            axs[2].set_xlabel('Time (s)')
            fig.suptitle(f'Open-Loop Rollout: {os.path.basename(traj["file"])} (Length: {N} steps)')
            plt.tight_layout()
            
            save_path = f'results/rollout_{idx_count+1}.png'
            plt.savefig(save_path)
            plt.close()
            print(f"  Saved plot to {save_path}")

if __name__ == "__main__":
    pass
