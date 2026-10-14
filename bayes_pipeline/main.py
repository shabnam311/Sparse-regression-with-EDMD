import numpy as np
import pandas as pd
import json
import os
import sys

# Ensure bayes_pipeline is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from adapter import get_data
from windows import accumulate_stats, build_eval_windows
from bayes_multistep import BayesMultiStep
from lifting_np import lift_poly2

def main():
    print("=== Bayesian Koopman Operator Sparse Regression Pipeline ===")
    
    # 1. Load Data
    data = get_data('repo')
    mx, sx, mu, su = data['scaler']
    
    # 2. Accumulate Streaming Statistics & Fit Bayesian Model via ARD
    print("\nAccumulating streaming statistics and fitting Bayesian model via ARD...")
    st = accumulate_stats(data['train'], 1, lift_poly2)
    bm = BayesMultiStep().fit(st)
    
    # 3. Extract 1-Step Model Parameters (k=0)
    # M0 shape: (L, L + m) = (9, 11)
    M0 = bm.mu[0]
    A_bayes = M0[:, :9]
    B_bayes = M0[:, 9:]
    
    # Parameter Covariance Tensor (L, d0, d0) = (9, 11, 11)
    Sig0 = bm.Sig[0]
    
    # Measurement / Residual Noise Precision (L,) = (9,)
    rho0 = bm.rho[0]
    noise_variance_scaled = 1.0 / rho0
    noise_std_scaled = np.sqrt(noise_variance_scaled)
    
    # Physical noise units (for states vx, vy, omega)
    noise_variance_phys = np.zeros(3)
    noise_std_phys = np.zeros(3)
    for i in range(3):
        noise_variance_phys[i] = noise_variance_scaled[i] * (sx[i] ** 2)
        noise_std_phys[i] = np.sqrt(noise_variance_phys[i])
    
    # 4. Held-out 1-Step Validation on Test Windows
    Pt, Yt = build_eval_windows(data['test'], 1, lift_poly2, stride=1)
    # 1-step prediction: Z_pred = Phi @ M0.T
    Z_pred = Pt @ M0.T
    X_pred_scaled = Z_pred[:, :3]
    X_true_scaled = Yt[:, 0, :3]
    # Unscale to physical units for RMSE calculation
    X_pred_phys = X_pred_scaled * sx + mx
    X_true_phys = X_true_scaled * sx + mx
    err_phys = X_pred_phys - X_true_phys
    rmse_1step = np.sqrt(np.mean(err_phys ** 2, axis=0))
    mae_1step = np.mean(np.abs(err_phys), axis=0)

    # 5. Display Summary
    features = [
        'vx_scaled', 'vy_scaled', 'omega_scaled',
        'vx_scaled^2', 'vy_scaled^2', 'omega_scaled^2',
        'vx_scaled*vy_scaled', 'vx_scaled*omega_scaled', 'vy_scaled*omega_scaled'
    ]
    controls = ['Fx_scaled', 'delta_scaled']
    regressor_names = features + controls
    units = ['m/s', 'm/s', 'rad/s']
    
    active_A = np.count_nonzero(A_bayes)
    active_B = np.count_nonzero(B_bayes)
    
    print("\n=== Bayesian Model Summary ===")
    print(f"Active terms in A_bayes: {active_A} / {A_bayes.size}")
    print(f"Active terms in B_bayes: {active_B} / {B_bayes.size}")
    
    print("\nEstimated Measurement / Residual Noise Variance:")
    print(f"{'State Channel':<16} {'Scaled Var':<16} {'Scaled Std':<16} {'Physical Std':<16} {'Units':<10}")
    print("-" * 76)
    for i, name in enumerate(['vx', 'vy', 'omega']):
        print(f"{name:<16} {noise_variance_scaled[i]:<16.4e} {noise_std_scaled[i]:<16.4e} {noise_std_phys[i]:<16.4e} {units[i]:<10}")
        
    print("\nHeld-out Test Set 1-Step Prediction Errors:")
    print(f"{'State Channel':<16} {'1-Step RMSE':<20} {'1-Step MAE':<20} {'Units':<10}")
    print("-" * 66)
    for i, name in enumerate(['vx', 'vy', 'omega']):
        print(f"{name:<16} {rmse_1step[i]:<20.6e} {mae_1step[i]:<20.6e} {units[i]:<10}")
        
    print("\nParameter Uncertainty (SE = sqrt(diag(Sigma)) for key states):")
    print(f"{'State Channel':<24} {'Max Param SE':<20} {'Min Nonzero Param SE':<20}")
    print("-" * 64)
    for i, name in enumerate(features[:3]):
        diag_sig = np.sqrt(np.diag(Sig0[i]))
        nonzero_se = diag_sig[diag_sig > 0]
        min_nz = np.min(nonzero_se) if len(nonzero_se) > 0 else 0.0
        max_se = np.max(diag_sig)
        print(f"{name:<24} {max_se:<20.4e} {min_nz:<20.4e}")
        
    # 6. Save Artifacts
    out_dir = os.path.join(SCRIPT_DIR, 'models_bayes')
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, 'A_bayes.npy'), A_bayes)
    np.save(os.path.join(out_dir, 'B_bayes.npy'), B_bayes)
    np.save(os.path.join(out_dir, 'covariance_Sigma.npy'), Sig0)
    np.save(os.path.join(out_dir, 'noise_variance.npy'), noise_variance_scaled)
    np.save(os.path.join(out_dir, 'noise_variance_physical.npy'), noise_variance_phys)
    
    # Save scalers & metadata so downstream MPC can reuse safely
    np.savez(os.path.join(out_dir, 'scaler_params.npz'),
             mean_x=mx, scale_x=sx, mean_u=mu, scale_u=su)
    with open(os.path.join(out_dir, 'feature_metadata.json'), 'w') as f:
        json.dump({
            'state_dim': 3,
            'lifted_dim': 9,
            'control_dim': 2,
            'features': features,
            'controls': controls,
            'lifting_order': ['vx', 'vy', 'omega', 'vx^2', 'vy^2', 'omega^2', 'vx*vy', 'vx*omega', 'vy*omega'],
            'one_step_rmse_physical': {
                'vx': float(rmse_1step[0]),
                'vy': float(rmse_1step[1]),
                'omega': float(rmse_1step[2])
            }
        }, f, indent=2)
    
    # Save CSVs
    df_A = pd.DataFrame(A_bayes, index=features, columns=features)
    df_B = pd.DataFrame(B_bayes, index=features, columns=controls)
    df_A.to_csv(os.path.join(out_dir, 'A_bayes.csv'))
    df_B.to_csv(os.path.join(out_dir, 'B_bayes.csv'))
    
    # Save human-readable summary text
    with open(os.path.join(out_dir, 'bayesian_model_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("========================================================================\n")
        f.write("           BAYESIAN KOOPMAN OPERATOR MODEL SUMMARY\n")
        f.write("========================================================================\n\n")
        f.write(f"Active terms in A_bayes: {active_A} / {A_bayes.size}\n")
        f.write(f"Active terms in B_bayes: {active_B} / {B_bayes.size}\n\n")
        
        f.write("Held-out 1-Step Prediction Errors (Physical Units):\n")
        for i, name in enumerate(['vx', 'vy', 'omega']):
            f.write(f"  {name:<6}: RMSE = {rmse_1step[i]:.6e} {units[i]}, MAE = {mae_1step[i]:.6e} {units[i]}\n")
            
        f.write("\nEstimated Measurement / Residual Noise Variance:\n")
        for i, name in enumerate(['vx', 'vy', 'omega']):
            f.write(f"  {name:<6}: Scaled Var = {noise_variance_scaled[i]:.4e}, Scaled Std = {noise_std_scaled[i]:.4e} | Phys Std = {noise_std_phys[i]:.4e} {units[i]}\n")
            
        f.write("\n=== MATRIX A_bayes (9x9 Posterior Mean) ===\n")
        f.write(df_A.round(6).to_string())
        f.write("\n\n=== MATRIX B_bayes (9x2 Posterior Mean) ===\n")
        f.write(df_B.round(6).to_string())
        
        f.write("\n\n=== PARAMETER STANDARD ERRORS (sqrt(diag(Sigma))) ===\n")
        df_se = pd.DataFrame(
            np.array([np.sqrt(np.diag(Sig0[i])) for i in range(len(features))]),
            index=features, columns=regressor_names
        )
        f.write(df_se.round(6).to_string())
        f.write("\n")
        
    print(f"\nSaved Bayesian model matrices, physical noise, and scalers to: {out_dir}")

if __name__ == '__main__':
    main()
