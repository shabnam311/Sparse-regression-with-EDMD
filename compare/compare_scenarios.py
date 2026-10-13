import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import sys

# Add paths for imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'bayes_pipeline'))
sys.path.insert(0, os.path.join(ROOT, 'baseline_stlsq'))

from adapter import get_data
from windows import accumulate_stats
from bayes_multistep import BayesMultiStep
from lifting_np import lift_poly2

def run_scenario_comparison():
    print("==========================================================================")
    print(" COMPARING BASELINE STLSQ & BAYESIAN KOOPMAN ON BENCHMARK SCENARIO 1 & 2")
    print("==========================================================================")

    # 1. Load trained Baseline STLSQ matrices & scalers
    A_stlsq = np.load(os.path.join(ROOT, 'baseline_stlsq', 'models', 'A.npy'))
    B_stlsq = np.load(os.path.join(ROOT, 'baseline_stlsq', 'models', 'B.npy'))

    with open(os.path.join(ROOT, 'baseline_stlsq', 'models', 'scaler_x.pkl'), 'rb') as f:
        scaler_x = pickle.load(f)
    with open(os.path.join(ROOT, 'baseline_stlsq', 'models', 'scaler_u.pkl'), 'rb') as f:
        scaler_u = pickle.load(f)

    # 2. Load or fit Bayesian Koopman model (step k=0) with parameter covariance & noise variance
    bayes_models_dir = os.path.join(ROOT, 'bayes_pipeline', 'models_bayes')
    if (os.path.exists(os.path.join(bayes_models_dir, 'A_bayes.npy')) and
        os.path.exists(os.path.join(bayes_models_dir, 'covariance_Sigma.npy')) and
        os.path.exists(os.path.join(bayes_models_dir, 'noise_variance.npy'))):
        print("Loading saved Bayesian Koopman model, covariance, and noise variance...")
        A_bayes = np.load(os.path.join(bayes_models_dir, 'A_bayes.npy'))
        B_bayes = np.load(os.path.join(bayes_models_dir, 'B_bayes.npy'))
        Sig_bayes = np.load(os.path.join(bayes_models_dir, 'covariance_Sigma.npy'))
        noise_var_bayes = np.load(os.path.join(bayes_models_dir, 'noise_variance.npy'))
    else:
        print("Fitting Bayesian Koopman model from training statistics...")
        data = get_data('repo')
        st = accumulate_stats(data['train'], 1, lift_poly2)
        bm = BayesMultiStep().fit(st)
        M0 = bm.mu[0]
        A_bayes = M0[:, :9]
        B_bayes = M0[:, 9:]
        Sig_bayes = bm.Sig[0]
        noise_var_bayes = 1.0 / bm.rho[0]

    print(f"Baseline STLSQ nonzeros  -> A: {np.count_nonzero(A_stlsq)}/81, B: {np.count_nonzero(B_stlsq)}/18")
    print(f"Bayesian Koopman nonzeros -> A: {np.count_nonzero(A_bayes)}/81, B: {np.count_nonzero(B_bayes)}/18")
    print(f"Estimated Noise Variance (vx, vy, omega): {noise_var_bayes[:3]}")

    # 3. Setup simulation parameters
    N = 400  # 400 steps (4.0 seconds @ 100 Hz)
    Ts = 0.01
    t_vec = np.arange(N) * Ts
    checkpoints = [10, 80, 200, 400]
    x0 = np.array([25.0, 0.0, 0.0])  # Initial state [vx, vy, omega]
    sx = scaler_x.scale_

    # Scenario 1 inputs: Constant deceleration (Fx = -1000 N, delta = 0 rad)
    Fx_seq_1 = -1000.0 * np.ones(N)
    delta_seq_1 = np.zeros(N)

    # Scenario 2 inputs: Sine-wave lane change (Fx = 5000*sin(t)+1000 N, delta = 5 deg * sin(t))
    Fx_seq_2 = 5000.0 * np.sin(t_vec) + 1000.0
    delta_seq_2 = 5.0 * np.sin(t_vec) * np.pi / 180.0

    # Simulation rollout helper with Bayesian covariance & noise uncertainty
    def simulate_model(A, B, Fx_seq, delta_seq, compute_uncertainty=False):
        x_pred = np.zeros((N + 1, 3))
        x_pred[0] = x0
        x_curr_raw = x0.copy()
        sigma_phys = np.zeros((N + 1, 3))

        for k in range(N):
            x_curr_sc = scaler_x.transform(x_curr_raw.reshape(1, -1))
            u_phys = np.array([[Fx_seq[k], delta_seq[k]]])
            u_curr_sc = scaler_u.transform(u_phys)

            z_curr = lift_poly2(x_curr_sc)[0]

            if compute_uncertainty:
                phi = np.concatenate([z_curr, u_curr_sc[0]])
                for j in range(3):
                    var_param = float(phi @ Sig_bayes[j] @ phi)
                    var_total = var_param + noise_var_bayes[j]
                    sigma_phys[k + 1, j] = np.sqrt(max(var_total, 0.0)) * sx[j]

            z_next = A @ z_curr + B @ u_curr_sc[0]
            x_next_sc = z_next[:3].reshape(1, -1)
            x_next_raw = scaler_x.inverse_transform(x_next_sc)[0]

            x_pred[k + 1] = x_next_raw
            x_curr_raw = x_next_raw

        return x_pred, sigma_phys

    # Run predictions
    pred_stlsq_1, _ = simulate_model(A_stlsq, B_stlsq, Fx_seq_1, delta_seq_1, compute_uncertainty=False)
    pred_bayes_1, sig_bayes_1 = simulate_model(A_bayes, B_bayes, Fx_seq_1, delta_seq_1, compute_uncertainty=True)

    pred_stlsq_2, _ = simulate_model(A_stlsq, B_stlsq, Fx_seq_2, delta_seq_2, compute_uncertainty=False)
    pred_bayes_2, sig_bayes_2 = simulate_model(A_bayes, B_bayes, Fx_seq_2, delta_seq_2, compute_uncertainty=True)

    # 4. Load TruckMaker Ground Truth
    gt1_path = os.path.join(ROOT, 'baseline_stlsq', 'Test 1', 'scenario_1_simulation_data.csv')
    gt2_path = os.path.join(ROOT, 'baseline_stlsq', 'Test 1', 'scenario_2_simulation_data.csv')

    gt1 = pd.read_csv(gt1_path)
    gt2 = pd.read_csv(gt2_path)

    x_true_1 = gt1[['vx', 'vy', 'YawRate']].values[:N+1]
    x_true_2 = gt2[['vx', 'vy', 'YawRate']].values[:N+1]

    # Compute Relative RMSE (%)
    def compute_rmse_pct(x_pred, x_true):
        res = []
        for cp in checkpoints:
            p = x_pred[:cp+1]
            t = x_true[:cp+1]
            err = p - t
            rmse_pct = (np.sqrt(np.sum(err**2)) / np.sqrt(np.sum(t**2))) * 100.0
            res.append(rmse_pct)
        return res

    rmse_stlsq_1 = compute_rmse_pct(pred_stlsq_1, x_true_1)
    rmse_bayes_1 = compute_rmse_pct(pred_bayes_1, x_true_1)

    rmse_stlsq_2 = compute_rmse_pct(pred_stlsq_2, x_true_2)
    rmse_bayes_2 = compute_rmse_pct(pred_bayes_2, x_true_2)

    dnn_ref_1 = [0.06, 0.13, 0.26, 0.39]
    dnn_ref_2 = [0.11, 1.00, 1.50, 1.51]

    # Print summary table
    print("\n==========================================================================")
    print(" SCENARIO 1: DECELERATION — RELATIVE RMSE (%)")
    print("==========================================================================")
    print(f"{'Timestep (T)':<18}" + "".join([f"{cp:<14}" for cp in checkpoints]))
    print(f"{'Baseline STLSQ':<18}" + "".join([f"{v:<14.4f}" for v in rmse_stlsq_1]))
    print(f"{'Bayesian Koopman':<18}" + "".join([f"{v:<14.4f}" for v in rmse_bayes_1]))
    print(f"{'DNN-30 Reference':<18}" + "".join([f"{v:<14.2f}" for v in dnn_ref_1]))

    print("\n==========================================================================")
    print(" SCENARIO 2: SINE-WAVE LANE CHANGE — RELATIVE RMSE (%)")
    print("==========================================================================")
    print(f"{'Timestep (T)':<18}" + "".join([f"{cp:<14}" for cp in checkpoints]))
    print(f"{'Baseline STLSQ':<18}" + "".join([f"{v:<14.4f}" for v in rmse_stlsq_2]))
    print(f"{'Bayesian Koopman':<18}" + "".join([f"{v:<14.4f}" for v in rmse_bayes_2]))
    print(f"{'DNN-30 Reference':<18}" + "".join([f"{v:<14.2f}" for v in dnn_ref_2]))
    print("==========================================================================\n")

    # 5. Generate Combined 4-Panel Figures with Bayesian Uncertainty Bands
    t_axis = np.linspace(0, N * Ts, N + 1)
    out_dir = os.path.join(ROOT, 'compare')
    os.makedirs(out_dir, exist_ok=True)

    def plot_scenario_comparison(title_prefix, x_true, p_stlsq, p_bayes, sig_bayes, r_stlsq, r_bayes, r_dnn, filename):
        fig, axs = plt.subplots(4, 1, figsize=(8, 12), gridspec_kw={'height_ratios': [1.2, 1, 1, 1]})

        # Panel 1: Relative RMSE vs Timestep
        axs[0].plot(checkpoints, r_stlsq, 'o-', color='#1f77b4', linewidth=1.8, markersize=6, label='Baseline STLSQ (Dim 9)')
        axs[0].plot(checkpoints, r_bayes, '^-', color='#2ca02c', linewidth=1.8, markersize=6, label='Bayesian Koopman (Dim 9)')
        axs[0].plot(checkpoints, r_dnn, 's--', color='black', linewidth=1.5, markersize=6, label='DNN-30 Reference')
        axs[0].set_title(f'{title_prefix} — Open-Loop Relative RMSE (%)', fontsize=12, fontweight='bold')
        axs[0].set_xlabel('Prediction Timestep T', fontsize=10)
        axs[0].set_ylabel('Relative RMSE (%)', fontsize=10)
        axs[0].set_xticks(checkpoints)
        axs[0].legend(fontsize=9, loc='upper left')
        axs[0].grid(True, linestyle=':', alpha=0.6)

        # Panel 2: Longitudinal Velocity vx
        axs[1].plot(t_axis, x_true[:, 0], 'k-', label='TruckMaker Ground Truth', linewidth=1.8)
        axs[1].plot(t_axis, p_stlsq[:, 0], '--', color='#1f77b4', label='Baseline STLSQ', linewidth=1.6)
        axs[1].plot(t_axis, p_bayes[:, 0], '-', color='#2ca02c', label='Bayesian Mean', linewidth=1.8)
        axs[1].fill_between(t_axis, p_bayes[:, 0] - 2 * sig_bayes[:, 0], p_bayes[:, 0] + 2 * sig_bayes[:, 0],
                            color='#2ca02c', alpha=0.25, label=r'Bayesian $\pm 2\sigma$ Band (Covariance + Noise)')
        axs[1].set_title(f'{title_prefix} — Longitudinal Velocity $v_x$', fontsize=11, fontweight='bold')
        axs[1].set_ylabel(r'$v_x$ (m/s)', fontsize=10)
        axs[1].legend(fontsize=9, loc='lower left')
        axs[1].grid(True, linestyle=':', alpha=0.6)

        # Panel 3: Lateral Velocity vy
        axs[2].plot(t_axis, x_true[:, 1], 'k-', label='TruckMaker Ground Truth', linewidth=1.8)
        axs[2].plot(t_axis, p_stlsq[:, 1], '--', color='#1f77b4', label='Baseline STLSQ', linewidth=1.6)
        axs[2].plot(t_axis, p_bayes[:, 1], '-', color='#2ca02c', label='Bayesian Mean', linewidth=1.8)
        axs[2].fill_between(t_axis, p_bayes[:, 1] - 2 * sig_bayes[:, 1], p_bayes[:, 1] + 2 * sig_bayes[:, 1],
                            color='#2ca02c', alpha=0.25, label=r'Bayesian $\pm 2\sigma$ Band (Covariance + Noise)')
        axs[2].set_title(f'{title_prefix} — Lateral Velocity $v_y$', fontsize=11, fontweight='bold')
        axs[2].set_ylabel(r'$v_y$ (m/s)', fontsize=10)
        axs[2].legend(fontsize=9, loc='lower left')
        axs[2].grid(True, linestyle=':', alpha=0.6)

        # Panel 4: Yaw Rate omega
        axs[3].plot(t_axis, x_true[:, 2], 'k-', label='TruckMaker Ground Truth', linewidth=1.8)
        axs[3].plot(t_axis, p_stlsq[:, 2], '--', color='#1f77b4', label='Baseline STLSQ', linewidth=1.6)
        axs[3].plot(t_axis, p_bayes[:, 2], '-', color='#2ca02c', label='Bayesian Mean', linewidth=1.8)
        axs[3].fill_between(t_axis, p_bayes[:, 2] - 2 * sig_bayes[:, 2], p_bayes[:, 2] + 2 * sig_bayes[:, 2],
                            color='#2ca02c', alpha=0.25, label=r'Bayesian $\pm 2\sigma$ Band (Covariance + Noise)')
        axs[3].set_title(f'{title_prefix} — Yaw Rate $\\omega$', fontsize=11, fontweight='bold')
        axs[3].set_xlabel('Time (s)', fontsize=10)
        axs[3].set_ylabel(r'$\omega$ (rad/s)', fontsize=10)
        axs[3].legend(fontsize=9, loc='lower left')
        axs[3].grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        save_path = os.path.join(out_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    plot_scenario_comparison('Scenario 1: Deceleration', x_true_1, pred_stlsq_1, pred_bayes_1, sig_bayes_1,
                             rmse_stlsq_1, rmse_bayes_1, dnn_ref_1, 'scenario_1_comparison.png')
    plot_scenario_comparison('Scenario 2: Lane Change', x_true_2, pred_stlsq_2, pred_bayes_2, sig_bayes_2,
                             rmse_stlsq_2, rmse_bayes_2, dnn_ref_2, 'scenario_2_comparison.png')

if __name__ == '__main__':
    run_scenario_comparison()
