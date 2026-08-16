import numpy as np
from pysindy.optimizers import STLSQ
import os

def fit_koopman_model(Z_train, U_train, Z_prime_train, Z_val, U_val, Z_prime_val, lambda_vals=[0.001, 0.01, 0.05, 0.1, 0.5]):
    """
    Fits the discrete-time Koopman model Z' = A*Z + B*U using sparse regression (STLSQ).
    Sweeps over lambda_vals to find the best balance of sparsity and validation error.
    """
    Theta_train = np.hstack([Z_train, U_train])
    Theta_val = np.hstack([Z_val, U_val])
    
    best_lam = None
    best_error = float('inf')
    
    results = []
    
    print("\nSweeping STLSQ threshold (lambda):")
    for lam in lambda_vals:
        # Fit with STLSQ on training data.
        # We set fit_intercept=False because the Koopman formulation is strictly linear Z' = A*Z + B*U
        opt = STLSQ(threshold=lam)
        opt.fit(Theta_train, Z_prime_train)
        
        # Predict on validation data
        Z_prime_val_pred = opt.predict(Theta_val)
        
        # Calculate RMSE on validation
        val_rmse = np.sqrt(np.mean((Z_prime_val - Z_prime_val_pred)**2))
        
        # Count non-zero coefficients
        active_terms = np.count_nonzero(opt.coef_)
        
        results.append({
            'lambda': lam,
            'val_rmse': val_rmse,
            'active_terms': active_terms
        })
        
        print(f"  lambda = {lam:.3f} | Val RMSE: {val_rmse:.6f} | Active terms: {active_terms}")
        
        if val_rmse < best_error:
            best_error = val_rmse
            best_lam = lam
            
    print(f"Selected lambda: {best_lam} with Val RMSE: {best_error:.6f}")
    
    # Retrain final model on Train + Val combined
    print("Refitting final model on Train + Val...")
    Theta_train_val = np.vstack([Theta_train, Theta_val])
    Z_prime_train_val = np.vstack([Z_prime_train, Z_prime_val])
    
    final_opt = STLSQ(threshold=best_lam)
    final_opt.fit(Theta_train_val, Z_prime_train_val)
    
    # coef_ shape is (n_targets, n_features) -> (z_dim, z_dim + u_dim)
    # coef_ = [A B]
    coef = final_opt.coef_
    
    n_states = Z_train.shape[1]
    A = coef[:, :n_states]
    B = coef[:, n_states:]
    
    # Save the matrices
    os.makedirs('models', exist_ok=True)
    np.save('models/A.npy', A)
    np.save('models/B.npy', B)
    
    print(f"Final model active terms: {np.count_nonzero(coef)} / {coef.size}")
    
    return A, B, best_lam, results

if __name__ == "__main__":
    pass
