import numpy as np
from data_loader import load_and_preprocess_data
from lifting import lift_state, get_feature_names
from koopman_fit import fit_koopman_model
from validate import validate_one_step, validate_rollout

def main():
    print("=== Koopman Operator Sparse Regression Pipeline ===")
    
    # 1. Load Data
    data_dir = 'data/dataset'
    data = load_and_preprocess_data(data_dir, split_ratio=(60, 5, 5))
    
    X_train, U_train, X_prime_train, train_scaled = data['train']
    X_val, U_val, X_prime_val, val_scaled = data['val']
    X_test, U_test, X_prime_test, test_scaled = data['test']
    scaler_x, scaler_u = data['scalers']
    
    # 2. Lift States
    print("\nLifting states (Degree 2 Polynomials)...")
    Z_train = lift_state(X_train)
    Z_prime_train = lift_state(X_prime_train)
    
    Z_val = lift_state(X_val)
    Z_prime_val = lift_state(X_prime_val)
    
    Z_test = lift_state(X_test)
    
    print(f"Lifted state dimension: {Z_train.shape[1]}")
    
    # 3. Fit Koopman Model via Sparse Regression (STLSQ)
    lambda_vals = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    A, B, best_lam, sweep_results = fit_koopman_model(
        Z_train, U_train, Z_prime_train,
        Z_val, U_val, Z_prime_val,
        lambda_vals=lambda_vals
    )
    
    # 4. Validation
    # Get raw target for one-step validation
    # X_prime_test is scaled. We need raw for true physical metric reporting.
    # The loader didn't explicitly store X_prime_raw in matrix form, but we can unscale it.
    X_prime_test_raw = scaler_x.inverse_transform(X_prime_test)
    
    validate_one_step(A, B, Z_test, U_test, X_prime_test_raw, scaler_x)
    
    validate_rollout(A, B, test_scaled, scaler_x, num_plots=3)
    
    # 5. Final Summary
    feature_names = get_feature_names()
    print("\n=== Final Model Summary ===")
    print(f"Chosen Sparsity Threshold (lambda): {best_lam}")
    
    # Analyze sparsity of A and B
    active_A = np.count_nonzero(A)
    active_B = np.count_nonzero(B)
    print(f"Active terms in A: {active_A} / {A.size}")
    print(f"Active terms in B: {active_B} / {B.size}")
    
    print("\nActive Lifted Features used by the model:")
    # A column in A corresponds to a feature in Z
    # If a column is entirely zero, that feature is effectively pruned entirely.
    active_cols = np.any(A != 0, axis=0)
    for i, active in enumerate(active_cols):
        status = "KEEP" if active else "PRUNED"
        print(f"  {feature_names[i]:12s} : {status}")

if __name__ == "__main__":
    main()
