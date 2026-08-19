import numpy as np

def lift_state(X):
    """
    Lifts the state X into a higher dimensional observable space Z.
    X shape: (N, 3) where columns are vx, vy, omega
    Z shape: (N, 9) including linear and quadratic terms.
    """
    vx = X[:, 0:1]
    vy = X[:, 1:2]
    omega = X[:, 2:3]
    
    # Quadratic terms
    vx_sq = vx ** 2
    vy_sq = vy ** 2
    omega_sq = omega ** 2
    
    # Cross terms
    vx_vy = vx * vy
    vx_omega = vx * omega
    vy_omega = vy * omega
    
    # Concatenate all to form Z
    Z = np.hstack([vx, vy, omega, vx_sq, vy_sq, omega_sq, vx_vy, vx_omega, vy_omega])
    return Z

def get_feature_names():
    """
    Returns the string names of the lifted features for interpretability.
    """
    return [
        'vx_scaled', 'vy_scaled', 'omega_scaled',
        'vx_scaled^2', 'vy_scaled^2', 'omega_scaled^2',
        'vx_scaled*vy_scaled', 'vx_scaled*omega_scaled', 'vy_scaled*omega_scaled'
    ]

if __name__ == "__main__":
    # Quick test
    X = np.array([[1.0, 2.0, 3.0]])
    Z = lift_state(X)
    print("Feature names:", get_feature_names())
    print("Lifted Z:", Z[0])
