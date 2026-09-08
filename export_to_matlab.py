import numpy as np
import pickle
import scipy.io as sio
import os
from lifting import lift_state

def export_model_to_matlab():
    print("Exporting trained Koopman model to MATLAB .mat file...")
    
    A = np.load('models/A.npy')
    B = np.load('models/B.npy')
    
    with open('models/scaler_x.pkl', 'rb') as f:
        scaler_x = pickle.load(f)
    with open('models/scaler_u.pkl', 'rb') as f:
        scaler_u = pickle.load(f)
        
    x_mean = np.zeros(3) # Pure variance scaling (mean = 0)
    x_std = scaler_x.scale_
    u_mean = np.zeros(2) # Pure variance scaling (mean = 0)
    u_std = scaler_u.scale_
    
    mat_data = {
        'A_l': A,
        'B_l': B,
        'C_l': np.hstack([np.eye(3), np.zeros((3, 6))]), # Linear output selector matrix (3x9)
        'x_mean': x_mean,
        'x_std': x_std,
        'u_mean': u_mean,
        'u_std': u_std,
        'Ts': 0.01,
        'lifting_dim': 9
    }
    
    os.makedirs('Test 1', exist_ok=True)
    out_path = 'Test 1/mpc_design_data_dim9.mat'
    sio.savemat(out_path, mat_data)
    print(f"Successfully exported model to: {out_path}")

if __name__ == '__main__':
    export_model_to_matlab()
