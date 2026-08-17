import os
import glob
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import pickle

def load_and_preprocess_data(data_dir, split_ratio=(60, 5, 5), seed=42):
    """
    Loads trajectories from CSV files, drops irregular timesteps, splits by trajectory,
    scales data, and generates one-step transition pairs.
    """
    random.seed(seed)
    
    # 1. Get all trajectory files
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not all_files:
        all_files = glob.glob(os.path.join(data_dir, "*")) # In case they don't have .csv extension
        
    print(f"Found {len(all_files)} trajectory files.")
    
    # Shuffle and split
    random.shuffle(all_files)
    n_train, n_val, n_test = split_ratio
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train : n_train + n_val]
    test_files = all_files[n_train + n_val :]
    
    print(f"Split sizes -> Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # Define columns based on investigation
    cols_to_use = ['Car.Gen.vx_1', 'Car.Gen.vy_1', 'Car.Virtual.Frc_1.x', 'Car.YawRate', 'Time', 'Vhcl.Steer.Ang']
    
    def process_file_list(file_list, desc=""):
        trajectories = []
        for file in file_list:
            # Read CSV skipping the first 3 lines of header
            df = pd.read_csv(file, skiprows=3, header=None)
            # Assuming headers map to columns 0-5 as:
            # 0: vx, 1: vy, 2: Fx, 3: omega, 4: time, 5: delta
            df.columns = cols_to_use
            
            # Check dt
            time = df['Time'].values
            dt = np.diff(time, prepend=time[0]-0.01) # First diff will be 0.01 normally
            
            invalid_mask = dt <= 0.0
            num_invalid = np.sum(invalid_mask)
            
            if num_invalid > 0:
                print(f"[{desc}] File {os.path.basename(file)}: dropped {num_invalid} rows with dt <= 0")
                df = df[~invalid_mask]
                
            # States and Controls
            # x = [vx, vy, omega]
            # u = [Fx, delta]
            x_raw = df[['Car.Gen.vx_1', 'Car.Gen.vy_1', 'Car.YawRate']].values
            u_raw = df[['Car.Virtual.Frc_1.x', 'Vhcl.Steer.Ang']].values
            
            trajectories.append({
                'file': file,
                'x_raw': x_raw,
                'u_raw': u_raw,
                'time': df['Time'].values
            })
            
        return trajectories

    print("\nProcessing Train Trajectories...")
    train_trajs = process_file_list(train_files, "Train")
    print("Processing Val Trajectories...")
    val_trajs = process_file_list(val_files, "Val")
    print("Processing Test Trajectories...")
    test_trajs = process_file_list(test_files, "Test")
    
    # Fit StandardScaler on train data only
    print("\nFitting Scalers on Train data...")
    all_train_x = np.vstack([traj['x_raw'] for traj in train_trajs])
    all_train_u = np.vstack([traj['u_raw'] for traj in train_trajs])
    
    scaler_x = StandardScaler().fit(all_train_x)
    scaler_u = StandardScaler().fit(all_train_u)
    
    # Save scalers for future use
    os.makedirs('models', exist_ok=True)
    with open('models/scaler_x.pkl', 'wb') as f:
        pickle.dump(scaler_x, f)
    with open('models/scaler_u.pkl', 'wb') as f:
        pickle.dump(scaler_u, f)
        
    def scale_and_build_pairs(trajectories):
        X_list, U_list, X_prime_list = [], [], []
        scaled_trajs = []
        
        for traj in trajectories:
            x_scaled = scaler_x.transform(traj['x_raw'])
            u_scaled = scaler_u.transform(traj['u_raw'])
            
            # Store scaled full trajectories for rollout validation
            scaled_trajs.append({
                'file': traj['file'],
                'x': x_scaled,
                'u': u_scaled,
                'time': traj['time'],
                'x_raw': traj['x_raw'] # Keep raw for evaluation metrics
            })
            
            # Build one-step pairs
            if len(x_scaled) > 1:
                X_list.append(x_scaled[:-1])
                U_list.append(u_scaled[:-1])
                X_prime_list.append(x_scaled[1:])
                
        # Stack all pairs vertically
        X_mat = np.vstack(X_list)
        U_mat = np.vstack(U_list)
        X_prime_mat = np.vstack(X_prime_list)
        
        return X_mat, U_mat, X_prime_mat, scaled_trajs

    print("Building one-step pairs...")
    X_train, U_train, X_prime_train, train_scaled = scale_and_build_pairs(train_trajs)
    X_val, U_val, X_prime_val, val_scaled = scale_and_build_pairs(val_trajs)
    X_test, U_test, X_prime_test, test_scaled = scale_and_build_pairs(test_trajs)
    
    print("\nDataset Statistics:")
    print(f"Total transition pairs - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    lengths = [len(t['time']) for t in train_trajs + val_trajs + test_trajs]
    print(f"Trajectory lengths - Min: {min(lengths)}, Max: {max(lengths)}, Mean: {np.mean(lengths):.1f}")
    
    return {
        'train': (X_train, U_train, X_prime_train, train_scaled),
        'val': (X_val, U_val, X_prime_val, val_scaled),
        'test': (X_test, U_test, X_prime_test, test_scaled),
        'scalers': (scaler_x, scaler_u)
    }

if __name__ == "__main__":
    # Test the loader
    data = load_and_preprocess_data('data/dataset', split_ratio=(60, 5, 5))
