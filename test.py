import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

from elope_modules.dataloader import DataLoader
from elope_modules.emmnet import create_model 

def run_realtime_prediction(
    model: nn.Module,
    data_loader_instance: DataLoader,
    sequence_id: str,
    event_integration_window_us: float = 1e5, # 100ms
    imu_seq_len: int = 50,
    H: int = 200, W: int = 200, T: int = 10,
    prediction_interval_s: float = 0.05, # How often to make a prediction (e.g., every 50ms)
    start_offset_s: float = 0.5, # Time to wait before first prediction (to fill LSTM buffers)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs "real-time" sequential prediction on a single test trajectory.

    Args:
        model (nn.Module): The trained MultiModalVelocityEstimator model.
        data_loader_instance (DataLoader): An instantiated DataLoader object.
        sequence_id (str): The ID of the test sequence (e.g., '0040').
        event_integration_window_us (float): Time window for events in microseconds.
        imu_seq_len (int): Sequence length for IMU and rangemeter data.
        H, W, T: Dimensions for the event tensor.
        prediction_interval_s (float): Interval in seconds at which to generate a prediction.
        start_offset_s (float): Initial offset in seconds to ensure enough data for seq_len.
                                This should be at least (imu_seq_len * trajectory_timestamp_interval).
        device (torch.device): The device to run inference on (cuda or cpu).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - predicted_states: Array of predicted (x,y,z,vx,vy,vz) at each prediction timestamp.
            - ground_truth_states: Array of corresponding ground truth (x,y,z,vx,vy,vz).
            - prediction_times_s: Array of timestamps at which predictions were made.
    """
    model.eval() # Set model to evaluation mode
    model.to(device)

    # Load the full test sequence data
    print(f"Loading test sequence {sequence_id}...")
    data_loader_instance.load_sequence(sequence_id, "train")
    
    # Store results
    predicted_states = []
    ground_truth_states = []
    prediction_times_s = []

    # Get the minimum timestamp from the trajectory to set an initial start time
    # This assumes trajectory timestamps are sorted and representative of the sequence
    min_traj_time_s = data_loader_instance.timestamps_full[0]
    
    # Determine the actual start time for predictions
    # This ensures that for the first prediction, we have enough history for the LSTMs.
    # The minimum required history time is roughly imu_seq_len * (avg_time_between_imu_samples)
    # A simple way to approximate avg_time_between_imu_samples:
    avg_imu_interval = np.mean(np.diff(data_loader_instance.timestamps_full[:imu_seq_len+5])) # Small sample for avg
    
    # Ensure start_offset_s covers at least the LSTM sequence length
    required_min_offset_s = imu_seq_len * avg_imu_interval
    if start_offset_s < required_min_offset_s:
        print(f"Warning: start_offset_s ({start_offset_s:.4f}s) is less than the recommended "
              f"minimum for IMU sequence length ({required_min_offset_s:.4f}s). Adjusting.")
        start_offset_s = required_min_offset_s * 1.1 # Add a small buffer

    # Start predictions after an initial warm-up period
    # Find the first timestamp in the trajectory that is at least `start_offset_s` past the beginning
    initial_prediction_time_s = min_traj_time_s + start_offset_s
    
    # Iterate through timestamps, simulating real-time
    # We will use data_loader_instance.timestamps_full as the master time index
    # We'll skip timestamps until we reach initial_prediction_time_s
    
    # Find the starting index for inference
    start_inference_idx = np.searchsorted(data_loader_instance.timestamps_full, initial_prediction_time_s, side='left')
    
    # Ensure we don't go out of bounds
    if start_inference_idx >= len(data_loader_instance.timestamps_full):
        print("No valid timestamps found for inference after initial offset.")
        return np.array([]), np.array([]), np.array([])
    
    # Select prediction timestamps at the specified interval
    current_t_idx = start_inference_idx
    
    print(f"Starting predictions from timestamp {data_loader_instance.timestamps_full[current_t_idx]:.4f}s")
    
    while current_t_idx < len(data_loader_instance.timestamps_full):
        t_current_s = data_loader_instance.timestamps_full[current_t_idx]
        
        # Get the synchronized data for the current time
        data_point = data_loader_instance.get_data_at_time(
            t_current_s,
            event_integration_window_us=event_integration_window_us,
            imu_seq_len=imu_seq_len,
            H=H, W=W, T=T
        )

        if data_point is None:
            # This can happen if t_current_s is out of bounds or data is insufficient
            print(f"Skipping prediction at {t_current_s:.4f}s due to insufficient data.")
            current_t_idx += 1 # Move to next timestamp
            continue
        
        # Unpack and move to device
        event_t = data_point['events_tensor'].unsqueeze(0).to(device) # Add batch dimension
        imu_s = data_point['imu_sequence'].unsqueeze(0).to(device)
        range_s = data_point['rangemeter_sequence'].unsqueeze(0).to(device)
        gt_pv = data_point['ground_truth'].unsqueeze(0).to(device)

        with torch.no_grad(): # No gradient calculations during inference
            outputs = model(event_t, imu_s, range_s)
            prediction = outputs['prediction']

        # Store the results
        print(f"Prediction at {t_current_s:.4f}s: {prediction.cpu().numpy().squeeze()}")
        predicted_states.append(prediction.cpu().numpy().squeeze()) # Remove batch dim
        print(f"Ground truth at {t_current_s:.4f}s: {gt_pv.cpu().numpy().squeeze()}")
        ground_truth_states.append(gt_pv.cpu().numpy().squeeze())
        prediction_times_s.append(t_current_s)

        # Move to the next prediction timestamp
        # Find the next timestamp in the trajectory that is at least `prediction_interval_s` later
        next_t_s = t_current_s + prediction_interval_s
        current_t_idx = np.searchsorted(data_loader_instance.timestamps_full, next_t_s, side='left')
        
    print(f"Finished predictions for sequence {sequence_id}. Total predictions: {len(predicted_states)}")
    
    return np.array(predicted_states), np.array(ground_truth_states), np.array(prediction_times_s)

# --- Main execution block for testing ---
if __name__ == "__main__":
    # --- Configuration ---
    DATAPATH = './elope_data' # Adjust as needed
    MODEL_PATH = 'checkpoints/model_epoch_50.pth' # Path to your trained model weights
    TEST_SEQUENCE_ID = '0020' # A test sequence not used in training (e.g., the first test trajectory)
    
    USE_ATTENTION = True # Must match how your trained model was created
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for inference: {DEVICE}")

    # --- 1. Initialize DataLoader and Model ---
    data_loader = DataLoader(datapath=DATAPATH)
    model = create_model(use_attention=USE_ATTENTION, device=DEVICE)

    # --- 2. Load Trained Model Weights ---
    if os.path.exists(MODEL_PATH):
        print(f"Loading model weights from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print(f"Warning: Trained model weights not found at {MODEL_PATH}. Using randomly initialized model.")
        print("Please train your model first or provide the correct path to weights.")
        # Optionally, exit or raise an error if model weights are essential
        # sys.exit(1)

    # --- 3. Run Real-Time Prediction ---
    print("\nStarting real-time prediction simulation...")
    predicted_states, ground_truth_states, prediction_times = run_realtime_prediction(
        model=model,
        data_loader_instance=data_loader,
        sequence_id=TEST_SEQUENCE_ID,
        event_integration_window_us=100000, # Same as training
        imu_seq_len=5, # Same as training
        H=200, W=200, T=5, # Same as training
        prediction_interval_s=0.05, # Make a prediction every 50ms
        start_offset_s=0.5, # Ensure at least 0.5s of data for LSTM warm-up
        device=DEVICE
    )

    if len(predicted_states) > 0:
        print("\nPrediction Results:")
        print(f"Total predictions: {len(predicted_states)}")
        print(f"Shape of predicted_states: {predicted_states.shape}")
        print(f"Shape of ground_truth_states: {ground_truth_states.shape}")

        # Calculate Mean Squared Error (MSE) on the test sequence for the whole vector
        mse_total = np.mean((predicted_states - ground_truth_states)**2)
        print(f"Overall Mean Squared Error on test sequence {TEST_SEQUENCE_ID}: {mse_total:.6f}")

        # Calculate individual MSE for each component
        component_labels = ['X (m)', 'Y (m)', 'Z (m)', 'Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)']
        for i, label in enumerate(component_labels):
            mse_component = np.mean((predicted_states[:, i] - ground_truth_states[:, i])**2)
            print(f"MSE for {label}: {mse_component:.6f}")

        # --- Plotting All 6 DoF Components ---
        # Create a 3x2 grid of subplots for position (x, y, z) and velocity (vx, vy, vz)
        fig, axes = plt.subplots(3, 2, figsize=(15, 15)) # Increased figsize for better readability
        axes = axes.flatten() # Flatten the 2D array of axes for easier iteration

        plot_info = [
            {'idx': 0, 'label': 'X', 'ylabel': 'Position (m)'},
            {'idx': 1, 'label': 'Y', 'ylabel': 'Position (m)'},
            {'idx': 2, 'label': 'Z', 'ylabel': 'Altitude (m)'},
            {'idx': 3, 'label': 'Vx', 'ylabel': 'Velocity (m/s)'},
            {'idx': 4, 'label': 'Vy', 'ylabel': 'Velocity (m/s)'},
            {'idx': 5, 'label': 'Vz', 'ylabel': 'Velocity (m/s)'}
        ]

        for i, info in enumerate(plot_info):
            ax = axes[i]
            idx = info['idx']
            label = info['label']
            ylabel = info['ylabel']

            ax.plot(prediction_times, ground_truth_states[:, idx], label=f'GT {label}', color='blue', linewidth=2)
            ax.plot(prediction_times, predicted_states[:, idx], label=f'Pred {label}', color='red', linestyle='--', linewidth=1.5)
            
            ax.set_title(f'{label} Prediction vs. Ground Truth')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True)

        plt.suptitle(f'6-DOF State Prediction vs. Ground Truth for Sequence {TEST_SEQUENCE_ID}', fontsize=16, y=1.02) # Add a main title
        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to make space for suptitle
        plt.savefig(f"lunar_descent_predictions_{TEST_SEQUENCE_ID}.png", dpi=300) # Save with sequence ID in filename and higher DPI

        # You can add more plots for X, Y, Vx, Vy if needed.
    else:
        print("No predictions were made. Check sequence ID, data path, or time offsets.")