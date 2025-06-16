import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

from elope_modules.event_proc import EventProcessor

class DataLoader:
    """Data loader for elope dataset"""
    
    def __init__(self, datapath: str = './elope_data'):
        
        self.datapath = datapath
        self.processor = EventProcessor()

        self.events_full = None
        self.timestamps_full = None
        self.trajectory_full = None
        self.rangemeter_full = None
    
    def load_sequence(self, sequence_id: str = '0000', source: str = "train") -> None:
        """
        Load a single sequence from the dataset
        
        Loads a sequence from the dataset, which consists of a set of events, timestamps,
        trajectory and optionally rangemeter data.
        
        Parameters
        ----------
        sequence_id : str, optional
            Sequence ID to load, by default '0000'
        
        Returns
        -------
        Dict
            Dictionary with the following keys:
            - events: Events array [x, y, p, t]
            - timestamps: Timestamps array [t]
            - trajectory: Trajectory array [x, y, z, vx, vy, vz]
            - rangemeter: Rangemeter array [t, range], or None if not available
        """
        fn = os.path.join(self.datapath, source, f'{sequence_id}.npz')
        
        if not os.path.exists(fn):
            raise FileNotFoundError(f"Dataset file not found: {fn}")
        
        print(f"Loading sequence from: {fn}")
        sequence = np.load(fn)
        
        # Extract data from the loaded sequence
        self.events_full = sequence['events']
        self.timestamps_full = sequence['timestamps'] 
        self.trajectory_full = sequence['traj']
        self.rangemeter_full = sequence['range_meter']
        
        print(f"Loaded:")
        print(f"  - Events: {len(self.events_full)} events")
        print(f"  - Timestamps: {len(self.timestamps_full)} steps")
        print(f"  - Trajectory: {self.trajectory_full.shape}")
        print(f"  - Rangemeter: {len(self.rangemeter_full)} measurements")
        
        return 
    
    def get_data_at_time(self, t_current: float,
                         event_integration_window_us: float = 1e5,
                         imu_seq_len: int = 50,
                         H: int = 200, W: int = 200, T: int = 10, event_encoder_method: str = "last_timestamp") -> Optional[Dict]:
        """
        Extracts preprocessed data for a given t_current.

        Args:
            t_current (float): The current timestamp in seconds.
            event_integration_window_us (float): Time window for events in microseconds.
            imu_seq_len (int): Sequence length for IMU and rangemeter data.
            H, W, T: Dimensions for the event tensor.

        Returns:
            Dict: Contains 'events_tensor', 'imu_sequence', 'rangemeter_sequence',
                  and 'ground_truth' for the given t_current.
                  Returns None if t_current is out of range for the full dataset.
        """
        if self.events_full is None:
            raise RuntimeError("Data not loaded. Call load_sequence() first.")

        # 1. Find the index for t_current in the trajectory timestamps
        # We need to find the closest timestamp in our trajectory array to t_current
        # This will be the index for our ground truth and the end point for IMU/rangemeter sequences.
        traj_idx = np.searchsorted(self.timestamps_full, t_current, side='right') - 1
        
        if traj_idx < 0 or traj_idx >= len(self.timestamps_full):
            print(f"Warning: t_current {t_current:.4f}s is out of trajectory timestamp range.")
            return None
        
        #print(f"Timestamps for t_current {t_current:.4f}s at index {traj_idx} corresponding to {self.timestamps_full[traj_idx]:.4f}s")
        
        # Ensure t_current is aligned with a trajectory timestamp for ground truth
        # We could interpolate ground truth if t_current is arbitrary
        # For simplicity, we'll align it to the closest past trajectory timestamp.
        actual_t_current = self.timestamps_full[traj_idx]
        #print(f"Actual t_current aligned to trajectory timestamp: {actual_t_current:.4f}s at index {traj_idx}")
        # --- Extract Events ---
        # Convert actual_t_current (seconds) to microseconds for event filtering
        events_end_time_us = actual_t_current * 1e6
        #print(f"Events end time in microseconds: {events_end_time_us:.0f}us")
        # Filter events up to events_end_time_us within the integration window
        events_mask = (self.events_full['t'] >= (events_end_time_us - event_integration_window_us)) & \
                      (self.events_full['t'] <= events_end_time_us)
        
        current_events = self.events_full[events_mask]
        
        # Preprocess events to tensor
        # Pass actual_t_current in microseconds for the preprocess_events logic
        events_tensor = self.preprocess_events(current_events, events_end_time_us,
                                               time_window=event_integration_window_us,
                                               H=H, W=W, T=T, method=event_encoder_method)

        # --- Extract IMU Sequence ---
        # IMU data corresponds to trajectory data directly
        imu_start_idx = max(0, traj_idx - imu_seq_len + 1)
        imu_sequence_raw = self.trajectory_full[imu_start_idx : traj_idx + 1, 6:12] # phi,theta,psi,p,q,r
        
        # Pad if the sequence is shorter than imu_seq_len
        if imu_sequence_raw.shape[0] < imu_seq_len:
            padding_needed = imu_seq_len - imu_sequence_raw.shape[0]
            # Repeat the first available IMU measurement for padding
            if imu_sequence_raw.shape[0] > 0:
                padding = np.tile(imu_sequence_raw[0], (padding_needed, 1))
            else: # If no IMU data at all, pad with zeros
                padding = np.zeros((padding_needed, 6))
            imu_sequence = np.vstack([padding, imu_sequence_raw])
        else:
            imu_sequence = imu_sequence_raw

        # --- Extract Rangemeter Sequence ---
        # Find relevant rangemeter data points
        rangemeter_end_time = actual_t_current
        rangemeter_start_time = self.timestamps_full[max(0, traj_idx - imu_seq_len + 1)] # Align with IMU start
        
        rangemeter_mask = (self.rangemeter_full[:, 0] >= rangemeter_start_time) & \
                          (self.rangemeter_full[:, 0] <= rangemeter_end_time)
        
        current_rangemeter = self.rangemeter_full[rangemeter_mask]

        # Interpolate rangemeter data to match the IMU sequence timestamps
        # We need the timestamps corresponding to the IMU sequence for interpolation
        target_imu_timestamps = self.timestamps_full[imu_start_idx : traj_idx + 1]

        if current_rangemeter.shape[0] > 1:
            interpolated_distances = np.interp(target_imu_timestamps, 
                                               current_rangemeter[:, 0], 
                                               current_rangemeter[:, 1])
            rangemeter_sequence_raw = interpolated_distances.reshape(-1, 1)
        elif current_rangemeter.shape[0] == 1:
            # If only one rangemeter reading, use it for all target timestamps
            rangemeter_sequence_raw = np.full((len(target_imu_timestamps), 1), current_rangemeter[0, 1])
        else: # No rangemeter data in window, fill with a default value (e.g., 0 or a large number)
            rangemeter_sequence_raw = np.zeros((len(target_imu_timestamps), 1)) # Or np.full(..., some_default_range)

        # Pad if necessary for rangemeter sequence length
        if rangemeter_sequence_raw.shape[0] < imu_seq_len:
            padding_needed = imu_seq_len - rangemeter_sequence_raw.shape[0]
            # Repeat the first available rangemeter measurement for padding
            if rangemeter_sequence_raw.shape[0] > 0:
                padding = np.tile(rangemeter_sequence_raw[0], (padding_needed, 1))
            else:
                padding = np.zeros((padding_needed, 1))
            rangemeter_sequence = np.vstack([padding, rangemeter_sequence_raw])
        else:
            rangemeter_sequence = rangemeter_sequence_raw
        
        # --- Ground Truth ---
        # Retrieve the position and velocity values (x, y, z, vx, vy, vz)
        ground_truth = self.trajectory_full[traj_idx, :6]
        
        return {
            'events_tensor': torch.from_numpy(events_tensor), 
            'imu_sequence': torch.from_numpy(imu_sequence.astype(np.float32)), 
            'rangemeter_sequence': torch.from_numpy(rangemeter_sequence.astype(np.float32)), 
            'ground_truth': torch.from_numpy(ground_truth.astype(np.float32)),
            'time': torch.tensor(np.float32(actual_t_current))
        }
    
    def preprocess_events(self, events: np.ndarray, end_time: float,
                         time_window: float = 1e5,
                         H: int = 200, W: int = 200, T: int = 10, method="last_timestamp") -> np.ndarray:
        """
        Preprocess events into a 4D tensor representation (EVFlownet-like).

        Args:
            events: Raw events array with columns [x, y, p, t].
            time_window: Time window in microseconds to filter events.
            H, W, T: Dimensions of the output tensor (Height, Width, Time bins).

        Returns:
            4D tensor in PyTorch format with dimensions (C, T, H, W).
        """
        
        # Create a copy of the events to avoid modifying the original data
        events_copy = events.copy()

        # Check if the events are in a structured array format
        if events_copy.dtype == [('x', '<i2'), ('y', '<i2'), ('p', '?'), ('t', '<i8')]:
            # Convert structured array to a regular ndarray with integer polarity
            events_array = np.column_stack([
                events_copy['x'],
                events_copy['y'], 
                events_copy['p'].astype(int),
                events_copy['t']
            ])
        else:
            # Assume events are already in a regular array format
            events_array = events_copy
            # Handle potential 1D array by converting it into a 2D array
            if len(events_array.shape) == 1:
                events_array = np.array([[e[0], e[1], int(e[2]), e[3]] for e in events])
        
        # Ensure events_array is a 2D array
        if events_array.shape[0] == 0:
            print("No events found in the specified time window.")
            # Return an empty tensor with the expected shape
            if method == "count":
                return np.zeros((2, T, H, W), dtype=np.float32)
            elif method == "last_timestamp":
                return np.zeros((2, 2, H, W), dtype=np.float32)
            
        # Filter events based on the given time window (most recent events)
        if time_window > 0:
            t_max = end_time
            t_min = t_max - time_window 
            mask = events_array[:, 3] >= t_min
            events_array = events_array[mask]

        # Convert the filtered events into a 4D tensor
        tensor = self.processor.events_to_tensor(events_array, H, W, T, method=method,
                                                 end_time=end_time, time_window=time_window)

        # Normalize the tensor using standard normalization
        tensor = self.processor.normalize_tensor(tensor, method='standard')

        # Rearrange tensor dimensions to PyTorch format: (Channels, Time, Height, Width)
        tensor = np.transpose(tensor, (3, 0, 1, 2))

        return tensor.astype(np.float32)
    
    # Legacy methods for modular processing of events, IMU, and rangemeter data (i.e. not using end-to-end deep network processing).
    def preprocess_imu(self, trajectory: np.ndarray, seq_len: int = 50) -> np.ndarray:
        """
        Extract IMU data (Euler angles + angular velocities) from trajectory

        Args:
            trajectory: Full trajectory array [x,y,z,vx,vy,vz,phi,theta,psi,p,q,r]
            seq_len: Sequence length for LSTM input

        Returns:
            IMU sequence of shape (seq_len, 6) with columns [phi, theta, psi, p, q, r]
        """
        
        # Extract Euler angles and angular velocities (columns 6-11)
        imu_data = trajectory[:, 6:12]  # [phi, theta, psi, p, q, r]
        
        # Take last seq_len steps
        if len(imu_data) >= seq_len:
            imu_sequence = imu_data[-seq_len:]
        else:
            # Pad if necessary
            padding = np.tile(imu_data[0], (seq_len - len(imu_data), 1))
            imu_sequence = np.vstack([padding, imu_data])
        
        return imu_sequence.astype(np.float32)
    
    def preprocess_rangemeter(self, rangemeter: Optional[np.ndarray], 
                             timestamps: np.ndarray, seq_len: int = 50) -> np.ndarray:
        """
        Preprocess rangemeter data
        
        If rangemeter data is not available, create dummy data.
        
        Args:
            rangemeter: Rangemeter measurements [t, d] or None
            timestamps: Trajectory timestamps
            seq_len: Sequence length
        """
        
        # Interpolate rangemeter to trajectory timestamps
        range_times = rangemeter[:, 0]
        range_distances = rangemeter[:, 1]
        
        # Take last seq_len timestamps
        target_times = timestamps[-seq_len:] if len(timestamps) >= seq_len else timestamps
        
        # Interpolate
        interpolated_distances = np.interp(target_times, range_times, range_distances)
        range_sequence = interpolated_distances.reshape(-1, 1)
    
        return range_sequence.astype(np.float32)
