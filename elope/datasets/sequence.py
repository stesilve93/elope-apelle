
import numpy as np 
import torch

from pathlib import Path 

from elope.utils import LOGGER

from .events import EventProcessor


class SequenceLoader: 
    """DataLoader for Elope's trajectory sequences."""
    
    def __init__(
        self, 
        datapath: str | Path, 
        event_integration_window: float = 1e5, 
        event_encoder_method: str = "last_timestamp", 
        event_H : int = 200,
        event_W : int = 200,
        event_T : int = 200,
        imu_seq_len: int = 5, 
        imu_padding: str = "static"
    ): 
        """Initialise a SequenceLoader class.
        
        Parameters
        ----------
        datapath : str or Path 
            Path to the folder containing the sequence data.
        event_integration_window : float 
            Integration time window for the events, in microseconds. Defaults to `1e5`. 
        event_encoder_method : str 
            Type of event encoding. Defaults to `last_timestamp`.
        event_H, event_W, event_T : int 
            Height, width and time bins dimensions for the event tensor. 
        imu_seq_len : int, optional 
            Length of the IMU and rangemeter data, defaults to 5. 
        imu_padding: str, optional
            Type of IMU padding, supported values are "static" or "copy".
        """

        self.datapath = Path(datapath) 
        self.processor = EventProcessor() 
        
        self.events_full = None 
        self.times_full  = None 
        self.trajectory_full = None 
        self.rangemeter_full = None
        
        self.seq_len = 0
        
        # Store the settings for the events processing
        self.event_integration_window = float(event_integration_window)
        self.event_encoder_method = event_encoder_method
        self.H = int(event_H)
        self.W = int(event_W)
        self.T = int(event_T)
        
        self.imu_seq_len = int(imu_seq_len)
        self.imu_padding = imu_padding
    
    def load_sequence(self, sequence_id: str='0000') -> None:
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
        
        fn = self.datapath / (f"{sequence_id}.npz")
        if not fn.exists(): 
            raise FileNotFoundError(f"Dataset file not found: {fn}")
        
        LOGGER.info(f"Loading sequence from: \033[33m{fn}\033[0m:")
        sequence = np.load(fn)
        
        # Extract data from the loaded sequence
        self.events_full = sequence['events']
        self.timestamps_full = sequence['timestamps'] 
        self.trajectory_full = sequence['traj']
        self.rangemeter_full = sequence['range_meter']
        
        # Store the length of the sequence
        if self.timestamps_full is not None:
            self.seq_len = len(self.timestamps_full)
        else: 
            self.seq_len = 0
        
        print(f"\t- Events: {len(self.events_full)} events")
        print(f"\t- Timestamps: {len(self.timestamps_full)} steps")
        print(f"\t- Trajectory: {self.trajectory_full.shape}")
        print(f"\t- Rangemeter: {len(self.rangemeter_full)} measurements")
        
        return 
    
    def get_data_at_time(self, t_current: float) -> dict: 
        """Extract preprocessed data for a given trajectory timestamp.
        
        Parameters
        ----------
        t_current : float
            Desired trajectory timestamp, in seconds. 
        
        Returns 
        -------
        # DOCME
        """
        
        if self.events_full is None: 
            raise RuntimeError("Data not loaded. Call `load_sequence` first.")
        
        # 1. Find the index for t_current in the trajectory timestamps
        # We need to find the closest timestamp in our trajectory array to t_current. This 
        # will be the index for the trajectory states and the end point for the IMU and 
        # rangemeter sequences.
        traj_idx = np.searchsorted(self.timestamps_full, t_current, side='right') - 1
        
        if traj_idx < 0 or traj_idx >= self.seq_len:
            LOGGER.warning(f"`Time {t_current:.4f}s is out of sequence timestamp range.")
            return None
        
        # Ensure `t_current` is aligned with a trajectory timestamp for ground thruth. 
        # Another option is to interpolate the ground-truth. In this case, we simply align 
        # it to the closest point past the trajectory timestamp. 
        time = self.timestamps_full[traj_idx]
        
        # ====== Extract Events 
        events_end_time = 1e6 * time 
        
        # Filter the events up to events_end_time within the integration window 
        events_mask = (
            (self.events_full['t'] >= (events_end_time - self.event_integration_window)) & 
            (self.events_full['t'] <= events_end_time)
        )
        
        current_events = self.events_full[events_mask]
        
        # Preprocess the events to a tensor
        events_tensor = self.preprocess_events(current_events, events_end_time)
        
        # ====== Extract the IMU and groundtruth data sequences
        
        # IMU data corresponds to the trajectory data 
        imu_start_idx = max(0, traj_idx - self.imu_seq_len + 1)
        imu_times = self.timestamps_full[imu_start_idx:traj_idx+1]
        
        # Retrieve the angular measures (phi, theta, psi, p, q, r)
        imu_seq = self.trajectory_full[imu_start_idx:traj_idx+1, 6:12] 
        targets = self.trajectory_full[imu_start_idx:traj_idx+1, 0:6]
        
        # Pad if the IMU sequence is shorter than imu_seq_len 
        if imu_seq.shape[0] < self.imu_seq_len: 
            npads = self.imu_seq_len - imu_seq.shape[0]
            
            if imu_seq.shape[0] > 0: 
                # Repeat the first available IMU measure for padding 
                padding_imu = np.tile(imu_seq[0], (npads, 1))
                padding_trg = np.tile(targets[0], (npads, 1))
                
            else: 
                # If no IMU data at all, pad with the zeros
                padding_imu = np.zeros((npads, 6))
                padding_trg = np.zeros((npads, 6))
                
            if self.imu_padding == "static":
                # Sets the lander to a static scenario in which it is not moving. 
                padding_imu[:, 3:6] = 0.0 
                padding_trg[:, 3:6] = 0.0

            # Apply the padding to the IMU and position/velocity data
            imu_seq = np.vstack([padding_imu, imu_seq])
            targets = np.vstack([padding_trg, targets])

        # ====== Extract the rangemeter sequence 
        
        # Find the rangemeter data points and align the start time with the IMU data 
        range_end_time = time 
        range_start_time = imu_times[0]
        
        range_mask = (self.rangemeter_full[:, 0] >= range_start_time) & \
                     (self.rangemeter_full[:, 0] <= range_end_time)
        
        current_range = self.rangemeter_full[range_mask]
        
        # Interpolate rangemeter data to match the IMU sequence timestamps
        # We need the timestamps corresponding to the IMU sequence for interpolation
        if current_range.shape[0] > 1:
            interpolated_distances = np.interp(
                imu_times, current_range[:, 0], current_range[:, 1]
            )
            
            range_seq = interpolated_distances.reshape(-1, 1)
            
        elif current_range.shape[0] == 1:
            # If only one rangemeter reading, use it for all target timestamps
            range_seq = np.full((len(imu_times), 1), current_range[0, 1])
            
        else: 
            # No rangemeter data in window, fill with a default value (e.g., 0 or a large number)
            range_seq = np.zeros((len(imu_times), 1)) 

        # Pad if necessary for rangemeter sequence length
        if range_seq.shape[0] < self.imu_seq_len:
            npads = self.imu_seq_len - range_seq.shape[0]
            
            if range_seq.shape[0] > 0:
                # Repeat the first available rangemeter measurement for padding
                padding = np.tile(range_seq[0], (npads, 1))
            else:
                # If no altimeter data, pad with zeros
                padding = np.zeros((npads, 1))
                
            range_seq = np.vstack([padding, range_seq])
        
        return {
            'events_tensor': torch.from_numpy(events_tensor), 
            'imu_sequence': torch.from_numpy(imu_seq.astype(np.float32)), 
            'rangemeter_sequence': torch.from_numpy(range_seq.astype(np.float32)), 
            'ground_truth': torch.from_numpy(targets.astype(np.float32)),
            'time': torch.tensor(np.float32(time))
        }
    
    def preprocess_events(self, events: np.ndarray, end_time: float) -> np.ndarray:
        """Preprocess events into a 4D tensor representation (EVFlownet-like).

        Parameters
        ----------
        events : np.ndarray
            Array of raw events with columns (x,y,p,t). 
        end_time : float

        Returns
        -------
        events : np.ndarray 
            Event tensor of shape (C, T, H, W).
        """
        
        # Retrieve event tensor settings
        H, W, T = self.H, self.W, self.T
        
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
            LOGGER.warning("No events found in the specified time window.")
            # Return an empty tensor with the expected shape
            if self.event_encoder_method == "count":
                return np.zeros((2, T, H, W), dtype=np.float32)
            elif self.event_encoder_method == "last_timestamp":
                return np.zeros((2, 2, H, W), dtype=np.float32)
            else: 
                raise ValueError("Unsupported event encoding method.")
            
        # Filter events based on the given time window (most recent events)
        if self.event_integration_window > 0:
            t_max = end_time
            t_min = t_max - self.event_integration_window 
            mask = events_array[:, 3] >= t_min
            events_array = events_array[mask]

        # Convert the filtered events into a 4D tensor
        tensor = self.processor.events_to_tensor(
            events_array, H, W, T, method=self.event_encoder_method,
            end_time=end_time, time_window=self.event_integration_window
        )

        # Normalize the tensor using standard normalization
        tensor = self.processor.normalize_tensor(tensor, method='standard')

        # Rearrange tensor dimensions to PyTorch format: (Channels, Time, Height, Width)
        tensor = np.transpose(tensor, (3, 0, 1, 2))

        return tensor.astype(np.float32)