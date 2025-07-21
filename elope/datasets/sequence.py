
import numpy as np 
import torch

from pathlib import Path 

from scipy.interpolate import UnivariateSpline

from elope.utils import LOGGER

from .events import EventProcessor


class SequenceLoader: 
    """DataLoader for Elope's trajectory sequences."""
    
    def __init__(
        self, 
        datapath: str | Path, 
        event_integration_window: float = 1e5, 
        event_encoder_method: str = "last_timestamp", 
        event_clamp: int = -1,
        event_H : int = 200,
        event_W : int = 200,
        event_T : int = 1,
        imu_seq_len: int = 5, 
        imu_padding: str = "static"
    ): 
        """Initialise a SequenceLoader class.
        
        Parameters
        ----------
        datapath : str or Path 
            Path to the folder containing the sequence data.
        event_integration_window : float, optional
            Integration time window for the events, in microseconds. Defaults to `1e5`. 
        event_encoder_method : str, optional
            Type of event encoding. Defaults to `last_timestamp`.
        event_clamp : int, optional 
            Maximum value for the time bins. Defaults to 10.
        event_H, event_W, event_T : int, optional
            Height, width and time bins dimensions for the event tensor. 
        imu_seq_len : int, optional 
            Length of the IMU and rangemeter data, defaults to 5. 
        imu_padding: str, optional
            Type of IMU padding, supported values are "static" or "copy". Defaults to static.
        """

        self.datapath = Path(datapath) 
        self.processor = EventProcessor() 
        
        self.events_full = None 
        self.timestamps_full  = None 
        self.trajectory_full = None 
        self.rangemeter_full = None
        
        self.seq_len = 0
        
        # Store the settings for the events processing
        self.event_integration_window = float(event_integration_window)
        self.event_encoder_method = event_encoder_method
        
        self.event_clamp = event_clamp
        
        self.H = int(event_H)
        self.W = int(event_W)
        self.T = int(event_T)
        
        # Ensure the encoder method is supported
        assert self.event_encoder_method in (
            "first_timestamp", "last_timestamp", "timestamp", "count", "hybrid"
        )
        
        # Ensure the number of time bins is coherent with the encoder method 
        if self.event_encoder_method in ("first_timestamp", "last_timestamp") and self.T != 1: 
            raise ValueError(
                f"Event encoder {self.event_encoder_method} supports only 1 event channel."
            )
                
        elif self.event_encoder_method == "timestamp" and self.T != 2: 
            raise ValueError("Event encoder `timestamp` supports only 2 event channels.")
        
        elif self.event_encoder_method == "hybrid" and self.T != 3: 
            raise ValueError("Event encoder `hybrid` supports only 3 event channels.")
        
        self.imu_seq_len = int(imu_seq_len)
        self.imu_padding = imu_padding
        
        self.events_pre_side = None
    
    def load_sequence(self, sequence_id: str='0000'):
        """
        Load a single sequence from the dataset
        
        Loads a sequence from the dataset, which consists of a set of events, timestamps,
        trajectory and optionally rangemeter data.
        
        Parameters
        ----------
        sequence_id : str, optional
            Sequence ID to load, by default '0000'
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
            
        # Update the event pre-processing flag
        self.events_pre_side = None
        
        print(f"\t- Events: {len(self.events_full)} events")
        print(f"\t- Timestamps: {len(self.timestamps_full)} steps")
        print(f"\t- Trajectory: {self.trajectory_full.shape}")
        print(f"\t- Rangemeter: {len(self.rangemeter_full)} measurements")
        
    def get_data_at_time(self, time: float, flip: bool=False) -> dict: 
        """Extract preprocessed data for a given trajectory timestamp.
        
        Parameters
        ----------
        time : float
            Desired trajectory timestamp, in seconds. 
        flip : bool, optional
            True if the motion should be flipped backwards. Defaults to `False`. 
        
        Returns 
        -------
        data : dict 
            Sample sequence dictionary with fields: 
            - times: an array of shape (n,) with the trajectory timestamps.
            - ground_truth: an array of shape (n,6) with the true positions and velocities.
            - imu_sequence: an array of shape (n,6) with the IMU angles and angular velocities.
            - rangemeter_sequence: an array of shape (n,) with the rangemeter data.
            - events_tensor: an array of shape (n, 2, C, H, W) with the events polarity, 
              channels, height an width data.
        """
        
        if self.events_full is None: 
            raise RuntimeError("Data not loaded. Call `load_sequence` first.")
        
        # Find the index for t_current in the trajectory timestamps
        # We need to find the closest timestamp in our trajectory array to t_current. This 
        # will be the index for the trajectory states and the end point for the IMU and 
        # rangemeter sequences.
        traj_idx = np.searchsorted(self.timestamps_full, time, side='right') - 1
        
        if traj_idx < 0 or traj_idx >= self.seq_len:
            LOGGER.warning(f"`Time {time:.4f}s is out of sequence timestamp range.")
            return None
        
        # Compute the time step for the trajectory data 
        dt = np.mean(np.diff(self.timestamps_full))
        
        # ====== Extract the IMU and groundtruth data sequences
        
        # Retrieve the sampling times at the desired IMU times
        imu_start_idx = max(0, traj_idx - self.imu_seq_len + 1)
        imu_tms = self.timestamps_full[imu_start_idx:traj_idx+1].copy()
        
        # Retrieve the angular measures (phi, theta, psi, p, q, r)
        imu_seq = self.trajectory_full[imu_start_idx:traj_idx+1, 6:12].copy()
        targets = self.trajectory_full[imu_start_idx:traj_idx+1, 0:6].copy()
        
        # Pad if the IMU sequence is shorter than imu_seq_len 
        if imu_seq.shape[0] < self.imu_seq_len: 
            npads = self.imu_seq_len - imu_seq.shape[0]
            
            if imu_seq.shape[0] > 0: 
                # Repeat the first available IMU measure for padding 
                padding_imu = np.tile(imu_seq[0], (npads, 1))
                padding_trg = np.tile(targets[0], (npads, 1))
                padding_tms = np.tile(imu_tms[0], npads)
                
            else: 
                # If no IMU data at all, pad with the zeros
                padding_imu = np.zeros((npads, 6))
                padding_trg = np.zeros((npads, 6))
                padding_tms = np.zeros(npads)
                
            if self.imu_padding == "static":
                # Sets the lander to a static scenario in which it is not moving. 
                padding_imu[:, 3:6] = 0.0 
                padding_trg[:, 3:6] = 0.0
                padding_tms[:] = padding_tms[-1] + np.arange(-npads, 0, 1)*dt

            # Apply the padding to the IMU and position/velocity data
            imu_tms = np.hstack((padding_tms, imu_tms))
            imu_seq = np.vstack([padding_imu, imu_seq])
            targets = np.vstack([padding_trg, targets])
            
            
        # ====== Extract the rangemeter data 
        
        # Find the rangemeter data points and align the start time with the IMU data. 
        # We retrieve data with a margin of +- dt to prevent interpolation issues.
        range_mask = (self.rangemeter_full[:, 0] >= imu_tms[0] - dt) & \
                     (self.rangemeter_full[:, 0] <= imu_tms[-1] + dt)
                     
        range_data = self.rangemeter_full[range_mask].copy()
        
        # Interpolate the data at all strictly positive times 
        range_seq = np.interp(imu_tms[imu_tms > 0], range_data[:,0], range_data[:,1])
        range_seq = range_seq.reshape(-1, 1)
        
        # Add padding for the initial missing values.
        if range_seq.shape[0] < self.imu_seq_len: 
            npads = self.imu_seq_len - range_seq.shape[0]
            
            if self.imu_padding == "static": 
                # We require knowledge of the altitude at time 0 (altimeter starts at 0.1), 
                # thus we interpolate the first points to retrieve it.
                range_beg = UnivariateSpline(range_data[:,0], range_data[:,1], k=1)(0.0)     
                padding_range = np.tile(range_beg, (npads, 1))
            
            else: 
                padding_range = np.zeros((npads, 1))
                
            range_seq = np.vstack([padding_range, range_seq])    
                
                
        # ====== Extract Events 
        
        # Compute the events within the integration window for each sequence point
        events_tensor = []
        for k in range(traj_idx - self.imu_seq_len + 1, traj_idx+1): 
            
            # Check the side that should be used to parse the events
            side = "right" if flip else "left"
            
            # Compute the event tensor for this trajectory state
            if k >= 0 and self.events_pre_side == side: 
                events_tensor.append(self.events_tensor[k])
            
            else:
                ik = k - traj_idx + self.imu_seq_len - 1
                events_k = self.process_events(imu_tms[ik], side=side)
                events_tensor.append(events_k)
                            
        events_tensor = np.stack(events_tensor, axis=0)
        
        if flip: 
            
            # Copies of the arrays are made because PyTorch does not support 
            # negative strides
            
            # Flip the rangemeter data and trajectory times
            range_seq = range_seq[::-1].copy()
            imu_tms = imu_tms[::-1].copy()
            
            # Flip the IMU sequence and invert the angular velocities
            imu_seq = imu_seq[::-1].copy()
            imu_seq[:, 3:6] = -imu_seq[:, 3:6]
            
            # Flip the trajectory data and invert the velocities
            targets = targets[::-1].copy()
            targets[:, 3:6] = -targets[:, 3:6]
            
            # Flip the events (we still have the first that is referred to the last state)
            events_tensor = events_tensor[::-1].copy()
        
        return {
            'events_tensor': torch.from_numpy(events_tensor), 
            'imu_sequence': torch.from_numpy(imu_seq.astype(np.float32)), 
            'rangemeter_sequence': torch.from_numpy(range_seq.astype(np.float32)), 
            'ground_truth': torch.from_numpy(targets.astype(np.float32)),
            'times': torch.tensor(np.float32(imu_tms))
        }
        
    def preprocess_events(self, side: str="left"): 
        """Preprocess the event tensors at the trajectory states. 
        
        Parameters
        ----------
        side : str
            Direction towards which we are approaching the states, either "left" or "right". 
        """
        
        if self.events_full is None: 
            raise RuntimeError("Data not loaded. Call `load_sequence` first.")
        
        LOGGER.info(f"Pre-processing events tensors on {side} side.")
        events_tensor = []
        for k in range(self.seq_len): 
            events_tensor.append(
                self.process_events(self.timestamps_full[k], side=side)
            )
            
        # Update the event tensor and flags
        self.events_tensor = np.stack(events_tensor, axis=0)
        self.events_pre_side = side 
    
    def process_events(self, time: float, side: str="left") -> np.ndarray: 
        """Process events into a 4D tensor representation (EVFlownet-like).
        
        Parameters
        ----------
        time : float 
            Reference trajectory time, in seconds. 
        side : str, optional 
            Direction towards which we approach the time, either "left" or "right". 
            Defaults to "left.
        
        Returns 
        -------
        tensor : np.ndarray
            An array of shape (2, C, H, W) with the events polarity, channels, 
            height and width data.
        """
        
        # Compute the event window start and end time 
        if side == "left": 
            t_end = 1e6*time
            t_beg = t_end - self.event_integration_window     
        else: 
            t_beg = 1e6*time
            t_end = t_beg + self.event_integration_window
        
        # Filter the events within the integration window and make a copy
        events_mask = (self.events_full['t'] >= t_beg) & (self.events_full['t'] <= t_end)
        events = self.events_full[events_mask].copy()
        
        # Check if the events are in a structured array format
        if events.dtype == [('x', '<i2'), ('y', '<i2'), ('p', '?'), ('t', '<i8')]:
            # Convert a structured array to a regular ndarray with integer polarity 
            events_array = np.column_stack([
                events['x'], events['y'], events['p'].astype(int), events['t']
            ])
            
        else: 
            # Events are already in a regular array format. We handle potential 1D arrays
            # by converting them to 2D representations 
            if events.ndim == 1: 
                events_array = np.array([[e[0], e[1], int(e[2]), e[3]] for e in events])
        
        # Retrieve event tensor settings
        H, W, T = self.H, self.W, self.T 
        
        # Ensure `events` is a 2D array 
        if events_array.shape[0] == 0: 
            # LOGGER.warning("No events found in the specified time window.")
            # Return an empty tensor with the expected shape 
            return np.zeros((2, T, H, W), dtype=np.float32)
        
        # Convert the filtered events into a 4D tensor 
        tensor = self.processor.events_to_tensor(
            events_array, 1e6*time, H, W, T, method=self.event_encoder_method, 
            time_window=self.event_integration_window, side=side, clamp=self.event_clamp
        )
        
        # Re-arrange the tensor dimensions to PyTorch format (Channels, Time, Height, Width)
        tensor = np.transpose(tensor, (3, 0, 1, 2))
        return tensor.astype(np.float32)