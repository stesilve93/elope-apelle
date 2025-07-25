
import numpy as np 
import torch

from abc import ABC
from pathlib import Path 

from scipy.interpolate import PchipInterpolator

from elope.utils import LOGGER

from .events import EventProcessor

class SequenceLoader(ABC): 
    """Abstract SequenceLoader for Elope's trajectory sequences."""
    
    def __init__(
        self, 
        datapath: str | Path, 
        event_integration_window: float = 1e5, 
        event_encoder_method: str = "last_timestamp", 
        event_clamp: int = -1, 
        event_H: int = 200, 
        event_W: int = 200, 
        event_T: int = 1,
        sequence_len: int = 5, 
        sequence_pad: str = "static",
        **kwargs
    ): 
        
        self.datapath = Path(datapath)
        self.processor = EventProcessor() 
        
        # Vectors in which to store the sequence data
        self.full_events = None 
        self.full_states = None 
        self.full_times = None 
        self.full_imu = None 
        self.full_rangemeter = None 
        
        # Vectors in which to store the interpreted sequence data 
        self.seq_events_left  = None
        self.seq_events_right = None
        
        self.seq_states = None 
        self.seq_times  = None 
        self.seq_imu    = None 
        self.seq_ranges = None
        
        # Length of the sequence 
        self.seq_len = 0
        
        # Sequence timestep 
        self.seq_dt = 0.0
        
        # Store the number of states to provide for a given output
        self.out_len = int(sequence_len)
        
        # Currently support only static paddings for the IMU 
        assert sequence_pad == "static"
        self.padding = sequence_pad
        
        # Store event tensor channels 
        self.H = int(event_H)
        self.W = int(event_W)
        self.T = int(event_T)
        
        # Check that the event inputs make sense
        assert event_encoder_method in (
            "first_timestamp", "last_timestamp", "timestamp", "count", "hybrid"
        )
        
        self.event_integration_window = float(event_integration_window)
        self.event_encoder_method = event_encoder_method
        self.event_clamp = float(event_clamp)
        
        # Check the encoder is coherent with the dimensions 
        if self.event_encoder_method in ("first_timestamp", "last_timestamp") and self.T != 1: 
            raise ValueError(
                f"Event encoder {self.event_encoder_method} supports only 1 event channel."
            )
                
        elif self.event_encoder_method == "timestamp" and self.T != 2: 
            raise ValueError("Event encoder `timestamp` supports only 2 event channels.")
        
        elif self.event_encoder_method == "hybrid" and self.T != 3: 
            raise ValueError("Event encoder `hybrid` supports only 3 event channels.")
                
    @property 
    def timestep(self) -> float: 
        return self.seq_dt
    
    def load_sequence(self, sequence_id: str): 
        """Load a single sequence file from the datapath directory. 
        
        Parameters
        ----------
        sequence_id : str 
            Sequence ID to load, e.g., '0000'.
        """ 
        
        fn = self.datapath / (f"{sequence_id}.npz")
        if not fn.exists(): 
            raise FileNotFoundError(f"Sequence file not found: {fn}.")
        
        LOGGER.info(f"Loading sequence from: \033[33m{fn}\033[0m:")
        sequence = np.load(fn)
        
        # Extract data from the loaded sequence 
        self.full_events = sequence["events"]
        self.full_times = sequence["timestamps"]
        self.full_rangemeter = sequence["range_meter"]

        trajectory = sequence["traj"]
        self.full_states = trajectory[:, 0:6]
        self.full_imu = trajectory[:, 6:12]
        
        # Reset all the states
        self.seq_events_left = None 
        self.seq_events_right = None 
        
        self.seq_states = None 
        self.seq_times = None 
        self.seq_imu = None 
        self.seq_ranges = None 
        
        print(f"\t- Events: {len(self.full_events)} events")
        print(f"\t- Timestamps: {len(self.full_times)} steps")
        print(f"\t- States: {self.full_states.shape} values")
        print(f"\t- IMU: {self.full_imu.shape} measurements")
        print(f"\t- Rangemeter: {len(self.full_rangemeter)} measurements")
        
    def get_data_at_index(self, idx: int, flip: bool = False) -> dict: 
        """Extract preprocessed data for a given trajectory time.
        
        Parameters
        ----------
        idx : float 
            Desired trajectory timestamp index. 
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
        
        # Ensure the trajectory data has been loaded
        if self.seq_times is None: 
            raise RuntimeError("Data not loaded. Call `load_sequence` first.")
        
        # Retrieve the starting index for the data    
        idx_beg = max(0, idx - self.out_len + 1)
        
        # Extract the sequence data
        out_times = self.seq_times[idx_beg:idx+1]
        out_imu = self.seq_imu[idx_beg:idx+1]
        out_states = self.seq_states[idx_beg:idx+1]
        out_ranges = self.seq_ranges[idx_beg:idx+1].reshape(-1, 1)
        
        if flip:     
            if self.seq_events_right is None:
                LOGGER.warning("Right-side events not loaded. Processing.")
                self.preprocess_events(side="right")
                
            out_events = self.seq_events_right[idx_beg:idx+1]
            
        else:
            if self.seq_events_left is None: 
                LOGGER.warning("Left-side events not loaded. Processing.")        
                self.preprocess_events(side="left")
            
            out_events = self.seq_events_left[idx_beg:idx+1]
            
        # Add padding to all data if the initial states are missing.
        npads = self.out_len - len(out_times)
        if npads > 0: 
            
            pad_imu    = np.tile(out_imu[0], (npads, 1))
            pad_states = np.tile(out_states[0], (npads, 1))
            pad_ranges = np.tile(out_ranges[0], (npads, 1))
            pad_events = np.tile(out_events[0], (npads, 1, 1, 1, 1))
            
            pad_times = out_times[0] + np.arange(-npads, 0, 1)*self.timestep
                
            if self.padding == "static": 
                # Sets the lander to static scenario in which it is not moving 
                pad_imu[:, 3:6] = 0.0 
                pad_states[:, 3:6] = 0.0 
                
            # Apply the padding to the IMU and position/velocity data
            out_times  = np.hstack((pad_times, out_times))
            out_events = np.vstack([pad_events, out_events])
            out_states = np.vstack([pad_states, out_states])
            out_ranges = np.vstack([pad_ranges, out_ranges])
            out_imu    = np.vstack([pad_imu, out_imu])
            
        if flip: 
            
            # Copies of the arrays are made because PyTorch does not support 
            # negative states 
            
            # Flip the rangemeter data and trajectory times 
            out_ranges = out_ranges[::-1].copy() 
            out_times = out_times[::-1].copy() 
            
            # Flip the IMU sequence and invert the velocities
            out_imu = out_imu[::-1].copy() 
            out_imu[:, 3:6] = -out_imu[:, 3:6]
            
            # Flip the trajectory data and invert the velocities 
            out_states = out_states[::-1].copy() 
            out_states[:, 3:6] = -out_states[:, 3:6]
            
            # Flip the events (we still have the first that is referred to the last state)
            out_events = out_events[::-1].copy() 
            
        return {
            'events': torch.from_numpy(out_events), 
            'imu': torch.from_numpy(out_imu.astype(np.float32)), 
            'rangemeter': torch.from_numpy(out_ranges.astype(np.float32)), 
            'states': torch.from_numpy(out_states.astype(np.float32)),
            'times': torch.tensor(np.float32(out_times))
        }

    def preprocess_events(self, side: str ="left"): 
        """Preprocess the event tensors at the trajectory states.
        
        Parameters
        ----------
        side : str 
            Direction towards which we are approaching the times, either "left" or "right".
        """
        
        # Ensure a trajectory has been loaded
        if self.seq_times is None:
            raise RuntimeError("Data not loaded. Call `load_sequence` first.")
        
        LOGGER.info(f"Pre-processing events tensors on {side} side.")
        
        # Events are processed near the times at which we have IMU data
        events_tensor = []
        for k in range(self.seq_len): 
            events_tensor.append(self.process_events(self.seq_times[k], side=side))
            
        events_tensor = np.stack(events_tensor, axis=0)
        if side == "left": 
            self.seq_events_left  = events_tensor
        else: 
            self.seq_events_right = events_tensor
            
        # Update the event tensor and flags 
        self.seq_events = np.stack(events_tensor, axis=0)
        
    def process_events(self, time: float, side: str = "left") -> np.ndarray: 
        """Process events into a 4D tensor representation.
        
        Parameters
        ----------
        time : float 
            Reference trajectory time, in seconds. 
        side : str, optional 
            Direction towards which we approach the time, either "left" or "right". 
            Defaults to "left".
        
        Returns 
        -------
        events_tensor : np.ndarray
            An array of shape (2, C, H, W) with the events polarity, channels, height and 
            width data.
        """
        
        # Compute the event window start and end time 
        if side == "left": 
            t_end = 1e6*time 
            t_beg = t_end - self.event_integration_window
        else: 
            t_beg = 1e6*time 
            t_end = t_beg + self.event_integration_window
            
        # Filter the events within the integration window and make a copy 
        events_mask = (self.full_events['t'] >= t_beg) & (self.full_events['t'] <= t_end)
        events = self.full_events[events_mask].copy() 
        
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
            # Return an empty tensor with the expected shape 
            return np.zeros((2, T, H, W), dtype=np.float32)
        
        # Convert the filtered events into a 4D tensor (T, H, W, 2)
        tensor = self.processor.events_to_tensor(
            events_array, 1e6*time, H, W, T, method=self.event_encoder_method, 
            time_window=self.event_integration_window, side=side, clamp=self.event_clamp
        )
        
        # Re-arrange the tensor dimensions
        tensor = np.transpose(tensor, (3, 0, 1, 2)) # (2, T, H, W)
        return tensor.astype(np.float32)
    
    def __len__(self) -> int: 
        return self.seq_len
        

class VariableSequenceLoader(SequenceLoader): 

    def load_sequence(self, sequence_id: str, events_side: str = "both"): 
        
        # Run the initial method 
        super().load_sequence(sequence_id)    
        
        # In this case the trajectory times are the ones aligned with the IMU data 
        self.seq_times = self.full_times.copy()
        self.seq_states = self.full_states.copy() 
        self.seq_imu = self.full_imu.copy()
        
        # Interpolate the range to retrieve it at the trajectory times
        self.seq_ranges = PchipInterpolator(
            self.full_rangemeter[:, 0], self.full_rangemeter[:, 1]
        )(self.seq_times)
        
        # Update the sequence length.
        self.seq_len = len(self.seq_times)
        
        # Update the sequence time step
        self.seq_dt = self.seq_times[1] - self.seq_times[0]
        
        # Pre-load all the events tensors.
        if events_side == "left": 
            self.preprocess_events(side="left")
        elif events_side == "right": 
            self.preprocess_events(side="right")
        elif events_side == "both": 
            self.preprocess_events(side="left")
            self.preprocess_events(side="right")
        else: 
            raise ValueError(f"Unsupported value for `events_side`: {events_side}.")


class FixedSequenceLoader(SequenceLoader): 
    
    def __init__(self, *args, time_step: float = 0.1, **kwargs): 
        super().__init__(*args, **kwargs)
        
        assert time_step > 0 
        
        # Store a fixed sampling time for the sequence
        self.seq_dt = time_step
        
    def load_sequence(self, sequence_id: str, events_side: str = "both"): 
        
        # Run the initial method 
        super().load_sequence(sequence_id)
        
        # Lets create the time vector at which we interpolate the data
        self.seq_times = np.arange(0.0, self.full_rangemeter[-1, 0], self.seq_dt)
        
        self.seq_ranges = PchipInterpolator(
            self.full_rangemeter[:, 0], self.full_rangemeter[:, 1]
        )(self.seq_times)
        
        # Interpolate the states data at the rangemeter times
        self.seq_states = np.stack([
            PchipInterpolator(self.full_times, self.full_states[:, i])(self.seq_times) 
            for i in range(6)
        ]).T.copy()
        
        # Interpolate the IMU data at the rangemeter times
        self.seq_imu = np.stack([
            PchipInterpolator(self.full_times, self.full_imu[:, i])(self.seq_times) 
            for i in range(6)
        ]).T.copy()
        
        # Update the sequence length.
        self.seq_len = len(self.seq_times)
        
        # Update the sequence time step
        self.seq_dt = self.seq_times[1] - self.seq_times[0]
        
        # Pre-load all the events tensors.
        if events_side == "left": 
            self.preprocess_events(side="left")
        elif events_side == "right": 
            self.preprocess_events(side="right")
        elif events_side == "both": 
            self.preprocess_events(side="left")
            self.preprocess_events(side="right")
        else: 
            raise ValueError(f"Unsupported value for `events_side`: {events_side}.")