
import numpy as np 
import torch

from numba import njit

class EventProcessor:
    """Process event camera data for optical flow estimation"""
    
    def __init__(self, width: int=200, height: int=200):
        """
        Initialize the EventProcessor with given width and height.
        
        Parameters
        ----------
        width : int, optional
            The width of the event frame, default is 200.
        height : int, optional
            The height of the event frame, default is 200.
        """
        # Set the width of the event frame
        self.width = width
        # Set the height of the event frame
        self.height = height
        
    def events_to_frames(self, events, end_time=0, time_window=1e5, method='count'):
        """Convert event stream to frame-like representations"""
        timestamps = []
        # Get time range
        t_start = end_time - time_window
        t_end = end_time
   
        assert time_window >= 0
        assert t_start >= 0
        
        # slice event dataframe
        ev_slice = events.loc[events['t'].between(t_start,t_end)]
        
        if len(ev_slice) == 0:
            return Exception("No events in the current time window")
            
        if method == 'count':
            frame = self._events_to_count_frame(ev_slice)
        elif method == 'time_surface':
            frame = self._events_to_time_surface(ev_slice, t_end)
        else:
            raise ValueError("Method must be either 'count' or 'time_surface'")
            
        timestamps.append(t_start / 1e6)  # Convert to seconds
        timestamps.append(t_end / 1e6)  # Convert back to seconds
            
        return frame, np.array(timestamps)
    
    def _events_to_event_frame(self, ev_slice):
        """Generate event frame representation from event stream.

        Parameters
        ----------
        ev_slice : pandas.DataFrame
            Event stream data with columns ['x', 'y', 'p', 't']
        start_t : float
            Start time of the time window
        acc_t : float
            Accumulation time of the time window

        Returns
        -------
        ev_frame : numpy.ndarray
            Event frame representation with shape (height, width, 3)
        """
        
        # TODO: This function is not used in the current implementation, but it can be 
        # used to visualize events in a frame-like representation.
        
        # Create empty frame
        ev_frame = np.zeros((self.height, self.width), dtype=np.float32)
        # Determine latest events (previous events of the slice will be ignored in this representation)
        max_entries = ev_slice.loc[ev_slice.groupby(['x', 'y'])['t'].idxmax()]

        # Split events by polarity
        pos_ev = max_entries.loc[(max_entries['p'] == True), ['x', 'y']]
        neg_ev = max_entries.loc[(max_entries['p'] == False), ['x', 'y']]

        # Create empty event frame
        ev_frame = np.zeros([200, 200, 3], dtype=np.uint8)

        # To be easy on the eye, we will use a blueish color for negative polarity and 
        # white for positive polarity
        ev_frame[pos_ev['x'], pos_ev['y']] = [255, 255, 255]
        ev_frame[neg_ev['x'], neg_ev['y']] = [80, 137, 204]

        return ev_frame
    
    def _events_to_count_frame(self, events):
        """
        Create count-based frame from events
        
        Parameters
        ----------
        events : pandas.DataFrame
            Event stream data with columns ['x', 'y', 'p', 't']
        
        Returns
        -------
        frame : numpy.ndarray
            Count-based frame representation with shape (height, width)
        """
        # Create empty frame
        frame = np.zeros((self.height, self.width), dtype=np.float32)
        
        if len(events) > 0:
            # Get event coordinates and polarities
            x_coords = events['x'].values.astype(int)
            y_coords = events['y'].values.astype(int)
            polarities = events['p'].values.astype(float)
            
            # Clip coordinates to image bounds
            x_coords = np.clip(x_coords, 0, self.width - 1)
            y_coords = np.clip(y_coords, 0, self.height - 1)
            
            # Accumulate events
            for x, y, p in zip(x_coords, y_coords, polarities):
                frame[y, x] += 1 if p else -1
                
        return frame
    
    def _events_to_time_surface(self, events, current_time: float):
        """
        Create time surface representation from events
        
        The time surface is a 2D representation of the event stream where each pixel
        value represents the time since the last event at that location occurred.
        
        Parameters
        ----------
        events : pandas.DataFrame
            Event stream data with columns ['x', 'y', 'p', 't']
        current_time : float
            Current time of the event stream
        
        Returns
        -------
        frame : numpy.ndarray
            Time surface representation with shape (height, width)
        """
        frame = np.zeros((self.height, self.width), dtype=np.float32)
        
        if len(events) > 0:
            x_coords = events['x'].values.astype(int)
            y_coords = events['y'].values.astype(int)
            timestamps = events['t'].values
            
            # Clip coordinates to image bounds
            x_coords = np.clip(x_coords, 0, self.width - 1)
            y_coords = np.clip(y_coords, 0, self.height - 1)
            
            # Time decay
            time_diff = current_time - timestamps
            decay = np.exp(-time_diff / 10000)  # Decay constant
            
            # For each event, update the time surface by taking the maximum of
            # the current decay value and the existing value at that location
            for x, y, decay_val in zip(x_coords, y_coords, decay):
                frame[y, x] = max(frame[y, x], decay_val)
                
        return frame
    
    @staticmethod 
    def events_to_tensor(
        events: np.ndarray, 
        time: float, 
        H: int=200, 
        W: int=200, 
        T: int=10, 
        method: str="count", 
        time_window: float=1e5, 
        side: str="left",
        clamp: int=10,
    ) -> np.ndarray: 
        # DOCME: Time must be expressed in microseconds!!
        
        # Ensure the size of the time-window is greater than 0
        assert time_window > 0
    
        if method == "count": 
            return _events_to_tensor_count(events, time, H, W, T, time_window, side, clamp)
        
        elif method == "first_timestamp": 
            return _events_to_tensor_timestamp(events, time, H, W, time_window, side)[0]
        
        elif method == "last_timestamp": 
            return _events_to_tensor_timestamp(events, time, H, W, time_window, side)[1]
        
        elif method == "timestamp": 
            return _events_to_tensor_timestamp(events, time, H, W, time_window, side)
        
        elif method == "hybrid":             
            
            # Combine the event counts with the freshness values
            ev_1 = _events_to_tensor_count(events, time, H, W, 1, time_window, side, clamp)
            ev_2 = _events_to_tensor_timestamp(events, time, H, W, time_window, side)
            
            return torch.vstack((ev_1, ev_2))
        
        else: 
            raise ValueError(f"`{method}` is not a valid event encoding method.")

    @staticmethod
    def normalize_tensor(
        tensor: torch.Tensor, method: str='standard', min_val: float=None, max_val: float=None
    ) -> torch.Tensor:
        """Normalize event tensor"""
            
        if method == 'standard':
            # Z-score normalization
            mean = tensor.mean()
            std = tensor.std()
            if std > 0:
                tensor = (tensor - mean) / std
                
        elif method == 'minmax':
            # Min-max normalization
            min_val = min_val if min_val is not None else tensor.min() 
            max_val = max_val if max_val is not None else tensor.max() 

            if max_val > min_val:
                tensor = (tensor - min_val) / (max_val - min_val)
                
        else: 
            raise ValueError(f"`{method}` is an unsupported normalization method.")
        
        return tensor

def _events_to_tensor_count(
    events: np.ndarray, time: float, H: int, W: int, T: int, time_window: float, side: str, 
    clamp: int
) -> np.ndarray: 
    
    # events is an array of size (n, 4) with (x, y, p, t)
    
    # Create the event tensor
    tensor = np.zeros((T, H, W, 2), dtype=np.float32)
    if len(events) == 0: 
        return tensor
    
    # Compute the size of a time-bin. This way the size of each bin is fixed for a given 
    # integration window, regardless of how many events are incoming.
    dt = time_window/T 
    
    if side == "left": 
        t_min = time - time_window
        t_max = time
    
    else: 
        t_min = time 
        t_max = time + time_window
    
    for i in range(T): 
        
        if side == "left": 
            # Compute the bin edges
            t_beg = t_min + i*dt
            t_end = t_beg + dt
            
            # Filter the events within the bin
            mask_bin  = events[:, 3] >= t_beg 
            mask_bin &= events[:, 3] < t_end if i < T - 1 else events[:, 3] <= t_end

        else: 
            # Compute the bin edges
            t_beg = t_max - i*dt
            t_end = t_beg - dt
            
            # Filter the events within the bin
            mask_bin  = events[:, 3] <= t_beg 
            mask_bin &= events[:, 3] > t_end if i < T - 1 else events[:, 3] >= t_end
             
        # Retrieve the events in the current time bin
        events_bin = events[mask_bin]
        
        if len(events_bin) == 0: 
            # There are no events to be added in this time-bin
            continue 
        
        # Separate the events by polarity (true and false)
        events_pos = events_bin[events_bin[:, 2] == 1]
        events_neg = events_bin[events_bin[:, 2] == 0]
        
        # Accumulate the positive events 
        if len(events_pos) > 0: 
            # Retrieve the events coordinates
            x_pos = events_pos[:, 0].astype(np.int64)
            y_pos = events_pos[:, 1].astype(np.int64)

            # Ensure the coordinates are within the bounds 
            mask_pos = (x_pos >= 0) & (x_pos < W) & (y_pos >= 0) & (y_pos < H)
            if np.any(mask_pos): 
                # Add one event at the position identified by these coordinates
                np.add.at(tensor[i, :, :, 0], (y_pos[mask_pos], x_pos[mask_pos]), 1)
        
        # Accumulate the negative events 
        if len(events_neg) > 0: 
            # Retrieve the events coordinates
            x_neg = events_neg[:, 0].astype(np.int64)
            y_neg = events_neg[:, 1].astype(np.int64)
            
            # Ensure the coordinates are within the bounds 
            mask_neg = (x_neg >= 0) & (x_neg < W) & (y_neg >= 0) & (y_neg < H)
            if np.any(mask_neg): 
                # Add one event at the position identified by these coordinates
                np.add.at(tensor[i, :, :, 1], (y_neg[mask_neg], x_neg[mask_neg]), 1)
    
    if side == "right": 
        # Invert the polarity of the events if we are looking backwards 
        tensor = tensor[:, :, :, ::-1]
    
    if clamp > 0:
        # Clamp the maximum values within each bin.
        tensor = np.clip(tensor, 0, clamp)
         
    return tensor     
    
@njit(cache=True)
def _events_to_tensor_timestamp(
    events: np.ndarray, time: float, H: int, W: int, time_window: float, side: str
) -> np.ndarray:
    
    # Initialize the latest timestamp tensor, with positive events on channel 0 and 
    # negative events on channel 1
    if side == "left": 
        t_beg = time - time_window
        first_timestamp = np.ones((H, W, 2))
        last_timestamp  = np.zeros((H, W, 2))
        
    else: 
        t_beg = time  
        first_timestamp = np.zeros((H, W, 2))
        last_timestamp  = np.ones((H, W, 2))
    
    # Filter positive events
    mask_pos = events[:, 2] == 1
    
    # Convert positive coordinates and normalize times
    pos_x = np.clip(events[mask_pos, 0].astype(np.int64), 0, W-1)
    pos_y = np.clip(events[mask_pos, 1].astype(np.int64), 0, H-1)
    pos_t = (events[mask_pos, 3] - t_beg)/time_window
    
    # Parse positive events
    for k in range(len(pos_x)): 
        # Retrieve event position and time
        x, y, t = pos_x[k], pos_y[k], pos_t[k]
        
        # Update the latest timestamp for this event
        if side == "left":
            
            if last_timestamp[y, x, 0] < t: 
                last_timestamp[y, x, 0] = t
                
            if first_timestamp[y, x, 0] > t: 
                first_timestamp[y, x, 0] = t
             
        elif side == "right": 
            
            if last_timestamp[y, x, 0] > t: 
                last_timestamp[y, x, 0] = t

            if first_timestamp[y, x, 0] < t: 
                first_timestamp[y, x, 0] = t
    
    # Filter negative events
    mask_neg = events[:, 2] == 0
    
    # Convert negative coordinates and normalize times
    neg_x = np.clip(events[mask_neg, 0].astype(np.int64), 0, W-1)
    neg_y = np.clip(events[mask_neg, 1].astype(np.int64), 0, H-1)
    neg_t = (events[mask_neg, 3] - t_beg)/time_window
    
    # Parse negative events 
    for k in range(len(neg_x)): 
        # Retrieve event position and time
        x, y, t = neg_x[k], neg_y[k], neg_t[k]
        
        # Update the latest timestamp for this event
        if side == "left":
            
            if last_timestamp[y, x, 1] < t: 
                last_timestamp[y, x, 1] = t
                
            if first_timestamp[y, x, 1] > t: 
                first_timestamp[y, x, 1] = t
             
        elif side == "right":
            
            if last_timestamp[y, x, 1] > t: 
                last_timestamp[y, x, 1] = t

            if first_timestamp[y, x, 1] < t: 
                first_timestamp[y, x, 1] = t
         
    if side == "right":
        
        # Invert the values such that the most recent event is the one closest to the point
        last_timestamp  = 1 - last_timestamp
        first_timestamp = 1 - first_timestamp

        # Invert the polarities 
        last_timestamp  = last_timestamp[:, :, ::-1]
        first_timestamp = first_timestamp[:, :, ::-1]
    
    # Stack the two channels 
    return np.vstack((first_timestamp, last_timestamp)).astype(np.float32)
