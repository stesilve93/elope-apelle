import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class EventProcessor:
    """Process event camera data for optical flow estimation"""
    
    def __init__(self, width=200, height=200):
        self.width = width
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
        # FIXME: This function is not used in the current implementation, but it can be used to visualize events in a frame-like representation.
        # Create empty frame
        ev_frame = np.zeros((self.height, self.width), dtype=np.float32)
        # Determine latest events (previous events of the slice will be ignored in this representation)
        max_entries = ev_slice.loc[ev_slice.groupby(['x', 'y'])['t'].idxmax()]

        # Split events by polarity
        pos_ev = max_entries.loc[(max_entries['p'] == True), ['x', 'y']]
        neg_ev = max_entries.loc[(max_entries['p'] == False), ['x', 'y']]

        # Create empty event frame
        ev_frame = np.zeros([200, 200, 3], dtype=np.uint8)

        # To be easy on the eye, we will use a blueish color for negative polarity and white for positive polarity
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
    
    def _events_to_time_surface(self, events, current_time):
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
    def events_to_tensor(events: np.ndarray, 
                        H: int = 200, 
                        W: int = 200, 
                        T: int = 10,
                        method: str = 'count') -> np.ndarray:
        """
        Convert event stream to 4D tensor representation
        
        Args:
            events: Array with columns [x, y, p, t]
            H, W: Spatial dimensions
            T: Number of time bins
            method: 'count' or 'time_surface'
            
        Returns:
            4D tensor of shape (T, H, W, 2) - [time, height, width, polarity]
            The tensor values represent the count of events at each location and time bin, separated by polarity.
        """
        if len(events) == 0:
            return np.zeros((T, H, W, 2), dtype=np.float32)
            
        # Create time bins
        t_min, t_max = events[:, 3].min(), events[:, 3].max()
        t_bins = np.linspace(t_min, t_max, T + 1)
        
        tensor = np.zeros((T, H, W, 2), dtype=np.float32)
        
        for i in range(T):
            # Find events in this time bin
            mask = (events[:, 3] >= t_bins[i]) & (events[:, 3] < t_bins[i + 1])
            if i == T - 1:  # Include last timestamp
                mask = (events[:, 3] >= t_bins[i]) & (events[:, 3] <= t_bins[i + 1])
            
            bin_events = events[mask]
            
            if len(bin_events) > 0:
                # Separate by polarity
                pos_events = bin_events[bin_events[:, 2] == 1]  # True polarity
                neg_events = bin_events[bin_events[:, 2] == 0]  # False polarity
                
                # Accumulate positive events
                if len(pos_events) > 0:
                    x_pos, y_pos = pos_events[:, 0].astype(int), pos_events[:, 1].astype(int)
                    # Ensure coordinates are within bounds
                    valid_pos = (x_pos >= 0) & (x_pos < W) & (y_pos >= 0) & (y_pos < H)
                    if np.any(valid_pos):
                        np.add.at(tensor[i, :, :, 0], (y_pos[valid_pos], x_pos[valid_pos]), 1)
                
                # Accumulate negative events
                if len(neg_events) > 0:
                    x_neg, y_neg = neg_events[:, 0].astype(int), neg_events[:, 1].astype(int)
                    valid_neg = (x_neg >= 0) & (x_neg < W) & (y_neg >= 0) & (y_neg < H)
                    if np.any(valid_neg):
                        np.add.at(tensor[i, :, :, 1], (y_neg[valid_neg], x_neg[valid_neg]), 1)
        
        return tensor
    
    @staticmethod
    def normalize_tensor(tensor: np.ndarray, method: str = 'standard') -> np.ndarray:
        """Normalize event tensor"""
        if method == 'standard':
            # Z-score normalization
            mean = tensor.mean()
            std = tensor.std()
            if std > 0:
                tensor = (tensor - mean) / std
        elif method == 'minmax':
            # Min-max normalization
            min_val, max_val = tensor.min(), tensor.max()
            if max_val > min_val:
                tensor = (tensor - min_val) / (max_val - min_val)
        
        return tensor.astype(np.float32)
