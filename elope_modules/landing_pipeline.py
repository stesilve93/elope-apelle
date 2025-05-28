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

from evflow.EVFlowNet import EVFlowNet
from elope_modules.event_proc import EventProcessor
from elope_modules.of_estimator import OpticalFlowEstimator
from elope_modules.traj_estimator import TrajectoryEstimator

class LunarDescentPipeline:
    """Complete pipeline for lunar descent velocity estimation"""
    
    def __init__(self, use_gpu=False, acc_time=1e5):
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.event_processor = EventProcessor()
        self.flow_estimator = OpticalFlowEstimator(device=self.device)
        self.trajectory_estimator = TrajectoryEstimator()
        self.acc_time = acc_time
        print(f"Accumulation time for event frames: {self.acc_time / 1e6} seconds")
        print("Initializing pipeline components...")
        
        # Initialize EVFlowNet with random weights (in practice, you'd load pre-trained weights)
        print("Initializing EVFlowNet...")
        self._initialize_evflownet()
        
    def _initialize_evflownet(self):
        """Initialize EVFlowNet weights (placeholder for pre-trained weights)"""
        # In practice, you would load pre-trained weights here:
        # self.flow_estimator.evflownet.load_state_dict(torch.load('evflownet_weights.pth'))
        
        print("EVFlowNet initialized with random weights")
        print("Note: In practice, you should load pre-trained weights for better performance")
        
    def load_pretrained_weights(self, weights_path):
        """Load pre-trained EVFlowNet weights"""
        try:
            self.flow_estimator.evflownet.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"Loaded pre-trained weights from {weights_path}")
        except Exception as e:
            print(f"Failed to load weights: {str(e)}")
            print("Using randomly initialized weights")
    
    ## TODO: review the code to include end-to-end emmnet processing and visualization
    def process_sequence(self, events, timestamps, trajectory, rangemeter):
        """Process complete sequence and estimate velocities
        
        Parameters
        ----------
        events : pandas.DataFrame
            Event data with columns ['x', 'y', 'p', 't']
        timestamps : numpy.ndarray
            Timestamps of the events
        trajectory : numpy.ndarray
            Trajectory data with columns [x, y, z, r, p, y, vx, vy, vz]
        rangemeter : numpy.ndarray
            Rangemeter data with columns [range, timestamp]
        
        Returns
        -------
        results : dict
            Dictionary containing results of the processing:
                - 'frames': list of frames
                - 'frame_timestamps': list of timestamps for the frames
                - 'flows': list of optical flow fields
                - 'flow_timestamps': list of timestamps for the flow fields
                - 'depth_maps': list of depth maps
                - 'estimated_velocities': list of estimated velocities
                - 'estimated_timestamps': list of timestamps for the estimated velocities
        """
        print("Processing event stream...")
        
        # Convert events to DataFrame if needed
        if isinstance(events, np.ndarray):
            ev_data = pd.DataFrame(events, columns=['x', 'y', 'p', 't'])
        else:
            ev_data = events
        
        # Estimate optical flow using EVFlowNet
        flows = []
        flow_timestamps = []
        
        print("Estimating optical flow using EVFlowNet...")
        frames = []
        frames_timestamps = []

        # Get time range of the simulation
        t_start = events['t'].min()
        t_end = events['t'].max()
        t_range = np.arange(t_start, t_end, self.acc_time)

        for i in range(len(t_range) - 1):
            if i == 0:
                continue  # Skip the first iteration to avoid index error
            tkm1 = t_range[i]
            tk = t_range[i + 1]
            # Convert events to frames
            frame, frame_timestamps = self.event_processor.events_to_frames(ev_data, tkm1, time_window=self.acc_time, method='time_surface')
            frames.append(frame)
            frames_timestamps.append(frame_timestamps[0])  # Use the first timestamp for the frame
            frame, frame_timestamps = self.event_processor.events_to_frames(ev_data, tk, time_window=self.acc_time)
            frames.append(frame)
            frames_timestamps.append(frame_timestamps[0])  # Use the first timestamp for the frame
            

            print(f"Processed frames {i} and {i+1} for timestamps {tkm1} and {tk}")
            try:
                raise Exception("Simulating EVFlowNet failure")  # Simulate failure for testing
                # Use EVFlowNet for optical flow estimation
                flow = self.flow_estimator.estimate_flow_evflownet(frames[i], frames[i+1])
                flows.append(flow)
                flow_timestamps.append((frame_timestamps[i] + frame_timestamps[i+1]) / 2)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i+1}/{len(frames)-1} flow pairs")
                    
            except Exception as e:
                print(f"EVFlowNet failed for frame {i}, falling back to Farneback: {str(e)}")
                # Fallback to traditional method
                flow = self.flow_estimator.estimate_flow_farneback(frames[0], frames[1])
                flows.append(flow)
                flow_timestamps.append((frame_timestamps[0] + frame_timestamps[1]) / 2)
            
        flows = np.array(flows)
        flow_timestamps = np.array(flow_timestamps)
        
        print(f"Computed {len(flows)} optical flow fields")
        
        # Estimate depth maps
        depth_maps = self.trajectory_estimator.depth_from_flow_and_range(
            flows, rangemeter, flow_timestamps)
        
        # Estimate velocities
        estimated_velocities = []
        estimated_timestamps = []
        
        for i in range(len(flows)):
            if i == 0:
                dt = flow_timestamps[1] - flow_timestamps[0] if len(flow_timestamps) > 1 else 0.05
            else:
                dt = flow_timestamps[i] - flow_timestamps[i-1]
                
            # Get angular velocity from trajectory (interpolated)
            traj_interp = interpolate.interp1d(timestamps, trajectory[:, 9:12].T, 
                                             bounds_error=False, fill_value=0)
            angular_vel = traj_interp(flow_timestamps[i])
            
            velocity = self.trajectory_estimator.estimate_velocity_from_flow(
                flows[i], depth_maps[i], dt, angular_vel)
            
            estimated_velocities.append(velocity)
            estimated_timestamps.append(flow_timestamps[i])
            
        estimated_velocities = np.array(estimated_velocities)
        estimated_timestamps = np.array(estimated_timestamps)
        
        return {
            'frames': frames,
            'frame_timestamps': frames_timestamps,
            'flows': flows,
            'flow_timestamps': flow_timestamps,
            'depth_maps': depth_maps,
            'estimated_velocities': estimated_velocities,
            'estimated_timestamps': estimated_timestamps
        }
    
    def visualize_results(self, results, trajectory, timestamps):
        """Visualize the results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot sample frames
        if len(results['frames']) > 0:
            axes[0, 0].imshow(results['frames'][0], cmap='gray')
            axes[0, 0].set_title('Sample Event Frame')
            axes[0, 0].axis('off')
        
        # Plot optical flow
        if len(results['flows']) > 0:
            flow_vis = np.linalg.norm(results['flows'][0], axis=2)
            im = axes[0, 1].imshow(flow_vis, cmap='hot')
            axes[0, 1].set_title('Optical Flow Magnitude')
            plt.colorbar(im, ax=axes[0, 1])
        
        # Plot depth map
        if len(results['depth_maps']) > 0:
            im = axes[0, 2].imshow(results['depth_maps'][0], cmap='plasma')
            axes[0, 2].set_title('Estimated Depth Map')
            plt.colorbar(im, ax=axes[0, 2])
        
        # Plot velocity comparison
        for i, (label, color) in enumerate(zip(['Vx', 'Vy', 'Vz'], ['r', 'g', 'b'])):
            # Ground truth
            axes[1, i].plot(timestamps, trajectory[:, 3+i], color=color, label=f'GT {label}', linewidth=2)
            
            # Estimated
            if len(results['estimated_velocities']) > 0:
                axes[1, i].plot(results['estimated_timestamps'], 
                              results['estimated_velocities'][:, i], 
                              color=color, linestyle='--', label=f'Est {label}', alpha=0.7)
            
            axes[1, i].set_xlabel('Time [s]')
            axes[1, i].set_ylabel(f'Velocity {label} [m/s]')
            axes[1, i].legend()
            axes[1, i].grid(True)
        
        plt.tight_layout()
        plt.show()