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

class TrajectoryEstimator:
    """Estimate trajectory from optical flow, range, and IMU data"""
    
    def __init__(self):
        self.camera_matrix = np.array([[100, 0, 100], [0, 100, 100], [0, 0, 1]])  # Simplified camera matrix
        
    def depth_from_flow_and_range(self, flow, range_data, timestamps):
        """Estimate depth map using flow and range measurements
        
        :param flow: (N, H, W, 2) array of optical flow fields
        :param range_data: (M, 2) array of range measurements with timestamps
        :param timestamps: (N,) array of timestamps for the optical flow fields
        :return: (N, H, W) array of depth maps
        """
        # Interpolate range data to flow timestamps
        range_interp = interpolate.interp1d(range_data[:, 0], range_data[:, 1], 
                                          bounds_error=False, fill_value='extrapolate')
        
        depth_maps = []
        for i, t in enumerate(timestamps):
            depth_center = range_interp(t)
            
            # Simple depth propagation (in practice, would use more sophisticated methods)
            depth_map = np.full((200, 200), depth_center, dtype=np.float32)
            
            # Modulate depth based on flow magnitude (assumption: faster flow = closer objects)
            if i < len(flow):
                flow_magnitude = np.linalg.norm(flow[i], axis=2)
                depth_map *= (1 + 0.1 * flow_magnitude / (flow_magnitude.max() + 1e-8))
            
            depth_maps.append(depth_map)
            
        return np.array(depth_maps)
    
    def estimate_velocity_from_flow(self, flow, depth, dt, angular_velocity):
        """
        Estimate 3D velocity from optical flow and depth
        
        :param flow: (N, H, W, 2) array of optical flow fields
        :param depth: (N, H, W) array of depth maps
        :param dt: time step (seconds)
        :param angular_velocity: (N, 3) array of angular velocities
        :return: estimated 3D velocity (m/s)
        """
        if len(flow) < 2 or dt <= 0:
            return np.zeros(3)
        
        # Get flow at center pixel
        h, w = flow.shape[:2]
        center_flow = flow[h//2, w//2]
        center_depth = depth[h//2, w//2]
        
        # Convert flow to angular velocity (rad/s)
        # 100.0 is a simplified conversion factor
        flow_angular = center_flow / 100.0
        
        # Compensate for camera rotation
        compensated_flow = flow_angular - angular_velocity[:2]
        
        # Estimate translational velocity
        # v = Z * ω (where ω is compensated angular flow)
        velocity_x = center_depth * compensated_flow[0] / dt
        velocity_y = center_depth * compensated_flow[1] / dt
        
        # Z velocity from range rate (if available)
        velocity_z = 0  # Would need range rate for this
        
        return np.array([velocity_x, velocity_y, velocity_z])
