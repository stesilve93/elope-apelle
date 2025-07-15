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

class OpticalFlowEstimator:
    """Estimate optical flow from event frames using EVFlowNet"""
    
    def __init__(self, device='cpu'):
        self.device = device
        from evflow.config import configs  # Assuming configs is defined in evflow/config.py
        args = configs()
        print("Initializing EVFlowNet with args:", args)
        self.evflownet = EVFlowNet(args).to(device)
        print("EVFlowNet initialized")
        self.evflownet.eval()  # Set to evaluation mode
        self.flow_history = []
        
    def prepare_input_tensor(self, frame1, frame2):
        """Prepare input tensor for EVFlowNet from two consecutive frames"""
        # Normalize frames
        frame1_norm = (frame1 - frame1.min()) / (frame1.max() - frame1.min() + 1e-8)
        frame2_norm = (frame2 - frame2.min()) / (frame2.max() - frame2.min() + 1e-8)
        
        # Stack frames as input channels
        input_tensor = np.stack([frame1_norm, frame2_norm], axis=0)
        
        # Add batch dimension and convert to tensor
        input_tensor = torch.FloatTensor(input_tensor).unsqueeze(0).to(self.device)
        
        return input_tensor
    
    def estimate_flow_evflownet(self, frame1, frame2):
        """Estimate optical flow using EVFlowNet"""
        with torch.no_grad():
            # Prepare input
            input_tensor = self.prepare_input_tensor(frame1, frame2)
            
            # Forward pass through EVFlowNet
            flow_tensor = self.evflownet(input_tensor)
            
            # Convert back to numpy
            flow = flow_tensor.squeeze(0).cpu().numpy()
            flow = np.transpose(flow, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)
            
        return flow
    
    def estimate_flow_lucas_kanade(self, frame1, frame2):
        """Estimate optical flow using Lucas-Kanade method (fallback)"""
        # Convert to uint8 for OpenCV
        frame1_uint8 = ((frame1 - frame1.min()) / (frame1.max() - frame1.min() + 1e-8) * 255).astype(np.uint8)
        frame2_uint8 = ((frame2 - frame2.min()) / (frame2.max() - frame2.min() + 1e-8) * 255).astype(np.uint8)
        
        # Detect corners
        corners = cv2.goodFeaturesToTrack(frame1_uint8, maxCorners=100, qualityLevel=0.01, minDistance=10)
        
        if corners is None or len(corners) == 0:
            return np.zeros((frame1.shape[0], frame1.shape[1], 2))
        
        # Calculate optical flow
        flow_points, status, error = cv2.calcOpticalFlowPyrLK(frame1_uint8, frame2_uint8, corners, None)
        
        # Create dense flow field
        flow_field = np.zeros((frame1.shape[0], frame1.shape[1], 2))
        
        valid_corners = corners[status.flatten() == 1]
        valid_flow = flow_points[status.flatten() == 1] - valid_corners
        
        if len(valid_corners) > 0:
            # Interpolate sparse flow to dense field
            y_coords, x_coords = np.mgrid[0:frame1.shape[0], 0:frame1.shape[1]]
            
            for i, (corner, flow_vec) in enumerate(zip(valid_corners, valid_flow)):
                x, y = int(corner[0][0]), int(corner[0][1])
                if 0 <= x < frame1.shape[1] and 0 <= y < frame1.shape[0]:
                    flow_field[y, x] = flow_vec[0]
        
        return flow_field
    
    def estimate_flow_farneback(self, frame1, frame2):
        """Estimate dense optical flow using Farneback method (fallback)"""
        frame1_uint8 = ((frame1 - frame1.min()) / (frame1.max() - frame1.min() + 1e-8) * 255).astype(np.uint8)
        frame2_uint8 = ((frame2 - frame2.min()) / (frame2.max() - frame2.min() + 1e-8) * 255).astype(np.uint8)
        
        flow = cv2.calcOpticalFlowFarneback(frame1_uint8, frame2_uint8, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow