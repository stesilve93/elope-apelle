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

class EventProcessor:
    """Process event camera data for optical flow estimation"""
    
    def __init__(self, width=200, height=200):
        self.width = width
        self.height = height
        
    def events_to_frames(self, events, time_window=0.01, method='count'):
        """Convert event stream to frame-like representations"""
        frames = []
        timestamps = []
        
        # Get time range
        t_start = events['t'].min()
        t_end = events['t'].max()
        
        # Convert microseconds to seconds for processing
        t_range = np.arange(t_start, t_end, time_window * 1e6)
        
        for i in range(len(t_range) - 1):
            t_curr = t_range[i]
            t_next = t_range[i + 1]
            
            # Get events in current time window
            mask = (events['t'] >= t_curr) & (events['t'] < t_next)
            window_events = events[mask]
            
            if len(window_events) == 0:
                continue
                
            if method == 'count':
                frame = self._events_to_count_frame(window_events)
            elif method == 'time_surface':
                frame = self._events_to_time_surface(window_events, t_curr)
            else:
                frame = self._events_to_polarity_frame(window_events)
                
            frames.append(frame)
            timestamps.append(t_curr / 1e6)  # Convert back to seconds
            
        return np.array(frames), np.array(timestamps)
    
    def _events_to_count_frame(self, events):
        """Create count-based frame from events"""
        frame = np.zeros((self.height, self.width), dtype=np.float32)
        
        if len(events) > 0:
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
        """Create time surface representation"""
        frame = np.zeros((self.height, self.width), dtype=np.float32)
        
        if len(events) > 0:
            x_coords = events['x'].values.astype(int)
            y_coords = events['y'].values.astype(int)
            timestamps = events['t'].values
            
            x_coords = np.clip(x_coords, 0, self.width - 1)
            y_coords = np.clip(y_coords, 0, self.height - 1)
            
            # Time decay
            time_diff = current_time - timestamps
            decay = np.exp(-time_diff / 10000)  # Decay constant
            
            for x, y, decay_val in zip(x_coords, y_coords, decay):
                frame[y, x] = max(frame[y, x], decay_val)
                
        return frame

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


class TrajectoryEstimator:
    """Estimate trajectory from optical flow, range, and IMU data"""
    
    def __init__(self):
        self.camera_matrix = np.array([[100, 0, 100], [0, 100, 100], [0, 0, 1]])  # Simplified camera matrix
        
    def depth_from_flow_and_range(self, flow, range_data, timestamps):
        """Estimate depth map using flow and range measurements"""
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
        """Estimate 3D velocity from optical flow and depth"""
        if len(flow) < 2 or dt <= 0:
            return np.zeros(3)
        
        # Get flow at center pixel
        h, w = flow.shape[:2]
        center_flow = flow[h//2, w//2]
        center_depth = depth[h//2, w//2]
        
        # Convert flow to angular velocity (rad/s)
        flow_angular = center_flow / 100.0  # Simplified conversion
        
        # Compensate for camera rotation
        compensated_flow = flow_angular - angular_velocity[:2]
        
        # Estimate translational velocity
        # v = Z * ω (where ω is compensated angular flow)
        velocity_x = center_depth * compensated_flow[0] / dt
        velocity_y = center_depth * compensated_flow[1] / dt
        
        # Z velocity from range rate (if available)
        velocity_z = 0  # Would need range rate for this
        
        return np.array([velocity_x, velocity_y, velocity_z])

class LunarDescentPipeline:
    """Complete pipeline for lunar descent velocity estimation"""
    
    def __init__(self, use_gpu=False):
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.event_processor = EventProcessor()
        self.flow_estimator = OpticalFlowEstimator(device=self.device)
        self.trajectory_estimator = TrajectoryEstimator()
        
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
        
    def process_sequence(self, events, timestamps, trajectory, rangemeter):
        """Process complete sequence and estimate velocities"""
        print("Processing event stream...")
        
        # Convert events to DataFrame if needed
        if isinstance(events, np.ndarray):
            ev_data = pd.DataFrame(events, columns=['x', 'y', 'p', 't'])
        else:
            ev_data = events
            
        # Convert events to frames
        frames, frame_timestamps = self.event_processor.events_to_frames(ev_data, time_window=0.05)
        
        if len(frames) < 2:
            print("Not enough frames generated from events")
            return None
            
        print(f"Generated {len(frames)} frames from events")
        
        # Estimate optical flow using EVFlowNet
        flows = []
        flow_timestamps = []
        
        print("Estimating optical flow using EVFlowNet...")
        
        for i in range(len(frames) - 1):
            try:
                # Use EVFlowNet for optical flow estimation
                flow = self.flow_estimator.estimate_flow_evflownet(frames[i], frames[i+1])
                flows.append(flow)
                flow_timestamps.append((frame_timestamps[i] + frame_timestamps[i+1]) / 2)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i+1}/{len(frames)-1} flow pairs")
                    
            except Exception as e:
                print(f"EVFlowNet failed for frame {i}, falling back to Farneback: {str(e)}")
                # Fallback to traditional method
                flow = self.flow_estimator.estimate_flow_farneback(frames[i], frames[i+1])
                flows.append(flow)
                flow_timestamps.append((frame_timestamps[i] + frame_timestamps[i+1]) / 2)
            
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
            'frame_timestamps': frame_timestamps,
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

# Main execution function
def run_lunar_descent_pipeline():
    """Run the complete pipeline"""
    
    # Load data
    datapath = '/home/stesilve/Documents/github/elope/elope_data'
    fn = os.path.join(datapath, 'train', '0000.npz')
    
    try:
        sequence = np.load(fn)
        events = sequence['events']
        timestamps = sequence['timestamps']
        trajectory = sequence['traj']
        rangemeter = sequence.get('rangemeter', np.column_stack([timestamps, trajectory[:, 2]]))  # Use Z as range if not available
        
        print("Data loaded successfully!")
        print(f"Events shape: {events.shape}")
        print(f"Trajectory shape: {trajectory.shape}")
        print(f"Timestamps shape: {timestamps.shape}")
        
        # Initialize and run pipeline
        pipeline = LunarDescentPipeline(use_gpu=True)  # Enable GPU if available
        
        # Load pre-trained weights from TensorFlow checkpoint
        pipeline.load_pretrained_weights('./weights/ev-flownet/ev-flownet/model.pth')  
        
        # Or inspect TensorFlow checkpoint structure first
        # pipeline.inspect_checkpoint('path/to/model.ckpt-600023.meta')
        
        results = pipeline.process_sequence(events, timestamps, trajectory, rangemeter)
        
        if results is not None:
            print("\nPipeline execution completed successfully!")
            print(f"Estimated velocities for {len(results['estimated_velocities'])} time steps")
            
            # Visualize results
            pipeline.visualize_results(results, trajectory, timestamps)
            
            # Print some statistics
            if len(results['estimated_velocities']) > 0:
                print("\nVelocity Estimation Statistics:")
                print(f"Mean estimated velocity: {np.mean(results['estimated_velocities'], axis=0)}")
                print(f"Std estimated velocity: {np.std(results['estimated_velocities'], axis=0)}")
        else:
            print("Pipeline failed to generate results")
            
    except FileNotFoundError:
        print(f"Data file not found at {fn}")
        print("Please ensure the data file exists at the specified path")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    run_lunar_descent_pipeline()