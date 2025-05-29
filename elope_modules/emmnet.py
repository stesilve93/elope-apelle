import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

#for testing purposes
from elope_modules.dataloader import DataLoader


class ResNet3DBlock(nn.Module):
    """Basic 3D ResNet block for event processing"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Basic 3D ResNet block for event processing.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int, optional): Stride of the convolutional layers. Defaults to 1.
        """
        super().__init__()
        # Convolutional layer 1
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        # Convolutional layer 2
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class EventEncoder(nn.Module):
    """3D CNN encoder for event data"""
    def __init__(self, input_channels: int = 2, output_dim: int = 128):
        """
        3D CNN encoder for event data

        Args:
            input_channels (int): Number of channels in input data. Defaults to 2.
            output_dim (int): Dimensionality of the output. Defaults to 128.
        """
        super().__init__()
        
        # Initial conv layer
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=(3,7,7), 
                              stride=(1,2,2), padding=(1,3,3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        
        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=(1,2,2))
        self.layer3 = self._make_layer(128, 256, 2, stride=(1,2,2))
        self.layer4 = self._make_layer(256, 512, 2, stride=(1,2,2))
        
        # Global pooling and final projection
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(0.2)
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int = 1):
        layers = [ResNet3DBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResNet3DBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Input shape: (B, C, T, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class IMUEncoder(nn.Module):
    """
    LSTM encoder for IMU data (Euler angles + angular velocities)
    
    Args:
        input_dim (int, optional): Input dimension (default: 6 - [phi, theta, psi, p, q, r])
        hidden_dim (int, optional): Hidden dimension for LSTM (default: 64)
        output_dim (int, optional): Output dimension for the final projection (default: 32)
        num_layers (int, optional): Number of LSTM layers (default: 2)
    """
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, output_dim: int = 32, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Input shape: (B, T, 6) - [phi, theta, psi, p, q, r]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        x = h_n[-1]  # Take last layer's final hidden state
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class RangemeterEncoder(nn.Module):
    """Simple encoder for rangemeter distance measurements"""
    """
    Simple encoder for rangemeter distance measurements
    
    The rangemeter distance measurements are fed into a single-layer LSTM
    followed by a fully connected layer to produce a feature vector.
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, output_dim: int = 16, num_layers: int = 1):
        """
        Initialize the rangemeter encoder
        
        Args:
        input_dim: The number of input features (default: 1)
        hidden_dim: The number of hidden units in the LSTM layer (default: 32)
        output_dim: The number of output features (default: 16)
        num_layers: The number of LSTM layers (default: 1)
        """
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Input shape: (B, T, 1) - distance measurements
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        x = h_n[-1]
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class CrossModalAttention(nn.Module):
    """Cross-modal attention for feature fusion"""
    def __init__(self, event_dim: int, imu_dim: int, range_dim: int, hidden_dim: int = 128):
        """
        Initialize the CrossModalAttention module.

        Args:
            event_dim (int): Dimension of the event features.
            imu_dim (int): Dimension of the IMU features.
            range_dim (int): Dimension of the rangemeter features.
            hidden_dim (int, optional): Dimension for the projected features and attention layer. Defaults to 128.
        """
        super().__init__()
        
        # Linear projections to transform input features to a common hidden dimension
        self.event_proj = nn.Linear(event_dim, hidden_dim)
        self.imu_proj = nn.Linear(imu_dim, hidden_dim)
        self.range_proj = nn.Linear(range_dim, hidden_dim)
        
        # Multihead attention mechanism for self-attention across modalities
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1)
        
        # Layer normalization for stabilizing the learning process
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, event_feat, imu_feat, range_feat):
        # Project all features to same dimension
        e_proj = self.event_proj(event_feat)
        i_proj = self.imu_proj(imu_feat)
        r_proj = self.range_proj(range_feat)
        
        # Stack for attention (seq_len=3, batch, hidden_dim)
        features = torch.stack([e_proj, i_proj, r_proj], dim=0)
        
        # Self-attention across modalities
        attended, attention_weights = self.attention(features, features, features)
        attended = self.norm(attended + features)  # Residual connection
        
        # Global pooling across modalities
        fused = attended.mean(dim=0)  # Average across the 3 modalities
        
        return fused, attention_weights


class MultiModalVelocityEstimator(nn.Module):
    """
    End-to-end multi-modal network for lunar descent velocity estimation
    """
    def __init__(self, 
                 event_channels: int = 2,
                 event_output_dim: int = 128,
                 imu_output_dim: int = 32,
                 range_output_dim: int = 16,
                 use_attention: bool = False,
                 output_dim: int = 6):  # [x, y, z, vx, vy, vz]
        """
        Initialize the MultiModalVelocityEstimator with the given parameters.
        
        Args:
            event_channels (int, optional): Number of event channels. Defaults to 2.
            event_output_dim (int, optional): Output dimension of the event encoder. Defaults to 128.
            imu_output_dim (int, optional): Output dimension of the IMU encoder. Defaults to 32.
            range_output_dim (int, optional): Output dimension of the rangemeter encoder. Defaults to 16.
            use_attention (bool, optional): Whether to use cross-modal attention for fusion. Defaults to False.
            output_dim (int, optional): Dimension of the output velocity vector. Defaults to 6.
        """
        super().__init__()
        
        # Individual encoders
        self.event_encoder = EventEncoder(event_channels, event_output_dim)
        self.imu_encoder = IMUEncoder(output_dim=imu_output_dim)
        self.range_encoder = RangemeterEncoder(output_dim=range_output_dim)
        
        # Fusion strategy
        self.use_attention = use_attention
        if use_attention:
            self.fusion = CrossModalAttention(event_output_dim, imu_output_dim, range_output_dim)
            fusion_input_dim = 128  # Hidden dim from attention
        else:
            # Simple concatenation
            fusion_input_dim = event_output_dim + imu_output_dim + range_output_dim
        
        # Final regression head
        self.regressor = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, event_tensor, imu_tensor, range_tensor):
        # Extract features from each modality
        event_feat = self.event_encoder(event_tensor)
        imu_feat = self.imu_encoder(imu_tensor)
        range_feat = self.range_encoder(range_tensor)
        
        # Fusion
        if self.use_attention:
            fused_feat, attention_weights = self.fusion(event_feat, imu_feat, range_feat)
        else:
            fused_feat = torch.cat([event_feat, imu_feat, range_feat], dim=1)
            attention_weights = None
        
        # Final prediction
        output = self.regressor(fused_feat)
        
        return {
            'prediction': output,
            'event_features': event_feat,
            'imu_features': imu_feat,
            'range_features': range_feat,
            'attention_weights': attention_weights
        }


def create_model(use_attention: bool = False, device: str = 'cpu') -> MultiModalVelocityEstimator:
    """Factory function to create the model"""
    model = MultiModalVelocityEstimator(use_attention=use_attention)
    return model.to(device)










###########     testing code    ###########
if __name__ == "__main__":
    loader = DataLoader("./elope_data")
    loader.load_sequence(sequence_id='0000', source="train")
    
    # Preprocess
    print("\nPreprocessing data...")
    event_tensor = loader.preprocess_events(loader.events_full, 10)
    imu_tensor = loader.preprocess_imu(loader.trajectory_full)
    range_tensor = loader.preprocess_rangemeter(loader.rangemeter_full, loader.timestamps_full)
    
    print(f"Preprocessed shapes:")
    print(f"  Events: {event_tensor.shape}")
    print(f"  IMU: {imu_tensor.shape}")
    print(f"  Rangemeter: {range_tensor.shape}")
    
    # Convert to PyTorch tensors and add batch dimension
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_point = loader.get_data_at_time(10,
                                            event_integration_window_us=100000, # 100ms
                                            imu_seq_len=3,
                                            H=200, W=200, T=10)
    
    # event_batch = torch.from_numpy(event_tensor).unsqueeze(0).to(device)
    # imu_batch = torch.from_numpy(imu_tensor).unsqueeze(0).to(device)
    # range_batch = torch.from_numpy(range_tensor).unsqueeze(0).to(device)

    event_batch = data_point['events_tensor'].unsqueeze(0).to(device)
    imu_batch = data_point['imu_sequence'].unsqueeze(0).to(device)
    range_batch = data_point['rangemeter_sequence'].unsqueeze(0).to(device)
    
    
    # Load model and run inference
    model = create_model(use_attention=True, device=device)
    model.eval()
    
    print("\nRunning inference...")
    with torch.no_grad():
        output = model(event_batch, imu_batch, range_batch)
    
    prediction = output['prediction'][0].cpu().numpy()
    
    print(f"\nPredicted state:")
    print(f"  Position (x,y,z): [{prediction[0]:.2f}, {prediction[1]:.2f}, {prediction[2]:.2f}] m")
    print(f"  Velocity (vx,vy,vz): [{prediction[3]:.2f}, {prediction[4]:.2f}, {prediction[5]:.2f}] m/s")
    
        # Compare with ground truth (last trajectory point)
    gt_state = data_point['ground_truth'].numpy()
    gt_pos = gt_state[:3]
    gt_vel = gt_state[3:6]
    
    print(f"\nGround truth (last trajectory point):")
    print(f"  Position (x,y,z): [{gt_pos[0]:.2f}, {gt_pos[1]:.2f}, {gt_pos[2]:.2f}] m")
    print(f"  Velocity (vx,vy,vz): [{gt_vel[0]:.2f}, {gt_vel[1]:.2f}, {gt_vel[2]:.2f}] m/s")
    
    # Calculate errors
    pos_error = np.linalg.norm(prediction[:3] - gt_pos)
    vel_error = np.linalg.norm(prediction[3:6] - gt_vel)
    
    print(f"\nErrors (untrained model):")
    print(f"  Position error: {pos_error:.2f} m")
    print(f"  Velocity error: {vel_error:.2f} m/s")
    
    print("Model test successful!")