
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from pathlib import Path

from elope.utils import *

class BaseEmbedding(nn.Module): 
    """Base MLP layer for basic feature expansion.""" 
    
    def __init__(
        self, 
        input_dim: int = 2,
        hidden_dim: int | list = 32,
        activation: nn.Module = nn.ReLU(), 
        norm: bool = False,
        dropout: float = 0.1, 
    ):
        # Initialize the base class 
        super().__init__() 
        
        # Ensure it is a list and retrieve layers
        hidden_dim = [hidden_dim] if isinstance(hidden_dim, int) else hidden_dim
        num_layers = len(hidden_dim)
        
        # Retrieve dimensions of first FC layer
        c1 = input_dim
        
        # Add FC layer
        self.layers = nn.ModuleList()
        for k in range(num_layers): 
            
            # Update the channel dimension 
            c2 = hidden_dim[k] 
            
            self.layers.append(
                nn.Sequential(
                    nn.Linear(c1, c2), 
                    nn.LayerNorm(c2) if norm else nn.Identity(), 
                    deepcopy(activation), 
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity()
                )
            )
            
            # Update the next layer input dims
            c1 = c2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is a tensor of shape (B, S, N)
        for layer in self.layers: 
            x = layer(x)
        
        return x
    
    
class DepthEncoder(nn.Module): 
    
    def __init__(
        self, 
        coords: torch.Tensor,
        channels: list=[8, 16, 2],
        dropout: float=0.1
    ):   
        super().__init__() 
        
        # Generate the coordinates
        self.register_buffer("coords", coords)
        
        # Create a small MLP layer for the coordinates 
        self.mlp = BaseEmbedding(
            input_dim=2, 
            hidden_dim=channels,
            activation=nn.ReLU(), 
            norm=True, 
            dropout=dropout
        )
        
    def forward(self, angles: torch.Tensor, ranges: torch.Tensor) -> torch.Tensor: 
        
        # angles: (B, 3)
        # range:  (B)
        
        # Pass the coordinates through the MLP layer 
        coords = self.coords.reshape(-1, 2)
        coords = self.mlp(coords)                               # (C, 2)
        
        return self.inverse_depthmap(coords, angles, ranges)    # (B, C)
    
    @staticmethod
    def inverse_depthmap(xys: torch.Tensor, angles: torch.Tensor, ranges: torch.Tensor): 
        
        # angles: (B, 3)        
        # coords: (C, 2)
        # ranges: (B)
        
        # Retrieve the spacecraft altitude 
        altitude = estimate_altitude(ranges.unsqueeze(-1), angles)
        
        cos_phi = torch.cos(angles[..., 0])
        sin_phi = torch.sin(angles[..., 0])

        cos_theta = torch.cos(angles[..., 1])
        sin_theta = torch.sin(angles[..., 1])
                
        a = (-sin_theta).unsqueeze(1)
        b = ( sin_phi*cos_theta).unsqueeze(1) 
        g = ( cos_phi*cos_theta).unsqueeze(1)
        
        # Retrieve the point coordinates
        xs = xys[..., 0]
        ys = xys[..., 1]
        
        # Compute the inverse pixel depth map
        return (a*xs + b*ys + g)/altitude


class RotationalFlowEncoder(nn.Module): 
    
    def __init__(
        self, 
        coords: torch.Tensor,
        channels: list = [16, 32, 64, 2],
        dropout: float=0.1
    ): 
        super().__init__() 
        
        # Generate the coordinates
        self.register_buffer("coords", coords)
        
        # Initialize the base embedding layer
        self.mlp = BaseEmbedding(
            input_dim=5, 
            hidden_dim=channels,
            activation=nn.ReLU(), 
            norm=True,
            dropout=dropout
        )
    
    def forward(self, omega: torch.Tensor) -> torch.Tensor: 
        
        # omega: (B, 3)
        B, _ = omega.shape
        
        coords = self.coords.reshape(-1, 2)          # (C, 2)
        coords = coords.expand(B, -1, 2)             # (B, C, 2)
        
        C = coords.shape[1]
        omega = omega.unsqueeze(1).expand(-1, C, 3)  # (B, C, 3)

        # Assemble the global tensor
        x = torch.cat([omega, coords], dim=-1)       # (B, C, 5)
        
        # Run the MLP layer
        x = self.mlp(x)                              # (B, C, 2)
        return x
        

class ResNet3DBlock(nn.Module):

    def __init__(
        self, 
        cin: int, 
        cout: int, 
        stride: int = 1, 
        dropout: float = 0.1
    ):
        
        super().__init__()
        
        # Convolutional layer 1
        self.conv1 = nn.Conv3d(
            cin, cout, kernel_size=3, stride=stride, padding=1, bias=False
        )
        
        self.bn1 = nn.BatchNorm3d(cout)
        self.dropout1 = nn.Dropout3d(dropout)
        
        # Convolutional layer 2
        self.conv2 = nn.Conv3d(cout, cout, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(cout)
        self.dropout2 = nn.Dropout3d(dropout)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(
                nn.Conv3d(cin, cout, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(cout)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class EventEncoder(nn.Module):

    def __init__(
        self, 
        cin: int = 2, 
        channels: list = [512, 256, 64, 2],
        dropout: float = 0.15
    ):
        super().__init__()
        
        # Initial convolution with stochastic depth
        # The kernel size is (3,7,7) to capture the spatial and temporal
        # information in the event data
        self.conv1 = nn.Conv3d(
            cin, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False
        )
        
        # Batch normalization to normalize the output of the convolution
        self.bn1 = nn.BatchNorm3d(64)
        # Max pooling to downsample the data
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        
        # ResNet blocks with increased dropout
        self.layer1 = self._make_layer(64, 64, 2, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(64, 128, 2, stride=(1,2,2), dropout=dropout)
        self.layer3 = self._make_layer(128, 256, 2, stride=(1,2,2), dropout=dropout)
        self.layer4 = self._make_layer(256, 512, 2, stride=(1,2,2), dropout=dropout)
        
        # Global pooling and projection with regularization
        # The AdaptiveAvgPool3d is used to reduce the spatial and temporal
        # dimensions to 1
        self.avgpool = nn.AdaptiveAvgPool3d((1, 4, 4))
        
        # Compute a measure of the flow at each location 
        self.mlp = BaseEmbedding(
            input_dim=512, 
            hidden_dim=channels,
            activation=nn.GELU(), 
            norm=True, 
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # x: torch.Size([B, 2, 1, 200, 200])
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)         # [B, 64, 1, 100, 100]
        
        x = self.layer1(x)          # [B, 64, 1, 50, 50]
        x = self.layer2(x)          # [B, 64, 1, 50, 50]
        x = self.layer3(x)          # [B, 256, 1, 13, 13]
        x = self.layer4(x)          # [B, 512, 1, 7, 7]
        
        x = self.avgpool(x)         # [B, 512, 1, 4, 4]
        x = x.squeeze()             # [B, 512, 4, 4]
        x = x.permute(0, 2, 3, 1)   # [B, 4, 4, 512]
        
        x = self.mlp(x)             # [B, 4, 4, H] 
        
        B, _, _, H = x.shape
        x = x.reshape(B, -1, H)     # [B, 16, H]
    
        return x
        
    def _make_layer(
        self, cin: int, cout: int, num_blocks: int, stride: int=1, dropout: float=0.1
    ):

        layers = [ResNet3DBlock(cin, cout, stride, dropout)]
        for _ in range(1, num_blocks):
            # The dropout in the ResNet block is the same as the input dropout
            layers.append(ResNet3DBlock(cout, cout, dropout=dropout))
            
        return nn.Sequential(*layers)
    
    
class CrossModalAttention(nn.Module):

    def __init__(
        self, 
        input_dim: int,
        channels: list = [256, 256, 512],
        dropout: float = 0.1
    ):
        
        super().__init__()
        
        # Feature projections with layer normalization
        self.proj = BaseEmbedding(
            input_dim=input_dim,
            hidden_dim=channels,
            activation=nn.ReLU(), 
            norm=False,
            dropout=dropout
        )

        # Multi-head attention with residual connections
        self.attention = nn.MultiheadAttention(
            channels[-1], num_heads=8, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network with residual connections
        self.ffn = nn.Sequential(
            nn.Linear(channels[-1], channels[-1] * 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(channels[-1] * 2, channels[-1])
        )
        
        # Layer normalization for stabilizing the learning process
        self.norm1 = nn.LayerNorm(channels[-1])
        self.norm2 = nn.LayerNorm(channels[-1])
        
        # Dropout regularization
        self.dropout = nn.Dropout(dropout)

        # Learnable weights for each modality
        #self.modal_weights = nn.Parameter(torch.ones(3)) # Learnable weights for each modality

    def forward(self, feats: torch.Tensor) -> tuple: 
        
        # feats: (B, C, I)
        
        # Project features
        feats = self.proj(feats)    # (B, C, D)

        # Self-attention with residual
        attended, attention_weights = self.attention(feats, feats, feats)
        attended = self.norm1(attended + feats)
        
        # Feed-forward with residual
        ffn_out = self.ffn(attended)
        ffn_out = self.norm2(ffn_out + attended)
        
        # Use learnable weights if available, otherwise simple average
        fused = torch.mean(ffn_out, dim=1)  # Simple average fallback
        return fused, attention_weights


class RegularizedRegressor(nn.Module):


    def __init__(self, input_dim: int, output_dim: int = 3, dropout: float = 0.3):

        super().__init__()
        
        # Simplified architecture with skip connections
        self.layers = nn.ModuleList([
            # First layer
            nn.Linear(input_dim, 128),
            # Second layer with skip connection
            nn.Linear(128, 64),
            # Final layer
            nn.Linear(64, output_dim)
        ])
        
        # Batch normalization for the first two layers
        self.norms = nn.ModuleList([
            nn.LayerNorm(128),
            nn.LayerNorm(64)
        ])
        
        # Dropout regularization for the first two layers
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout),
            nn.Dropout(dropout * 0.7)
        ])
        
        # Skip connection projection
        self.skip_proj = nn.Linear(input_dim, 64) if input_dim != 64 else nn.Identity()
        
        
    def forward(self, x):
        # First layer
        out = F.gelu(self.norms[0](self.layers[0](x)))
        out = self.dropouts[0](out)
        
        # Second layer with skip connection
        identity = self.skip_proj(x)
        out = F.gelu(self.norms[1](self.layers[1](out)))
        out = self.dropouts[1](out + identity)  # Skip connection
        
        # Final layer (no activation for regression)
        out = self.layers[2](out)
        
        return out


class MultiModalVelocityEstimatorAngles(nn.Module):
    
    def __init__(
        self, 
        channels_event: int = [512, 256, 64],
        channels_omega: list = [16, 32, 64],
        channels_depth: list = [8, 16, 2],
        channels_fusion: list = [256, 256],
        output_dim: int = 3,
        dropout: float = 0.15,
    ):

        super().__init__()

        # Generate teh coordinate grid
        coords = generate_pixelgrid(4, 4)
        
        # Initialize event encoder
        self.encoder_event = EventEncoder(
            2,
            channels=channels_event,
            dropout=dropout
        )
                
        # Create the encoder for the depth map
        self.encoder_depth = DepthEncoder(
            coords,
            channels_depth, 
            dropout=dropout
        )
        
        # Create the encoder for the rotational flow 
        self.encoder_omega = RotationalFlowEncoder(
            coords, 
            channels_omega, 
            dropout=dropout
        )

        # Use cross-modal attention mechanism for feature fusion
        self.fusion = CrossModalAttention(
            input_dim=channels_event[-1] + channels_omega[-1] + 1,
            channels=channels_fusion,
            dropout=dropout, 
        )
        
        # Final regression layer
        self.regressor = RegularizedRegressor(channels_fusion[-1], output_dim, dropout)
        
        # Initialize weights to prevent overfitting
        self._init_weights()
    
    @staticmethod 
    def create_model(cfg: str | Path | dict, device: str="cpu", **kwargs):
        """Factory function to create the improved model"""
        
        # Retrieve the model configuration
        if isinstance(cfg, (str, Path)): 
            cfg = load_yaml(cfg)
            
        # assert cfg["output_type"] == "central_state"
        
        model = MultiModalVelocityEstimatorAngles(
            dropout=float(cfg["dropout"]),
            channels_event=cfg["channels_event"], 
            channels_omega=cfg["channels_omega"], 
            channels_depth=cfg["channels_depth"],
            channels_fusion=cfg["channels_fusion"],
            **kwargs
        )
        
        return model.to(device)
    
    def _init_weights(self):
        """
        Initialize weights properly to prevent overfitting.

        This function initializes weights using Xavier initialization with a smaller gain
        (0.5) and kaiming_normal initialization for convolutional layers.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize weights using Xavier initialization with a smaller gain (0.5)
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    # Initialize bias to zero
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                # Initialize weights using kaiming_normal initialization
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, times, event_tensor, imu_tensor, range_tensor):
        
        # Adjust times dimension
        times = times.unsqueeze(2)
        
        # Keep only the latest event 
        tensor_event = event_tensor[:, -1]      # (B, 2, C, H, W)
        tensor_angle = imu_tensor[:, -1, 0:3]   # (B, 3)
        tensor_omega = imu_tensor[:, -1, 3:6]   # (B, 3)
        tensor_range = range_tensor[:, -1, 0]   # (B)
        
        # Compute the inverse depth map 
        depth_map = self.encoder_depth(tensor_angle, tensor_range)  # (B, C)
        depth_map = depth_map.unsqueeze(2)                          # (B, C, 1)
        
        # Compute the rotational flow map 
        feat_rot  = self.encoder_omega(tensor_omega)                 # (B, C, R)
        
        # Estimate the optical flow from the events 
        # TODO: evaluate whether to provide a confidence score for these flows
        feat_flow = self.encoder_event(tensor_event)                 # (B, C, F)
        
        # Concatenate the different features and fuse them together
        feats = torch.cat([feat_flow, feat_rot, depth_map], dim=-1)  # (B, C, D)
        fused_feat, _ = self.fusion(feats)  # (B, M)

        # Final prediction
        output = self.regressor(fused_feat)

        # Compute DCM rotation from camera to inertial frame
        dcm = angles_to_dcm(tensor_angle) # (B, 3, 3)
        
        # Output has shape (B, 3)
        output = output.unsqueeze(-1)   # (B, 3, 1)
        output = (dcm@output).squeeze() # (B, 3)
        
        return {
            'prediction': output,
        }
