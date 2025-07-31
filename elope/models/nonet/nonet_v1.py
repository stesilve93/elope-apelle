
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

from elope.models.blocks.resnet import _make_layer
from elope.evflow import EVFlowNet
from elope.utils import load_yaml, angles_to_quat, quat_rotate


class EVFlowNetHead(nn.Module):

    def __init__(self, ):
        # Initialize the base class
        super().__init__()
        
        # Initialize the EVFlowNet model
        self.model = EVFlowNet(batch_norm=True)
        
    def freeze_weights(self): 
        """Freeze the weights of the original EVFlowNet model."""
        self.model.eval() 
        for p in self.model.parameters(): 
            p.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> tuple:
                         
        # x is of shape (B, 2, C, H, W)
        B, _, C, H, W = x.shape

        # Invert the dimensions to have the channels (counts, stamps) on the first dim
        x = x.permute(0, 2, 1, 3, 4) # (B, C, 2, H, W)
        x = x.reshape(B, -1, H, W)   # (B, 2*C, H, W)
            
        # Upsample the images to ensure the shape matches the one expected from the model
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        
        # Run the EVFlowNet model on the input
        dict_flow = self.model(x)
        
        # Extract the optical flow from the last image 
        flow = dict_flow["flow3"]   # (B, 2, H, W)

        # Downsample the optical flow to the original (200, 200) resolution
        flow = F.interpolate(flow, size=(200, 200), mode="bilinear", align_corners=False)
        
        return flow, dict_flow
    
class FlowEncoder(nn.Module): 
    
    def __init__(
        self, 
        dim_output: int,
        dropout: float = 0.1, 
    ): 
        
        super().__init__() 
        
        # Event image size
        h, w = 200, 200
        
        y_coords = torch.linspace(-1, 1, h)
        x_coords = torch.linspace(-1, 1, w)
        
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([x_grid, y_grid], dim=0).unsqueeze(0) # (1, 2, H, W)
        
        # Register the tensor as a buffer 
        self.register_buffer("coords", coords)
        
        c1, c2, c3, c4, c5 = 7, 64, 128
        
        # Create the encoder/decoder layers 
        self.conv1 = nn.Conv2d(
            c1, out_channels=c2, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        self.bn1 = nn.BatchNorm2d(c2)
        self.relu = nn.Relu()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        
        # Create a convolutional layer 
        self.layer1 = _make_layer(c2, c2, n_blocks=1, stride=1) # (B, 64, 100, 100)
        self.layer2 = _make_layer(c2, c3, n_blocks=1, stride=2) # (B, 128, 100, 100)
        self.layer3 = _make_layer(c3, c4, n_blocks=1, stride=2) # (B, 256, 100, 100)
        self.layer4 = _make_layer(c4, c5, n_blocks=1, stride=2) # (B, 512, 100, 100)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        # Small fully connected layer with dropout to reduce overfitting
        self.fc = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(), 
            nn.Linear(c5, dim_output), 
            nn.LayerNorm(dim_output)
        )
        
    def forward(self, flow: torch.Tensor, w: torch.Tensor) -> torch.Tensor: 
        
        # Flow is a tensor of shape (B, 2, H, W)
        # w is a tensor of shape (B, 3)
        B, _, H, W = flow.shape
        
        # Expand the angular velocity vector to be (B, 3, H, W) 
        w = w.unsqueeze(-1).unsqueeze(-1)   # (B, 3, 1, 1)
        w = w.expand(B, 3, H, W)            # (B, 3, H, W)
        
        # Create the stacked feature vector of shape (B, 7, H, W)
        x = torch.cat([flow, w, self.coords], dim=1)
        
        # Pass through the initial convolutional layer 
        x = self.conv1(x) 
        x = self.bn1(x) 
        x = self.relu(x) 
        x = self.maxpool(x)     # (B, 64, 100, 100)
        
        # Pass through the different encoder layers
        x = self.layer1(x)      # (B, 64, 100, 100) 
        x = self.layer2(x)      # (B, 128, 100, 100)
        x = self.layer3(x)      # (B, 256, 100, 100)
        x = self.layer4(x)      # (B, 512, 100, 100) 
        
        # Perform global average pooling
        x = self.avgpool(x)     # (B, 512, 1, 1)
        x = torch.flatten(x)    # (B, 512)
        
        # Apply fully connected layer 
        x = self.fc(x)          # (B, dim_output) 
        return x


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
                    activation, 
                    # Layer Norm is deactivated by default because when inputs consists 
                    # of difference physical units we would be losing that. 
                    nn.LayerNorm(c2) if norm else nn.Identity(), 
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
    
class RegularizedRegressor(nn.Module):

    def __init__(
        self, 
        input_dim: int, 
        output_dim: int = 3, 
        dropout: float = 0.3
    ):

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
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        
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


class NoFlowNet(nn.Module):
    
    def __init__(
        self, 
        sequence_len: int,
        dim_flow: int=128,
        dim_range: int=[32],
        dropout: float = 0.15,
    ):

        super().__init__()
        
        # Initialize encoders with dropout for regularization
        self.evflownet = EVFlowNetHead(
            dropout=dropout
        )
        
        # Initialize the translational flow encoder
        self.flow_encoder = FlowEncoder(
            dim_output=dim_flow,
            dropout=dropout
        )
        
        # Initialize the encoder for the rangemeter 
        self.range_encoder = BaseEmbedding(
            dim_input=1, 
            hidden_dim=dim_range, 
            activation=nn.SiLU(),
            norm=True, 
            dropout=dropout
        )
        
        # Create the regressor for the velocity in the camera frame
        dim_regressor = dim_flow + dim_range[-1]
        self.regressor = RegularizedRegressor(
            dim_regressor,
            output_dim=3,
            dropout=dropout
        )
        
        # Initialize weights to prevent overfitting
        self._init_weights()
    
    @staticmethod 
    def create_model(cfg: str | Path | dict, device: str="cpu", **kwargs):
        """Factory function to create the improved model"""
        
        # Retrieve the model configuration
        if isinstance(cfg, (str, Path)): 
            cfg = load_yaml(cfg)
        
        model = NoFlowNet(
            dropout=float(cfg["dropout"]),
            sequence_len=int(cfg["sequence_length"]),
            dim_flow=int(cfg["dim_flow"]), 
            dim_range=cfg["dim_range"],
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
        
        # Times is a tensor of size (B, T) 
        # Event is a tensor of size (B, T, 2, C, H, W)
        # IMU is a tensor of size (B, T, 6)
        # Rangemeter is a tensor of size (B, T, 1)
        
        # Adjust times dimension
        times = times.unsqueeze(2)
        
        # Keep only the last tensor for the events 
        event_tensor = event_tensor[:, -1]  # (B, 2, C, H, W)
        range_tensor = range_tensor[:, -1]  # (B, 1)
        
        # Keep only the first angular velocity 
        w_tensor = imu_tensor[:, -1, 3:6]   # (B, 3)

        # Estimate the optical flow using EVFlowNet 
        flow, dict_flow = self.evflownet(event_tensor) # (B, 2, H, W)
        
        # Estimate the features of the translational flow
        feats_flow = self.flow_encoder(flow, w_tensor)        # (B, dim_flow)
        
        # Get the feature from the rangemeter 
        feats_range = self.range_encoder(range_tensor)        # (B, dim_range)
        
        # Simple fusion between the two sets of features
        # TODO: this could be updated with an improved fusion strategy....
        feats = torch.cat([feats_flow, feats_range], dim=-1)  # (B, dim_regressor)
        
        # The regressor estimates the velocity in the camera axes. 
        out = self.regressor(feats)         # (B, 3)
        
        # We then use the orientation angles to rotate it to the inertial frame 
        # knowing the transformation matrix
        angles = imu_tensor[..., 0:3]       # (B, T, 3)
        quat = angles_to_quat(angles)       # (B, T, 4)
        quat = quat[:, -1]                  # (B, 4)
        
        out = quat_rotate(quat, out)
        
        return {
            'prediction': out,
            'optical_flow': dict_flow,
        }
