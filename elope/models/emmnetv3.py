
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

from elope.utils import load_yaml

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
        
        # Ensure it is a lsit
        if isinstance(hidden_dim, int): 
            hidden_dim = [hidden_dim]
        
        # Retrieve number of layers 
        num_layers = len(hidden_dim)
        
        # Retrieve dimensions of first FC layer
        c1 = input_dim
        
        # Add FC layer
        self.layers = []
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
    
class VelocityRegressor(nn.Module): 
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int,
        hidden_dim: int | list = [128, 64],
        activation: nn.Module = nn.GELU(),
        norm: bool = False, 
        dropout: float = 0.0, 
    ): 
        super().__init__()
        
        # Ensure the hidden dim is a list 
        if isinstance(hidden_dim, int): 
            hidden_dim = [hidden_dim]
            
        # Retrieve the number of layers 
        num_layers = len(hidden_dim)
        
        # Retrieve the dimenions of the first FC layer 
        c1 = input_dim
        
        # Add the FC layers 
        self.layers = []
        for k in range(num_layers): 
            
            # Update the channel dimension 
            c2 = hidden_dim[k]
            
            self.layers.append(
                nn.Sequential(
                    nn.Linear(c1, c2), 
                    activation, 
                    nn.LayerNorm(c2) if norm else nn.Identity(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity() 
                )
            )
            
            # Update the next layer input dims
            c1 = c2 
        
        # Add a final fully connected layer 
        # TODO: evaluate whether this can be improved by adding skip connections
        self.proj_head = nn.Linear(c2, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        
        # Apply the different layers
        for layer in self.layers: 
            x = layer(x)
        
        # Final layer (without activation for the regression)
        out = self.proj_head(x)
        return out
        

class MultiModalTransformerEstimator(nn.Module):
    """ Multi-modal transformer-based network with better regularization."""
    
    def __init__(
        self, 
        dropout: float = 0.1,
    ):

        super().__init__()
        
        # Initialise the BaseEmbedding block for the rangemeter data 
        self.proj_range = BaseEmbedding(
            input_dim = 2, 
            hidden_dim = [32, 16], 
            activation = nn.SiLU(),
            norm = False, 
            dropout = dropout
        )
        
        # Initialise the BaseEmbedding block for the IMU data 
        self.proj_imu = BaseEmbedding(
            input_dim = 7, 
            hidden_dim = [64, 64, 32], 
            activation = nn.SiLU(),
            norm = False, 
            dropout = dropout
        )
        
        self.regressor_head = VelocityRegressor(
            input_dim = , # TODO: add dimension of transfomer output.
            output_dim = 3,
            hidden_dim=[128, 64]
            activation=nn.SiLU(), 
            norm=False, 
            dropout = dropout,
        )

        # Initialize weights to prevent overfitting
        self._init_weights()
    
    @staticmethod 
    def create_model(cfg: str | Path | dict, device: str="cpu", **kwargs):
        """Factory function to create the improved model"""
        
        # Retrieve the model configuration
        if isinstance(cfg, (str, Path)): 
            cfg = load_yaml(cfg)
        
        model = MultiModalTransformerEstimator(
            dropout=float(cfg["dropout"]), 
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
    
    def forward(
        self, 
        times: torch.Tensor, 
        tensor_event: torch.Tensor, 
        tensor_imu: torch.Tensor, 
        tensor_range: torch.Tensor
    ) -> dict:
        
        # Times is a tensor of size (B, S)
        # Event is a tensor of size (B, S, 2, H, W, C)
        # IMU is a tensor of size (B, S, 6)
        # Rangemeter is a tensor of size (B, S, 1)
        
        # Adjust times dimension
        times = times.unsqueeze(2)  # (B, S, 1)
        
        # Compute the timesteps 
        time_step = torch.diff(times, dim=1)
        time_step = torch.cat((time_step, time_step[:, 0].unsqueeze(1)), dim=1) # (B, S, 1)

        # Stack the timestep to the rangemeter data 
        tensor_ext_range = torch.cat((tensor_range, time_step))  # (B, S, 2)

        # Stack the timestep to the IMU data 
        tensor_ext_imu = torch.cat((tensor_imu, time_step)) # (B, S, 7)
        
        # Compute the range embeddings
        feat_range = self.proj_range(tensor_ext_range) # (B, S, 16)
        
        # Compute the IMU embeddings 
        feat_imu = self.proj_imu(tensor_ext_imu) # (B, S, 32)
        
        # TODO: add event-frame encoder 
        
        # TODO: add transformer fuser 
        
        # TODO: add final regressor
        output= self.regressor_head()
        
        return {
            'prediction': output,
        }