
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

from elope.utils import load_yaml

class BaseMLP(nn.Module): 
    """Base MLP layer for basic feature expansion.""" 
    
    def __init__(
        self, 
        input_dim: int = 2,
        hidden_dim: int = 32,
        num_layers: int = 1,
        activation: nn.Module = nn.ReLU(), 
        norm: bool=True,
        dropout: float = 0.1, 
    )
        # Initialize the base class 
        super().__init__() 
        
        c1, c2 = input_dim, hidden_dim
        
        # Add FC layer
        self.layers = []
        for _ in num_layers: 
            
            layer = nn.Module(
                nn.Linear()
            )
            
             
        self.layer = nn.Linear(input_dim, hidden_dim)
        
        # Add the activation function 
        self.act = activation
        
        # Create dropout and layer norms
        self.bn1 = nn.LayerNorm(hidden_dim) if norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity() 
    

class MultiModalTransformerEstimator(nn.Module):
    """ Multi-modal transformer-based network with better regularization."""
    
    def __init__(
        self, 
        dropout: float = 0.1,
    ):

        super().__init__()

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
        
        # TODO: add rangemeter embedding 
        # TODO: add IMU embedding 
        # TODO: add event-frame encoder 
        
        # TODO: add transformer fuser 
        # TODO: add final regressor
        
        return {
            'prediction': output,
        }