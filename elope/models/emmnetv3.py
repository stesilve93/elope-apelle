
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

from elope.utils import load_yaml

from .blocks import ResNet, ResNet18, ResNet34
from .blocks import vAPE, tAPE, lPE

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
    
class ResNetEventEncoder(nn.Module): 
    
    def __init__(
        self, 
        resnet: str, 
        event_channels: int,
        output_dim: int, 
        dropout: float = 0.0,
    ):
        
        super().__init__() 
        
        # Store the ResNet model
        if resnet == "resnet-18": 
            self.encoder = ResNet18(event_channels)
        elif resnet == "resnet-34": 
            self.encoder = ResNet34(event_channels)
        else: 
            raise ValueError(f"`{resnet}` is not a supported ResNet model.")
        
        # Create a fully-connected layer to project the feature map. 
        self.fc = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(512, output_dim), 
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        
        # The input is a tensor of shape (B, T, C, H, W)
        B, T, C, H, W = x.shape
        
        # We first get rid of the time dimension to process each time frame independetly.
        x = x.reshape(-1, C, H, W)  # (B*T, C, H, W)
        
        # Run the input tensor through the encoder 
        feats, x = self.encoder(x) 
        
        # TODO: concatenate features, what we can do here is to use fully-connected 
        # layers to stack the feature map features together.
        
        # Reshape the output features to be (B, T, 512)
        x = x.reshape(B, T, -1)
        
        # Pass the encoder output through a fully connected layer 
        x = self.fc(x) # (B, T, output_dim)
        return x    

    
class SequenceTransfomer(nn.Module): 
    
    def __init__(
        self, 
        sequence_len: int,
        input_dim: int, 
        output_dim: int, 
        n_heads: int = 8, 
        n_layers: int = 2, 
        dropout: int = 0.1, 
        encoding: str = "tAPE"
    ):
        super().__init__()
        
        # Sequence length is the number of "states" (i.e., timesteps). `input_dim` is 
        # the size of the feature vector at each timestep.
        
        # Create the positional encoding 
        assert encoding in ("vAPE", "tAPE", "lPE")
        if encoding == "vAPE": 
            self.pos_encoding = vAPE(input_dim, sequence_len, dropout=dropout)
        elif encoding == "tAPE":
            self.pos_encoding = tAPE(input_dim, sequence_len, dropout=dropout)
        elif encoding == "lPE": 
            self.pos_encoding = lPE(input_dim, sequence_len, dropout=dropout)
            
        # Create the transformer encoder: 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=n_heads, 
            dim_feedforward=input_dim * 2, 
            dropout=dropout, 
            activation="gelu", 
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection with regularization if the sizes differ 
        self.output_proj = nn.Identity() 
        if output_dim != input_dim: 
            self.output_proj = nn.Sequential( 
                nn.Linear(input_dim, output_dim), 
                nn.LayerNorm(output_dim), 
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:         
        # Input shape has size (B, T, N)

        # Add positional encoding to the model 
        x = self.pos_encoding(x)   # (B, T, N)

        # Apply the transformer 
        x = self.transformer(x)    # (B, T, D)
        
        # Extract the features associated to the last state 
        x = x[:, -1]    # (B, D)
        
        # Final projection 
        x = self.output_proj(x) # (B, D)
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
        
        # Ensure the hidden dim is a list and retrieve layers
        hidden_dim = [hidden_dim] if isinstance(hidden_dim, int) else hidden_dim
        num_layers = len(hidden_dim)
        
        # Retrieve the dimenions of the first FC layer 
        c1 = input_dim
        
        # Add the FC layers 
        self.layers = nn.ModuleList()
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
        sequence_length: int,
        event_channels: int, 
        event_output_dim: int = 256,
        range_channels: list = [32, 8], 
        imu_channels: list = [64, 64, 32],
        resnet: str = "resnet-18",
        dropout: float = 0.1,
        stack_time: bool = True,
    ):

        super().__init__()
        
        self.stack_time = stack_time
        t_dim = 1 if self.stack_time else 0
        
        # Initialise the BaseEmbedding block for the rangemeter data 
        self.proj_range = BaseEmbedding(
            input_dim = 1 + t_dim, 
            hidden_dim = range_channels, 
            activation = nn.GELU(),
            norm = False, 
            dropout = dropout
        )
        
        # Initialise the BaseEmbedding block for the IMU data 
        self.proj_imu = BaseEmbedding(
            input_dim = 6 + t_dim, 
            hidden_dim = imu_channels,
            activation = nn.GELU(),
            norm = False, 
            dropout = dropout
        )
        
        # Create the encoder for the events tensor
        self.encoder_event = ResNetEventEncoder(
            resnet=resnet, event_channels=event_channels, output_dim=event_output_dim
        )
        
        # Total size of the sequence entering the transformer
        token_size = range_channels[-1] + imu_channels[-1] + event_output_dim
        transformer_out_dim = token_size
        
        self.transformer = SequenceTransfomer(
            sequence_len = sequence_length, 
            input_dim = token_size,
            output_dim = transformer_out_dim,
            dropout = dropout,
            encoding = "vAPE",
            n_heads = 4,
        )
        
        self.regressor_head = VelocityRegressor(
            input_dim = transformer_out_dim,
            output_dim = 3,
            hidden_dim=[128, 64],
            activation=nn.GELU(), 
            norm=False, 
            dropout = dropout,
        )

        # Initialize weights to prevent overfitting
        self._init_weights()
    
    @staticmethod 
    def create_model(
        cfg: str | Path | dict, 
        event_channels: int, 
        device: str="cpu", 
        **kwargs
    ):
        """Factory function to create the improved model"""
        
        # Retrieve the model configuration
        if isinstance(cfg, (str, Path)): 
            cfg = load_yaml(cfg)
            
        assert cfg["output_type"] == "final_state"
            
        cfg_arch = cfg["architecture"]
        model = MultiModalTransformerEstimator(
            sequence_length=int(cfg["sequence_length"]),
            resnet=cfg_arch["resnet_model"], 
            event_channels=2 * event_channels, # Include both polarities
            event_output_dim=int(cfg_arch["event_output_dim"]),
            range_channels=cfg_arch["range_channels"], 
            imu_channels=cfg_arch["imu_channels"],
            dropout=float(cfg["dropout"]), 
            stack_time=bool(cfg_arch["stack_time"]),
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
    
            elif isinstance(module, ResNet): 
                module.initialize()
    
    def forward(
        self, 
        times: torch.Tensor, 
        tensor_event: torch.Tensor, 
        tensor_imu: torch.Tensor, 
        tensor_range: torch.Tensor
    ) -> dict:
        
        # Times is a tensor of size (B, T)
        # Event is a tensor of size (B, T, 2, C, H, W)
        # IMU is a tensor of size (B, T, 6)
        # Rangemeter is a tensor of size (B, T, 1)
        
        # Adjust times dimension
        times = times.unsqueeze(2)  # (B, T, 1)

        if self.stack_time:
            
            # Compute the timesteps 
            time_step = torch.diff(times, dim=1)
            time_step = torch.cat((time_step, time_step[:, 0].unsqueeze(1)), dim=1) # (B, T, 1)

            # Stack the timestep to the rangemeter data 
            tensor_range = torch.cat([tensor_range, time_step], dim=-1)  # (B, T, 2)

            # Stack the timestep to the IMU data 
            tensor_imu = torch.cat([tensor_imu, time_step], dim=-1)      # (B, T, 7)
            
        # print("Range: ", tensor_range.shape)
        # print("IMU: ", tensor_imu.shape)
        
        # Reshape the event tensor to be (B, T, C, H, W)
        B, T, _, _, H, W = tensor_event.shape
        tensor_event = tensor_event.reshape(B, T, -1, H, W) 
        # print("Events:", tensor_event.shape, tensor_event.device)
                
        # Compute the range embeddings
        feat_range = self.proj_range(tensor_range) # (B, T, 8)
        # print("Feat range:" , feat_range.shape)
        
        # Compute the IMU embeddings 
        feat_imu = self.proj_imu(tensor_imu) # (B, T, 32)
        # print("Feat IMU:", feat_imu.shape)
        
        # Pass the event through the encoder and retrieve the feature vector. 
        feat_events = self.encoder_event(tensor_event) # (B, T, N)
        # print("Feat events: ", feat_events.shape)
        
        # Stack all the features according to their time-state 
        features = torch.cat([feat_range, feat_imu, feat_events], dim=-1)   # (B, T, F)
        # print("Stacked features: ", features.shape)
        
        # Pass the features through the transformer 
        features = self.transformer(features)   # (B, M)
        # print("Transformer output: ", features.shape)
        
        # Apply the final velocity regressor
        output = self.regressor_head(features)
        # print("Final output: ", output.shape)
        
        return {
            'prediction': output,
        }