
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from pathlib import Path

from elope.utils import load_yaml, angles_to_dcm
from elope.models.emmnet.emmnetOf import OpticalFlowHead, EventWarper



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
    
class GRURangemeterEncoder(nn.Module):
    """GRU-based rangemeter encoder with attention pooling"""
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, 
                 output_dim: int = 16, num_layers: int = 2, dropout: float = 0.1):
        """
        Initialize the GRU-based rangemeter encoder with attention pooling.

        Args:
        input_dim (int, optional): Input feature dimension (default: 1)
        hidden_dim (int, optional): GRU hidden dimension (default: 32)
        output_dim (int, optional): Output feature dimension (default: 16)
        num_layers (int, optional): Number of GRU layers (default: 2)
        dropout (float, optional): Dropout probability (default: 0.1)
        """
        super().__init__()
        
        # GRU layers with bidirectional processing
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Bidirectional for better context
        )
        
        # Attention mechanism for temporal pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # Output a single attention weight
        )
        
        # Output projection with regularization
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim)
        )
    
    def forward(self, x):
        # Input shape: (B, T, 1)
        gru_out, _ = self.gru(x)  # (B, T, hidden_dim*2)
        
        # Attention-based pooling
        attention_weights = self.attention(gru_out)  # (B, T, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        attended = torch.sum(gru_out * attention_weights, dim=1)  # (B, hidden_dim*2)
        
        # Final projection
        output = self.output_proj(attended)
        
        return output


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
        output_dim: int = 128, 
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
        self.avgpool = nn.AdaptiveAvgPool3d((1, 2, 2))
        
        # The output is flattened and passed through a fully connected layer
        # with a dropout layer to reduce overfitting
        self.fc = nn.Sequential(
            nn.Dropout(dropout * 2),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)  # Layer norm for better stability
        )
    
    
    def _make_layer(
        self, cin: int, cout: int, num_blocks: int, stride: int = 1, dropout: float = 0.1
    ):

        layers = [ResNet3DBlock(cin, cout, stride, dropout)]
        for _ in range(1, num_blocks):
            # The dropout in the ResNet block is the same as the input dropout
            layers.append(ResNet3DBlock(cout, cout, dropout=dropout))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)         # [B, 512, 1, 2, 2]
        # Only remove the temporal dimension; keep batch even when B=1.
        x = x.squeeze(2)            # [B, 512, 2, 2]
        x = x.permute(0, 2, 3, 1)   # [B, 2, 2, 512] 
        
        x = self.fc(x)              # [B, 2, 2, H]
        
        B, _, _, H = x.shape
        x = x.reshape(B, -1, H)     # [B, 4, H]
        
        return x

class CrossModalAttention(nn.Module):

    def __init__(
        self, 
        dim_angle: int, 
        dim_omega: int, 
        dim_range: int, 
        dim_event: int, 
        hidden_dim: int = 256, 
        dropout: float = 0.1
    ):
        
        super().__init__()
        
        # Feature projections with layer normalization
        self.event_proj = nn.Sequential(
            nn.Linear(dim_event, hidden_dim),
            # Normalize the projected features
            nn.LayerNorm(hidden_dim)
        )
        self.w_proj = nn.Sequential(
            nn.Linear(dim_omega, hidden_dim),
            # Normalize the projected features
            nn.LayerNorm(hidden_dim)
        )
        
        self.angles_proj = nn.Sequential(
            nn.Linear(dim_angle, hidden_dim), 
            nn.LayerNorm(hidden_dim),
        )
        
        self.range_proj = nn.Sequential(
            nn.Linear(dim_range, hidden_dim),
            # Normalize the projected features
            nn.LayerNorm(hidden_dim)
        )

        # Multi-head attention with residual connections
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=False
        )
        
        # Feed-forward network with residual connections
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            # GELU activation function
            nn.GELU(),
            # Dropout regularization
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Layer normalization for stabilizing the learning process
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout regularization
        self.dropout = nn.Dropout(dropout)

        # Learnable weights for each modality
        #self.modal_weights = nn.Parameter(torch.ones(3)) # Learnable weights for each modality

    def forward(self, event_feat, angle_feat, imu_feat, range_feat):

        # Project features
        e_proj = self.event_proj(event_feat)        # [B, 16, H]
        e_proj = e_proj.permute(1, 0, 2)            # [16, B, H]     
        
        i_proj = self.w_proj(imu_feat).unsqueeze(0)              # [1, B, H]
        r_proj = self.range_proj(range_feat).unsqueeze(0)        # [1, B, H]
        a_proj = self.angles_proj(angle_feat).unsqueeze(0)       # [1, B, H]
        
        # Stack for attention (seq_len=3, batch, hidden_dim)
        features = torch.cat([e_proj, i_proj, r_proj, a_proj], dim=0)

        # Self-attention with residual
        attended, attention_weights = self.attention(features, features, features)
        attended = self.norm1(attended + features)
        
        # Feed-forward with residual
        ffn_out = self.ffn(attended)
        ffn_out = self.norm2(ffn_out + attended)
        
        # Use learnable weights if available, otherwise simple average
        if hasattr(self, 'modal_weights'):
            weights = F.softmax(self.modal_weights, dim=0)  # (3,)
            fused = torch.sum(ffn_out * weights.view(3, 1, 1), dim=0)  # Weighted sum
        else:
            fused = torch.mean(ffn_out, dim=0)  # Simple average fallback
        
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


class MultiModalVelocityEstimatorAnglesOF(nn.Module):
    
    def __init__(
        self, 
        event_channels: int = 2,
        event_output_dim: int = 256,
        channels_angle: list = [32, 64, 64], 
        channels_omega: list = [32, 64, 64],
        channels_range: list = [32, 64, 64], 
        output_dim: int = 3,
        dropout: float = 0.15,
        flow_aux: bool = False,
        flow_height: int = 200,
        flow_width: int = 200,
        flow_from_events: bool = True,
    ):

        super().__init__()
        
        # Initialize  encoders with dropout for regularization
        self.encoder_event = EventEncoder(event_channels, event_output_dim, dropout)
        
        self.encoder_omega = BaseEmbedding(
            input_dim=3, 
            hidden_dim=channels_omega, 
            activation=nn.ReLU(), 
            norm=True, 
            dropout=dropout
        )
        
        self.encoder_angle = BaseEmbedding(
            input_dim=3,
            hidden_dim=channels_angle, 
            activation=nn.GELU(), 
            norm=True, 
            dropout=dropout
        )
        
        # self.encoder_range = BaseEmbedding(
        #     input_dim=1, 
        #     hidden_dim=channels_range, 
        #     activation=nn.Identity(), 
        #     norm=False, 
        #     dropout=dropout
        # )

        channels_range = [16]
        self.encoder_range = GRURangemeterEncoder(
            output_dim=channels_range[-1],
            dropout=dropout
        )

        # Use cross-modal attention mechanism for feature fusion
        self.fusion = CrossModalAttention(
            dim_angle=channels_angle[-1],
            dim_omega=channels_omega[-1], 
            dim_range=channels_range[-1], 
            dim_event=event_output_dim,
            dropout=dropout, 
        )
        
        fusion_input_dim = 256  # Must match CrossModalAttention hidden_dim

        # Initialize regularized regressor for final prediction
        self.regressor = RegularizedRegressor(fusion_input_dim, output_dim, dropout)

        # Optional flow head for self-supervised event warping
        self.flow_aux = bool(flow_aux)
        self.flow_from_events = bool(flow_from_events)
        if self.flow_aux:
            flow_in_dim = event_output_dim if self.flow_from_events else fusion_input_dim
            self.flow_head = OpticalFlowHead(
                flow_in_dim, flow_height=flow_height, flow_width=flow_width, dropout=dropout
            )
            self.event_warper = EventWarper()
        
        # Initialize weights to prevent overfitting
        self._init_weights()
    
    @staticmethod 
    def create_model(cfg: str | Path | dict, device: str="cpu", **kwargs):
        """Factory function to create the improved model"""
        
        # Retrieve the model configuration
        if isinstance(cfg, (str, Path)): 
            cfg = load_yaml(cfg)
            
        # assert cfg["output_type"] == "central_state"
        
        model = MultiModalVelocityEstimatorAnglesOF(
            dropout=float(cfg["dropout"]),
            channels_angle=cfg["channels_angle"], 
            channels_omega=cfg["channels_omega"], 
            channels_range=cfg["channels_range"],
            flow_aux=bool(cfg.get("flow_aux", False)),
            flow_height=int(cfg.get("flow_height", 200)),
            flow_width=int(cfg.get("flow_width", 200)),
            flow_from_events=bool(cfg.get("flow_from_events", True)),
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
        tensor_event = event_tensor[:, -1]    # (B, 2, C, H, W)
        tensor_angle = imu_tensor[:, -1, 0:3] # (B, 3)
        tensor_omega = imu_tensor[:, -1, 3:6] # (B, 3)
        tensor_range = range_tensor[:]
        
        # Extract features
        feat_event = self.encoder_event(tensor_event)
        feat_angle = self.encoder_angle(tensor_angle)
        feat_omega = self.encoder_omega(tensor_omega)
        feat_range = self.encoder_range(tensor_range)
        
        # Fusion
        fused_feat, attention_weights = self.fusion(
            feat_event, feat_angle, feat_omega, feat_range
        )
       
        # Final prediction
        output = self.regressor(fused_feat)
        flow_pred = None
        if self.flow_aux:
            if self.flow_from_events:
                # Pool event tokens into a single vector for flow prediction
                flow_feat = feat_event.mean(dim=1)  # (B, H)
                flow_pred = self.flow_head(flow_feat)
            else:
                flow_pred = self.flow_head(fused_feat)
        
        # Compute DCM rotation from camera to inertial frame
        dcm = angles_to_dcm(tensor_angle) # (B, 3, 3)
        
        # Output has shape (B, 3)
        output = output.unsqueeze(-1)   # (B, 3, 1)
        output = (dcm@output).squeeze() # (B, 3)
        
        return {
            'prediction': output,
            'attention_weights': attention_weights,
            'flow_prediction': flow_pred,
        }
