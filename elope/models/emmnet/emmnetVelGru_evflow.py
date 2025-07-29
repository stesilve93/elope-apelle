
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

from elope.evflow import EVFlowNet
from elope.utils import load_yaml

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-based encoders"""
    """
    Positional encoding for transformer-based encoders
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize the PositionalEncoding module

        Args:
        d_model (int): The number of features in the model
        max_len (int, optional): The maximum length of the input sequence. Defaults to 5000.
        """
        super().__init__()
        # Create a tensor of shape (max_len, d_model) filled with zeros
        pe = torch.zeros(max_len, d_model)
        # Create a tensor of shape (max_len) with the position of each element
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Create a tensor of shape (d_model/2) with the divisors for the sine and cosine
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        # Fill the pe tensor with the sine and cosine of the position
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Reshape the tensor to (1, max_len, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Register the tensor as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class EVFLowNetEncoder(nn.Module):

    def __init__(
        self, 
        weights_path: str, 
        output_dim: int = 128, 
        dropout: float = 0.15
    ):
        # Initialize the base class
        super().__init__()
        
        # Initialize the EVFlowNet model
        self.evflownet = EVFlowNet(batch_norm=True)
        
        # Load the pre-trained model 
        assert weights_path.exists() == True 
        data = torch.load(weights_path)
        self.evflownet.load_state_dict(data)
        
        # Number of output channels at the first decoder layer
        cout = 1024
        
        # The output is flattened and passed through a fully connected layer
        # with a dropout layer to reduce overfitting
        self.fc = nn.Sequential(
            nn.Dropout(dropout * 2),
            nn.Linear(cout, 128),
            nn.LayerNorm(output_dim)  # Layer norm for better stability
        )
    
    def forward(self, x: torch.Tensor) -> dict:
                         
        # x is of shape (B, 2, C, H, W)
        B, _, C, H, W = x.shape

        # Invert the dimensions to have the channels (counts, stamps) on the first dim
        x = x.permute(0, 2, 1, 3, 4) # (B, C, 2, H, W)
        x = x.reshape(B, -1, H, W)   # (B, 2*C, H, W)
            
        # Upsample the images to ensure the shape matches the one expected from the model
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        
        # Run the EVFlowNet model on the input
        dict_flow = self.evflownet(x)
        
        # Extract the optical flow from the first decoder layer 
        feats = dict_flow["flow0"] # (B, 2, 32, 32)
        
        # Flatten the x-y coordinates
        feats = feats.reshape(B, 2, -1) # (B, 2, 1024)
        
        # Pass through the FC layer 
        feats = self.fc(feats) # (B, 2, 128)        
        return dict_flow, feats    


class TransformerIMUEncoder(nn.Module):

    def __init__(
        self, 
        input_dim: int = 6, 
        d_model: int = 64, 
        output_dim: int = 32, 
        n_heads: int = 4, 
        n_layers: int = 2, 
        dropout: float = 0.1
    ):
        
        super().__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',  # GELU often works better than ReLU
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection with regularization
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )
        
    def forward(self, x):
        # Input shape: (B, T, 6)
        B, T, _ = x.shape
        
        # Project to model dimension
        x = self.input_proj(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (T, B, d_model) for pos encoding
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # Back to (B, T, d_model)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global average pooling over time dimension
        x = x.mean(dim=1)  # (B, d_model)
        
        # Final projection
        x = self.output_proj(x)
        
        return x


class GRURangemeterEncoder(nn.Module):
    """GRU-based rangemeter encoder with attention pooling"""
    
    def __init__(
        self, 
        input_dim: int = 1, 
        hidden_dim: int = 32, 
        output_dim: int = 16, 
        num_layers: int = 2,
        dropout: float = 0.1
    ):

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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
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


class CrossModalAttention(nn.Module):

    def __init__(
        self, 
        event_dim: int, 
        imu_dim: int, 
        range_dim: int, 
        hidden_dim: int = 128, 
        dropout: float = 0.1, 
    ):
        super().__init__()
        
        # Feature projections with layer normalization
        self.event_proj = nn.Sequential(
            nn.Linear(event_dim, hidden_dim),
            # Normalize the projected features
            nn.LayerNorm(hidden_dim)
        )
        
        self.imu_proj = nn.Sequential(
            nn.Linear(imu_dim, hidden_dim),
            # Normalize the projected features
            nn.LayerNorm(hidden_dim)
        )
        
        self.range_proj = nn.Sequential(
            nn.Linear(range_dim, hidden_dim),
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
        #self.modal_weights = nn.Parameter(torch.ones(4)) # Learnable weights for each modality

        
    def forward(self, event_feat, imu_feat, range_feat) -> tuple:
        
        # Project features
        e_proj = self.event_proj(event_feat)    # (B, 2, D)
        i_proj = self.imu_proj(imu_feat)        # (B, D)
        r_proj = self.range_proj(range_feat)    # (B, D)
        
        # Stack for attention (seq_len=3, batch, hidden_dim)
        features = torch.stack([e_proj[:, 0], e_proj[:, 1], i_proj, r_proj], dim=0)

        # Self-attention with residual
        attended, attention_weights = self.attention(features, features, features)
        attended = self.norm1(attended + features)
        
        # Feed-forward with residual
        ffn_out = self.ffn(attended)
        ffn_out = self.norm2(ffn_out + attended)
        
        # Use learnable weights if available, otherwise simple average
        if hasattr(self, 'modal_weights'):
            weights = F.softmax(self.modal_weights, dim=0)  # (4,)
            fused = torch.sum(ffn_out * weights.view(4, 1, 1), dim=0)  # Weighted sum
        else:
            fused = torch.mean(ffn_out, dim=0)  # Simple average fallback
        
        return fused, attention_weights


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


class MultiModalVelocityEstimatorEVFlow(nn.Module):
    """ Multi-modal network with better regularization"""
    def __init__(
        self, 
        evflownet_weights: str, 
        event_output_dim: int = 128,
        imu_output_dim: int = 32,
        range_output_dim: int = 16,
        output_dim: int = 3,
        dropout: float = 0.15,
    ):

        super().__init__()
        
        # Initialize  encoders with dropout for regularization
        self.event_encoder = EVFLowNetEncoder(
            weights_path=evflownet_weights,
            output_dim=output_dim,
            dropout=dropout
        )
        
        # Initialize IMU encoder
        self.imu_encoder = TransformerIMUEncoder(
            output_dim=imu_output_dim, 
            dropout=dropout
        )
        
        # Initialize range encoder
        self.range_encoder = GRURangemeterEncoder(
            output_dim=range_output_dim, 
            dropout=dropout
        )

        # Use cross-modal attention mechanism for feature fusion
        self.fusion = CrossModalAttention(
            event_output_dim, 
            imu_output_dim, 
            range_output_dim, 
            dropout=dropout, 
        )
        
        fusion_input_dim = 128  # Hidden dimension for attention

        # Initialize regularized regressor for final prediction
        self.regressor = RegularizedRegressor(fusion_input_dim, output_dim, dropout)
        
        # Initialize weights to prevent overfitting
        self._init_weights()
    
    @staticmethod 
    def create_model(cfg: str | Path | dict, device: str="cpu", **kwargs):
        """Factory function to create the improved model"""
        
        # Retrieve the model configuration
        if isinstance(cfg, (str, Path)): 
            cfg = load_yaml(cfg)
        
        model = MultiModalVelocityEstimatorEVFlow(
            dropout=float(cfg["dropout"]),
            evflownet_weights=Path(cfg["evflownet_weights"]),
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
        
        # Keep only the events of the last state
        event_tensor = event_tensor[:, -1]
        
        # Adjust times dimension
        times = times.unsqueeze(2)

        # Extract features
        flow, event_feat = self.event_encoder(event_tensor)

        imu_feat = self.imu_encoder(imu_tensor)
        range_feat = self.range_encoder(range_tensor)
        
        # Fusion
        fused_feat, attention_weights = self.fusion(event_feat, imu_feat, range_feat)

        # Final prediction
        output = self.regressor(fused_feat)
        
        return {
            'prediction': output,
            'event_features': event_feat,
            'imu_features': imu_feat,
            'range_features': range_feat,
            'attention_weights': attention_weights,
            'optical_flow': flow,
        }
