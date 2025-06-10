import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


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

class ResNet3DBlock(nn.Module):
    """ 3D ResNet block with better regularization"""
    """
     3D ResNet block with better regularization

    Args:
    in_channels (int): The number of input channels
    out_channels (int): The number of output channels
    stride (int): The stride of the convolutional layers. Defaults to 1.
    dropout (float): The dropout probability. Defaults to 0.1.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        # Convolutional layer 1
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.dropout1 = nn.Dropout3d(dropout)
        
        # Convolutional layer 2
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout2 = nn.Dropout3d(dropout)
        
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
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class EventEncoder(nn.Module):
    """ event encoder with better regularization"""
    """
     event encoder with better regularization

    Args:
    input_channels (int): The number of input channels. Defaults to 2.
    output_dim (int): The number of output dimensions. Defaults to 128.
    dropout (float): The dropout probability. Defaults to 0.15.
    """
    def __init__(self, input_channels: int = 2, output_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        
        # Initial convolution with stochastic depth
        # The kernel size is (3,7,7) to capture the spatial and temporal
        # information in the event data
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=(2,5,5),
                              stride=(1,1,1), padding=(0,2,2), bias=False)
        # Batch normalization to normalize the output of the convolution
        self.bn1 = nn.BatchNorm3d(64)
        
        # Spatial-temporal attention before pooling
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1),  
            nn.Sigmoid()
        )   

        # Max pooling to downsample the data
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        
        # ResNet blocks with increased dropout
        # The dropout probability is increased to 0.2 to reduce overfitting
        self.layer1 = self._make_layer(64, 64, 2, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(64, 128, 2, stride=(1,2,2), dropout=dropout)
        self.layer3 = self._make_layer(128, 256, 2, stride=(1,2,2), dropout=dropout)
        self.layer4 = self._make_layer(256, 512, 2, stride=(1,2,2), dropout=dropout)
        
        # Temporal attention mechanism
        self.temporal_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),  # Pool spatial dims only
            nn.Conv1d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, 512, kernel_size=1),
            nn.Sigmoid()
        )
        # Global pooling and projection with regularization
        # The AdaptiveAvgPool3d is used to reduce the spatial and temporal
        # dimensions to 1
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # The output is flattened and passed through a fully connected layer
        # with a dropout layer to reduce overfitting
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, 
                   stride: int = 1, dropout: float = 0.1):
        layers = [ResNet3DBlock(in_channels, out_channels, stride, dropout)]
        for _ in range(1, num_blocks):
            layers.append(ResNet3DBlock(out_channels, out_channels, dropout=dropout))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Spatial attention
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att
        
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Temporal attention
        B, C, T, H, W = x.shape
        if T > 1:
            # Reshape for temporal attention
            temp_x = x.view(B, C, T, H*W).mean(dim=-1)  # (B, C, T)
            temp_att = self.temporal_attention(temp_x)  # (B, C, T)
            temp_att = temp_att.unsqueeze(-1).unsqueeze(-1)  # (B, C, T, 1, 1)
            x = x * temp_att
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

class RangemeterEncoder(nn.Module):
    """Enhanced rangemeter encoder with velocity estimation and physics constraints"""
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, output_dim: int = 32, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        # Add range velocity (derivative) as additional feature
        self.range_processor = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),  # [range, range_velocity]
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Bidirectional GRU for temporal modeling
        self.gru = nn.GRU(
            hidden_dim // 2, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Multi-head attention for temporal relationships
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # Physics-informed layers
        self.physics_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2 + hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # Input shape: (B, T, 1) - range measurements
        B, T, _ = x.shape
        
        # Compute range velocity (numerical derivative)
        if T > 1:
            range_vel = torch.zeros_like(x)
            range_vel[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
            # For the first timestep, use the same velocity as the second
            range_vel[:, 0, :] = range_vel[:, 1, :]
        else:
            range_vel = torch.zeros_like(x)
        
        # Combine range and range velocity
        range_features = torch.cat([x, range_vel], dim=-1)  # (B, T, 2)
        
        # Process range features
        processed_range = self.range_processor(range_features)  # (B, T, hidden_dim//2)
        
        # GRU processing
        gru_out, _ = self.gru(processed_range)  # (B, T, hidden_dim*2)
        
        # Self-attention
        attended, _ = self.attention(gru_out, gru_out, gru_out)  # (B, T, hidden_dim*2)
        
        # Global average pooling
        temporal_features = attended.mean(dim=1)  # (B, hidden_dim*2)
        
        # Physics-informed processing
        physics_features = self.physics_layer(temporal_features)  # (B, hidden_dim//2)
        
        # Combine all features
        combined = torch.cat([temporal_features, physics_features], dim=-1)
        
        # Final projection
        output = self.output_proj(combined)
        
        return output

class IMUEncoder(nn.Module):
    """IMU encoder with better temporal modeling and physics awareness"""
    def __init__(self, input_dim: int = 6, d_model: int = 128, output_dim: int = 64,
                 n_heads: int = 8, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Separate processing for orientation and angular velocity
        self.orientation_proj = nn.Linear(3, d_model // 2)  # phi, theta, psi
        self.angular_vel_proj = nn.Linear(3, d_model // 2)  # p, q, r
        
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Enhanced transformer with more layers and heads
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Multi-scale temporal pooling
        self.temporal_pools = nn.ModuleList([
            nn.AdaptiveAvgPool1d(1),
            nn.AdaptiveMaxPool1d(1),
            nn.AdaptiveAvgPool1d(5)  # Multi-resolution
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model * 3),  # 3 pooling methods
            nn.Dropout(dropout),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, x):
        # Input shape: (B, T, 6) - [phi, theta, psi, p, q, r]
        B, T, _ = x.shape
        
        # Separate orientation and angular velocity
        orientation = x[:, :, :3]  # phi, theta, psi
        angular_vel = x[:, :, 3:]  # p, q, r
        
        # Project separately and combine
        orient_proj = self.orientation_proj(orientation) * math.sqrt(self.d_model // 2)
        angular_proj = self.angular_vel_proj(angular_vel) * math.sqrt(self.d_model // 2)
        
        # Combine features
        x_combined = torch.cat([orient_proj, angular_proj], dim=-1)  # (B, T, d_model)
        
        # Add positional encoding
        x_combined = x_combined.transpose(0, 1)  # (T, B, d_model)
        x_combined = self.pos_encoding(x_combined)
        x_combined = x_combined.transpose(0, 1)  # (B, T, d_model)
        
        # Transformer processing
        x_transformed = self.transformer(x_combined)  # (B, T, d_model)
        
        # Multi-scale temporal pooling
        pooled_features = []
        for pool in self.temporal_pools:
            # Transpose for 1D pooling: (B, d_model, T)
            pooled = pool(x_transformed.transpose(1, 2))
            if pooled.size(-1) > 1:
                pooled = pooled.mean(dim=-1)  # Average multiple outputs
            else:
                pooled = pooled.squeeze(-1)
            pooled_features.append(pooled)
        
        # Combine all pooled features
        combined_features = torch.cat(pooled_features, dim=-1)
        
        # Final projection
        output = self.output_proj(combined_features)
        
        return output

class AdaptiveCrossModalFusion(nn.Module):
    """Adaptive fusion with learnable modality weights and gating"""
    def __init__(self, event_dim: int, imu_dim: int, range_dim: int,
                 hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        
        # Feature projections
        self.event_proj = nn.Sequential(
            nn.Linear(event_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.imu_proj = nn.Sequential(
            nn.Linear(imu_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.range_proj = nn.Sequential(
            nn.Linear(range_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Learnable modality importance weights
        self.modality_weights = nn.Parameter(torch.ones(3))
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # Gating mechanism for adaptive fusion
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )
        
        # Final fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, event_feat, imu_feat, range_feat):
        # Project all features to same dimension
        e_proj = self.event_proj(event_feat)
        i_proj = self.imu_proj(imu_feat)
        r_proj = self.range_proj(range_feat)
        
        # Apply learnable modality weights
        weights = F.softmax(self.modality_weights, dim=0)
        e_proj = e_proj * weights[0]
        i_proj = i_proj * weights[1]
        r_proj = r_proj * weights[2]
        
        # Cross-modal attention
        features = torch.stack([e_proj, i_proj, r_proj], dim=1)  # (B, 3, hidden_dim)
        attended, attention_weights = self.cross_attention(features, features, features)
        
        # Flatten for gating
        attended_flat = attended.reshape(attended.size(0), -1)  # (B, 3*hidden_dim)
        
        # Adaptive gating
        gates = self.gate_network(attended_flat)  # (B, 3)
        
        # Apply gates to attended features
        gated_features = attended * gates.unsqueeze(-1)  # (B, 3, hidden_dim)
        gated_flat = gated_features.reshape(gated_features.size(0), -1) # (B, 3*hidden_dim)
        
        # Final fusion
        fused = self.fusion_layers(gated_flat)
        
        return fused, attention_weights, gates

class MultiScaleRegressor(nn.Module):
    """Multi-scale regressor with residual connections and uncertainty estimation"""
    def __init__(self, input_dim: int, output_dim: int = 3, dropout: float = 0.2):
        super().__init__()
        
        # Multi-scale processing branches
        self.branch1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.branch2 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        
        # Combine branches
        self.combiner = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Final prediction layers
        self.velocity_head = nn.Linear(128, output_dim)
        
        # Uncertainty estimation (optional)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, output_dim),
            nn.Softplus()  # Ensure positive uncertainty
        )

    def forward(self, x):
        # Multi-scale processing
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        
        # Combine branches
        combined = torch.cat([branch1_out, branch2_out], dim=-1)
        features = self.combiner(combined)
        
        # Predictions
        velocity = self.velocity_head(features)
        uncertainty = self.uncertainty_head(features)
        
        return velocity, uncertainty

class MultiModalVelocityEstimator(nn.Module):
    """Enhanced multi-modal velocity estimator with improved fusion and physics constraints"""
    def __init__(self,
                 event_channels: int = 2,
                 event_output_dim: int = 128,
                 imu_output_dim: int = 64,
                 range_output_dim: int = 32,
                 output_dim: int = 3,
                 dropout: float = 0.15):
        super().__init__()
        
        # Enhanced encoders
        self.event_encoder = EventEncoder(event_channels, event_output_dim, dropout)
        self.imu_encoder = IMUEncoder(output_dim=imu_output_dim, dropout=dropout)
        self.range_encoder = RangemeterEncoder(output_dim=range_output_dim, dropout=dropout)
        
        # Adaptive fusion
        self.fusion = AdaptiveCrossModalFusion(
            event_output_dim, imu_output_dim, range_output_dim, dropout=dropout
        )
        
        # Multi-scale regressor
        self.regressor = MultiScaleRegressor(256, output_dim, dropout)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.xavier_uniform_(module.weight, gain=0.8)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, event_tensor, imu_tensor, range_tensor):
        # Extract features from each modality
        event_feat = self.event_encoder(event_tensor)
        imu_feat = self.imu_encoder(imu_tensor)
        range_feat = self.range_encoder(range_tensor)
        
        # Adaptive fusion
        fused_feat, attention_weights, fusion_gates = self.fusion(event_feat, imu_feat, range_feat)
        
        # Final prediction with uncertainty
        velocity, uncertainty = self.regressor(fused_feat)
        
        return {
            'prediction': velocity,
            'uncertainty': uncertainty,
            'event_features': event_feat,
            'imu_features': imu_feat,
            'range_features': range_feat,
            'attention_weights': attention_weights,
            'fusion_gates': fusion_gates,
            'modality_weights': F.softmax(self.fusion.modality_weights, dim=0)
        }

def create_model(use_attention: bool = True, device: str = 'cpu', dropout: float = 0.15) -> MultiModalVelocityEstimator:
    """Factory function to create the enhanced model"""
    model = MultiModalVelocityEstimator(dropout=dropout)
    return model.to(device)