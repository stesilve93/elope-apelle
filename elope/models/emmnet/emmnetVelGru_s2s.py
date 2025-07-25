
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

from elope.utils import load_yaml

class PhysicsConsistencyGate(nn.Module):
    """
    Self-supervised physics gate that modulates features based on kinematic consistency
    """
    def __init__(self, hidden_dim: int = 64):
        """
        Initialize the PhysicsConsistencyGate module
        
        Parameters
        ----------
        hidden_dim : int, optional
            The number of hidden units in the physics consistency encoder, by default 64
        """
        super().__init__()
        
        # Learnable physics consistency encoder
        self.consistency_encoder = nn.Sequential(
            # Input residual vector (3 elements)
            nn.Linear(3, hidden_dim // 4),  
            # GELU activation function
            nn.GELU(),
            # Output consistency score (1 element)
            nn.Linear(hidden_dim // 4, 1),
            # Sigmoid activation function to output a score in [0, 1]
            nn.Sigmoid()  
        )
        
    def forward(self, angles, body_rates, times):
        """
        Calculate physics consistency and return gating weights
        """
        if angles.size(1) < 2:
            # Not enough temporal data for consistency check
            return torch.ones(angles.size(0), 1, device=angles.device)
        
        # Calculate actual angle derivatives
        angle_derivatives = torch.diff(angles, dim=1)  # (B, T-1, 3)
        time_diff = torch.diff(times, dim=1)  # (B, 1)
        angle_derivatives = angle_derivatives / time_diff
        
        # Calculate expected derivatives from kinematics
        expected_derivatives = self._body_to_world_rates(
            angles[:, :-1], body_rates[:, :-1]
        )  # (B, T-1, 3)
        
        # Physics residual (how much they disagree)
        physics_residual = torch.abs(angle_derivatives - expected_derivatives)
        
        # Aggregate over time and angles
        residual_magnitude = torch.mean(physics_residual, dim=[0, 1, 2])  # (B,)
        
        # Convert to consistency score (high residual = low consistency)
        consistency_score = self.consistency_encoder(
            physics_residual.mean(dim=1)  # (B, 3)
        )  # (B, 1)
        
        return consistency_score, residual_magnitude
    
    def _body_to_world_rates(self, angles, body_rates):
        """Convert body rates to world frame angle derivatives"""
        phi, theta, psi = angles[..., 0], angles[..., 1], angles[..., 2]
        p, q, r = body_rates[..., 0], body_rates[..., 1], body_rates[..., 2]
        
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        cos_theta_safe = torch.clamp(cos_theta, min=1e-6)
        tan_theta = sin_theta / cos_theta_safe
        sec_theta = 1.0 / cos_theta_safe
        
        phi_dot = p + q * sin_phi * tan_theta + r * cos_phi * tan_theta
        theta_dot = q * cos_phi - r * sin_phi
        psi_dot = (q * sin_phi + r * cos_phi) * sec_theta
        
        return torch.stack([phi_dot, theta_dot, psi_dot], dim=-1)
    
class PhysicsModulatedTransformerLayer(nn.Module):
    """
    Transformer layer with physics-modulated attention
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize a single layer of the transformer model with physics-modulated attention.
        
        Args:
            d_model (int): The dimensionality of the model.
            n_heads (int): The number of attention heads.
            dropout (float): The dropout probability (default: 0.1).
        """
        super().__init__()
        
        # Standard self-attention
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Physics-modulated attention weights
        self.physics_attention_gate = nn.Sequential(
            # Project the consistency score to the model dimensionality
            nn.Linear(1, d_model),
            # Apply a sigmoid activation function to get a gating value in [0, 1]
            nn.Sigmoid()
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            # First linear layer
            nn.Linear(d_model, d_model * 2),
            # GELU activation function
            nn.GELU(),
            # Dropout
            nn.Dropout(dropout),
            # Second linear layer
            nn.Linear(d_model * 2, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, consistency_score):
        # Standard self-attention
        attn_out, _ = self.attention(x, x, x)
        
        # Physics-modulated attention gating
        physics_gate = self.physics_attention_gate(consistency_score).unsqueeze(1)  # (B, 1, d_model)
        
        # Apply physics modulation
        attn_out = attn_out * physics_gate
        
        # Residual connection
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class PhysicsAwareIMUEncoder(nn.Module):
    """
    Modified IMU encoder that outputs sequential predictions
    """
    def __init__(self, input_dim: int = 6, d_model: int = 64, output_dim: int = 32, 
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        """
        Initialize the physics-aware IMU encoder with a transformer architecture.

        Args:
            input_dim (int): The number of input features (default: 6).
            d_model (int): The dimension of the model (default: 64).
            output_dim (int): The dimension of the output features (default: 32).
            n_heads (int): The number of attention heads (default: 4).
            n_layers (int): The number of transformer layers (default: 2).
            dropout (float): The dropout probability (default: 0.1).
        """
        super().__init__()
        self.d_model = d_model
        
        # Physics consistency gate
        self.physics_gate = PhysicsConsistencyGate(d_model)
        
        # Input projection with physics-aware scaling
        self.input_proj = nn.Linear(input_dim, d_model)
        self.physics_proj = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer with physics-modulated attention
        self.transformer_layers = nn.ModuleList([
            PhysicsModulatedTransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Sequential output projection (no pooling)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )
        
        
    def forward(self, x, times):
        # Input shape: (B, T, 6) - [phi, theta, psi, p, q, r]
        angles = x[..., :3]
        body_rates = x[..., 3:]
        
        # Get physics consistency score
        consistency_score, kin_loss = self.physics_gate(angles, body_rates, times)  # (B, 1)
        
        # Project input features
        x_proj = self.input_proj(x) * math.sqrt(self.d_model)
        
        # Create physics-aware feature modulation
        physics_modulation = self.physics_proj(consistency_score).unsqueeze(1)  # (B, 1, d_model)
        
        # Apply physics modulation to input
        x_modulated = x_proj + physics_modulation
        
        # Add positional encoding
        x_modulated = x_modulated.transpose(0, 1)
        x_modulated = self.pos_encoding(x_modulated)
        x_modulated = x_modulated.transpose(0, 1)
        
        # Apply transformer layers with physics awareness
        for layer in self.transformer_layers:
            x_modulated = layer(x_modulated, consistency_score)
        
        # Sequential output (no temporal pooling): (B, T, output_dim)
        output = self.output_proj(x_modulated)
        
        return output, consistency_score, kin_loss



class KinematicConstraintLayer(nn.Module):
    """
    Layer that enforces kinematic constraints through feature rectification
    """
    def __init__(self, feature_dim: int):
        """
        Initialize the KinematicConstraintLayer.
        
        Args:
            feature_dim (int): Dimension of the input features.
        """
        super().__init__()
        
        # Learnable constraint enforcement
        self.constraint_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Tanh()  # Bounded correction
        )
        
        # Adaptive mixing parameter
        self.mixing_param = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, features, raw_imu_data, times):
        """
        Apply kinematic constraint rectification to features.

        Kinematic constraints are used to enforce the physical relationship
        between the angular velocities and the Euler angles.

        Args:
            features (torch.Tensor): The input features to rectify.
            raw_imu_data (torch.Tensor): The raw IMU data used to enforce
                the kinematic constraints.

        Returns:
            The corrected features after applying kinematic constraint
            rectification.
        """
        # Extract physics information
        angles = raw_imu_data[..., :3]
        body_rates = raw_imu_data[..., 3:]
        
        if angles.size(1) < 2:
            return features
        
        # Calculate kinematic consistency
        angle_derivatives = torch.diff(angles, dim=1)
        time_diff = torch.diff(times, dim=1)  # (B, 1)
        angle_derivatives = angle_derivatives / time_diff

        expected_derivatives = self._body_to_world_rates(
            angles[:, :-1], body_rates[:, :-1]
        )
        
        # Compute correction signal
        kinematic_residual = torch.mean(
            torch.abs(angle_derivatives - expected_derivatives), 
            dim=[1, 2]
        )  # (B,)
        
        # Generate feature correction
        correction = self.constraint_encoder(features)
        correction_weight = torch.sigmoid(-kinematic_residual).unsqueeze(-1)  # (B, 1)
        
        # Apply adaptive correction
        corrected_features = features + correction_weight * correction
        
        return corrected_features
    
    def _body_to_world_rates(self, angles: torch.Tensor, body_rates: torch.Tensor) -> torch.Tensor:
        """
        Convert body rates to world frame angle derivatives

        This function takes in the body rates and the Euler angles and returns
        the derivatives of the Euler angles with respect to time.

        Args:
            angles (torch.Tensor): A tensor of shape (B, T, 3) where B is the
                batch size, T is the sequence length, and 3 is the number of
                Euler angles.
            body_rates (torch.Tensor): A tensor of shape (B, T, 3) where B is
                the batch size, T is the sequence length, and 3 is the number of
                body rates.

        Returns:
            A tensor of shape (B, T, 3) where B is the batch size, T is the
            sequence length, and 3 is the number of Euler angle derivatives.
        """
        phi, theta, psi = angles[..., 0], angles[..., 1], angles[..., 2]
        p, q, r = body_rates[..., 0], body_rates[..., 1], body_rates[..., 2]
        
        # Calculate trigonometric terms
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        # Avoid division by zero
        cos_theta_safe = torch.clamp(cos_theta, min=1e-6)
        tan_theta = sin_theta / cos_theta_safe
        sec_theta = 1.0 / cos_theta_safe
        
        # Calculate world frame angle derivatives
        phi_dot = p + q * sin_phi * tan_theta + r * cos_phi * tan_theta
        theta_dot = q * cos_phi - r * sin_phi
        psi_dot = (q * sin_phi + r * cos_phi) * sec_theta
        
        return torch.stack([phi_dot, theta_dot, psi_dot], dim=-1)

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
    """
    3D ResNet block with regularization

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
    """
    Event encoder to process event tensors for each sequence timestamp.
    It combines a 3D CNN for spatio-temporal feature extraction within each IMU sequence interval
    and a GRU layer to process the sequence of these event features.
    """
    def __init__(self, input_channels: int = 2, output_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        
        # --- 3D CNN for per-IMUseq-timestamp event feature extraction ---
        # This part processes a single event sub-sequence (C, T_sub_event, H, W)
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=(3,7,7), 
                              stride=(1,2,2), padding=(1,3,3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(64, 128, 2, stride=(1,2,2), dropout=dropout)
        self.layer3 = self._make_layer(128, 256, 2, stride=(1,2,2), dropout=dropout)
        self.layer4 = self._make_layer(256, 512, 2, stride=(1,2,2), dropout=dropout)
        
        # Pools only spatial dimensions, collapses T_sub_event if it's not already 1
        # This needs to yield a single feature vector *per IMU timestamp* for the GRU.
        self.spatial_temporal_pool = nn.AdaptiveAvgPool3d((1, 1, 1)) # Pool down to 1x1x1 for each feature map
        
        # Linear layer to project the CNN output to GRU's input dimension
        self.cnn_output_proj = nn.Linear(512, output_dim)
        
        # --- GRU Layer for processing the sequence of event features ---
        # input_size: output_dim from the CNN after projection
        # hidden_size: output_dim for the EventEncoder
        self.gru = nn.GRU(
            input_size=output_dim, 
            hidden_size=output_dim, 
            num_layers=1, # GRU layer
            batch_first=True, # Input/output will be (B, T_imu, feature_dim)
            dropout=dropout # Apply dropout if needed
        )
        
        # Layer norm after GRU
        self.norm_gru = nn.LayerNorm(output_dim)
        self.dropout_gru = nn.Dropout(dropout)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, 
                   stride: int | tuple = 1, dropout: float = 0.1): 
        layers = [ResNet3DBlock(in_channels, out_channels, stride, dropout)]
        for _ in range(1, num_blocks):
            layers.append(ResNet3DBlock(out_channels, out_channels, dropout=dropout))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x shape: (B, T_imu, C, T_sub_event, H, W)
        B, T_seq, C, T_sub_event, H, W = x.shape
        
        # Reshape for CNN processing: (B * T_imu, C, T_sub_event, H, W)
        x_reshaped = x.view(B * T_seq, C, T_sub_event, H, W)
        
        # --- CNN Feature Extraction (per IMU timestamp's event data) ---
        cnn_out = F.relu(self.bn1(self.conv1(x_reshaped)))
        cnn_out = self.maxpool(cnn_out)
        
        cnn_out = self.layer1(cnn_out)
        cnn_out = self.layer2(cnn_out)
        cnn_out = self.layer3(cnn_out)
        cnn_out = self.layer4(cnn_out)
        
        # Pool down the CNN output for each (B*T_imu) entry
        # From (B * T_imu, 512, T_processed, H_processed, W_processed) to (B * T_imu, 512, 1, 1, 1)
        cnn_out = self.spatial_temporal_pool(cnn_out) 
        
        # Squeeze to (B * T_imu, 512) and project to desired dimension
        cnn_out = cnn_out.squeeze(-1).squeeze(-1).squeeze(-1) # (B * T_imu, 512)
        cnn_features_per_imu_timestep = self.cnn_output_proj(cnn_out) # (B * T_imu, output_dim)
        
        # --- GRU Sequence Processing ---
        # Reshape for GRU: (B, T_imu, output_dim)
        gru_input = cnn_features_per_imu_timestep.view(B, T_seq, -1)
        
        # Pass through GRU
        gru_output, _ = self.gru(gru_input) # (B, T_imu, output_dim)
        
        # Apply layer norm and dropout
        output = self.norm_gru(gru_output)
        output = self.dropout_gru(output)
        
        return output


class TransformerIMUEncoder(nn.Module):
    """
    Modified Transformer-based IMU encoder for sequential output
    """
    def __init__(self, input_dim: int = 6, d_model: int = 64, output_dim: int = 32, 
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
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
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Sequential output projection (no temporal pooling)
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
        
        # Sequential output (no temporal pooling): (B, T, output_dim)
        x = self.output_proj(x)
        
        return x


class GRURangemeterEncoder(nn.Module):
    """Modified GRU-based rangemeter encoder for sequential output"""
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, 
                 output_dim: int = 16, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        # GRU layers with bidirectional processing
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Sequential output projection (no temporal pooling)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim)
        )
    
    def forward(self, x):
        # Input shape: (B, T, 1)
        gru_out, _ = self.gru(x)  # (B, T, hidden_dim*2)
        
        # Sequential output projection: (B, T, output_dim)
        output = self.output_proj(gru_out)
        
        return output


class CrossModalAttention(nn.Module):
    """
    Modified cross-modal attention for sequential fusion
    """
    def __init__(self, event_dim: int, imu_dim: int, range_dim: int, 
                 hidden_dim: int = 128, dropout: float = 0.1, use_physics_aware: bool = False):
        super().__init__()
        
        # Feature projections
        self.event_proj = nn.Sequential(
            nn.Linear(event_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.imu_proj = nn.Sequential(
            nn.Linear(imu_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.range_proj = nn.Sequential(
            nn.Linear(range_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.use_physics_aware = use_physics_aware
        if self.use_physics_aware:
            # Physics-informed modality weighting
            self.modality_weighting = nn.Sequential(
                nn.Linear(1, 16),
                nn.GELU(),
                nn.Linear(16, 3),
                nn.Softmax(dim=-1)
            )

        # Multi-head attention for each timestep
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=False
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, event_feat, imu_feat, range_feat, consistency_score):
        # Input shapes: (B, T, feature_dim)
        B, T, _ = event_feat.shape
        
        # Project features for each timestep
        e_proj = self.event_proj(event_feat)  # (B, T, hidden_dim)
        i_proj = self.imu_proj(imu_feat)      # (B, T, hidden_dim)
        r_proj = self.range_proj(range_feat)  # (B, T, hidden_dim)
        
        # Prepare for attention: (T, B*3, hidden_dim)
        # Stack modalities for each timestep
        features = torch.stack([e_proj, i_proj, r_proj], dim=2)  # (B, T, 3, hidden_dim)
        features = features.permute(1, 0, 2, 3).reshape(T, B*3, -1)  # (T, B*3, hidden_dim)

        if self.use_physics_aware:
            # Physics-informed modality weighting
            modality_weights = self.modality_weighting(consistency_score)  # (B, 3)
            # Expand for all timesteps: (T, B, 3, 1)
            modality_weights = modality_weights.unsqueeze(0).unsqueeze(-1).expand(T, B, 3, 1)
            modality_weights = modality_weights.reshape(T, B*3, 1)
            
            # Apply physics-informed weighting
            weighted_features = features * modality_weights
            
            # Self-attention
            attended, attention_weights = self.attention(
                weighted_features, weighted_features, weighted_features
            )
            attended = self.norm1(attended + weighted_features)
        else:
            # Self-attention with residual
            attended, attention_weights = self.attention(features, features, features)
            attended = self.norm1(attended + features)
        
        # Feed-forward with residual
        ffn_out = self.ffn(attended)
        ffn_out = self.norm2(ffn_out + attended)
        
        # Reshape back and fuse modalities: (T, B*3, hidden_dim) -> (B, T, hidden_dim)
        ffn_out = ffn_out.reshape(T, B, 3, -1)  # (T, B, 3, hidden_dim)
        fused = torch.mean(ffn_out, dim=2)      # (T, B, hidden_dim)
        fused = fused.transpose(0, 1)           # (B, T, hidden_dim)
        
        return fused, attention_weights


class RegularizedRegressor(nn.Module):
    """
    Modified regressor for sequential output
    """
    def __init__(self, input_dim: int, output_dim: int = 3, dropout: float = 0.3):
        super().__init__()
        
        # Sequential layers (applied to each timestep independently)
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, 128),
            nn.Linear(128, 64),
            nn.Linear(64, output_dim)
        ])
        
        # Batch normalization
        self.norms = nn.ModuleList([
            nn.LayerNorm(128),
            nn.LayerNorm(64)
        ])
        
        # Dropout regularization
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout),
            nn.Dropout(dropout * 0.7)
        ])
        
        # Skip connection projection
        self.skip_proj = nn.Linear(input_dim, 64) if input_dim != 64 else nn.Identity()
        
    def forward(self, x):
        # Input shape: (B, T, input_dim)
        # First layer
        out = F.gelu(self.norms[0](self.layers[0](x)))
        out = self.dropouts[0](out)
        
        # Second layer with skip connection
        identity = self.skip_proj(x)
        out = F.gelu(self.norms[1](self.layers[1](out)))
        out = self.dropouts[1](out + identity)
        
        # Final layer: (B, T, output_dim)
        out = self.layers[2](out)
        
        return out


class MultiModalVelocityEstimatorS2S(nn.Module):
    """
    Modified multi-modal network for sequential output
    """
    def __init__(self, 
                 event_channels: int = 2,
                 event_output_dim: int = 128,
                 imu_output_dim: int = 32,
                 range_output_dim: int = 16,
                 use_attention: bool = True,
                 output_dim: int = 3,
                 dropout: float = 0.15,
                 use_physics_aware: bool = True):
        super().__init__()
        
        # Initialize encoders (modified for sequential output)
        self.event_encoder = EventEncoder(event_channels, event_output_dim, dropout)
        if use_physics_aware:
            self.imu_encoder = PhysicsAwareIMUEncoder(output_dim=imu_output_dim, dropout=dropout)
        else:
            self.imu_encoder = TransformerIMUEncoder(output_dim=imu_output_dim, dropout=dropout)

        self.range_encoder = GRURangemeterEncoder(output_dim=range_output_dim, dropout=dropout)
        
        # Fusion strategy
        self.use_attention = use_attention
        self.use_physics_aware = use_physics_aware

        if use_attention:
            self.fusion = CrossModalAttention(
                event_output_dim, imu_output_dim, range_output_dim, 
                dropout=dropout, use_physics_aware=self.use_physics_aware
            )
            fusion_input_dim = 128
        else:
            fusion_input_dim = event_output_dim + imu_output_dim + range_output_dim

        if use_physics_aware:
            # Note: KinematicConstraintLayer might need modification for sequential processing
            self.kinematic_constraint = KinematicConstraintLayer(fusion_input_dim)

        # Sequential regressor
        self.regressor = RegularizedRegressor(fusion_input_dim, output_dim, dropout)
        
        # Initialize weights
        self._init_weights()
    
    @staticmethod 
    def create_model(cfg: str | Path | dict, device: str="cpu"):
        if isinstance(cfg, (str, Path)): 
            cfg = load_yaml(cfg)
        
        model = MultiModalVelocityEstimatorS2S(
            use_attention=bool(cfg["use_attention"]), 
            dropout=float(cfg["dropout"]),
            use_physics_aware=bool(cfg["physics_aware"])
        )
        
        return model.to(device)
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, times, event_tensor, imu_tensor, range_tensor):

        # Adjust times dimension
        times = times.unsqueeze(2)

        # Extract sequential features
        event_feat = self.event_encoder(event_tensor)  # (B, T, event_output_dim)
        if self.use_physics_aware:
            imu_feat, consistency_score, kin_loss = self.imu_encoder(imu_tensor, times)  # (B, T, imu_output_dim)
        else:
            imu_feat = self.imu_encoder(imu_tensor)
            consistency_score = torch.zeros(imu_tensor.size(0), 1, device=imu_tensor.device)
            kin_loss = 0
        range_feat = self.range_encoder(range_tensor)  # (B, T, range_output_dim)
        
        # Fusion
        if self.use_attention:
            fused_feat, attention_weights = self.fusion(event_feat, imu_feat, range_feat, consistency_score)
        else:
            fused_feat = torch.cat([event_feat, imu_feat, range_feat], dim=-1)  # Concatenate along feature dim
            attention_weights = None
        
        if self.use_physics_aware:
            # Note: This may need modification to work with sequential data
            # For now, we'll apply it to each timestep independently
            B, T, F = fused_feat.shape
            fused_flat = fused_feat.reshape(B*T, F)
            imu_flat = imu_tensor.reshape(B*T, -1)
            constrained_flat = self.kinematic_constraint(fused_flat, imu_flat, times)
            constrained_feat = constrained_flat.reshape(B, T, F)
            output = self.regressor(constrained_feat)
        else:             
            output = self.regressor(fused_feat)
        
        return {
            'prediction': output,  # (B, T, 3) - sequential predictions
            'event_features': event_feat,
            'imu_features': imu_feat,
            'range_features': range_feat,
            'attention_weights': attention_weights,
            'kin_loss': kin_loss if self.use_physics_aware else 0
        }
