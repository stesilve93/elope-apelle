import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpatialRangemeterFusion(nn.Module):
    """
    Fuse rangemeter depth information directly with event tensor
    """
    def __init__(self, event_channels=2, H=200, W=200):
        super().__init__()
        self.H = H
        self.W = W
        self.center_pixel = [H//2, W//2]  # [100, 100] for 200x200
        
        # Learnable spatial encoding for depth information
        self.depth_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # Spatial attention to weight pixels based on distance from center
        self.spatial_attention = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        
        # Project depth features to match event tensor channels
        self.depth_projection = nn.Conv2d(64, event_channels, kernel_size=1)
        
    def create_depth_map(self, depth_values, batch_size, T):
        """
        Create spatial depth maps from rangemeter measurements
        
        Args:
            depth_values: (B, seq_len, 1) - rangemeter measurements
            batch_size: batch size
            T: temporal dimension (should match event tensor)
        
        Returns:
            depth_maps: (B, T, H, W) - spatial depth representations
        """
        # Create Gaussian-like depth influence centered at [100,100]
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.H, device=depth_values.device),
            torch.arange(self.W, device=depth_values.device),
            indexing='ij'
        )
        
        # Distance from center pixel
        center_y, center_x = self.center_pixel
        distances = torch.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        
        # Create Gaussian weight (stronger at center, weaker at edges)
        sigma = min(self.H, self.W) / 6  # Adjust spread
        weights = torch.exp(-distances**2 / (2 * sigma**2))
        weights = weights.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # Handle temporal dimension mismatch by interpolating rangemeter data
        B, seq_len, _ = depth_values.shape
        if seq_len != T:
            # Interpolate rangemeter sequence to match event tensor temporal dimension
            depth_values_interp = F.interpolate(
                depth_values.transpose(1, 2),  # (B, 1, seq_len)
                size=T, 
                mode='linear', 
                align_corners=True
            ).transpose(1, 2)  # (B, T, 1)
        else:
            depth_values_interp = depth_values
        
        # Broadcast depth values across spatial dimensions
        depth_maps = []
        for t in range(T):
            depth_t = depth_values_interp[:, t, :]  # (B, 1)
            # Reshape and broadcast: (B, 1) -> (B, 1, H, W)
            depth_spatial = depth_t.unsqueeze(-1).unsqueeze(-1) * weights
            depth_maps.append(depth_spatial)
        
        # Stack temporal dimension: (B, T, H, W)
        depth_maps = torch.stack(depth_maps, dim=1)
        return depth_maps
    
    def forward(self, depth_sequence, target_temporal_dim):
        """
        Args:
            depth_sequence: (B, seq_len, 1) - rangemeter measurements
            target_temporal_dim: int - temporal dimension to match (from event tensor)
        
        Returns:
            depth_features: (B, event_channels, T, H, W) - spatial depth features
        """
        B, seq_len, _ = depth_sequence.shape
        T = target_temporal_dim  # Use the target temporal dimension of event time bins
        
        # Create spatial depth maps with matching temporal dimension
        depth_maps = self.create_depth_map(depth_sequence, B, T)  # (B, T, H, W)
        
        # Apply spatial attention
        depth_flat = depth_maps.view(B*T, 1, self.H, self.W)
        attention_weights = torch.sigmoid(self.spatial_attention(depth_flat))
        depth_attended = depth_flat * attention_weights
        depth_attended = depth_attended.view(B, T, self.H, self.W)
        
        # Embed depth values temporally (interpolate first if needed)
        if seq_len != T:
            depth_sequence_interp = F.interpolate(
                depth_sequence.transpose(1, 2),  # (B, 1, seq_len)
                size=T,
                mode='linear',
                align_corners=True
            ).transpose(1, 2)  # (B, T, 1)
        else:
            depth_sequence_interp = depth_sequence
            
        depth_embedded = self.depth_embed(depth_sequence_interp)  # (B, T, 64)
        
        # Broadcast temporal features to spatial dimensions
        depth_features_list = []
        for t in range(T):
            temp_feat = depth_embedded[:, t, :]  # (B, 64)
            temp_feat = temp_feat.unsqueeze(-1).unsqueeze(-1)  # (B, 64, 1, 1)
            temp_feat = temp_feat.expand(-1, -1, self.H, self.W)  # (B, 64, H, W)
            
            # Combine with spatial depth map
            spatial_depth = depth_attended[:, t:t+1, :, :]  # (B, 1, H, W)
            combined = temp_feat * spatial_depth  # Element-wise multiplication
            
            depth_features_list.append(combined)
        
        # Stack temporal dimension: (B, 64, T, H, W)
        depth_features = torch.stack(depth_features_list, dim=2)
        
        # Project to match event channels
        B, C, T, H, W = depth_features.shape
        depth_features_flat = depth_features.view(B*T, C, H, W)
        depth_projected = self.depth_projection(depth_features_flat)  # (B*T, event_channels, H, W)
        depth_projected = depth_projected.view(B, -1, T, H, W)  # (B, event_channels, T, H, W)
        
        return depth_projected


class EnhancedEventEncoder(nn.Module):
    """
    Enhanced event encoder with rangemeter fusion
    """
    def __init__(self, input_channels=2, output_dim=128, H=200, W=200):
        super().__init__()
        
        # Rangemeter fusion module
        self.rangemeter_fusion = SpatialRangemeterFusion(input_channels, H, W)
        
        # Original event encoder components
        self.conv1 = nn.Conv3d(input_channels*2, 64, kernel_size=(3,7,7), 
                              stride=(1,2,2), padding=(1,3,3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        
        # ResNet blocks (reuse from original)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=(1,2,2))
        self.layer3 = self._make_layer(128, 256, 2, stride=(1,2,2))
        self.layer4 = self._make_layer(256, 512, 2, stride=(1,2,2))
        
        # Global pooling and final projection
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(0.2)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        from elope_modules.emmnet import ResNet3DBlock  # Import the existing ResNet3DBlock
        layers = [ResNet3DBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResNet3DBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, event_tensor, rangemeter_sequence):
        """
        Args:
            event_tensor: (B, C, T, H, W) - event data
            rangemeter_sequence: (B, seq_len, 1) - rangemeter measurements
        
        Returns:
            Enhanced event features with geometric depth information
        """
        # Extract target temporal dimension from event tensor
        event_time_bins = event_tensor.shape[2]

        # Get spatial depth features
        depth_features = self.rangemeter_fusion(rangemeter_sequence, event_time_bins)  # (B, C, T, H, W)
        
        # Fuse with event tensor
        # Option 1: Concatenation along channel dimension
        fused_input = torch.cat([event_tensor, depth_features], dim=1)  # (B, 2*C, T, H, W)
        
        # Option 2: Element-wise combination (alternative)
        # fused_input = event_tensor + depth_features
        
        # Process through existing CNN architecture
        x = F.relu(self.bn1(self.conv1(fused_input)))
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


class EnhancedMultiModalVelocityEstimator(nn.Module):
    """
    Enhanced multimodal network with spatial rangemeter fusion
    Estimates only 3D linear velocity.
    """
    def __init__(self, 
                 event_channels=2,
                 event_output_dim=128,
                 imu_output_dim=32,
                 use_attention=False,
                 output_dim=3, # <--- CHANGED FROM 6 TO 3 FOR VELOCITY ONLY
                 H=200, W=200):
        super().__init__()
        
        # Enhanced event encoder with rangemeter fusion
        self.event_encoder = EnhancedEventEncoder(event_channels, event_output_dim, H, W)
        
        # IMU encoder (unchanged)
        # Import the existing IMUEncoder here, e.g.:
        from elope_modules.emmnet import IMUEncoder 
        self.imu_encoder = IMUEncoder(output_dim=imu_output_dim)
        
        fusion_input_dim = event_output_dim + imu_output_dim
        
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
        # Extract enhanced event features (with rangemeter fusion)
        event_feat = self.event_encoder(event_tensor, range_tensor)
        imu_feat = self.imu_encoder(imu_tensor)
    
        fused_feat = torch.cat([event_feat, imu_feat], dim=1)
        attention_weights = None
        
        # Final prediction
        output = self.regressor(fused_feat)
        
        return {
            'prediction': output,
            'event_features': event_feat,
            'imu_features': imu_feat,
            'attention_weights': attention_weights
        }


def create_model(use_attention=False, device='cpu'):
    """Factory function for enhanced model, specifically for velocity estimation."""
    # Ensure to pass output_dim=3 here when creating the model instance
    model = EnhancedMultiModalVelocityEstimator(use_attention=use_attention, output_dim=3)
    return model.to(device)