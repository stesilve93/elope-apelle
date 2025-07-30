
import torch 
from torch import nn

from elope.models.blocks.encodings import vAPE

class SoftMaskAttention(nn.Module): 
    
    def __init__(
        self, 
        cin: int,
        dropout: float = 0.1
    ): 
        super().__init__() 
        
        self.mask_x = nn.Sequential(
            nn.Conv2d(2*cin, cin, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity,
            nn.Conv2d(cin, cin, kernel_size=1), 
            nn.Sigmoid()
        )
        
        self.mask_y = nn.Sequential(
            nn.Conv2d(2*cin, cin, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity,
            nn.Conv2d(cin, cin, kernel_size=1), 
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: 
        
        # x is a tensor of shape (B, C, H, W)
        # y is a tensor of shape (B, C, H, W)
        
        # Stack them to get a tensor of shape (B, 2C, H, W)
        z = torch.cat([x, y], dim=1)
        
        # Apply the two soft-mask layers 
        m1 = self.mask_x(z)  # (B, C, H, W)
        m2 = self.mask_y(z)  # (B, C, H, W)

        return m1*x + m2*y