
import math 

import matplotlib.pyplot as plt
import torch 

from torch import nn 


class vAPE(nn.Module): 
    """Vanilla positional encoding for transformer-based encoders."""
    
    def __init__(
        self, 
        d_model: int, 
        max_len: int, 
        dropout: float = 0.1
    ): 
        # Initialise the base model
        super().__init__()
        
        # Ensure the dimensionality of the model is even
        assert d_model % 2 == 0
        
        # Create a dropout layer 
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Create the tensor with the encodings 
        pe = torch.zeros(max_len, d_model)
        
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        k = torch.exp(-torch.arange(0, d_model, 2).float() * math.log(10000.0) / d_model)
        
        # Create the tensor using the vanilla formulation 
        pe[:, 0::2] = torch.sin(pos * k)
        pe[:, 1::2] = torch.cos(pos * k)
        
        # Reshape the tensor to (1, max_len, d_model) 
        pe = pe.unsqueeze(0)
        
        # Register the tensor as a buffer 
        self.register_buffer('pe', pe) 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        
        x = x + self.pe 
        x = self.dropout(x) 
        return x    
    
    
class lPE(nn.Module): 
    """Learnable Positional Encoding (lPE)."""
    
    def __init__(
        self, 
        d_model: int, 
        max_len: int, 
        dropout: float = 0.1
    ): 
        # Initialize the base model 
        super().__init__() 
        
        # Create a dropout layer
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity() 
        
        # Create the embedding vector
        self.pe = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)
        self.pe = self.pe.unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        
        x = x + self.pe.unsqueeze(0)
        x = self.dropout(x)
        return x
    

class tAPE(nn.Module): 
    """Time Absolute Positional Encoding (tAPE)."""
    
    def __init__(
        self,
        d_model: int, 
        max_len: int, 
        dropout: float = 0.1
    ): 
        # Initialize the base model 
        super().__init__() 
        
        # Ensure the dimensionality is even 
        assert d_model % 2 == 0 
        
        # Create a dropout layer 
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Create the tensor with the encodings 
        pe = torch.zeros(max_len, d_model)
        
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        k = torch.exp(-torch.arange(0, d_model, 2).float() * math.log(10000.0) / d_model)
        
        # Apply a correction to the term 
        k = k * (d_model/max_len)
        
        # Create the tensor using the vanilla formulation 
        pe[:, 0::2] = torch.sin(pos * k)
        pe[:, 1::2] = torch.cos(pos * k)
        
        # Reshape the tensor to (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register the tensor as a buffer 
        self.register_buffer('pe', pe) 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        
        x = x + self.pe 
        x = self.dropout(x) 
        return x
        
        
vape = vAPE(128, 1001)
tape = tAPE(128, 1001)

p1 = vape.pe[0]
p2 = tape.pe[0]

# Reference positional embedding
i = p1.shape[0] // 2 

s_vape = torch.zeros(p1.shape[0])
s_tape = torch.zeros(p2.shape[0])

for k in range(len(s_vape)): 
    s_vape[k] = torch.dot(p1[k], p1[i])
    s_tape[k] = torch.dot(p2[k], p2[i])

fig, ax = plt.subplots() 
t = torch.arange(0, len(s_vape), 1) - i

ax.plot(t, s_tape, label='tAPE')
ax.plot(t, s_vape, label='Vanilla APE')
ax.legend()

plt.show()

# fig, ax = plt.subplots() 
# ax.plot(torch.arange(0, p1.shape[1], 1), p1[1])
# ax.plot(torch.arange(0, p1.shape[1], 1), p1[-1])

# plt.show()
