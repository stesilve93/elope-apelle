
import torch

from torch import nn 
from torchvision.models.resnet import BasicBlock

class ResNet(nn.Module): 
    
    def __init__(self, cin: int, n_blocks: int):
        
        super().__init__() 
        
        # Define all the channel dimensions
        c1, c2, c3, c4, c5 = cin, 64, 128, 256, 512
        
        # Initialise the stem layer 
        self.conv1 = nn.Conv2d(
            c1, out_channels=c2, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        self.bn1 = nn.BatchNorm2d(c2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Create the 4 convolutional layers 
        self.layer1 = _make_layer(c2, c2, n_blocks=n_blocks, stride=1)
        self.layer2 = _make_layer(c2, c3, n_blocks=n_blocks, stride=2)
        self.layer3 = _make_layer(c3, c4, n_blocks=n_blocks, stride=2) 
        self.layer4 = _make_layer(c4, c5, n_blocks=n_blocks, stride=2)
        
        # Create final average pooling 
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        # Flattener
        self.flatten = nn.Flatten()
        
    def initialize(self, zero_init: bool = False): 
        
        # Initialise the model
        for m in self.modules(): 
            if isinstance(m, nn.Conv2d): 
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialise the last BN in each residual branch so that the residual branch 
        # starts with zeros, and each residual block behaves like an identity. 
        if zero_init: 
            for m in self.modules(): 
                if isinstance(m, BasicBlock) and m.bn2.weight is not None: 
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward(self, x: torch.Tensor) -> tuple: 
        
        x = self.conv1(x) 
        x = self.bn1(x) 
        x = self.relu(x) 
        x = self.maxpool(x) 
        
        f1 = self.layer1(x) 
        f2 = self.layer2(f1) 
        f3 = self.layer3(f2) 
        f4 = self.layer4(f3)
        
        x = self.avgpool(f4) 
        x = self.flatten(x)
        
        return (f1, f2, f3, f4), x
        
class ResNet18(ResNet): 
    """ResNet-18 model with 2 Residual blocks per layer."""
    def __init__(self, input_dim: int): 
        super().__init__(input_dim, n_blocks=2)
        
class ResNet34(ResNet): 
    """ResNet-34 model with 3 residual blocks per layer."""
    def __init__(self, input_dim: int): 
        super().__init__(input_dim, n_blocks=3)
        
        
def _make_layer(cin: int, cout: int, n_blocks: int, stride: int) -> nn.Sequential: 
    
    downsample = None
    if stride != 1 or cin != cout: 
        downsample = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(cout)
        )
    
    layers = []
    layers.append(
        BasicBlock(
            cin, 
            cout, 
            stride=stride, 
            downsample=downsample, 
            norm_layer=nn.BatchNorm2d
        )
    )
    
    cin = cout 
    for _ in range(1, n_blocks): 
        layers.append(BasicBlock(cin, cout, norm_layer=nn.BatchNorm2d))
    
    return nn.Sequential(*layers)
