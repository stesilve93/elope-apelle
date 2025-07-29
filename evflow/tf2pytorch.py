
import numpy as np 
import torch 

from pathlib import Path
from torch import nn

from .EVFlowNet import EVFlowNet

def load_conv2d(conv: nn.Module, path: Path, name: str):

    # Retrieve the weight and bias files
    weight = np.load(str(path / (name + "-kernel.npy")))
    bias   = np.load(str(path / (name + "-bias.npy")))

    # In PyTorch weights are (Cout, Cin, H, W)
    # In TensorFlow weights are (H, W, Cin, Cout)
    conv.weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
    conv.bias.data = torch.from_numpy(bias) 
    
def load_batchnorm(norm: nn.Module, path: Path, name: str): 
    
    # Retrieve the norm parameters 
    gamma = np.load(str(path / (name + "-gamma.npy")))
    beta  = np.load(str(path / (name + "-beta.npy")))
    mean  = np.load(str(path / (name + "-moving_mean.npy")))
    var   = np.load(str(path / (name + "-moving_variance.npy"))) 
    
    norm.weight.data  = torch.from_numpy(gamma) 
    norm.bias.data    = torch.from_numpy(beta)
    norm.running_mean = torch.from_numpy(mean) 
    norm.running_var  = torch.from_numpy(var)
    
def load_encoder_layer(layer: nn.Module, path: Path, index: int): 
    
    name = "encoder"
    path = path / name
    suffix = f"_{index}" if index > 0 else ""
    
    load_conv2d(layer[0], path, name + "-conv2d" + suffix)
    load_batchnorm(layer[2], path, name + f"-conv{index}_bn")
    
def load_decoder_layer(layer: nn.Module, path: Path, index: int): 
    
    name = "decoder"
    path = path / name
    
    i1 = f"_{2*index}" if index > 0 else ""
    i2 = f"_{2*index+1}"
    
    load_conv2d(layer.general_conv2d[0], path, name + "-conv2d" + i1)
    load_batchnorm(layer.general_conv2d[2], path, name + f"-deconv{index}_bn")
    
    load_conv2d(layer.predict_flow[0], path, name + "-conv2d" + i2)
    
def load_residual_layer(layer: nn.Module, path: Path, index: int): 
    
    name = "transition"
    path = path / name 
    
    i1 = f"_{2*index}" if index > 0 else "" 
    i2 = f"_{2*index+1}"
    
    m0 = layer.res_block[0]
    load_conv2d(m0[0], path, name + "-conv2d" + i1)
    load_batchnorm(m0[2], path, name + f"-res{index}_res1_bn")
    
    m1 = layer.res_block[1]
    load_conv2d(m1[0], path, name + "-conv2d" + i2)
    load_batchnorm(m1[2], path, name + f"-res{index}_res2_bn")
    
# High-level path with the model weights    
path = Path("weights")

# Create the network model
model = EVFlowNet(batch_norm=True)

# Load the encoder layers
load_encoder_layer(model.encoder1, path, 0)
load_encoder_layer(model.encoder2, path, 1)
load_encoder_layer(model.encoder3, path, 2)
load_encoder_layer(model.encoder4, path, 3)

# Load the residual layers
load_residual_layer(model.resnet_block[0], path, 0)

# Load the decoder layers
load_decoder_layer(model.decoder1, path, 0)
load_decoder_layer(model.decoder1, path, 1)
load_decoder_layer(model.decoder1, path, 2)
load_decoder_layer(model.decoder1, path, 3)

# Save the PyTorch model of EV-FlowNet
torch.save(model.state_dict(), "weights/evflownet.pth")
