
from torch import nn

from .emmnet import * 
from .setnet import * 

def build_model(
    cfg_model: dict,
    cfg_dataset: dict,
    device: str = "cuda", 
) -> nn.Module: 
    
    # Retrieve number of event channels per polarity
    event_channels = int(cfg_dataset["events"]["channels"])
    
    # Retrieve the model architecture
    model = cfg_model["model"]
    
    if "emmnet" in model: 
        # EMMNET-models
        
        if model == "emmnet-base": 
            return MultiModalVelocityEstimator.create_model(cfg_model, device)
        
        elif model == "emmnet-nopool": 
            return MultiModalVelocityEstimatorNoPool.create_model(
                cfg_model, 
                event_channels=event_channels, 
                device=device
            )
            
        elif model == "emmnet-seq2seq": 
            assert cfg_model["output_type"] == "sequence"
            return MultiModalVelocityEstimatorS2S.create_model(cfg_model, device)
        
        elif model == "emmnet-transformer": 
            return MultiModalVelocityEstimatorTransformer.create_model(cfg_model, device)
        
        elif model == "emmnet-evflownet": 
            return MultiModalVelocityEstimatorEVFlow.create_model(cfg_model, device)
        
        else: 
            raise ValueError(f"Unsupported emmnet model: {model}")
    
    elif "setnet" in model:
        # SETNET-models 
        
        if model == "setnet-v1": 
            return SETNetV1.create_model(
                cfg_model, 
                event_channels=event_channels, 
                device=device
            )
            
        else: 
            raise ValueError(f"Unsupported setnet model: {model}")
    
    else: 
        raise ValueError(f"Unrecognised network model: {model}")
