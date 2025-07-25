
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
    
    # Lets check what kind of model we are dealing with 
    if "architecture" in cfg_model.keys(): 
        # SETNet type 
        return SETNetV1.create_model(
            cfg_model, 
            event_channels=event_channels, 
            device=device
        )
        
    # EMMNET type
    if cfg_model["output_type"] == "sequence": 
        return MultiModalVelocityEstimatorS2S.create_model(cfg_model, device)
    else: 
        
        if cfg_model.get("use_nopool", False): 
            return MultiModalVelocityEstimatorNoPool.create_model(
                cfg_model, 
                event_channels=event_channels,
                device=device,
            )
        
        else: 
            return MultiModalVelocityEstimator.create_model(cfg_model, device)

