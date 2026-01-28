
import torch
from torch import nn

from .emmnet import * 
# from .setnet import * 
# from .nonet import *

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
            model = MultiModalVelocityEstimatorEVFlow.create_model(cfg_model, device)

            # Update the EVFlowNet weights
            weights_path = cfg_model["evflownet_weights"]
            data = torch.load(weights_path)
            model.event_encoder.evflownet.load_state_dict(data)
            
            # Check whether to freeze the weights
            if cfg_model["freeze_weights"]:
                model.event_encoder.evflownet.eval() 
                for p in model.event_encoder.evflownet.parameters(): 
                    p.requires_grad = False
            
            return model
        
        elif model == "emmnet-angles": 
            return MultiModalVelocityEstimatorAngles.create_model(cfg_model, device)

        else: 
            raise ValueError(f"Unsupported emmnet model: {model}")
    
    # elif "setnet" in model:
    #     # SETNET-models 
        
    #     if model == "setnet-v1": 
    #         return SETNetV1.create_model(
    #             cfg_model, 
    #             event_channels=event_channels, 
    #             device=device
    #         )
            
    #     else: 
    #         raise ValueError(f"Unsupported setnet model: {model}")
    
    # elif "nonet" in model: 
        
    #     if model == "nonet-v1": 
    #         # Create the model
    #         model = NoFlowNet.create_model(cfg_model, device=device)

    #     elif model == "nonet-v2":
    #         model = NoFlowNet2.create_model(cfg_model, device=device)
        
    #     # Update the EVFlowNet weights 
    #     weights_path = cfg_model["evflownet_weights"]
    #     data = torch.load(weights_path) 
    #     model.evflownet.model.load_state_dict(data)
        
    #     # Check whether to freeze the weights: 
    #     if cfg_model["freeze_weights"]: 
    #         model.evflownet.freeze_weights() 
            
    #     return model
        
    
    else: 
        raise ValueError(f"Unrecognised network model: {model}")
