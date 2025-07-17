
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
import torch.nn as nn

from pathlib import Path
from typing import Dict, Tuple, Optional

from elope.datasets import SequenceLoader, ElopeDataLoader, ElopeDataset
from elope.trainers import LunarTrainer
from elope.models.emmnetVelGru import MultiModalVelocityEstimator
from elope.utils import LOGGER, load_yaml, getfiles
    

# Name of the file in which the weights are stored
WEIGHTS_NAME = "elope-emmnet-v1_20250717_083150.pth"

# Path to the folder from which to retrieve the weights 
WEIGHTS_PATH = Path("weights") 

# Use physics aware IMU encoder
USE_PHYSICS_AWARE = False 

# Path to the yaml file containing the dataset settings
MODEL_CONFIG = "cfg/v1-rnd-cfg.yml"

# Path in which the sequence data is stored
DATAPATH = Path("elope_data") / "test"

model_cfg = load_yaml(MODEL_CONFIG)

# Retrieve all the dataset config values 
event_H = int(model_cfg["events"]["height"])
event_W = int(model_cfg["events"]["width"])
event_T = int(model_cfg["events"]["time_bins"])

# Retrieve event parsing options
event_encoder_method = model_cfg["events"]["encoder_method"]
event_integration_window = float(model_cfg["events"]["integration_window"])

imu_seq_len = int(model_cfg["imu_sequence_length"])

# Device configuration 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
LOGGER.info(f"Using device: {device}")

# Create the model 
model = MultiModalVelocityEstimator.create_model(
    use_attention=True, device=device, use_physics_aware=USE_PHYSICS_AWARE
)

# Load model weights 
weights_fullpath = WEIGHTS_PATH / WEIGHTS_NAME
if weights_fullpath.exists(): 
    LOGGER.info(f"Loading weights from: {weights_fullpath}")
    data = torch.load(str(weights_fullpath), map_location=device) 
    model.load_state_dict(data, strict=False)
    
else: 
    raise ValueError(f"Weights file {weights_fullpath} does not exist.")

# Set model to evaluation mode
model.eval() 
model.to(device)
    
sequence_files = getfiles(DATAPATH)
sequences = [s.stem for s in sequence_files]
sequences.sort() 

# Create the sequence loader 
seq_loader = SequenceLoader(datapath=DATAPATH)

bogus = dict()
for seq in sequences: 

    # Load the sequence
    seq_loader.load_sequence(seq)
    # Retrieve its length
    seq_len = len(seq_loader.timestamps_full)

    # Recover initial trajectory time
    t_beg = seq_loader.timestamps_full[0]

    dt_imu = np.diff(seq_loader.timestamps_full)[0]
    t_beg = seq_loader.timestamps_full[0] + (imu_seq_len-1)*dt_imu

    t_idx = np.searchsorted(seq_loader.timestamps_full, t_beg, side='left')
    t0 = seq_loader.timestamps_full[t_idx]
        
    # Ensure we don't go out of bounds
    if t_idx >= len(seq_loader.timestamps_full):
        raise RuntimeError("Che cazzo di lunghezza di IMU hai preso.")

    vx, vy, vz = [], [], []    
    for k in range(t_idx, seq_len):
        
        data_k = seq_loader.get_data_at_time(
            seq_loader.timestamps_full[k], 
            event_integration_window=event_integration_window, 
            event_encoder_method=event_encoder_method,
            imu_seq_len=imu_seq_len, 
            H=event_H, 
            W=event_W, 
            T=event_T, 
        )
        
        tk =  seq_loader.timestamps_full[k]
        
        event_t = data_k['events_tensor'].unsqueeze(0).to(device)
        imu_s   = data_k['imu_sequence'].unsqueeze(0).to(device)
        range_s = data_k['rangemeter_sequence'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Run inference 
            outputs = model(event_t, imu_s, range_s)
            # Retrieve the prediction
            pred_k = outputs['prediction']
            
        vels = pred_k.cpu().numpy().squeeze()
        
        vx.append(vels[0])
        vy.append(vels[1])
        vz.append(vels[2])

    vx = [vx[0]] * t_idx + vx 
    vy = [vy[0]] * t_idx + vy 
    vz = [vz[0]] * t_idx + vz
        
    bogus[int(seq)] = {"vx": vx, "vy": vy, "vz": vz}
    LOGGER.info(f"Tested sequence: {seq}")

# writing submission-file to drive
with open('bogus_submission.json', 'wt') as f:
    json.dump(bogus, f)
