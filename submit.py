
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
WEIGHTS_PATH = Path("weights") / "elope-emmnet-v1_20250717_164721" / "best.pt"

CFG_PATH_DATASET = "cfg/dataset/rng-5s.yml"

# Path to the yaml file containing the dataset settings
CFG_PATH_MODEL = "cfg/training/emmnet-v1.yml"

# Path in which the sequence data is stored
DATAPATH = Path("elope_data") / "test"

cfg_dataset = load_yaml(CFG_PATH_DATASET)
cfg_model = load_yaml(CFG_PATH_MODEL)

# Retrieve all the dataset config values 
event_H = int(cfg_dataset["events"]["height"])
event_W = int(cfg_dataset["events"]["width"])
event_T = int(cfg_dataset["events"]["time_bins"])

# Retrieve event parsing options
event_encoder_method = cfg_dataset["events"]["encoder_method"]
event_integration_window = float(cfg_dataset["events"]["integration_window"])

imu_seq_len = int(cfg_dataset["imu_sequence_length"])

# Device configuration 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
LOGGER.info(f"Using device: {device}")

# Create the model 
model = MultiModalVelocityEstimator.create_model(cfg_model, device=device)

# Load model weights 
if WEIGHTS_PATH.exists(): 
    LOGGER.info(f"Loading weights from: {WEIGHTS_PATH}")
    data = torch.load(str(WEIGHTS_PATH), map_location=device) 
    model.load_state_dict(data, strict=False)
    
else: 
    raise ValueError(f"Weights file {WEIGHTS_PATH} does not exist.")

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
        
        vx.append(float(vels[0]))
        vy.append(float(vels[1]))
        vz.append(float(vels[2]))

    vx = [vx[0]] * t_idx + vx 
    vy = [vy[0]] * t_idx + vy 
    vz = [vz[0]] * t_idx + vz
        
    bogus[int(seq)] = {"vx": vx, "vy": vy, "vz": vz}
    LOGGER.info(f"Tested sequence: {seq}")

# writing submission-file to drive
with open('apelle_submission.json', 'wt') as f:
    json.dump(bogus, f)
