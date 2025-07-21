
import matplotlib.pyplot as plt
import numpy as np
import json
import torch

from pathlib import Path

from elope.datasets import SequenceLoader, EventProcessor
from elope.models.emmnetVelGru import MultiModalVelocityEstimator
from elope.utils import (
    LOGGER, 
    getfiles, 
    gridminor,
    increment_path,
    load_yaml, 
    compute_posz, 
    compute_posvelz
)
    
# Name of the file in which the weights are stored
WEIGHTS_PATH = Path("weights") / "elope-emmnet-v1_20250717_164721" / "best.pt"

# Path to the file containing the trained dataset
CFG_PATH_DATASET = "cfg/dataset/rng-5s.yml"

# Path to the yaml file containing the dataset settings
CFG_PATH_MODEL = "cfg/training/emmnet-v1.yml"

# Path in which the sequence data is stored
DATAPATH = Path("elope_data") / "test"

# Ture if the plots of the predictions should be saved for each test traj 
SAVE_PLOTS = True

# True if the output of the z-velocity should be taken from the geometry constraint 
OUTPUT_ANALYTICAL_VZ = True

# Load the configurations
cfg_dataset = load_yaml(CFG_PATH_DATASET)
cfg_model = load_yaml(CFG_PATH_MODEL)

# This script is working only for seq2one models
assert cfg_model["seq2seq"] == False

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

if SAVE_PLOTS: 
    
    # Generate the path in which to store the plots
    PLOTS_PATH = increment_path(
        Path("plots") / "submissions" / WEIGHTS_PATH.parent.name, exist_ok=False
    )
    
    PLOTS_PATH.mkdir()

# Set model to evaluation mode
model.eval() 
model.to(device)
    
sequence_files = getfiles(DATAPATH)
sequences = [s.stem for s in sequence_files]
sequences.sort() 

# Create the sequence loader 
cfg_events = cfg_dataset["events"]
seq_loader = SequenceLoader(
    cfg_dataset["datapath"], 
    event_integration_window=cfg_events["integration_window"],
    event_encoder_method=cfg_events["encoder_method"],
    event_clamp=cfg_events.get("clamp", -1),
    event_H=cfg_events["height"],
    event_W=cfg_events["width"],
    event_T=cfg_events["channels"], 
    imu_seq_len=cfg_dataset["imu_sequence_length"], 
    imu_padding=cfg_dataset["imu_padding"]
)

# Retrieve the type of event normalization 
event_normalization = cfg_model["event_normalization"]

# Retrieve the starting index 
idx_beg = seq_loader.imu_seq_len - 1

bogus = dict()
for seq_id in sequences: 

    # Load the sequence
    LOGGER.info(f"Loading test sequence: {seq_id}")
    seq_loader.load_sequence(seq_id)
    seq_loader.preprocess_events(side="left")
    
    # Retrieve the sequence times 
    times = seq_loader.timestamps_full
    seq_len = len(times)

    # Ensure we don't go out of bounds
    if idx_beg >= seq_len:
        raise RuntimeError("Invalid IMU sequence length")

    # Store output velocities
    vx, vy, vz = [], [], []    
    
    # Store ranges and angles for post-computations 
    rangemeters, angles = [], []
    
    for k in range(idx_beg, seq_len):
        
        data_k = seq_loader.get_data_at_time(times[k])
        
        event_t = data_k['events_tensor'].unsqueeze(0).to(device)
        imu_s   = data_k['imu_sequence'].unsqueeze(0).to(device)
        range_s = data_k['rangemeter_sequence'].unsqueeze(0).to(device)
        
        # Normalize events if requested 
        if event_normalization != "null": 
            for i in range(event_t.shape[0]): 
                
                # Normalize the event tensor
                event_clamp = seq_loader.event_clamp
                max_val = event_clamp if event_clamp > 0 else None       
                                
                event_t[i] = EventProcessor.normalize_tensor(
                    event_t[i], method=event_normalization, max_val=max_val
                )
                
        if not cfg_model["seq2seq"]: 
            event_t = event_t[:, -1]
        
        with torch.no_grad():
            # Run inference 
            outputs = model(event_t, imu_s, range_s)
            # Retrieve the prediction
            pred_k = outputs['prediction']
            
        vels = pred_k.cpu().numpy().squeeze()
        
        vx.append(float(vels[0]))
        vy.append(float(vels[1]))
        vz.append(float(vels[2]))
        
        # Store additional values 
        rangemeters.append(range_s.cpu().numpy().squeeze()[-1])
        angles.append(imu_s.cpu().numpy().squeeze()[-1, :3])

    vx = [vx[0]] * (idx_beg) + vx 
    vy = [vy[0]] * (idx_beg) + vy 
    vz = [vz[0]] * (idx_beg) + vz
    
    # We are missing the initial values for angles and rangemeters, so we retrieve them.
    range_init, angles_init = [], []
    for k in range(idx_beg): 
        data_k = seq_loader.get_data_at_time(times[k])  

        range_init.append(data_k['rangemeter_sequence'].cpu().numpy()[-1])
        angles_init.append(data_k['imu_sequence'].cpu().numpy()[-1, :3])        
    
    # Concatenate the lists
    angles = angles_init + angles
    rangemeters = range_init + rangemeters
        
    angles = np.array(angles)
    rangemeters = np.array(rangemeters)
    
    if OUTPUT_ANALYTICAL_VZ: 
        # Replace the output velocity on the Z-direction with the one from the 
        # geometrical constraints 
        pos_z, vel_z = compute_posvelz(
            times, rangemeters, angles, fp_window_length=30, fv_window_length=30
        )
        
        # Substitude the values
        vz = [float(v) for v in vel_z.tolist()]
        
    if SAVE_PLOTS:  
        # Create the plots with the predicted velocities 
        fig, axes = plt.subplots(nrows=3, figsize=(15,15), sharex=True)
        for i, vi in enumerate(zip(vx, vy, vz)): 
            axes[i].plot(times, vi)
            axes[i].set_xlim(times[0], times[-1])
            gridminor(axes[i])
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Velocity (m/s)')
        
        fig.tight_layout() 
        fig.savefig(PLOTS_PATH / f"{seq_id}.png", dpi=300)
        plt.close(fig)
        
        
    bogus[int(seq_id)] = {"vx": vx, "vy": vy, "vz": vz}
    LOGGER.info(f"Tested sequence: {seq_id}")

# writing submission-file to drive
with open('apelle_submission.json', 'wt') as f:
    json.dump(bogus, f)
