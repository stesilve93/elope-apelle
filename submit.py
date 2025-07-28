
import matplotlib.pyplot as plt
import numpy as np
import json
import torch

from pathlib import Path

from scipy.interpolate import PchipInterpolator

from elope.datasets import EventProcessor, VariableSequenceLoader, FixedSequenceLoader
from elope.models import build_model
from elope.utils import (
    LOGGER, 
    getfiles, 
    gridminor,
    increment_path,
    load_yaml, 
    compute_posvelz
)

SUBMISSION_NAME = "elope-emmnet-v1_20250725_104838"

MODEL_PATH = Path("weights") / SUBMISSION_NAME

# Path to the yaml file containing the dataset settings
DATASET_CFG = MODEL_PATH / "dataset-cfg.yml"

# Path to the yaml file containing the model settings
MODEL_CFG = MODEL_PATH / "model-cfg.yml"

# Path to PyTorch's weight file
WEIGHTS_PATH = MODEL_PATH / "best.pth"

# Path in which the sequence data is stored
DATAPATH = Path("elope_data") / "test"

# Ture if the plots of the predictions should be saved for each test traj 
SAVE_PLOTS = True

# True if the output of the z-velocity should be taken from the geometry constraint 
OUTPUT_ANALYTICAL_VZ = True

# Device configuration 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
LOGGER.info(f"Using device: {device}")

# Load the configurations for the model and dataset
model_cfg = load_yaml(MODEL_CFG)
dataset_cfg = load_yaml(DATASET_CFG)

# Check the model outputs only positions 
assert model_cfg.get("velocity_only", True) == True

# Retrieve the type of model 
out_type = model_cfg["output_type"]
assert out_type in ("initial_state", "final_state", "central_state", "sequence")

# Retrieve the type of event normalization 
event_normalization = model_cfg["event_normalization"]

# Create the network model
model = build_model(model_cfg, dataset_cfg, device=device)
LOGGER.info(f"Model type: {type(model)}")

# Load the model weights 
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
    
    PLOTS_PATH.mkdir(parents=True)

# Set model to evaluation mode
model.eval() 
model.to(device)
    
sequence_files = getfiles(DATAPATH)
sequences = [s.stem for s in sequence_files]
sequences.sort() 

# Create the sequence loader 
dataset_cfg = load_yaml(DATASET_CFG)

# Retrieve the type of sequence loader
if dataset_cfg["sequence_type"] == "fixed": 
    seq_cls = FixedSequenceLoader
else: 
    seq_cls = VariableSequenceLoader

events_cfg = dataset_cfg["events"]

# Check whether the integration window should be updated 
int_window = float(events_cfg["integration_window"])
event_integration_window = model_cfg.get("event_integration_window", int_window)

seq_loader = seq_cls(
    DATAPATH, 
    time_step=float(dataset_cfg.get("time_step", -1)),
    event_integration_window=float(event_integration_window),
    event_encoder_method=events_cfg["encoder_method"],
    event_clamp=int(events_cfg.get("clamp", -1)),
    event_H=int(events_cfg["height"]),
    event_W=int(events_cfg["width"]),
    event_T=int(events_cfg["channels"]), 
    sequence_len=int(model_cfg["sequence_length"]), 
    sequence_pad=model_cfg["padding"]
)

# Retrieve the starting index 
idx_beg = seq_loader.out_len - 1

bogus = dict()
for seq_id in sequences: 

    # Load the sequence
    LOGGER.info(f"Loading test sequence: {seq_id}")
    seq_loader.load_sequence(seq_id, events_side="left", test=True)
    
    # Store the velocities and their timestamps
    times, velocities = [], []
    for k in range(idx_beg, len(seq_loader)):
        
        data_k = seq_loader.get_data_at_index(k)
        
        # Unpack and move to device after adding the batch dimension 
        tms    = data_k['times'].unsqueeze(0).to(device)
        imu    = data_k['imu'].unsqueeze(0).to(device)
        ranges = data_k['rangemeter'].unsqueeze(0).to(device)
        events = data_k['events'].unsqueeze(0).to(device)
        
        # Check whether events should be normalized
        if event_normalization != "null":         
            for i in range(events.shape[0]): 
    
                # Normalize the event tensor
                event_clamp = seq_loader.event_clamp
                max_val = event_clamp if event_clamp > 0 else None       
                
                events[i] = EventProcessor.normalize_tensor(
                    events[i], method=event_normalization, max_val=max_val
                )
                        
        # Normalize the input times
        tms_in = tms - tms[..., 0:1]
    
        with torch.no_grad():
            # Run inference and retrieve the predictions 
            outputs = model(tms_in, events, imu, ranges)
            pred_k = outputs['prediction']
              
        # Check which output we need to retrieve 
        if out_type == "initial_state": 
            tms = tms[:, 0]
        elif out_type == "final_state": 
            tms = tms[:, -1]
        elif out_type == "central_state":
            tms = tms[:, seq_loader.seq_len // 2]
        elif out_type == "sequence":
            tms = tms[:, -1]
            pred_k = pred_k[:, -1]
            
        # Store the predicted velocity at this timing
        times.append(tms.cpu().numpy().squeeze())
        velocities.append(pred_k.cpu().numpy().squeeze())
    
    # Transform the lists into arrays 
    times = np.array(times)
    velocities = np.array(velocities)    
    
    # Retrieve the timings at which we actually want to compute the velocity 
    out_times = seq_loader.full_times
    
    # We interpolate the predicted times and velocities and sample them at the 
    # actual trajectory timestamps 
    out_vel = np.stack([
        PchipInterpolator(times, velocities[:, i])(out_times) for i in range(3)
    ]).T.copy() 
    
    # Create the lists for the Python dictionary 
    vx = [float(vi) for vi in out_vel[:, 0]]
    vy = [float(vi) for vi in out_vel[:, 1]]
    vz = [float(vi) for vi in out_vel[:, 2]]

    if OUTPUT_ANALYTICAL_VZ: 
        # Replace the output velocity on the Z-direction with the one from the 
        # geometrical constraints 
        
        # Interpolate the rangemeter data at the one at which we have the trajectory 
        ranges = PchipInterpolator(
            seq_loader.full_rangemeter[:, 0], seq_loader.full_rangemeter[:, 1], 
        )(out_times)
        
        # Estimate the vertical position and velocity
        _, vel_z = compute_posvelz(
            out_times, ranges, seq_loader.full_imu[:, 0:3], 
            fp_window_length=30, fv_window_length=30
        )
        
        # Substitude the values
        vz = [float(v) for v in vel_z.tolist()]
    
    if SAVE_PLOTS:  
        # Create the plots with the predicted velocities 
        fig, axes = plt.subplots(nrows=3, figsize=(15,15), sharex=True)
        for i, vi in enumerate((vx, vy, vz)): 
            axes[i].plot(out_times, vi)
            axes[i].set_xlim(out_times[0], out_times[-1])
            gridminor(axes[i])
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Velocity (m/s)')
        
        fig.tight_layout() 
        fig.savefig(PLOTS_PATH / f"{seq_id}.png", dpi=300)
        plt.close(fig)
        
    bogus[int(seq_id)] = {"vx": vx, "vy": vy, "vz": vz}
    LOGGER.info(f"Tested sequence: {seq_id}")

# writing submission-file to drive
with open(f'apelle_{SUBMISSION_NAME}.json', 'wt') as f:
    json.dump(bogus, f)
