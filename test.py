
import matplotlib.pyplot as plt 
import numpy as np
import torch 

from pathlib import Path 

from tabulate import tabulate

from elope.datasets import ElopeDataLoader, SequenceLoader, EventProcessor
from elope.models.emmnetVelGru import MultiModalVelocityEstimator
from elope.trainers import LunarTrainer
from elope.utils import (
    LOGGER, 
    gridminor, 
    increment_path,
    load_yaml, 
    compute_posz, 
    compute_posvelz,
)

MODEL_PATH = Path("weights") / "elope-emmnet-v2_20250722_135946"

# Path to the yaml file containing the dataset settings
DATASET_CFG = MODEL_PATH / "dataset-cfg.yml"

# Path to the yaml file containing the model settings
MODEL_CFG = MODEL_PATH / "model-cfg.yml"

# Path to PyTorch's weight file
WEIGHTS_PATH = MODEL_PATH / "best.pth"

# True if the plots of the predictions / groundtruth should be saved for each test traj.
SAVE_PLOTS = True

# True if the output of the z-velocity should be taken from the geometry constraint
OUTPUT_ANALYTICAL_VZ = False 

# Device configuration 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info(f"Using device: {device}\n")

# Split the sequences between train/val 
all_sequences = [str(i).zfill(4) for i in range(28)]
# seq_train = all_sequences[:20] + ['0023', '0027'] # 80% for training 
# seq_val = all_sequences[20:23] + all_sequences[24:27]    # 20% for validation
seq_train = all_sequences[:22] # 80% for training 
seq_val = all_sequences[22:]   # 20% for validation

# Load the model config 
model_cfg = load_yaml(MODEL_CFG)

# This script is working only for seq2one models 
assert model_cfg["seq2seq"] == False

# Load the dataset config and create a Sequence Loader
dataset_cfg = load_yaml(DATASET_CFG)
events_cfg = dataset_cfg["events"]

seq_loader = SequenceLoader(
    dataset_cfg["datapath"], 
    event_integration_window=events_cfg["integration_window"],
    event_encoder_method=events_cfg["encoder_method"],
    event_clamp=events_cfg.get("clamp", -1),
    event_H=events_cfg["height"],
    event_W=events_cfg["width"],
    event_T=events_cfg["channels"], 
    imu_seq_len=int(model_cfg["imu_sequence_length"]), 
    imu_padding=model_cfg["imu_padding"]
)


# Retrieve the type of event normalization 
event_normalization = model_cfg["event_normalization"]

# Create the network model 
model = MultiModalVelocityEstimator.create_model(model_cfg, device=device)

# Load the model weights 
if WEIGHTS_PATH.exists(): 
    LOGGER.info(f"Loading weights from: {WEIGHTS_PATH}")
    data = torch.load(str(WEIGHTS_PATH), map_location=device)
    model.load_state_dict(data, strict=False)

else: 
    raise ValueError(f"Weights file {WEIGHTS_PATH} does not exist.")

# Set the model in evaluation mode
model.eval()
model.to(device)

# Retrieve the starting index 
idx_beg = seq_loader.imu_seq_len - 1

tab_headers = ["sequence", "time_step", "vel_mse_abs", "vel_mse_rel", "elope_score"]
tab_values  = []

if SAVE_PLOTS: 
    
    # Generate the path in which to store the plots
    PLOTS_PATH = increment_path(
        Path("plots") / "testing" / WEIGHTS_PATH.parent.name, exist_ok=False
    )
    
    PLOTS_PATH.mkdir(parents=True)

for seq_id in seq_val: 
    
    # Load the sequence
    LOGGER.info(f"Loading test sequence: {seq_id}")
    seq_loader.load_sequence(seq_id)
    seq_loader.preprocess_events(side="left")
    
    # Compute the timestep of this sequence 
    seq_dt = seq_loader.timestamps_full[1] - seq_loader.timestamps_full[0]
    
    # Initialize the arrays for the results
    predictions, targets, times = [], [], []
    rangemeters, angles = [], []
    for k in range(idx_beg, seq_loader.seq_len): 
        
        # Retrieve the data at the current time
        data_k = seq_loader.get_data_at_time(seq_loader.timestamps_full[k])

        # Unpack and move to device after adding the batch dimension 
        tms       = data_k['times'].unsqueeze(0).to(device)
        events    = data_k['events_tensor'].unsqueeze(0).to(device)
        imu_seq   = data_k['imu_sequence'].unsqueeze(0).to(device)
        range_seq = data_k['rangemeter_sequence'].unsqueeze(0).to(device)
        states    = data_k['ground_truth'].unsqueeze(0).to(device)
        
        # Check whether events should be normalized
        if event_normalization != "null":         
            for i in range(events.shape[0]): 
    
                # Normalize the event tensor
                event_clamp = seq_loader.event_clamp
                max_val = event_clamp if event_clamp > 0 else None       
                
                events[i] = EventProcessor.normalize_tensor(
                    events[i], method=event_normalization, max_val=max_val
                )
        
        if not model_cfg["seq2seq"]: 
            events = events[:, -1]
            states = states[:, -1]
        
        # Normalize the times 
        tms = tms - tms[..., 0:1]
        
        with torch.no_grad(): 
            # Run inference and retrieve the predictions 
            outputs = model(tms, events, imu_seq, range_seq)
            pred_k = outputs['prediction']
            
        # Store the results 
        times.append(seq_loader.timestamps_full[k])
        targets.append(states.cpu().numpy().squeeze())
        predictions.append(pred_k.cpu().numpy().squeeze())
        
        # Store additional values 
        rangemeters.append(range_seq.cpu().numpy().squeeze()[-1])
        angles.append(imu_seq.cpu().numpy().squeeze()[-1, :3])
            
    predictions = np.array(predictions)
    targets = np.array(targets)
    times = np.array(times)
    
    angles = np.array(angles)
    rangemeters = np.array(rangemeters)
        
    if OUTPUT_ANALYTICAL_VZ: 
        
        # Replace the output velocity on the Z-direction with the one from the 
        # geometrical constraints 
        pos_z, vel_z = compute_posvelz(
            times, rangemeters, angles, fp_window_length=30, fv_window_length=30
        )
        
        # Replace the network output with our data
        predictions[:, -1] = vel_z
    
    # Compute the test metrics
    test_metrics = LunarTrainer.compute_metrics(
        torch.tensor(predictions), 
        torch.tensor(targets), 
        velocity_only=model_cfg["velocity_only"]
    )
    
    # Store the statistics of this trajectory
    seq_metrics = [seq_id, seq_dt]
    for header in tab_headers[2:]: 
        seq_metrics.append(float(test_metrics[header]))
        
    tab_values.append(seq_metrics)
    
    if SAVE_PLOTS: 
        # Get the offest index
        ioff = 0 if model_cfg["velocity_only"] else 3
        
        # Create the plots with the predicted velocities 
        fig, axes = plt.subplots(nrows=3, figsize=(15,15), sharex=True)
        for i in range(3): 
            
            axes[i].plot(times, targets[:, i+3], label='Target')
            axes[i].plot(times, predictions[:, i+ioff], label='Prediction')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_xlim(times[0], times[-1])
            gridminor(axes[i])
            
            axes[i].legend() 
            
        fig.tight_layout() 
        fig.savefig(PLOTS_PATH / f"{seq_id}.png", dpi=300)
        plt.close(fig)
    
    # Display the validation losses (e.g., each entry in the dictonary)
    loss_names = tuple(test_metrics.keys())
    loss_values = tuple([test_metrics[ln] for ln in loss_names])

    print(("Test Metrics: " + '%15s' * len(loss_names)) % loss_names)
    print((" " * 14 + '%15.5f' * len(loss_names)) % loss_values)
    print("\n")
                
                
LOGGER.info("Test statistics summary:")
table = tabulate(tab_values, headers=tab_headers, tablefmt="fancy_outline")
print("\n".join(" "*7 + line for line in table.splitlines()))
