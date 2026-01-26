
import matplotlib.pyplot as plt 
import numpy as np
import torch 

from pathlib import Path 

from tabulate import tabulate

from elope.datasets import SequenceLoader
from elope.utils import LOGGER, load_yaml, gridminor, getfiles

MODEL_PATH = Path("weights") / "emmnet-angles_20250803_152906-best"

# Path to the yaml file containing the dataset settings
DATASET_CFG = MODEL_PATH / "dataset-cfg.yml"

# Path to the yaml file containing the model settings
MODEL_CFG = MODEL_PATH / "model-cfg.yml"

# Path to PyTorch's weight file
WEIGHTS_PATH = MODEL_PATH / "best.pth"

# True/False dependign on whether you would like to save the plots
SAVE_PLOTS = True

# Path to the folder in which to store the plots
PLOT_PATH = Path("plots") / "sequences"

# Device configuration 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info(f"Using device: {device}\n")

# Load the dataset config and create a Sequence Loader
dataset_cfg = load_yaml(DATASET_CFG)
events_cfg = dataset_cfg["events"]

# Load the model config 
model_cfg = load_yaml(MODEL_CFG)

data_path = Path("elope_data")
modes = ["train", "test"]

tab_headers = ["mode", "sequence", "time_step"]
tab_values = []

for mode in modes: 

    # Create a sequence loader
    seq_loader = SequenceLoader(
        data_path / mode,
        event_integration_window=events_cfg["integration_window"],
        event_encoder_method=events_cfg["encoder_method"],
        event_clamp=events_cfg.get("clamp", -1),
        event_H=events_cfg["height"],
        event_W=events_cfg["width"],
        event_T=events_cfg["channels"], 
        imu_seq_len=1, 
        imu_padding="static"
    )
   
    # Retrieve the type of event normalization 
    event_normalization = model_cfg["event_normalization"]
    
    # Retrieve all the sequences in that path 
    sequences = getfiles(data_path /  mode, ".npz")
    sequences = [s.stem for s in sequences]
    sequences.sort()
    
    for seq in sequences: 
        
        # Save the trajectory states and IMU readings
        seq_loader.load_sequence(seq)
        targets, imu, rangemeter = [], [], []
        
        times = seq_loader.full_times
        targets = np.array(seq_loader.full_states[:, 0:6])
        imu = np.array(seq_loader.full_imu[:, 6:12])
        rangemeter = np.interp(
            times, 
            seq_loader.full_rangemeter[:, 0], 
            seq_loader.full_rangemeter[:, 1],
        )
        
        # Store the simulation timestep
        tab_values.append([mode, seq, times[1]-times[0]])
        
        if not SAVE_PLOTS: 
            continue
        
        PLOT_PATH_SEQ = PLOT_PATH / mode / seq
        PLOT_PATH_SEQ.mkdir(exist_ok=True, parents=True)
        
        # Save plots for imu
        fig, ax = plt.subplots(figsize=(14,15), nrows=3, sharex=True)
        for k in range(3): 
            
            line_ang = ax[k].plot(times, np.rad2deg(imu[:, k]))[0]
            
            ax2 = ax[k].twinx() 
            line_w = ax2.plot(times, np.rad2deg(imu[:, k+3]), color='orange')[0]
            
            ax[k].set_xlim(times[0], times[-1])
            gridminor(ax[k])
            ax[k].set_xlabel('Time (s)')
            
            lines = [line_ang, line_w]
            labels = ['angle', 'w']
            ax[k].legend(lines, labels, loc='best')
            
        fig.tight_layout() 
        plt.savefig(PLOT_PATH_SEQ / "imu.png", dpi=300)
        plt.close(fig)
        
        # Display rangemeter data
        fig, ax = plt.subplots()
        ax.plot(times, rangemeter, label="Rangemeter data (m)")
        ax.set_xlim(times[0], times[-1])
        ax.set_xlabel('Time (s)') 
        
        if mode == "train": 
            ax.plot(times, -targets[:, 2], label="Altitude (m)") 
        
        ax.legend()
        gridminor(ax) 
        fig.tight_layout() 
        plt.savefig(PLOT_PATH_SEQ / "rangemeter.png", dpi=300)
        plt.close(fig)
        
        # Display position and velocity evolution (only for train)
        if mode == "test": 
            continue 
        
        fig, ax = plt.subplots(figsize=(14, 15), nrows=3, sharex=True)
        for k in range(3): 
            
            line_pos = ax[k].plot(times, targets[:, k], label='pos')[0]
            ax[k].set_xlim(times[0], times[-1])
            gridminor(ax[k])
            ax[k].set_xlabel('Time (s)')
            
            ax2 = ax[k].twinx() 
            line_vel = ax2.plot(times, targets[:, k+3], label='vel', color='orange')[0]
            
            lines = [line_pos, line_vel]
            labels = ['pos', 'vel']
            
            ax[k].legend(lines, labels, loc='best')
            
        fig.tight_layout() 
        plt.savefig(PLOT_PATH_SEQ / "trajectory.png", dpi=300) 
        plt.close(fig)
        
        
# Display the timesteps results
LOGGER.info("Timestep statistics:")
table = tabulate(tab_values, headers=tab_headers, tablefmt="fancy_outline")
print("\n".join(" "*7 + line for line in table.splitlines()))        
        
