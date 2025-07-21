
import matplotlib.pyplot as plt 
import numpy as np
import torch 

from pathlib import Path 

from elope.datasets import SequenceLoader
from elope.utils import LOGGER, load_yaml, gridminor, getfiles

# Path to the yaml file containing the dataset settings
DATASET_CFG = "cfg/dataset/dataset-5s-count-1b.yml"

# Path to the yaml file containing the model settings
MODEL_CFG = "cfg/training/emmnet-v1-mse-rel.yml"

# Path to PyTorch's weight file
WEIGHTS_PATH = Path("weights") / "elope-emmnet-v1-elope_20250719_123610" / "best.pth"

# Path to the folder in which to store the plots
PLOT_PATH = Path("plots") / "sequences"

# Device configuration 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info(f"Using device: {device}\n")

# Load the dataset config and create a Sequence Loader
dataset_cfg = load_yaml(DATASET_CFG)
events_cfg = dataset_cfg["events"]

data_path = Path("elope_data")
modes = ["train", "test"]

for mode in modes: 

    # Create a sequence loader
    seq_loader = SequenceLoader(
        data_path / mode,
        event_integration_window=events_cfg["integration_window"],
        event_encoder_method=events_cfg["encoder_method"],
        event_H=events_cfg["height"],
        event_W=events_cfg["width"],
        event_T=5, 
        imu_seq_len=dataset_cfg["imu_sequence_length"], 
        imu_padding=dataset_cfg["imu_padding"]
    )
    
    # Retrieve all the sequences in that path 
    sequences = getfiles(data_path /  mode, ".npz")
    sequences = [s.stem for s in sequences]
    sequences.sort()
    
    for seq in sequences: 
        
        # Save the trajectory states and IMU readings
        seq_loader.load_sequence(seq)
        seq_loader.preprocess_events(side="left")
        
        targets, imu, rangemeter = [], [], []
        
        for k in range(seq_loader.seq_len): 
            
            s = seq_loader.get_data_at_time(seq_loader.timestamps_full[k])
            
            rangemeter.append(s['rangemeter_sequence'].cpu().numpy()[-1])
            imu.append(s['imu_sequence'].cpu().numpy()[-1])
            targets.append(s['ground_truth'].cpu().numpy()[-1])

        # Retrieve all the states
        times = seq_loader.timestamps_full
        targets = np.array(targets)
        imu = np.array(imu)
        rangemeter = np.array(rangemeter).squeeze()
        
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
        
        
