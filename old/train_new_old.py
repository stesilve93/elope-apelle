
import os 

import numpy as np 
import torch 
import torch.optim as optim

from pathlib import Path

from torch import nn 
from torch.optim.lr_scheduler import ReduceLROnPlateau # or CosineAnnealingLR
from torch.utils.data import Dataset

from elope_modules.emmnetVelGru import create_model
from elope_modules.elopeDataset import LunarDescentDataset, LunarTrainer
from elope_modules.dataloader import ElopeSequenceLoader
from elope_modules.elopeDataset import ElopeDataLoader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preparation
datapath = './elope_data'
VELOCITY_ONLY = True  # Set to True for velocity-only training
data_loader = ElopeSequenceLoader(datapath=datapath)

# Split sequences for train/val
all_sequences = [str(i).zfill(4) for i in range(28)]
train_sequences = all_sequences[:22]  # 80% for training
val_sequences = all_sequences[22:]    # 20% for validation

BATCH_SIZE = 32

BATCH_INTERVAL = 8

# Path to the folder containing the desired dataset
DATASET_PATH = Path("dataset") / "posvel"

# Path to the folder in which to store the model weights
WEIGHTS_PATH = "weights"

INT_WINDOW_US = 1e5  # Integration window in microseconds
SEQ_LEN = 5  # Length of IMU sequence
H, W, T = 200, 200, 5  # Image dimensions and time steps
SAMPLE_INTERVAL = 1
EVENT_ENCODER_METHOD = 'last_timestamp' # count or last_timestamp
USE_PHYSICS_AWARE = False  # Use physics-aware imu encoder

# Form-up the dataset name
dataset_name = ("_dataset_integration_window_" \
               f"{INT_WINDOW_US}_imu_seq_len_{SEQ_LEN}_H_{H}_W_{W}_T_{T}" \
               f"_sample_interval_{SAMPLE_INTERVAL}_{EVENT_ENCODER_METHOD}.pth")

CREATE_DATASET = False  # Set to True to create datasets
if CREATE_DATASET:
    
    # Create datasets
    train_dataset = LunarDescentDataset(
        data_loader_instance=data_loader,
        sequence_ids=train_sequences,
        event_integration_window_us=INT_WINDOW_US,
        imu_seq_len=SEQ_LEN,
        H=H, W=W, T=T,
        sample_interval=SAMPLE_INTERVAL,
        batch_size=BATCH_SIZE, 
        batch_interval=BATCH_INTERVAL,
        sequential_samples=True,
        event_encoder_method= EVENT_ENCODER_METHOD
    )

    val_dataset = LunarDescentDataset(
        data_loader_instance=ElopeSequenceLoader(datapath),  # Fresh instance
        sequence_ids=val_sequences,
        event_integration_window_us=INT_WINDOW_US,
        imu_seq_len=SEQ_LEN,
        H=H, W=W, T=T,
        sample_interval=SAMPLE_INTERVAL,  # Sample less frequently for validation
        batch_size=BATCH_SIZE, 
        batch_interval=BATCH_INTERVAL,
        sequential_samples=True,  # Ensure validation dataset is also velocity-only
        event_encoder_method= EVENT_ENCODER_METHOD
    )
    
    # Check if the directory already exist, if not create it 
    if not DATASET_PATH.exists(): 
        DATASET_PATH.mkdir(parents=True)
        
    # Save the datasets 
    torch.save(train_dataset, DATASET_PATH / ("train" + dataset_name))
    torch.save(val_dataset, DATASET_PATH / ("val" + dataset_name))

else:
    
    # Load datasets if they already exist
    train_dataset = torch.load(DATASET_PATH / ("train" + dataset_name), weights_only=False)
    val_dataset = torch.load(DATASET_PATH / ("val" + dataset_name), weights_only=False)


# Create data loaders
train_loader = ElopeDataLoader(
    train_dataset, batch_size=train_dataset.batch_size, shuffle=True, num_workers=8
)

val_loader = ElopeDataLoader(
    val_dataset, batch_size=val_dataset.batch_size, shuffle=False, num_workers=4
)

# Create model (if emmnetSpatial is used, set use_attention=False)
model = create_model(use_attention=True, device=device, use_physics_aware=USE_PHYSICS_AWARE)
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

# Create trainer
trainer = LunarTrainer(model, train_loader, val_loader, device, velocity_only=VELOCITY_ONLY)

model_name = ("model_integration_window_" \
             f"{INT_WINDOW_US}_imu_seq_len_{SEQ_LEN}_H_{H}_W_{W}_T_{T}_" \
             f"{EVENT_ENCODER_METHOD}_physics_aware_{USE_PHYSICS_AWARE}.pth")

# Train model
trainer.train(
    num_epochs=100, 
    save_path=Path(WEIGHTS_PATH) / model_name, 
    max_patience=10
)

trainer.plot_training(save_figure=True, figure_name_prefix='./plots/training/training')

print("Training completed!")