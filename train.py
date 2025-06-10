import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau # or CosineAnnealingLR
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import os
import numpy as np

from elope_modules.emmnetVelGru import create_model
from elope_modules.elopeDataset import LunarDescentDataset, LunarTrainer
from elope_modules.dataloader import DataLoader
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preparation
datapath = './elope_data'
VELOCITY_ONLY = True  # Set to True for velocity-only training
data_loader = DataLoader(datapath=datapath, velocity_only=VELOCITY_ONLY)

# Split sequences for train/val
all_sequences = [str(i).zfill(4) for i in range(28)]
train_sequences = all_sequences[:22]  # 80% for training
val_sequences = all_sequences[22:]    # 20% for validation

INT_WINDOW_US = 1e5  # Integration window in microseconds
SEQ_LEN = 5  # Length of IMU sequence
H, W, T = 200, 200, 5  # Image dimensions and time steps
SAMPLE_INTERVAL = 1
EVENT_ENCODER_METHOD = 'last_timestamp' # count or last_timestamp
USE_PHYSICS_AWARE = True  # Use physics-aware imu encoder

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
        velocity_only=VELOCITY_ONLY,
        event_encoder_method= EVENT_ENCODER_METHOD
    )

    val_dataset = LunarDescentDataset(
        data_loader_instance=DataLoader(datapath),  # Fresh instance
        sequence_ids=val_sequences,
        event_integration_window_us=INT_WINDOW_US,
        imu_seq_len=SEQ_LEN,
        H=H, W=W, T=T,
        sample_interval=SAMPLE_INTERVAL,  # Sample less frequently for validation
        velocity_only=VELOCITY_ONLY,  # Ensure validation dataset is also velocity-only
        event_encoder_method= EVENT_ENCODER_METHOD
    )

    # Save
    torch.save(train_dataset, f'dataset/vel_only/train_dataset_integration_window_{INT_WINDOW_US}_imu_seq_len_{SEQ_LEN}_H_{H}_W_{W}_T_{T}_sample_interval_{SAMPLE_INTERVAL}_{EVENT_ENCODER_METHOD}.pth')
    torch.save(val_dataset, f'dataset/vel_only/val_dataset_integration_window_{INT_WINDOW_US}_imu_seq_len_{SEQ_LEN}_H_{H}_W_{W}_T_{T}_sample_interval_{SAMPLE_INTERVAL}_{EVENT_ENCODER_METHOD}.pth')
else:
    # Load datasets if they already exist
    train_dataset = torch.load(f'dataset/vel_only/train_dataset_integration_window_{INT_WINDOW_US}_imu_seq_len_{SEQ_LEN}_H_{H}_W_{W}_T_{T}_sample_interval_{SAMPLE_INTERVAL}_{EVENT_ENCODER_METHOD}.pth',
                               weights_only=False)
    val_dataset = torch.load(f'dataset/vel_only/val_dataset_integration_window_{INT_WINDOW_US}_imu_seq_len_{SEQ_LEN}_H_{H}_W_{W}_T_{T}_sample_interval_{SAMPLE_INTERVAL}_{EVENT_ENCODER_METHOD}.pth',
                             weights_only=False)


# Create data loaders
train_loader = TorchDataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
val_loader = TorchDataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Create model
model = create_model(use_attention=True, device=device, use_physics_aware=USE_PHYSICS_AWARE) # if emmnetSpatial is used, set use_attention=False
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

# Create trainer
trainer = LunarTrainer(model, train_loader, val_loader, device, velocity_only=VELOCITY_ONLY)

# Train model
trainer.train(num_epochs=100, save_path=f'model_integration_window_{INT_WINDOW_US}_imu_seq_len_{SEQ_LEN}_H_{H}_W_{W}_T_{T}_{EVENT_ENCODER_METHOD}_physics_aware_{USE_PHYSICS_AWARE}.pth', max_patience=10)
trainer.plot_training(save_figure=True, figure_name_prefix='./plots/training/training')

print("Training completed!")