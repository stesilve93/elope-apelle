import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau # or CosineAnnealingLR
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import os
import numpy as np

from elope_modules.emmnet import create_model
from elope_modules.elopeDataset import LunarDescentDataset, LunarTrainer
from elope_modules.dataloader import DataLoader
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preparation
datapath = './elope_data'
data_loader = DataLoader(datapath=datapath)

# Split sequences for train/val
all_sequences = [str(i).zfill(4) for i in range(28)]
train_sequences = all_sequences[:22]  # 80% for training
val_sequences = all_sequences[22:]    # 20% for validation

# Create datasets
train_dataset = LunarDescentDataset(
    data_loader_instance=data_loader,
    sequence_ids=train_sequences,
    event_integration_window_us=1e5,
    imu_seq_len=5,
    H=200, W=200, T=10,
    sample_interval=1
)

val_dataset = LunarDescentDataset(
    data_loader_instance=DataLoader(datapath),  # Fresh instance
    sequence_ids=val_sequences,
    event_integration_window_us=1e5,
    imu_seq_len=5,
    H=200, W=200, T=10,
    sample_interval=2  # Sample less frequently for validation
)

# Create data loaders
train_loader = TorchDataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = TorchDataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Create model
model = create_model(use_attention=True, device=device)
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

# Create trainer
trainer = LunarTrainer(model, train_loader, val_loader, device)

# Train model
trainer.train(num_epochs=100, save_path='best_lunar_pose_model_velocity.pth')

print("Training completed!")