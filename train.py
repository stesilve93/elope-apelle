

import torch 

from pathlib import Path 

from elope.datasets import ElopeDataset, ElopeDataLoader
from elope.models.emmnetVelGru import MultiModalVelocityEstimator
from elope.trainers import LunarTrainer
from elope.utils import LOGGER, increment_path

# Use physics aware IMU encoder
USE_PHYSICS_AWARE = False

# Path to the folder in which to store the model weights 
WEIGHTS_PATH = "weights"

# Name of the model 
MODEL_NAME = "elope-emmnet-v1"

# Path to the yaml file containing the dataset settings
DATASET_CONFIG = "cfg/v1-rnd-cfg.yml"

# True if the network should only provide the velocity as output
VELOCITY_ONLY = True

# Device configuration 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info(f"Using device: {device}\n")

# Split the sequences between train/val 
all_sequences = [str(i).zfill(4) for i in range(28)]
seq_train = all_sequences[:22]  # 80% for training
seq_val = all_sequences[22:]    # 20% for validation

# Create/load the training and validation datasets
train_dataset = ElopeDataset.from_yaml(DATASET_CONFIG, seq_train, mode="train")
val_dataset = ElopeDataset.from_yaml(DATASET_CONFIG, seq_val, mode="val")

# Create the PyTorch's dataloaders
train_loader = ElopeDataLoader(
    train_dataset, batch_size=train_dataset.batch_size, shuffle=True, num_workers=8
)

val_loader = ElopeDataLoader(
    val_dataset, batch_size=val_dataset.batch_size, shuffle=True, num_workers=4
)

# Create the network model 
model = MultiModalVelocityEstimator.create_model(
    use_attention=True, device=device, use_physics_aware=USE_PHYSICS_AWARE
)

# Create the trainer for the model 
# TODO: add configuration for the loss function
trainer = LunarTrainer(
    model, train_loader, val_loader, device, velocity_only=VELOCITY_ONLY
)

# raise RuntimeError

# Train the model 
trainer.train( 
    num_epochs=5, 
    save_path=increment_path(Path(WEIGHTS_PATH) / MODEL_NAME),
    max_patience=10
)

trainer.plot_training(save_figure=True, figure_name_prefix="./plots/training/training")
LOGGER.info("Training completed!")