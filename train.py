

import torch 

from pathlib import Path 

from elope.datasets import ElopeDataset, ElopeDataLoader
from elope.models.emmnetVelGru import MultiModalVelocityEstimator
from elope.trainers import LunarTrainer
from elope.utils import LOGGER, increment_path

# Path to the yaml file containing the dataset settings
DATASET_CFG = "cfg/dataset/dataset-5s.yml"

# Path to the yaml file containing the model settings
MODEL_CFG = "cfg/training/emmnet-v2.yml"

# Device configuration 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info(f"Using device: {device}\n")

# Split the sequences between train/val 
all_sequences = [str(i).zfill(4) for i in range(28)]
seq_train = all_sequences[:22]  # 80% for training
seq_val = all_sequences[22:]    # 20% for validation

# Create the PyTorch's dataloaders
train_loader = ElopeDataLoader(
    DATASET_CFG,
    seq_train, 
    augment=True, 
    batch_size=32,
    shuffle=True, 
    num_workers=8
)

val_loader = ElopeDataLoader(
    DATASET_CFG, 
    seq_val, 
    augment=False,
    batch_size=32, 
    shuffle=True, 
    num_workers=4
)

# Create the network model 
model = MultiModalVelocityEstimator.create_model(MODEL_CFG, device=device)

# Create the trainer for the model 
trainer = LunarTrainer(MODEL_CFG, model, train_loader, val_loader, device)

# Train the model 
trainer.train(num_epochs=25, max_patience=10)

trainer.plot_training(save_figure=True, figure_name_prefix="./plots/training/training")
LOGGER.info("Training completed!")