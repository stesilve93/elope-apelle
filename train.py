
import torch 

from pathlib import Path 

from elope.datasets import ElopeDataLoader
from elope.models.emmnetVelGru import MultiModalVelocityEstimator
from elope.trainers import LunarTrainer
from elope.utils import LOGGER, load_yaml


# Path to the yaml file containing the dataset settings
DATASET_CFG = "cfg/dataset/dataset-5s-stamp-left-1us.yml"

# Path to the yaml file containing the model settings
MODEL_CFG = "cfg/training/emmnet-v2.yml"

# Device configuration 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info(f"Using device: {device}\n")

# Split the sequences between train/val 
all_sequences = [str(i).zfill(4) for i in range(28)]
seq_train = all_sequences[:20] + ['0023', '0027'] # 80% for training 
seq_val = all_sequences[20:23] + all_sequences[24:27]    # 20% for validation

# Load the model config.
model_cfg = load_yaml(MODEL_CFG)

# Create the PyTorch's dataloaders
train_loader = ElopeDataLoader(
    DATASET_CFG,
    seq_train, 
    event_normalization=model_cfg["event_normalization"],
    augment=True, 
    batch_size=32,
    shuffle=True, 
    num_workers=8, 
    rangemeter_noise=0.005, 
    angles_noise=0.001, 
    angles_vel_noise=0.001
)

val_loader = ElopeDataLoader(
    DATASET_CFG, 
    seq_val, 
    event_normalization=model_cfg["event_normalization"],
    augment=False,
    batch_size=32, 
    shuffle=True, 
    num_workers=4
)

# Retrieve the model configuration
if isinstance(MODEL_CFG, (str, Path)): 
    cfg = load_yaml(MODEL_CFG)

if bool(cfg["seq2seq"]):
    from elope.models.emmnetVelGru_s2s import MultiModalVelocityEstimator
else:
    from elope.models.emmnetVelGru import MultiModalVelocityEstimator

LOGGER.info("Model seq2seq: %s", cfg["seq2seq"])
# Create the network model 
model = MultiModalVelocityEstimator.create_model(
    MODEL_CFG, 
    device=device, 
)


# Create the trainer for the model 
trainer = LunarTrainer(MODEL_CFG, model, train_loader, val_loader, device)

# Train the model 
trainer.train(num_epochs=500, max_patience=30)

trainer.plot_training(save_figure=True, figure_name_prefix="./plots/training/training")
LOGGER.info("Training completed!")