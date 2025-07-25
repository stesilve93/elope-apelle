
import datetime 
import shutil

import torch 

from pathlib import Path 

from elope.datasets import ElopeDataLoader
from elope.models import build_model
from elope.trainers import LunarTrainer
from elope.utils import LOGGER, load_yaml, increment_path

# Path to the yaml file containing the dataset settings
DATASET_CFG = "cfg/dataset/dataset-fix-last-1us.yml"

# Path to the yaml file containing the model settings
MODEL_CFG = "cfg/training/emmnet-v3.yml"

# Device configuration 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info(f"Using device: {device}\n")

# Split the sequences between train/val 
all_sequences = [str(i).zfill(4) for i in range(28)]
#seq_train = all_sequences[:20] + ['0023', '0027'] # 80% for training 
#seq_val = all_sequences[20:23] + all_sequences[24:27]    # 20% for 
seq_train = all_sequences[:22] # 80% for training 
seq_val = all_sequences[22:]   # 20% for validation

# Load the model and dataset config 
model_cfg = load_yaml(MODEL_CFG)
dataset_cfg = load_yaml(DATASET_CFG)

# Retrieve the dataset configs 
sequence_length = int(model_cfg["sequence_length"])
padding = str(model_cfg["padding"])
event_norm = str(model_cfg["event_normalization"])

# Create the PyTorch's dataloaders
train_loader = ElopeDataLoader(
    DATASET_CFG,
    seq_train, 
    sample_len=sequence_length,
    padding=padding,
    event_normalization=event_norm,
    augment=True, 
    flip=0.0,
    batch_size=32,
    shuffle=True, 
    num_workers=8, 
    rangemeter_noise=0.000, 
    angles_noise=0.000, 
    angles_vel_noise=0.000
)

val_loader = ElopeDataLoader(
    DATASET_CFG, 
    seq_val, 
    sample_len=sequence_length,
    padding=padding,
    event_normalization=event_norm,
    augment=False,
    flip=0.0,
    batch_size=32, 
    shuffle=True, 
    num_workers=4
)

# Create the model 
out_type = model_cfg["output_type"]
model = build_model(model_cfg, dataset_cfg, device=device)
LOGGER.info(f"Model type: {type(model)}")

# Create the trainer for the model 
trainer = LunarTrainer(MODEL_CFG, model, train_loader, val_loader, device)

# Create the folder in which to store the model data 
cfg_weights = model_cfg["weights"]

# Get current timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
# Generates the folder in which to store the data
SAVE_NAME = cfg_weights["name"] + f"_{timestamp}"
SAVE_PATH = increment_path(Path(cfg_weights["path"]) / SAVE_NAME, exist_ok=False)
SAVE_PATH.mkdir(parents=True)

LOGGER.info(f"Saving training output to {SAVE_PATH} directory.")

# Copy inside the folder the configuration yamls for the dataset and the model 
shutil.copy(DATASET_CFG, SAVE_PATH / "dataset-cfg.yml")
shutil.copy(MODEL_CFG, SAVE_PATH / "model-cfg.yml")

# Train the model 
trainer.train(num_epochs=100, max_patience=30, save_path=SAVE_PATH)
trainer.plot_training(save_figure=True, path=SAVE_PATH, filename=f"training.png")

LOGGER.info("Training completed!")