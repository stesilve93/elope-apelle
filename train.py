
import datetime 
import shutil

import torch 

from pathlib import Path 

from elope.datasets import ElopeDataLoader
from elope.models.emmnetVelGru import MultiModalVelocityEstimator
from elope.trainers import LunarTrainer
from elope.utils import LOGGER, load_yaml, increment_path


# Path to the yaml file containing the dataset settings
DATASET_CFG = "cfg/dataset/dataset-5s-stamp-left-1us.yml"

# Path to the yaml file containing the model settings
MODEL_CFG = "cfg/training/emmnet-v2-reduced.yml"

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

if bool(model_cfg["seq2seq"]):
    from elope.models.emmnetVelGru_s2s import MultiModalVelocityEstimator
else:
    from elope.models.emmnetVelGru import MultiModalVelocityEstimator

LOGGER.info("Model seq2seq: %s", model_cfg["seq2seq"])

# Create the network model 
model = MultiModalVelocityEstimator.create_model(
    MODEL_CFG, 
    device=device, 
)

# Create the trainer for the model 
trainer = LunarTrainer(MODEL_CFG, model, train_loader, val_loader, device)

# Create the folder in which to store the model data 
cfg_weights = model_cfg("weights")

# Get current timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
# Generates the folder in which to store the data
SAVE_NAME = cfg_weights["name"] + f"_{timestamp}"
SAVE_PATH = increment_path(Path(cfg_weights["path"]) / SAVE_NAME, exist_ok=False)
SAVE_PATH.mdkir(parents=True)

LOGGER.info(f"Saving training output to {SAVE_PATH} directory.")

# Copy inside the folder the configuration yamls for the dataset and the model 
shutil.copy(DATASET_CFG, SAVE_PATH / "dataset-cfg.yml")
shutil.copy(MODEL_CFG, SAVE_PATH / "model-cfg.yml")

# Train the model 
trainer.train(num_epochs=500, max_patience=30, save_path=SAVE_PATH)

trainer.plot_training(save_figure=True, figure_name_prefix="./plots/training/training")
LOGGER.info("Training completed!")