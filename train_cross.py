
import datetime 
import random
import shutil

import numpy as np
import torch 

from pathlib import Path 
from tabulate import tabulate

from elope.datasets import ElopeDataLoader
from elope.models.emmnetVelGru import MultiModalVelocityEstimator
from elope.trainers import LunarTrainer
from elope.utils import LOGGER, load_yaml, increment_path


# Path to the yaml file containing the dataset settings
DATASET_CFG = "cfg/dataset/dataset-hybrid-1us.yml"

# Path to the yaml file containing the model settings
MODEL_CFG = "cfg/training/emmnet-v1.yml"

# Number of groups in which to split the validation dataset
N_GROUPS = 7

# Maximum number of training epochs
MAX_EPOCHS = 2

# Maximum epoch patience per training
MAX_EPOCHS_PATIENCE = 30

# Set for random number generator to enable reproducibility
RANDOM_SEED = 0

# Device configuration 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info(f"Using device: {device}\n")

RNG = np.random.default_rng(seed=RANDOM_SEED)

# Split the sequences between train/val 
sequences = np.arange(0, 28, 1)
RNG.shuffle(sequences)

# Genearate the different groups
groups = np.array_split(sequences, N_GROUPS)
groups = [[f"{s:04d}" for s in g] for g in groups]

# Load the model config.
model_cfg = load_yaml(MODEL_CFG)

if bool(model_cfg["seq2seq"]):
    from elope.models.emmnetVelGru_s2s import MultiModalVelocityEstimator
else:
    from elope.models.emmnetVelGru import MultiModalVelocityEstimator

# Get current timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
# Create the folder in which to store the model data 
cfg_weights = model_cfg["weights"]

# Generates the folder in which to store the data
SAVE_NAME = cfg_weights["name"] + f"_{timestamp}"
SAVE_PATH = increment_path(Path(cfg_weights["path"]) / SAVE_NAME, exist_ok=False)
SAVE_PATH.mkdir(parents=True)

LOGGER.info("Model seq2seq: %s", model_cfg["seq2seq"])
LOGGER.info(f"Saving cross-training output to {SAVE_PATH} directory.")

# Copy inside the folder the configuration yamls for the dataset and the model 
shutil.copy(DATASET_CFG, SAVE_PATH / "dataset-cfg.yml")
shutil.copy(MODEL_CFG, SAVE_PATH / "model-cfg.yml")

tab_headers = ["group", "elope_score", "val_group"]
tab_values  = []

# Start the k-fold cross-validation
for k in range(N_GROUPS):
    
    # Retrieve the validation sequence
    seq_val = groups[k]
    
    # Retrieve the training sequence
    seq_train = groups[:k] + groups[k+1:]
    seq_train = [s for sublist in seq_train for s in sublist]

    # Create the PyTorch's dataloaders
    train_loader = ElopeDataLoader(
        DATASET_CFG,
        seq_train, 
        imu_seq_len=int(model_cfg["imu_sequence_length"]),
        imu_padding=model_cfg["imu_padding"],
        event_normalization=model_cfg["event_normalization"],
        verbose=False,
        augment=False,
        flip=0.0, 
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
        imu_seq_len=int(model_cfg["imu_sequence_length"]),
        imu_padding=model_cfg["imu_padding"],
        event_normalization=model_cfg["event_normalization"],
        verbose=False,
        augment=False,
        flip=0.0,
        batch_size=32, 
        shuffle=True, 
        num_workers=4
    )

    # Create the network model 
    model = MultiModalVelocityEstimator.create_model(MODEL_CFG, device=device)

    # Create the trainer for the model 
    model_cfg["weights"]["checkpoint_epochs"] = MAX_EPOCHS + 1
    trainer = LunarTrainer(model_cfg, model, train_loader, val_loader, device)

    path_k = SAVE_PATH / f"group-{k}"
    path_k.mkdir(parents=True)

    # Train the model (skipping the intermediate saving of all single groups)
    trainer.train(num_epochs=MAX_EPOCHS, max_patience=MAX_EPOCHS_PATIENCE, save_path=path_k)
    trainer.plot_training(save_figure=True, path=path_k, filename=f"training.png")
    LOGGER.info(f"Training completed for group {k}!")
    
    # Add this statistics to the table
    tab_values.append([k, trainer.best_val_loss, groups[k]])

LOGGER.info("Cross training completed.")
LOGGER.info("Cross-training statistics:")
table = tabulate(tab_values, headers=tab_headers, tablefmt="fancy_outline")
print("\n".join(" "*7 + line for line in table.splitlines()))

# Recover all the validation losses
val_losses = [val[1] for val in tab_values]
LOGGER.info(f"The model has a mean validation loss of: {np.mean(val_losses)}.") 

