
import torch 

from pathlib import Path 

from elope.datasets import ElopeDataLoader
from elope.models.emmnetVelGru import MultiModalVelocityEstimator
from elope.trainers import LunarTrainer
from elope.utils import LOGGER

# Path to the yaml file containing the dataset settings
DATASET_CFG = "cfg/dataset/dataset-5s-stamp.yml"

# Path to the yaml file containing the model settings
MODEL_CFG = "cfg/training/emmnet-v1-mse-rel.yml"

# Path to PyTorch's weight file
WEIGHTS_PATH = Path("weights") / "" / "best.pt"

# Device configuration 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info(f"Using device: {device}\n")

# Split the sequences between train/val 
all_sequences = [str(i).zfill(4) for i in range(28)]
seq_train = all_sequences[:22]  # 80% for training
seq_val = all_sequences[22:]    # 20% for validation

# Create the PyTorch's dataloaders
test_loader = ElopeDataLoader(
    DATASET_CFG,
    seq_val, 
    augment=True, 
    batch_size=32,
    shuffle=True, 
    num_workers=8, 
    rangemeter_noise=0.01, 
    angles_noise=0.005, 
    angles_vel_noise=0.005
)

# Create the network model 
model = MultiModalVelocityEstimator.create_model(MODEL_CFG, device=device)

# Load the model weights 
if WEIGHTS_PATH.exists(): 
    LOGGER.info(f"Loading weights from: {WEIGHTS_PATH}")
    data = torch.load(str(WEIGHTS_PATH), map_location=device)
    model.load_state_dict(data, strict=False)

else: 
    raise ValueError(f"Weights file {WEIGHTS_PATH} does not exist.")

# Set the model in evaluation mode
model.eval()
model.to(device)

# Retrieve the sequence loader from the dataset config 
seq_loader = test_loader.dataset.seq_loader

# Retrieve the starting index 
idx_beg = seq_loader.imu_seq_len - 1

for seq_id in seq_train: 
    
    # Load the sequence
    LOGGER.info(f"Loading test sequence: {seq_id}")
    seq_loader.load_sequence(seq_id)
    
    # Retrieve the sequence length 
    len_seq = len(seq_loader)
    
    # Initialize the arrays for the results
    predictions, targets, times = [], [], []
    for k in range(idx_beg, len_seq): 
        
        # Retrieve the data at the current time
        data_k = seq_loader.get_data_at_time(seq_loader.timestamps_full[k])

        # Unpack and move to device after adding the batch dimension 
        events    = data_k['events_tensor'].unsqueeze(0).to(device)
        imu_seq   = data_k['imu_sequence'].unsqueeze(0).to(device)
        range_seq = data_k['rangemeter_sequence'].unsqueeze(0).to(device)
        targets   = data_k['ground_truth'].unsqueeze(0).to(device)
        
        with torch.no_grad(): 
            # Run inference and retrieve the predictions 
            outputs = model(events, imu_seq, range_seq)
            pred_k = outputs['prediction']
            
        # Store the results 
        times.append(seq_loader.timestamps_full[k])
        targets.append(targets.cpu().numpy().squeeze())
        predictions.append(pred_k.cpu().numpy().squeeze())
        
    # Compute the test metrics
    test_metrics = LunarTrainer.compute_metrics(
        torch.tensor(predictions), 
        torch.tensor(targets), 
        velocity_only=predictions[0].shape[1] == 3
    )
    
    # Display the validation losses (e.g., each entry in the dictonary)
    loss_names = tuple(test_metrics.keys())
    loss_values = tuple([test_metrics[ln] for ln in loss_names])

    print(("Test Metrics: " + '%15s' * len(loss_names)) % loss_names)
    print((" " * 14 + '%15.5f' * len(loss_names)) % loss_values)
    print("\n")
                