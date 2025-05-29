import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau # or CosineAnnealingLR
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import os
import numpy as np

from elope_modules.emmnet import create_model
from elope_modules.elopeDataset import LunarDescentDataset
from elope_modules.dataloader import DataLoader
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Model Initialization ---
# Decide if you want to use attention
use_attention_in_model = True # Set to False for simple concatenation
model = create_model(use_attention=use_attention_in_model, device=device)
print(model)

# --- Loss Function and Optimizer ---
criterion = nn.MSELoss() # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=1e-4) # Adam optimizer is a good default
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# --- Training Parameters ---
num_epochs = 500 # Adjust as needed

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs, device):
    model.train() # Set model to training mode
    best_epoch_loss = float('inf') # Initialize best loss for early stopping

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (event_t, imu_s, range_s, gt_pv) in enumerate(train_loader):
            # Move data to the correct device
            event_t = event_t.to(device)
            imu_s = imu_s.to(device)
            range_s = range_s.to(device)
            gt_pv = gt_pv.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(event_t, imu_s, range_s)
            predictions = outputs['prediction']

            # Calculate loss
            loss = criterion(predictions, gt_pv)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * event_t.size(0) # Multiply by batch size
            
            # Print progress every now and then
            if (i + 1) % 50 == 0: # Print every 50 batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / ((i+1) * train_loader.batch_size):.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} finished. Average Loss: {epoch_loss:.4f}")

        # Step the learning rate scheduler
        scheduler.step(epoch_loss)
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            print(f"New best loss: {best_epoch_loss:.4f}. Saving model...")
            # Save the model state if it improves
            torch.save(model.state_dict(), "best_velocity_estimator_model.pth")
        else:
            print(f"No improvement in loss. Current best: {best_epoch_loss:.4f}")


# --- Start Training ---
print("Starting training...")
datapath = './elope_data' # Adjust as needed
data_loader = DataLoader(datapath=datapath)

# Generate a list of training sequence IDs
train_sequence_ids = [str(i).zfill(4) for i in range(28)]

# Create the dataset
train_dataset = LunarDescentDataset(
    data_loader_instance=data_loader,
    sequence_ids=train_sequence_ids,
    event_integration_window_us=1e5, 
    imu_seq_len=5,
    H=200, W=200, T=10,
    sample_interval=1 # Adjust sampling frequency
)

# Create a PyTorch DataLoader
batch_size = 64 # Or whatever fits your GPU memory
train_dataloader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4) # num_workers for parallel data loading
train_model(model, train_dataloader, criterion, optimizer, scheduler, num_epochs, device)
print("Training complete!")

# --- Optional: Save final model ---
# torch.save(model.state_dict(), "final_velocity_estimator_model.pth")