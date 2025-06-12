import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from elope_modules.dataloader import DataLoader
from typing import Dict, Tuple
import time
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

from pathlib import Path

class LunarDescentDataset(Dataset):
    def __init__(self, data_loader_instance: DataLoader,
                 sequence_ids: list,
                 event_integration_window_us: float = 1e5,
                 imu_seq_len: int = 50,
                 H: int = 200, W: int = 200, T: int = 10,
                 sample_interval: int = 10,
                 velocity_only: bool = True,
                 event_encoder_method: str = 'last_timestamp'):
        """
        Custom PyTorch Dataset for lunar descent data.

        This dataset class is responsible for generating samples from the lunar descent
        dataset. It takes in a DataLoader instance which contains the dataset metadata,
        and a list of sequence IDs to load. For each sequence, it loads the full sequence
        data into the DataLoader instance, and then generates samples from this loaded
        sequence.

        Each sample is a dictionary containing the following keys:
        - 'events_tensor': A tensor of shape (H, W, T) representing the event image.
        - 'imu_sequence': A tensor of shape (seq_len, 6) representing the IMU sequence.
        - 'rangemeter_sequence': A tensor of shape (seq_len, 2) representing the rangemeter sequence.
        - 'ground_truth': A tensor of shape (7,) representing the ground truth state (x, y, z, vx, vy, vz, t).

        The samples are stored in a list called `self.samples`, which is accessible
        through the `__getitem__` method.

        Args:
            data_loader_instance (DataLoader): An instantiated DataLoader object.
            sequence_ids (list): List of sequence IDs (e.g., ['0000', '0001', ...]) to load.
            event_integration_window_us (float): Time window for events in microseconds.
            imu_seq_len (int): Sequence length for IMU and rangemeter data.
            H, W, T: Dimensions for the event tensor.
            sample_interval (int): How often to sample a timestamp from the trajectory
                                   (e.g., 10 means every 10th timestamp).
        """
        self.data_loader = data_loader_instance
        self.samples = []
        self.event_integration_window_us = event_integration_window_us
        self.imu_seq_len = imu_seq_len
        self.H = H
        self.W = W
        self.T = T
        self.velocity_only = velocity_only

        print("Preparing dataset samples...")
        for seq_id in sequence_ids:
            print(f"Loading sequence {seq_id}...")
            # Load the full sequence data into the DataLoader instance
            self.data_loader.load_sequence(seq_id, source="train")

            # Generate samples from this loaded sequence
            # Iterate through timestamps from the loaded trajectory
            # Use data_loader.timestamps_full which was populated by load_sequence
            if self.data_loader.timestamps_full is not None:
                for i in range(sample_interval, len(self.data_loader.timestamps_full), sample_interval):
                    t_curr_s = self.data_loader.timestamps_full[i]
                    data_point = self.data_loader.get_data_at_time(
                        t_curr_s,
                        event_integration_window_us=self.event_integration_window_us,
                        imu_seq_len=self.imu_seq_len,
                        H=self.H, W=self.W, T=self.T, event_encoder_method=event_encoder_method
                    )
                    if data_point:
                        self.samples.append(data_point)
            print(f"  -> Added {len(self.samples)} samples so far.")
        print(f"Finished preparing dataset. Total samples: {len(self.samples)}")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        # Return the pre-processed tensors and ground truth
        sample = self.samples[idx]
        
        return (
            sample['events_tensor'], 
            sample['imu_sequence'], 
            sample['rangemeter_sequence'],
            sample['ground_truth']
        )


class LunarTrainer:
    
    def __init__(
        self, model, train_loader: DataLoader, val_loader: DataLoader=None, 
        device: str ='cuda', velocity_only: bool=True
    ):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.velocity_only = velocity_only
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, 
                                         patience=5, verbose=True)
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def weighted_pose_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Compute weighted loss for position and velocity components."""
        
        # Retrieve the position and velocity ground-thruths 
        pos_target = targets[:, 0:3]
        vel_target = targets[:, 3:6]
        
        if self.velocity_only:
            
            # If only velocity is used, we only compute the loss for velocity
            vel_pred = predictions[:, :3]
            vel_loss = self.criterion(vel_pred, vel_target)
            
            return {
                'total_loss': vel_loss,
                'position_loss': torch.tensor(0.0, device=self.device),
                'velocity_loss': vel_loss
            }
            
        # If both position and velocity are used, compute both losses
        pos_pred = predictions[:, 0:3] 
        vel_pred = predictions[:, 3:6]

        pos_loss = self.criterion(pos_pred, pos_target)
        vel_loss = self.criterion(vel_pred, vel_target)
        
        # Weight velocity more heavily than position
        #TODO: Adjust the weights based on specific requirements
        total_loss = pos_loss + 0.1*vel_loss
        
        return {
            'total_loss': total_loss,
            'position_loss': pos_loss,
            'velocity_loss': vel_loss
        }
    
    @staticmethod
    def compute_metrics(
        predictions: torch.Tensor, targets: torch.Tensor, velocity_only: bool
    ) -> Dict:
        """Compute pose estimation metrics."""
        
        with torch.no_grad():
            
            # Retrieve groundthruth values 
            pos_target = targets[:, 0:3]
            vel_target = targets[:, 3:6]
            
            metrics = {}
            
            # Add the position velocity metrics 
            if not velocity_only:
                
                # Retrieve network predictions
                pos_pred = predictions[:, 0:3]
                vel_pred = predictions[:, 3:6]
                
                err_pos = pos_pred - pos_target
                
                # Position error (L2 norm)
                pos_error = torch.norm(err_pos, dim=1).mean()
                
                # Component-wise RMSE errors
                pos_rmse = torch.sqrt(torch.mean(err_pos**2, dim=0))

                # Update the position metrics
                metrics["position_error"] = pos_error.item(), 
                metrics["pos_rmse_xyz"] = pos_rmse.cpu().numpy() 
                
            else:
                
                # Retrieve the velocity predictions
                vel_pred = predictions[:, 0:3]
                
            # Velocity error 
            err_vel = vel_pred - vel_target
            err_vel_norm = torch.norm(err_vel, dim=1)
            
            # Velocity error (L2 norm)
            vel_error = err_vel_norm.mean()
            
            # Velocity RMSE 
            vel_rmse = torch.sqrt(torch.mean(err_vel**2, dim=0))
                
            # ELOPE metrics
            np = len(predictions)
            elope_score = torch.sum(err_vel_norm/pos_target[:, 2])/np
            
            # Store the remaining velocity-related metrics
            metrics["velocity_error"] = vel_error.item() 
            metrics["vel_rmse_xyz"] = vel_rmse.cpu().numpy() 
            metrics["elope_score"] = elope_score.cpu().numpy() 
            
            return metrics
    
    def train_epoch(self) -> Dict:
        """Train for one epoch"""
        
        self.model.train()
        
        running_loss = 0.0
        running_pos_loss = 0.0
        running_vel_loss = 0.0
        num_batches = 0
        
        for i, (events, imu, rangemeter, targets) in enumerate(self.train_loader):
            
            events = events.to(self.device)
            imu = imu.to(self.device)
            rangemeter = rangemeter.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(events, imu, rangemeter)
            predictions = outputs['prediction']
            
            # Compute loss
            loss_dict = self.weighted_pose_loss(predictions, targets)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track losses
            running_loss += loss.item()
            running_pos_loss += loss_dict['position_loss'].item()
            running_vel_loss += loss_dict['velocity_loss'].item()
            num_batches += 1
            
            if (i + 1) % 50 == 0:
                avg_loss = running_loss / num_batches
                print(f"Batch {i+1}/{len(self.train_loader)}, Loss: {avg_loss:.4f}")
        
        return {
            'total_loss': running_loss / num_batches,
            'position_loss': running_pos_loss / num_batches,
            'velocity_loss': running_vel_loss / num_batches
        }
    
    def validate(self) -> Dict:
        """Validate the model"""
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        running_loss = 0.0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            
            for events, imu, rangemeter, targets in self.val_loader: 
                
                events = events.to(self.device)
                imu = imu.to(self.device)
                rangemeter = rangemeter.to(self.device)
                targets = targets.to(self.device)
                
                # Run inference
                outputs = self.model(events, imu, rangemeter)
                predictions = outputs['prediction']
                
                # Compute the loss 
                loss_dict = self.weighted_pose_loss(predictions, targets)
                running_loss += loss_dict['total_loss'].item()
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                
        # Stack all predictions and targets together
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute the global prediction metric
        metrics = self.compute_metrics(all_predictions, all_targets, self.velocity_only)
        
        # Compute the validation loss
        val_loss = running_loss / len(self.val_loader)
        metrics['total_loss'] = val_loss
        
        return metrics
    
    def train(self, num_epochs: int, save_path: str='best_model.pth', max_patience: int=10):
        """
        Main training loop.
        
        Args:
            num_epochs (int): Number of epochs to train the model.
            save_path (str): Path to save the best model (default: 'best_model.pth').
            max_patience (int): Maximum number of epochs to wait for improvement (default: 10).
        """
        print(f"Starting training for {num_epochs} epochs...")
        
        # Check if the folder in which to store the weights exists, else create it 
        save_path = Path(save_path)
        if not save_path.parent.exists(): 
            save_path.parent.mkdir(parents=True)

        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create the figure filename with timestamp
        save_name = save_path.stem + f"_{timestamp}.pth"
        save_path_model = str(save_path.parent / save_name)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['total_loss'])
            
            # Validate
            val_metrics = self.validate()
            if val_metrics:
                self.val_losses.append(val_metrics['total_loss'])
                val_loss = val_metrics['total_loss']
            else:
                val_loss = train_metrics['total_loss']
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path_model)
                print(f"✓ New best model saved! Val loss: {val_loss:.4f}")
                patience_counter = 0
                
            else:
                patience_counter+=1
                print(f"✗ No improvement. Current best val loss: {self.best_val_loss:.4f}")
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"Train Loss: {train_metrics['total_loss']:.4f} "
                  f"(Pos: {train_metrics['position_loss']:.4f}, "
                  f"Vel: {train_metrics['velocity_loss']:.4f})")
            
            if val_metrics and not self.velocity_only:
                print(f"Val Loss: {val_metrics['total_loss']:.4f}")
                print(f"Val Metrics - Pos Error: {val_metrics['position_error']:.2f}m, "
                      f"Vel Error: {val_metrics['velocity_error']:.2f}m/s", f"elope_score: {val_metrics['elope_score']:.4f}")
            else:
                print(f"Val Loss: {val_loss:.4f}")
                print(f"Val Metrics - Vel Error: {val_metrics['velocity_error']:.2f}m/s", f"elope_score: {val_metrics['elope_score']:.4f}")
            print("-" * 50)

            if patience_counter >= max_patience:
                print(" --> Early stopping triggered. No improvement for 10 epochs.")
                break

    def plot_training(self, save_figure=False, figure_name_prefix="training_plot"):
        """
        Plots the training and validation losses over epochs.
        Args:
            save_figure (bool): If True, saves the figure to a file.
            figure_name_prefix (str): Prefix for the filename if saving the figure.
        """
        if not self.train_losses:
            print("No training data to plot. Please run the 'train' method first.")
            return

        epochs = range(1, len(self.train_losses) + 1)

        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))

        plt.plot(epochs, self.train_losses, label='Training Loss', color='skyblue', linewidth=2)
        if self.val_losses:
            plt.plot(epochs, self.val_losses, label='Validation Loss', color='salmon', linewidth=2)
            
        plt.title('Training and Validation Loss Over Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_figure:
                
            # Get current timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Ensure the 
            outpath = Path(f"{figure_name_prefix}_{timestamp}.png")
            if not outpath.parent.exists(): 
                # Ensure the output directory exists
                outpath.parent.mkdir(parents=True)

            # Create the figure filename with timestamp
            plt.savefig(str(outpath), dpi=300)
            print(f"Figure saved as: {str(outpath)}")
        
        plt.show()
