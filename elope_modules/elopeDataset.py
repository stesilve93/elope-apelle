import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from elope_modules.dataloader import DataLoader
from typing import Dict, Tuple
import time


class LunarDescentDataset(Dataset):
    def __init__(self, data_loader_instance: DataLoader,
                 sequence_ids: list,
                 event_integration_window_us: float = 1e5,
                 imu_seq_len: int = 50,
                 H: int = 200, W: int = 200, T: int = 10,
                 sample_interval: int = 10):
        """
        Custom PyTorch Dataset for lunar descent data.

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
                        H=self.H, W=self.W, T=self.T
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
        return (sample['events_tensor'], sample['imu_sequence'],
                sample['rangemeter_sequence'], sample['ground_truth'])


class LunarTrainer:
    def __init__(self, model, train_loader, val_loader=None, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, 
                                         patience=5, verbose=True)
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def weighted_pose_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict:
        """
        Compute weighted loss for position and velocity components
        """
        pos_pred, vel_pred = predictions[:, :3], predictions[:, 3:]
        pos_target, vel_target = targets[:, :3], targets[:, 3:]
        
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
    def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict:
        """
        Compute pose estimation metrics
        """
        with torch.no_grad():
            pos_pred, vel_pred = predictions[:, :3], predictions[:, 3:]
            pos_target, vel_target = targets[:, :3], targets[:, 3:]
            
            # Position error (L2 norm)
            pos_error = torch.norm(pos_pred - pos_target, dim=1).mean()
            
            # Velocity error (L2 norm)
            vel_error = torch.norm(vel_pred - vel_target, dim=1).mean()
            
            # Component-wise errors
            pos_rmse = torch.sqrt(torch.mean((pos_pred - pos_target) ** 2, dim=0))
            vel_rmse = torch.sqrt(torch.mean((vel_pred - vel_target) ** 2, dim=0))

            # ELOPE metrics
            square_vel_errors = (vel_pred[:,0] - vel_target[:,0]) ** 2 + \
                                (vel_pred[:,1] - vel_target[:,1]) ** 2 + \
                                (vel_pred[:,2] - vel_target[:,2]) ** 2
            elope_score =  (1/len(predictions))*torch.sum( (torch.sqrt(square_vel_errors))/pos_target[:, 2])
            
            return {
                'position_error': pos_error.item(),
                'velocity_error': vel_error.item(),
                'pos_rmse_xyz': pos_rmse.cpu().numpy(),
                'vel_rmse_xyz': vel_rmse.cpu().numpy(),
                'elope_score': elope_score.cpu().numpy()
            }
    
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
                
                outputs = self.model(events, imu, rangemeter)
                predictions = outputs['prediction']
                
                # Compute loss
                loss_dict = self.weighted_pose_loss(predictions, targets)
                running_loss += loss_dict['total_loss'].item()
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Compute overall metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.compute_metrics(all_predictions, all_targets)
        
        val_loss = running_loss / len(self.val_loader)
        metrics['total_loss'] = val_loss
        
        return metrics
    
    def train(self, num_epochs: int, save_path: str = 'best_model.pth'):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        
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
                torch.save(self.model.state_dict(), save_path)
                print(f"✓ New best model saved! Val loss: {val_loss:.4f}")
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"Train Loss: {train_metrics['total_loss']:.4f} "
                  f"(Pos: {train_metrics['position_loss']:.4f}, "
                  f"Vel: {train_metrics['velocity_loss']:.4f})")
            
            if val_metrics:
                print(f"Val Loss: {val_metrics['total_loss']:.4f}")
                print(f"Val Metrics - Pos Error: {val_metrics['position_error']:.2f}m, "
                      f"Vel Error: {val_metrics['velocity_error']:.2f}m/s", f"elope_score: {val_metrics['elope_score']:.4f}")
            print("-" * 50)
###########     testing code    ###########
if __name__ == "__main__":
    datapath = './elope_data' # Adjust as needed
    data_loader = DataLoader(datapath=datapath)

    # Generate a list of your 40 train trajectory IDs
    # Assuming they are '0000.npz' to '0039.npz'
    train_sequence_ids = [str(i).zfill(4) for i in range(28)]

    # Create the dataset
    train_dataset = LunarDescentDataset(
        data_loader_instance=data_loader,
        sequence_ids=train_sequence_ids,
        event_integration_window_us=100000, # 100ms window
        imu_seq_len=50,
        H=200, W=200, T=10,
        sample_interval=5 # Adjust sampling frequency
    )

    # Create a PyTorch DataLoader
    batch_size = 16 # Or whatever fits your GPU memory
    train_dataloader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4) # num_workers for parallel data loading