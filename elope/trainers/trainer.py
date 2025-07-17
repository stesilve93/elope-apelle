
import datetime 
import time

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import tqdm 

from pathlib import Path 

from torch import nn 
from torch import optim  
from torch.optim.lr_scheduler import ReduceLROnPlateau

from elope.datasets import ElopeDataLoader
from elope.utils import LOGGER

from .losses import loss_elope, loss_mse_rel

class LunarTrainer: 
    
    def __init__(
        self, model, train_loader: ElopeDataLoader, val_loader: ElopeDataLoader=None, 
        device: str ='cuda', velocity_only: bool=True, integral_loss: bool=False
    ):
        
        # Ensure both datasets are set to the same sequential settings. This check is needed 
        # because if the validation set is not the same, the computation of integral 
        # cost functions may yield incoherent results.
        
        assert train_loader.sequential_samples == val_loader.sequential_samples
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.velocity_only = velocity_only
        
        self.sequential_samples = self.train_loader.sequential_samples
        
        # True if the position values should be used to evaluate an integral loss.
        self.integral_loss = integral_loss
        
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def weighted_pose_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """Compute weighted loss for position and velocity components."""
        
        pos_target = targets[:, 3:6]
        vel_target = targets[:, 3:6]
        
        # Retrieve the predicted velocities
        vel_pred = predictions[:, 0:3] if self.velocity_only else predictions[:, 3:6]
        
        loss = {
            'position_loss': torch.tensor(0.0, device=self.device), 
            'velocity_loss': torch.tensor(0.0, device=self.device), 
            # 'total_loss': loss_elope(vel_pred, vel_target, pos_target)
            'total_loss': loss_mse_rel(vel_pred, vel_target)
        }
        
        return loss
        
        # Retrieve the position and velocity ground-thruths 
        # pos_target = targets[:, 0:3]
        
        # loss = {} 
        
        # total_loss = torch.tensor(0.0, device=self.device)
        
        # # TODO: update losses computation using weights 
        
        # # MSE Position loss 
        # # MSE Velocity loss 
        # # Integral loss 
        # # Relative MSE Position loss 
        # # Relative MSE Velocity loss 
        # # Relative MSE Integral loss 
        # # ELOPE score 
        
        # if not self.velocity_only: 
            
        #     # Weight for the velocity loss: velocity is weighted more heavily than pos
        #     weight_vel = 0.1
            
        #     # Retrieve the preidctions
        #     pos_pred = predictions[:, 0:3]
        #     vel_pred = predictions[:, 3:6]
            
        #     # Compute and store the position loss
        #     pos_loss = self.criterion(pos_pred, pos_target)
        #     loss['position_loss'] = pos_loss 
            
        #     # Update the total loss with the position component 
        #     total_loss += pos_loss
        
        # else: 
            
        #     # Weight for the velocity loss 
        #     weight_vel = 1.0
        
        #     # Retrieve the velocity predictions
        #     vel_pred = predictions[:, 0:3]
            
        #     # Compute and store the position loss
        #     loss['position_loss'] = torch.tensor(0.0, device=self.device)
        
        # # Compute the velocity loss and update the total loss
        # vel_loss = self.criterion(vel_pred, vel_target)    
        # total_loss += weight_vel*vel_loss
        
        # # vel_loss_est = torch.sum(torch.norm(vel_pred-vel_target, dim=1)**2)/3/vel_pred.shape[0]
        # # print(vel_loss, vel_loss_est)
        # # raise RuntimeError
            
        # if self.integral_loss: 
            
        #     # TODO: this requires that I know the timings at which the predictions were made
        #     integral_loss = torch.tensor(0.0, device=self.device)
        #     # integral_loss = torch.trapezoid()
            
        #     # Compute an additional loss due to the integral of the position 
        #     loss['integral_loss'] = integral_loss   
        #     total_loss += integral_loss
            
        # else: 
        #     loss['integral_loss'] = torch.tensor(0.0, device=self.device)
            

        # # Store the velocity and total loss 
        # loss['velocity_loss'] = vel_loss
        # loss['total_loss'] = total_loss 
        
        # return loss
        
    @staticmethod
    def compute_metrics(
        predictions: torch.Tensor, targets: torch.Tensor, velocity_only: bool
    ) -> dict:
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
            elope_score = loss_elope(vel_pred, vel_target, pos_target)
            
            # elope_score = torch.sum(err_vel_norm/torch.abs(pos_target[:, 2]))/np
            
            # Store the remaining velocity-related metrics
            metrics["velocity_error"] = vel_error.item() 
            metrics["vel_rmse_xyz"] = vel_rmse.cpu().numpy() 
            metrics["elope_score"] = elope_score.cpu().numpy() 
            
            return metrics
    
    def train_epoch(self, epoch: int, num_epochs: int) -> dict:
        """Train for one epoch."""
        
        self.model.train()
        
        running_loss = 0.0
        running_pos_loss = 0.0
        running_vel_loss = 0.0
        num_batches = 0
        
        # Create the bar to display current iterations
        tbar = tqdm.tqdm(
            self.train_loader, desc=f"       Epoch {epoch:02d}/{num_epochs:02d}", unit="i", 
            ncols=90, miniters=5
        )
        
        for i, (events, imu, rangemeter, targets, times) in enumerate(tbar):
        
            times = times.to(self.device)
            targets = targets.to(self.device)
            rangemeter = rangemeter.to(self.device)
            events = events.to(self.device)
            imu = imu.to(self.device)
            
            if self.sequential_samples: 
                # Remove from all the vectors the first dimension as the 'batch'
                # dimension is already embedded in the samples sequentiality 
                times = times.squeeze(0)
                targets = targets.squeeze(0)
                
                rangemeter = rangemeter.squeeze(0)
                events = events.squeeze(0)
                imu = imu.squeeze(0)
            
            # print(imu.shape, rangemeter.shape, events.shape, times.shape, targets.shape)
            # raise RuntimeError
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
            
            if i % tbar.miniters == 0:
                tbar.set_postfix(avg_loss=f"{running_loss/num_batches:.3f}")
            
        return {
            'total_loss': running_loss / num_batches,
            'position_loss': running_pos_loss / num_batches,
            'velocity_loss': running_vel_loss / num_batches
        }
    
    def validate(self) -> dict:
        """Validate the model."""
        
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        running_loss = 0.0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            
            for events, imu, rangemeter, targets, times in self.val_loader: 
                
                times = times.to(self.device)
                targets = targets.to(self.device)
                rangemeter = rangemeter.to(self.device)
                events = events.to(self.device)
                imu = imu.to(self.device)
                
                if self.sequential_samples: 
                    # Remove from all the vectors the first dimension as the 'batch'
                    # dimension is already embedded in the samples sequentiality 
                    times = times.squeeze(0)
                    targets = targets.squeeze(0)
                    
                    rangemeter = rangemeter.squeeze(0)
                    events = events.squeeze(0)
                    imu = imu.squeeze(0)
                
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
        
        LOGGER.info(f"Starting training for {num_epochs} epochs.")
        
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
            
            # Train
            train_metrics = self.train_epoch(epoch+1, num_epochs)
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
                print(
                    " "*6, f"New best model saved! Val. Loss: {val_loss:.4f} /", 
                    f"Train Loss: {train_metrics['total_loss']:.4f}\n"
                )
                
                patience_counter = 0
                
            else:
                patience_counter+=1

            if patience_counter >= max_patience:
                LOGGER.warning(" --> Early stopping triggered. No improvement for 10 epochs.")
                break

    def plot_training(self, save_figure=False, figure_name_prefix="training_plot"):
        """
        Plots the training and validation losses over epochs.
        Args:
            save_figure (bool): If True, saves the figure to a file.
            figure_name_prefix (str): Prefix for the filename if saving the figure.
        """
        if not self.train_losses:
            LOGGER.warning("No training data to plot. Please run the 'train' method first.")
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
            LOGGER.info(f"Figure saved as: {str(outpath)}")
        
        plt.show()
