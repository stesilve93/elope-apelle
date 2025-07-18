
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
from elope.utils import LOGGER, load_yaml

from .losses import loss_elope, loss_mse_abs, loss_mse_rel

class LunarTrainer: 
    
    def __init__(
        self, 
        model_cfg: str | Path | dict, 
        model: nn.Module, 
        train_loader: ElopeDataLoader, 
        val_loader: ElopeDataLoader=None, 
        device: str ='cuda'
    ):
        
        # Retrieve the model configuration
        if isinstance(model_cfg, (str | Path)): 
            model_cfg = load_yaml(model_cfg)
        
        # Store the configuration for the model
        self.cfg = model_cfg 
        self.velocity_only = bool(self.cfg["velocity_only"])

        # Store the network model and hardware device 
        self.model = model
        self.device = device

        # Store the validation and training dataset loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Check whether the model is sequence to sequence.
        self.seq2seq = bool(self.cfg["seq2seq"])
        
        # Retrieve the loss weights
        cfg_losses = self.cfg["losses"]
        self.lmb_mse_abs = cfg_losses["lmb_mse_abs"] 
        self.lmb_mse_rel = cfg_losses["lmb_mse_rel"] 
        self.lmb_elope   = cfg_losses["lmb_elope"] 
        
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
        
        if not self.seq2seq: 
            # If we are only predicting the final state, retrieve only that part.
            targets = targets[:, -1]
            
        pos_target = targets[:, 0:3]
        vel_target = targets[:, 3:6]
        
        # Retrieve the predicted velocities
        vel_pred = predictions[:, 0:3] if self.velocity_only else predictions[:, 3:6]
        
        # Compute the different losses
        l_mse_abs = self.lmb_mse_abs*loss_mse_abs(vel_pred, vel_target)
        l_mse_rel = self.lmb_mse_rel*loss_mse_rel(vel_pred, vel_target)
        l_elope   = self.lmb_elope*loss_elope(vel_pred, vel_target, pos_target)
        
        # Compute the global loss function
        total_loss  = l_elope + l_mse_abs + l_mse_rel
        
        loss = {
            'vel_mse_abs_loss': l_mse_abs, 
            'vel_mse_rel_loss': l_mse_rel,
            'elope_loss': l_elope,
            'total_loss': total_loss
        }
        
        return loss
        
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
                
                # Compute the position absolute and relative MSE 
                metrics["pos_mse_abs"] = loss_mse_abs(pos_pred, pos_target)
                metrics["pos_mse_rel"] = loss_mse_rel(pos_pred, pos_target)
                
            else:
                
                # Retrieve the velocity predictions
                vel_pred = predictions[:, 0:3]
            
            # Compute the velocity absolute and relative MSE
            vel_mse_abs = loss_mse_abs(vel_pred, vel_target) 
            vel_mse_rel = loss_mse_rel(vel_pred, vel_target)
            
            # ELOPE metrics
            elope_score = loss_elope(vel_pred, vel_target, pos_target)
            
            # Store the remaining velocity-related metrics
            metrics["vel_mse_abs"] = vel_mse_abs.cpu().numpy()
            metrics["vel_mse_rel"] = vel_mse_rel.cpu().numpy() 
            metrics["elope_score"] = elope_score.cpu().numpy() 
            
            return metrics
    
    def train_epoch(self, epoch: int, num_epochs: int) -> dict:
        """Train for one epoch."""
        
        self.model.train()
        
        running_loss = 0.0
        running_elope_loss = 0.0 
        running_mse_abs_loss = 0.0 
        running_mse_rel_loss = 0.0 
        
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
            
            running_elope_loss += loss_dict['elope_loss'].item()
            running_mse_abs_loss += loss_dict['vel_mse_abs_loss'].item()
            running_mse_rel_loss += loss_dict['vel_mse_rel_loss'].item()
            
            num_batches += 1
            
            if i % tbar.miniters == 0:
                tbar.set_postfix(avg_loss=f"{running_loss/num_batches:.3f}")
            
        return {
            'total_loss': running_loss / num_batches,
            'elope_loss': running_elope_loss / num_batches,
            'vel_mse_abs_loss': running_mse_abs_loss / num_batches,
            'vel_mse_rel_loss': running_mse_rel_loss / num_batches
        }
    
    def validate(self) -> dict:
        """Validate the model."""
        
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        
        running_loss = 0.0
        running_elope_loss = 0.0 
        running_mse_abs_loss = 0.0 
        running_mse_rel_loss = 0.0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            
            for events, imu, rangemeter, targets, times in self.val_loader: 
                
                times = times.to(self.device)
                targets = targets.to(self.device)
                rangemeter = rangemeter.to(self.device)
                events = events.to(self.device)
                imu = imu.to(self.device)
                
                # Run inference
                outputs = self.model(events, imu, rangemeter)
                predictions = outputs['prediction']
                
                # Compute the loss 
                loss_dict = self.weighted_pose_loss(predictions, targets)
                
                running_loss += loss_dict['total_loss'].item()
                
                running_elope_loss   += loss_dict['elope_loss'].item()
                running_mse_abs_loss += loss_dict['vel_mse_abs_loss'].item()
                running_mse_rel_loss += loss_dict['vel_mse_rel_loss'].item()
                
                # Check whether we only care about the last target state 
                if not self.seq2seq: 
                    targets = targets[:, -1]
            
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                
        # Stack all predictions and targets together
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute the global prediction metric
        metrics = self.compute_metrics(all_predictions, all_targets, self.velocity_only)
        
        # Compute the validation loss
        metrics['total_loss'] = running_loss / len(self.val_loader)
        metrics['elope_loss'] = running_elope_loss / len(self.val_loader)
        metrics['vel_mse_abs_loss'] = running_mse_abs_loss / len(self.val_loader)
        metrics['vel_mse_rel_loss'] = running_mse_rel_loss / len(self.val_loader)
        return metrics
    
    def train(self, num_epochs: int, max_patience: int=10, **kwargs):
        """
        Main training loop.
        
        Args:
            num_epochs (int): Number of epochs to train the model.
            save_path (str): Path to save the best model (default: 'best_model.pth').
            max_patience (int): Maximum number of epochs to wait for improvement (default: 10).
        """
        
        LOGGER.info(f"Starting training for {num_epochs} epochs.")
        
        # Check if the folder in which to store the weights exists, else create it 
        cfg_weights = self.cfg["weights"]
                
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Retrieve the folder in which to store the results
        save_path = Path(kwargs.get("save_path", cfg_weights["path"]))
        save_name = kwargs.get("save_name", cfg_weights["name"]) + f"_{timestamp}"
        save_path_model = save_path / save_name
        save_path_model.mkdir(parents=True, exist_ok=False)
        
        # Retrieve the number of epochs between each saved checkpoint
        ckp_epochs = int(cfg_weights["checkpoint_epochs"])
        
        for epoch in range(num_epochs):
            
            # Train
            train_metrics = self.train_epoch(epoch+1, num_epochs)
            self.train_losses.append(train_metrics['total_loss'])
            
            # Validate
            val_metrics = self.validate()
            if val_metrics:
                self.val_losses.append(val_metrics['total_loss'])
                val_loss = val_metrics['total_loss']
                loss_metrics = val_metrics
            
            else:
                val_loss = train_metrics['total_loss']
                loss_metrics = train_metrics
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Check whether we are at a checkpoint for saving the weights
            if (epoch % ckp_epochs) == 0:
                torch.save(self.model.state_dict(), save_path_model / f"{epoch}.pth")
                print(
                    " "*6, f"Model weights saved! Val. Loss: {val_loss:.4f} / ", 
                    f"Train Loss: {train_metrics['total_loss']:.4f}"
                )
                
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path_model / "best.pth")
                print(
                    " "*6, f"New best model saved! Val. Loss: {val_loss:.4f} /", 
                    f"Train Loss: {train_metrics['total_loss']:.4f}"
                )
                
                patience_counter = 0
                
            else:
                patience_counter += 1
            
            # Display the validation losses (e.g., each entry in the dictionary)
            loss_names = tuple(loss_metrics.keys())
            loss_values = tuple([loss_metrics[ln] for ln in loss_names])
            
            print((" " * 6 + '%20s' * len(loss_names)) % loss_names)
            print((" " * 6 + '%20.5f' * len(loss_names)) % loss_values)
            print("\n")

            if patience_counter >= max_patience:
                LOGGER.warning("Early stopping triggered. No improvement for 10 epochs.")
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
