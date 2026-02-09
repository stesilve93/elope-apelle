
import datetime 
import time

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import tqdm 

from pathlib import Path 

from torch import nn 
from torch import optim  
from torch.optim.lr_scheduler import ReduceLROnPlateau

from elope.datasets import ElopeDataLoader
from elope.utils import LOGGER, load_yaml
from elope.models.emmnet.emmnetOf import EventWarper

from .losses import loss_elope, loss_mse_abs, loss_mse_rel

class LunarTrainer: 
    
    def __init__(
        self, 
        model_cfg: str | Path | dict, 
        model: nn.Module, 
        train_loader: ElopeDataLoader, 
        val_loader: ElopeDataLoader=None, 
        device: str ='cuda',
        val_metric: str="elope_score",
        latent_log: dict | None = None
    ):
        
        # Retrieve the model configuration
        if isinstance(model_cfg, (str | Path)): 
            model_cfg = load_yaml(model_cfg)
        
        # Store the configuration for the model
        self.cfg = model_cfg 
        
        # Store the network model and hardware device 
        self.model = model
        self.device = device
        
        # Hard-coded value for the velocity prediction
        self.velocity_only = True 
        
        # Store the validation and training dataset loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Store the name of the metric used to identify the best model 
        self.val_metric_key = val_metric

        # Retrieve which state the model gives in output 
        self.output_type = self.cfg["output_type"]
        assert self.output_type in (
            "initial_state", "final_state", "central_state", "sequence"
        )
        
        # Retrieve the sequence length
        self.seq_len = int(self.cfg["sequence_length"])
        if self.output_type == "central_state": 
            assert self.seq_len % 2 == 1
        
        # Retrieve the loss weights
        cfg_losses = self.cfg["losses"]
        self.lmb_mse_abs = cfg_losses["lmb_mse_abs"] 
        self.lmb_mse_rel = cfg_losses["lmb_mse_rel"] 
        self.lmb_elope   = cfg_losses["lmb_elope"] 

        # Optional self-supervised event-flow loss
        self.aux_flow_weight = float(self.cfg.get("aux_flow_weight", 0.0))
        self.aux_flow_smooth_weight = float(self.cfg.get("aux_flow_smooth_weight", 0.1))
        self.aux_flow_enabled = self.aux_flow_weight > 0.0
        self.event_warper = EventWarper().to(self.device) if self.aux_flow_enabled else None
        
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_val_epoch = None

        # Optional latent logging
        self.latent_log = latent_log or {}
        self.latent_log_enabled = bool(self.latent_log.get("enabled", False))
        self.latent_log_split = str(self.latent_log.get("split", "val"))
        self.latent_log_max_batches = int(self.latent_log.get("max_batches", 10))
        self.latent_log_every_n_epochs = int(self.latent_log.get("every_n_epochs", 1))
        self.latent_log_dir = self.latent_log.get("path", None)
        self._latent_cache = {}
        self._latent_buffers = None
        self._fusion_hook_handle = None
        if self.latent_log_enabled and hasattr(self.model, "fusion"):
            self._fusion_hook_handle = self.model.fusion.register_forward_hook(
                self._capture_fusion
            )

    def _capture_fusion(self, module, inputs, output):
        # Cache fused features and attention weights from the fusion module.
        if isinstance(output, (tuple, list)) and len(output) > 0:
            fused = output[0]
            self._latent_cache["fused"] = fused.detach()
            if len(output) > 1 and torch.is_tensor(output[1]):
                self._latent_cache["attention"] = output[1].detach()
        input_shapes = []
        event_tokens = None
        total_tokens = None
        if len(inputs) > 0 and torch.is_tensor(inputs[0]) and inputs[0].ndim >= 2:
            event_tokens = int(inputs[0].shape[1])
        if event_tokens is not None:
            total_tokens = event_tokens + 3
        for inp in inputs:
            input_shapes.append(tuple(inp.shape) if torch.is_tensor(inp) else None)
        self._latent_cache["input_shapes"] = input_shapes
        self._latent_cache["event_tokens"] = event_tokens
        self._latent_cache["total_tokens"] = total_tokens

    def _select_targets_times(self, targets: torch.Tensor, times: torch.Tensor) -> tuple:
        if self.output_type == "initial_state":
            return targets[:, 0], times[:, 0]
        if self.output_type == "final_state":
            return targets[:, -1], times[:, -1]
        if self.output_type == "central_state":
            return targets[:, self.seq_len // 2], times[:, self.seq_len // 2]
        return targets, times

    def _init_latent_buffers(self):
        self._latent_buffers = {
            "fused": [],
            "attention": [],
            "pred": [],
            "target_vel": [],
            "target_pos": [],
            "time": [],
            "event_tokens": [],
            "total_tokens": []
        }

    def _maybe_log_batch(self, outputs: dict, targets: torch.Tensor, times: torch.Tensor):
        if not self.latent_log_enabled or self._latent_buffers is None:
            return
        if len(self._latent_buffers["fused"]) >= self.latent_log_max_batches:
            return

        fused = self._latent_cache.get("fused", None)
        if fused is None:
            return

        attention = outputs.get("attention_weights", None)
        if attention is None:
            attention = self._latent_cache.get("attention", None)

        targets_sel, times_sel = self._select_targets_times(targets, times)
        target_pos = targets_sel[..., 0:3].detach().cpu()
        target_vel = targets_sel[..., 3:6].detach().cpu()

        self._latent_buffers["fused"].append(fused.detach().cpu())
        if attention is not None and torch.is_tensor(attention):
            self._latent_buffers["attention"].append(attention.detach().cpu())
        self._latent_buffers["pred"].append(outputs["prediction"].detach().cpu())
        self._latent_buffers["target_vel"].append(target_vel)
        self._latent_buffers["target_pos"].append(target_pos)
        self._latent_buffers["time"].append(times_sel.detach().cpu())
        self._latent_buffers["event_tokens"].append(
            torch.tensor(self._latent_cache.get("event_tokens", -1))
        )
        self._latent_buffers["total_tokens"].append(
            torch.tensor(self._latent_cache.get("total_tokens", -1))
        )

    def _save_latent_buffers(self, epoch: int, split: str):
        if not self.latent_log_enabled or self._latent_buffers is None:
            return
        if self.latent_log_dir is None:
            return

        out_dir = Path(self.latent_log_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"latents_{split}_epoch_{epoch:03d}.npz"

        fused = torch.cat(self._latent_buffers["fused"], dim=0).numpy()
        pred = torch.cat(self._latent_buffers["pred"], dim=0).numpy()
        target_vel = torch.cat(self._latent_buffers["target_vel"], dim=0).numpy()
        target_pos = torch.cat(self._latent_buffers["target_pos"], dim=0).numpy()
        times = torch.cat(self._latent_buffers["time"], dim=0).numpy()

        attention = None
        if len(self._latent_buffers["attention"]) > 0:
            first_attn = self._latent_buffers["attention"][0]
            if first_attn.ndim == 2:
                attention = torch.stack(self._latent_buffers["attention"], dim=0).numpy()
            else:
                attention = torch.cat(self._latent_buffers["attention"], dim=0).numpy()

        event_tokens = torch.stack(self._latent_buffers["event_tokens"]).numpy()
        total_tokens = torch.stack(self._latent_buffers["total_tokens"]).numpy()

        np.savez_compressed(
            out_path,
            fused=fused,
            pred=pred,
            target_vel=target_vel,
            target_pos=target_pos,
            times=times,
            attention=attention,
            event_tokens=event_tokens,
            total_tokens=total_tokens
        )
        
    def weighted_pose_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """Compute weighted loss for position and velocity components."""
        
        # Check which output we need to retrieve 
        if self.output_type == "initial_state":
            targets = targets[:, 0]               # (B, 6)
        elif self.output_type == "final_state": 
            targets = targets[:, -1]              # (B, 6)
        elif self.output_type == "central_state": 
            targets = targets[:, self.seq_len // 2]
            
        pos_target = targets[..., 0:3]
        vel_target = targets[..., 3:6]

        # Retrieve the predicted velocities
        vel_pred = predictions[..., 0:3]
        
        # Assemble the global loss function
        total_loss = torch.tensor(0.0, requires_grad=True).to(targets)
         
        # Initialize the loss functions
        l_mse_abs = torch.tensor(0.0, requires_grad=True).to(targets) 
        l_mse_rel = torch.tensor(0.0, requires_grad=True).to(targets)
        l_elope   = torch.tensor(0.0, requires_grad=True).to(targets)
         
        # Compute the different losses
        if self.lmb_mse_abs != 0: 
            l_mse_abs = self.lmb_mse_abs*loss_mse_abs(vel_pred, vel_target)
            total_loss += l_mse_abs
            
        if self.lmb_mse_rel != 0: 
            l_mse_rel = self.lmb_mse_rel*loss_mse_rel(vel_pred, vel_target)
            total_loss += l_mse_rel
        
        if self.lmb_elope != 0: 
            l_elope = self.lmb_elope*loss_elope(vel_pred, vel_target, pos_target)
            total_loss += l_elope
        
        loss = {
            'vel_mse_abs_loss': l_mse_abs, 
            'vel_mse_rel_loss': l_mse_rel,
            'elope_loss': l_elope,
            'total_loss': total_loss
        }
        
        return loss

    @staticmethod
    def _flow_smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
        grad_u_x = torch.abs(flow[:, 0, :, 1:] - flow[:, 0, :, :-1])
        grad_u_y = torch.abs(flow[:, 0, 1:, :] - flow[:, 0, :-1, :])
        grad_v_x = torch.abs(flow[:, 1, :, 1:] - flow[:, 1, :, :-1])
        grad_v_y = torch.abs(flow[:, 1, 1:, :] - flow[:, 1, :-1, :])
        return grad_u_x.mean() + grad_u_y.mean() + grad_v_x.mean() + grad_v_y.mean()

    def _compute_flow_aux_loss(
        self, events: torch.Tensor, flow_pred: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # events shape: (B, T, 2, C, H, W)
        if events.size(1) < 2:
            return torch.tensor(0.0, device=events.device), torch.tensor(0.0, device=events.device)

        events_t0 = events[:, -2]
        events_t1 = events[:, -1]
        B, P, C, H, W = events_t0.shape
        events_t0 = events_t0.reshape(B, P * C, H, W)
        events_t1 = events_t1.reshape(B, P * C, H, W)

        warped = self.event_warper(events_t0, flow_pred)
        photo_loss = F.l1_loss(warped, events_t1, reduction='mean')
        smooth_loss = self._flow_smoothness_loss(flow_pred)
        return photo_loss, smooth_loss
        
    @staticmethod
    def compute_metrics(
        predictions: torch.Tensor, targets: torch.Tensor, velocity_only: bool
    ) -> dict:
        """Compute pose estimation metrics."""
        
        with torch.no_grad():
            
            # Retrieve groundthruth values 
            pos_target = targets[..., 0:3]
            vel_target = targets[..., 3:6]
            
            metrics = {}
            
            # Add the position velocity metrics 
            if not velocity_only:
                
                # Retrieve network predictions
                pos_pred = predictions[..., 0:3]
                vel_pred = predictions[..., 3:6]
                
                # Compute the position absolute and relative MSE 
                metrics["pos_mse_abs"] = loss_mse_abs(pos_pred, pos_target)
                metrics["pos_mse_rel"] = loss_mse_rel(pos_pred, pos_target)
                
            else:
                
                # Retrieve the velocity predictions
                vel_pred = predictions[..., 0:3]
            
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
        running_flow_photo = 0.0
        running_flow_smooth = 0.0
        
        num_batches = 0

        do_latent_log = (
            self.latent_log_enabled
            and self.latent_log_split == "train"
            and (epoch % self.latent_log_every_n_epochs == 0)
        )
        if do_latent_log:
            self._init_latent_buffers()
        
        # Create the bar to display current iterations
        tbar = tqdm.tqdm(
            self.train_loader, desc=f"       Epoch {epoch:02d}/{num_epochs:02d}", unit="i", 
            ncols=120, miniters=5
        )
        
        for i, (events, imu, rangemeter, targets, times) in enumerate(tbar):
            
            times = times.to(self.device)
            targets = targets.to(self.device)
            rangemeter = rangemeter.to(self.device)
            events = events.to(self.device)
            imu = imu.to(self.device)

            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(times, events, imu, rangemeter)
            predictions = outputs['prediction']

            if do_latent_log:
                self._maybe_log_batch(outputs, targets, times)
            
            # Compute loss
            loss_dict = self.weighted_pose_loss(predictions, targets)
            loss = loss_dict['total_loss']

            if self.aux_flow_enabled and outputs.get("flow_prediction") is not None:
                flow_pred = outputs["flow_prediction"]
                if flow_pred is not None:
                    photo_loss, smooth_loss = self._compute_flow_aux_loss(events, flow_pred)
                    loss = loss + self.aux_flow_weight * (
                        photo_loss + self.aux_flow_smooth_weight * smooth_loss
                    )
                    loss_dict["flow_photo_loss"] = photo_loss
                    loss_dict["flow_smooth_loss"] = smooth_loss

            # Backward pass
            # with torch.autograd.detect_anomaly():
            loss.backward()
            
            # Gradient clipping for stability
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track losses
            running_loss += loss.item()
            
            running_elope_loss += loss_dict['elope_loss'].item()
            running_mse_abs_loss += loss_dict['vel_mse_abs_loss'].item()
            running_mse_rel_loss += loss_dict['vel_mse_rel_loss'].item()
            if "flow_photo_loss" in loss_dict:
                running_flow_photo += loss_dict["flow_photo_loss"].item()
                running_flow_smooth += loss_dict["flow_smooth_loss"].item()
            
            num_batches += 1
            
            if i % tbar.miniters == 0:
                tbar.set_postfix(avg_loss=f"{running_loss/num_batches:.6f}")

        if do_latent_log:
            self._save_latent_buffers(epoch, "train")

        return {
            'total_loss': running_loss / num_batches,
            'elope_loss': running_elope_loss / num_batches,
            'vel_mse_abs_loss': running_mse_abs_loss / num_batches,
            'vel_mse_rel_loss': running_mse_rel_loss / num_batches,
            'flow_photo_loss': (running_flow_photo / num_batches) if self.aux_flow_enabled else 0.0,
            'flow_smooth_loss': (running_flow_smooth / num_batches) if self.aux_flow_enabled else 0.0
        }
    
    def validate(self, epoch: int | None = None) -> dict:
        """Validate the model."""
        
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        
        running_loss = 0.0
        running_elope_loss = 0.0 
        running_mse_abs_loss = 0.0 
        running_mse_rel_loss = 0.0
        running_flow_photo = 0.0
        running_flow_smooth = 0.0
        
        all_predictions = []
        all_targets = []

        do_latent_log = (
            self.latent_log_enabled
            and self.latent_log_split == "val"
            and (epoch is not None)
            and (epoch % self.latent_log_every_n_epochs == 0)
        )
        if do_latent_log:
            self._init_latent_buffers()
        
        with torch.no_grad():
            
            for events, imu, rangemeter, targets, times in self.val_loader: 
                
                times = times.to(self.device)
                targets = targets.to(self.device)
                rangemeter = rangemeter.to(self.device)
                events = events.to(self.device)
                imu = imu.to(self.device)
                
                # Run inference
                outputs = self.model(times, events, imu, rangemeter)
                predictions = outputs['prediction']

                if do_latent_log:
                    self._maybe_log_batch(outputs, targets, times)
                
                # Compute the loss 
                loss_dict = self.weighted_pose_loss(predictions, targets)
                
                running_loss += loss_dict['total_loss'].item()
                
                running_elope_loss   += loss_dict['elope_loss'].item()
                running_mse_abs_loss += loss_dict['vel_mse_abs_loss'].item()
                running_mse_rel_loss += loss_dict['vel_mse_rel_loss'].item()

                if self.aux_flow_enabled and outputs.get("flow_prediction") is not None:
                    flow_pred = outputs["flow_prediction"]
                    if flow_pred is not None:
                        photo_loss, smooth_loss = self._compute_flow_aux_loss(events, flow_pred)
                        running_flow_photo += photo_loss.item()
                        running_flow_smooth += smooth_loss.item()
                
                # Check which output we need to retrieve 
                if self.output_type == "initial_state":
                    targets = targets[:, 0]               # (B, 6)
                elif self.output_type == "final_state": 
                    targets = targets[:, -1]              # (B, 6)
                elif self.output_type == "central_state": 
                    targets = targets[:, self.seq_len // 2]
                    
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                
        # Stack all predictions and targets together
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        if do_latent_log:
            self._save_latent_buffers(epoch, "val")
        
        # Compute the global prediction metric
        metrics = self.compute_metrics(all_predictions, all_targets, self.velocity_only)
        
        # Compute the validation loss
        metrics['total_loss'] = running_loss / len(self.val_loader)
        metrics['elope_loss'] = running_elope_loss / len(self.val_loader)
        metrics['vel_mse_abs_loss'] = running_mse_abs_loss / len(self.val_loader)
        metrics['vel_mse_rel_loss'] = running_mse_rel_loss / len(self.val_loader)
        if self.aux_flow_enabled:
            metrics['flow_photo_loss'] = running_flow_photo / len(self.val_loader)
            metrics['flow_smooth_loss'] = running_flow_smooth / len(self.val_loader)
        return metrics
    
    def train(self, num_epochs: int, max_patience: int=10, save_path: str=None, **kwargs):
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
        
        if save_path is None:
            
            # Get current timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
            # Retrieve the folder in which to store the results
            save_path = Path(kwargs.get("save_path", cfg_weights["path"]))
            save_name = kwargs.get("save_name", cfg_weights["name"]) + f"_{timestamp}"
            save_path_model = save_path / save_name
            save_path_model.mkdir(parents=True, exist_ok=False)
        
        else: 
            save_path_model = save_path    
        
            
        # Retrieve the number of epochs between each saved checkpoint
        ckp_epochs = int(cfg_weights["checkpoint_epochs"])
        
        patience_counter = 0
        for epoch in range(num_epochs):
            
            # Train
            train_metrics = self.train_epoch(epoch+1, num_epochs)
            self.train_losses.append(train_metrics['total_loss'])
            
            # Validate
            val_metrics = self.validate(epoch=epoch+1)
            if val_metrics:
                self.val_losses.append(val_metrics['total_loss'])
                val_loss = val_metrics[self.val_metric_key]
                loss_metrics = val_metrics
            
            else:
                val_loss = train_metrics[self.val_metric_key]
                loss_metrics = train_metrics
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Check whether we are at a checkpoint for saving the weights
            if epoch > 0 and (epoch % ckp_epochs) == 0:
                torch.save(self.model.state_dict(), save_path_model / f"{epoch}.pth")
                print(
                    " "*6, f"Model weights saved! Val. Metric ({self.val_metric_key}): " 
                    f"{val_loss:.6f} / Train Loss: {train_metrics['total_loss']:.6f}"
                )
                
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_epoch = epoch + 1
                torch.save(self.model.state_dict(), save_path_model / "best.pth")
                print(
                    " "*6, f"New best model saved! Val. Metric ({self.val_metric_key}): "
                    f"{val_loss:.6f} / Train Loss: {train_metrics['total_loss']:.6f}"
                )
                
                patience_counter = 0
                
            else:
                patience_counter += 1
            
            # Display the validation losses (e.g., each entry in the dictionary)
            loss_names = tuple(loss_metrics.keys())
            loss_values = tuple([loss_metrics[ln] for ln in loss_names])
            
            if self.best_val_epoch is None:
                best_epoch_label = "n/a"
                best_loss_label = "n/a"
            else:
                best_epoch_label = f"{self.best_val_epoch:02d}"
                best_loss_label = f"{self.best_val_loss:.6f}"

            print(
                " " * 7
                + f"Best ({self.val_metric_key}): {best_loss_label} @ epoch {best_epoch_label}"
            )
            print((" " * 6 + '%20s' * len(loss_names)) % loss_names)
            print((" " * 6 + '%20.5f' * len(loss_names)) % loss_values)
            print("\n")

            if patience_counter >= max_patience:
                LOGGER.warning(
                    "Early stopping triggered. No improvement for {max_patience} epochs."
                )
                break

    def plot_training(
        self, save_figure=False, path: str | Path="/plots/training/", filename: str=None
    ):
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
            
            if filename is None: 
                filename = f"training_{timestamp}.png"
            
            outpath = Path(path) / filename
            if not outpath.parent.exists(): 
                # Ensure the output directory exists
                outpath.parent.mkdir(parents=True)

            # Create the figure filename with timestamp
            plt.savefig(str(outpath), dpi=300)
            LOGGER.info(f"Figure saved as: {str(outpath)}")
        
        plt.show()
