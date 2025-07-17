
import torch 


def loss_mse_abs(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the Mean-Square Error Loss.
    
    Parameters
    ----------
    pred : torch.Tensor 
    
    target : torch.Tensor
    
    
    Returns
    -------
    loss : torch.Tensor
    
    """

    err_vel_abs = (pred - target)**2
    return torch.sum(err_vel_abs)/target.numel()
    
    
def loss_mse_rel(pred: torch.Tensor, target: torch.Tensor) -> torch.tensor: 
    # DOCME
    
    err_vel_rel = ((pred - target)/target)**2
    return torch.sum(err_vel_rel)/target.numel()

    
def loss_elope(
    vel_input: torch.Tensor, vel_target: torch.Tensor, pos_target: torch.Tensor
) -> torch.Tensor: 
    """Compute the ELOPE score."""
    # DOCME
    
    err_vel_norm = torch.norm(vel_input - vel_target, dim=1)
    return torch.sum(err_vel_norm/torch.abs(pos_target[:, 2]))/pos_target.shape[0]