
import torch 


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the Mean-Square Error Loss.
    
    Parameters
    ----------
    pred : torch.Tensor 
    
    target : torch.Tensor
    
    
    Returns
    -------
    loss : torch.Tensor
    
    """
    
    squared_error = (pred - target)**2
    torch.sum()
    
    
    
    
def loss_elope(
    vel_input: torch.Tensor, vel_target: torch.Tensor, pos_target: torch.Tensor
) -> torch.Tensor: 
    """Compute the ELOPE score."""
    # DOCME
    
    err_vel_norm = torch.norm(vel_input - vel_target, dim=1)
    return torch.sum(err_vel_norm/torch.abs(pos_target[:, 2]))/pos_target.shape[0]