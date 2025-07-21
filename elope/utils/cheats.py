
import numpy as np 
import torch 

from scipy.signal import savgol_filter

def angles2dcm(angles: torch.Tensor) -> torch.Tensor:
    """Transform a sequence of 3 angles in a DCM using a 123 rotation."""

    if isinstance(angles, np.ndarray): 
        angles = torch.from_numpy(angles.copy())

    c1, s1 = torch.cos(angles[0]), torch.sin(angles[0])
    c2, s2 = torch.cos(angles[1]), torch.sin(angles[1])
    c3, s3 = torch.cos(angles[2]), torch.sin(angles[2])

    return torch.tensor([
        [ c2*c3,  s1*s2*c3 + c1*s3, -c1*s2*c3 + s1*s3], 
        [-c2*s3, -s1*s2*s3 + c1*c3,  c1*s2*s3 + s1*c3], 
        [    s2,            -s1*c2,             c1*c2]
    ]).to(angles)
    
def compute_posz(rangemeter: torch.Tensor, angles: torch.Tensor) -> float: 
    """Return an estimate of the vertical position from the range and attitude."""    
    
    def compute_posz_scalar(dist, ang):     
        # Vertical direction of the rangemeter in the body frame
        uz = torch.tensor([0.0, 0.0, 1.0]).to(ang)
            
        # Compute the rotation from body to world frame 
        dcm_b2w = angles2dcm(ang)

        # Compute the rangemeter direction in the world frame 
        uw = dcm_b2w@uz
        return -abs(dist*uw[2])
    
    # Ensure the input is a torch array
    if isinstance(angles, np.ndarray): 
        angles = torch.from_numpy(angles.copy())
        
    if angles.ndim == 1: 
        return compute_posz_scalar(rangemeter, angles)

    # Ensure the dimensions match the expectations 
    assert angles.shape[1] == 3
    
    # Ensure rangemeter data is a PyTorch tensor 
    if isinstance(rangemeter, np.ndarray): 
        rangemeter = torch.from_numpy(rangemeter.copy())
   
    pos_z = torch.zeros_like(rangemeter)
    for k in range(pos_z.shape[0]): 
        pos_z[k] = compute_posz_scalar(rangemeter[k], angles[k])

    return pos_z 

def compute_posvelz(
    times: torch.Tensor, 
    rangemeter: torch.Tensor, 
    angles: torch.Tensor, 
    fp_window_length: int=5, 
    fp_poly_order: int=2,
    fv_window_length: int=5,
    fv_poly_order: int=2
) -> tuple: 
    
    # Compute the number of points
    n = len(times)
    
    # Retrieve an estimate of the vertical position
    pos = compute_posz(rangemeter, angles)
    
    assert len(times) > 1
    
    if fp_window_length > 0:
        # Ensure enough points are available
        assert fp_window_length <= n
        
        # If requested apply a savgol filter to the position to smooth it
        pos = savgol_filter(
            pos, window_length=fp_window_length, polyorder=fp_poly_order
        )
    
    # Compute the derivative of the position 
    vel = np.gradient(pos, times[1] - times[0])
    
    if fv_window_length == 0:
        return pos, vel 
    
    # Ensure enough points are available
    assert fv_window_length <= n
    
    # If requested apply a savgol filter to the velocity to smooth it
    return pos, savgol_filter(
        vel, window_length=fv_window_length, polyorder=fv_poly_order
    )