
import torch 

def derivate(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute the time-derivative of a tensor.
    
    Parameters
    ----------
    x : torch.Tensor 
        Input tensor, of shape (b, s, n). 
    t : torch.Tensor
        Time tensor, of shape (b, s, 1).
    
    Returns 
    -------
    dx : torch.Tensor 
        Time derivative. It is computed along the 1-st axis dimension.
    """
    # Retrieve the batch and sequence lenghts
    _, s, _ = x.shape
    
    # Initialize the derivative 
    dx = torch.zeros_like(x)
    
    # Compute the central order differences 
    dx[:, 1:s-1] = (x[:, 2:s] - x[:, 0:s-2])/(t[:, 2:s] - t[:, 0:s-2])
    
    # Use a 1-st order forward difference for the first point 
    dx[:, 0] = (x[:, 1] - x[:, 0])/(t[:, 1] - t[:, 0])
    
    # Use a 1-st order backward difference for the last point 
    dx[:, -1] = (x[:, -1] - x[:, -2])/(t[:, -1] - t[:, -2])
    return dx
    
    
def body_to_angles_rates(angles: torch.Tensor, body_rates: torch.Tensor) -> torch.Tensor: 
    """Convert body rates to world frame angle derivatives.
    
    Parameters
    ----------
    angles : torch.Tensor 
        Euler angles tensor, of shape (B, S, 3). 
    body_rates : torch.Tensor
        Body angular velocity, of shape (B, S, 3).
        
    Returns 
    -------
    dang : torch.Tensor
        Euler angles time derivatives, of shape (B, S, 3).
    """

    phi, theta = angles[..., 0], angles[..., 1]
    p, q, r = body_rates[..., 0], body_rates[..., 1], body_rates[..., 2]
    
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    cos_theta_safe = torch.clamp(cos_theta, min=1e-6)
    tan_theta = sin_theta / cos_theta_safe
    sec_theta = 1.0 / cos_theta_safe
    
    phi_dot = p + q * sin_phi * tan_theta + r * cos_phi * tan_theta
    theta_dot = q * cos_phi - r * sin_phi
    psi_dot = (q * sin_phi + r * cos_phi) * sec_theta
    
    return torch.stack([phi_dot, theta_dot, psi_dot], dim=-1)


def angles_to_body_rates(angles: torch.Tensor, t: torch.Tensor) -> torch.Tensor: 
    """Recover the body rates from the time evolution of the Euler angles.
    
    Parameters
    ----------
    angles : torch.Tensor 
        Euler angles tensor, of shape (B, S, 3). 
    times : torch.Tensor 
        Time tensor, of shape (B, S, 1).
    
    Returns 
    -------
    w : torch.Tensor
        Body rates angular velocities, of shape (B, S, 3).
    """
    
    # Compute the Euler angles derivatives
    da = derivate(angles, t)
    
    phi, theta = angles[..., 0], angles[..., 1]
    phi_dot, theta_dot, psi_dot = da[..., 0], da[..., 1], da[..., 2]
    
    cos_theta = torch.clamp(torch.cos(theta), min=1e-8)
    sin_theta = torch.sin(theta)
    tan_theta = sin_theta / cos_theta
    
    cos_phi = torch.clamp(torch.cos(phi), min=1e-8)
    sin_phi = torch.sin(phi)
    tan_phi = sin_phi / cos_phi
    
    d = torch.clamp(tan_phi*sin_phi + cos_phi, min=1e-8)
    
    r = (psi_dot*cos_theta - theta_dot*tan_phi)/d
    q = theta_dot/cos_phi + r*tan_phi 
    p = phi_dot - (q*sin_phi +r*cos_phi)*tan_theta
    
    return torch.stack([p, q, r], dim=-1)
    
    
def angles_to_dcm(angles: torch.Tensor) -> torch.Tensor: 
    """Compute the rotation matrix from camera to inertial frame. 
    
    Parameters
    ----------
    angles : torch.Tensor: 
        Roll, pitch and yaw rotation sequence (123), of shape (B, 3) or (B, T, 3). 
    
    Returns 
    -------
    dcm : torch.Tensor 
        DCM matrices, of shape (B, 3, 3) or (B, T, 3, 3).
    """
    
    assert angles.ndim <= 3
    
    # Retrieve the batch dimension
    B = angles.shape[0]
    
    is_sequence = angles.ndim == 3
    if is_sequence:
        angles = angles.view(-1, 3)
    
    # Pre-compute all trigonometric terms
    cos_phi = torch.cos(angles[..., 0])
    sin_phi = torch.sin(angles[..., 0])
    
    cos_theta = torch.cos(angles[..., 1])
    sin_theta = torch.sin(angles[..., 1])
    
    cos_psi = torch.cos(angles[..., 2])
    sin_psi = torch.sin(angles[..., 2])
    
    # Initialize the tensor
    dcm = torch.zeros(angles.shape[0], 9).to(angles)
    
    dcm[..., 0] = cos_phi*cos_psi
    dcm[..., 1] = sin_phi*sin_theta*cos_psi - cos_phi*sin_psi
    dcm[..., 2] = sin_phi*sin_psi + cos_phi*sin_theta*cos_psi
    dcm[..., 3] = cos_theta*sin_psi 
    dcm[..., 4] = cos_phi*cos_psi + sin_phi*sin_theta*sin_psi
    dcm[..., 5] = cos_phi*sin_theta*sin_psi - sin_phi*cos_psi
    dcm[..., 6] = -sin_theta
    dcm[..., 7] = sin_phi*cos_theta
    dcm[..., 8] = cos_phi*cos_theta
    
    # Transform into a 3x3 matrix
    dcm = dcm.view(-1, 3, 3)
    
    if is_sequence:
        # Recover the sequence length 
        dcm = dcm.view(B, -1, 3, 3)
    
    return dcm 
    
    
def rotate_velocity(velocity: torch.Tensor, angles: torch.Tensor) -> torch.Tensor: 
    """Rotate the lander velocity from the camera to the inertial frame.
    
    Parameters
    ----------
    velocity : torch.Tensor 
        Lander velocity in the camera frame, of shape (B, 3) or (B, T, 3).
    angles : torch.Tensor 
        Roll, pitch and yaw rotation sequence (123), of shape (B, 3) or (B, T, 3).
        
    Returns 
    -------
    out : torch.Tensor
        Lander velocity in the inertial frame, of shape (B, 3) or (B, T, 3)
    """
    
    assert velocity.shape[0] == angles.shape[0]
    assert velocity.ndim == angles.ndim 
    assert velocity.ndim <= 3 
    
    B = angles.shape[0]
    is_sequence = angles.ndim == 3
    if is_sequence: 
        angles   = angles.view(-1, 3)       # (D, 3)
        velocity = velocity.view(-1, 3)     # (D, 3)
    
    # Compute the rotation from camera to the inertial frame 
    dcm = angles_to_dcm(angles)             # (D, 3, 3)
        
    velocity = velocity.unsqueeze(-1)       # (D, 3, 1)
    velocity = (dcm@velocity).squeeze()     # (D, 3)
    
    if is_sequence: 
        velocity = velocity.view(B, -1, 3)
        
    return velocity


def estimate_altitude(ranges: torch.Tensor, angles: torch.Tensor) -> torch.Tensor: 
    """Return an estimate of the vertical position from the range and attitude.
    
    Parameters
    ----------
    ranges : torch.Tensor 
        Tensor of rangemeter values, of shape (B, T, 1)
    angles : torch.Tensor 
        XYZ sequence of Euler angles, of shape (B, T, 3)
    
    Returns
    -------
    posz : torch.Tensor 
        Vertical position component, of shape (B, T, 1). Negative by convention.
    """
    
    # Retrieve the batch dimensions 
    assert angles.shape[0] == ranges.shape[0]
    assert angles.ndim == ranges.ndim 
    assert angles.ndim <= 3 
    
    B = angles.shape[0]
    is_sequence = angles.ndim == 3 
    if is_sequence: 
        angles = angles.view(-1, 3)
        ranges = ranges.view(-1, 1)
        
    # Ensure this is a scalar tensor
    ranges = ranges.view(-1)

    cos_theta = torch.cos(angles[..., 0])
    cos_phi = torch.cos(angles[..., 1])
    g = cos_phi * cos_theta     # (B,)
    
    # Compute the altitude
    altitude = (g * ranges).view(-1, 1) # (B, 1)

    if is_sequence: 
        altitude = altitude.view(B, -1, 1)
    
    # Retrieve the vertical position component 
    return altitude
    
    
def compute_point_depth(
    coords: torch.Tensor, angles: torch.Tensor, altitude: torch.Tensor
) -> torch.Tensor: 
    """Compute the depth of a point given the lander attitude and altitude.
    
    Parameters
    ----------
    coords : torch.Tensor 
        Normalized pixel coordinates, of shape (C, 2)
    angles : torch.Tensor 
        XYZ sequence of Euler angles, of shape (B, 3). 
    altitude : torch.Tensor 
        Spacecraft altitude, of shape (B,).
    
    Returns
    -------
    depth : torch.Tensor
        Pixel depth, of shape (B, C)
    """
    
    cos_phi = torch.cos(angles[..., 0])
    sin_phi = torch.sin(angles[..., 0])

    cos_theta = torch.cos(angles[..., 1])
    sin_theta = torch.sin(angles[..., 1])
    
    a = (-sin_theta).unsqueeze(1)
    b = ( sin_phi*cos_theta).unsqueeze(1) 
    g = ( cos_phi*cos_theta).unsqueeze(1)
    
    # Retrieve the point coordinates
    xs = coords[..., 0]
    ys = coords[..., 1]
    
    # Compute the pixel depth
    depth = altitude.unsqueeze(1)/(a*xs + b*ys + g)
    return depth

def generate_pixelgrid(height: int, width: int) -> torch.Tensor: 
    """Return a normalized grid of pixel coordinates.
    
    Parameters
    ----------
    height : int 
        Camera height. 
    width : int 
        Camera width 
        
    Returns
    -------
    coords : torch.Tensor
        A Tensor of shape (H, W, 2) with the normalized pixel coords.
    """
    
    xc = (torch.arange(width) + 0.5) * (2 / width) - 1
    yc = (torch.arange(height) + 0.5) * (2 / height) - 1
    
    grid = torch.cartesian_prod(yc, xc).view(height, width, 2)
    grid = grid.flip(2)
    return grid

