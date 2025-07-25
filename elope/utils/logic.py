
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
    
    
def angles_to_quat(angles: torch.Tensor) -> torch.Tensor: 
    """Convert an XYZ sequence of angles to a quaternion.
    
    Parameters
    ----------
    angles : torch.Tensor 
        Euler angles, of shape (B, T, 3).
    
    Returns
    -------
    quat : torch.Tensor 
        Quaternion, in scalar-last format, of shape (B, T, 4).
    """
    
    # Retrieve the angles
    phi, theta, psi = angles[..., 0], angles[..., 1], angles[..., 2]
    
    c1 = torch.cos(phi / 2)
    s1 = torch.sin(phi / 2)
    
    c2 = torch.cos(theta / 2)
    s2 = torch.sin(theta / 2)
    
    c3 = torch.cos(psi / 2)
    s3 = torch.sin(psi / 2)
    
    # Compute the scalar component 
    q0 = c1 * c2 * c3 - s1 * s2 * s3
    
    s = torch.ones_like(q0)
    s[q0 < 0] = -1
    
    # Compute the vectorial part
    q1 = s1 * c2 * c3 + c1 * s2 * s3
    q2 = c1 * s2 * c3 - s1 * c2 * s3
    q3 = c1 * c2 * s3 + s1 * s2 * c3
    
    # Ensure the quaternon is normalized
    q = torch.stack([s*q1, s*q2, s*q3, s*q0], dim=-1)
    return q / torch.norm(q, dim=-1).unsqueeze(2)
    
    
def quat_multiply(qa: torch.Tensor, qb: torch.Tensor) -> torch.Tensor: 
    """Compute the product between two quaternions.
    
    .. note:: 
        All quaternions are assumed in scalar-last format. 
        
    Parameters
    ----------
    qa : torch.Tensor 
        First quaternion, of shape (B, T, 4).
    qb : torch.Tensor 
        Second quaternion, of shape (B, T, 4).

    Returns
    -------
    qc : torch.Tensor   
        Output quaternion, of shape (B, T, 4).

    """
    qa1, qa2, qa3, qa4 = qa[..., 0], qa[..., 1], qa[..., 2], qa[..., 3]
    qb1, qb2, qb3, qb4 = qb[..., 0], qb[..., 1], qb[..., 2], qb[..., 3]
    
    # Compute the quaternion components
    q1 = qa4*qb1 + qa3*qb2 - qa2*qb3 + qa1*qb4 
    q2 = qa4*qb2 - qa3*qb1 + qa1*qb3 + qa2*qb4
    q3 = qa2*qb1 - qa1*qb2 + qa4*qb3 + qa3*qb4 
    q4 = qa4*qb4 - qa1*qb1 - qa2*qb2 - qa3*qb3 
    
    return torch.stack([q1, q2, q3, q4], dim=-1)


def quat_conj(q: torch.Tensor) -> torch.Tensor: 
    """Compute the complex conjugate of a quaternion.
    
    Parameters
    ----------
    q : torch.Tensor
        Input quaternion, of shape (B, T, 4), in scalar-last format..
        
    Returns
    -------
    qj : torch.Tensor
        Quaternion conjugate, of shape (B, T, 4).
    """
    return torch.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], dim=-1)
    

def quat_divide(qn: torch.Tensor, qd: torch.Tensor) -> torch.Tensor: 
    """Compute the division between two quaternions.
        
    The convention adopted is such that given two elementary rotations `qn` and `qd`, 
    with Direction Cosine Matrices (DCM) `N` and `D` respectively, the result of ``qn / qd``
    equals ``N@D.T``.
    
    Parameters
    ----------
    qn, qd : torch.Tensor
        Input quaternions of shape (B, T, 4), in scalar-last format.
    
    Returns 
    -------
    qout : torch.Tensor 
        Division result, of shape (B, T, 4).
    """
    return quat_multiply(qn, quat_conj(qd))
    

def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor: 
    """Compute the passive rotation of a vector by a set of quaternions.
    
    Parameters
    ----------
    q : torch.Tensor 
        Input quaternions, of shape (B, T, 4), in scalar-last format.
    v : torch.Tensor
        Input vector, of shape (3,).
    
    Returns 
    -------
    out : torch.Tensor 
        Rotated vector, of shape (B, T, 3). 
    
    """
    
    # Transform the vector into a quaternion 
    qv = torch.zeros_like(q)
    qv[..., 0] = v[0]
    qv[..., 1] = v[1]
    qv[..., 2] = v[2]
    
    # Compute the quaternion result
    qout = quat_multiply(q, quat_divide(qv, q))
    return qout[..., :3]
    
    
def estimate_zpos(ranges: torch.Tensor, angles: torch.Tensor) -> torch.Tensor: 
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
    
    # Transform the angles into quaternions 
    q = angles_to_quat(angles)
    
    # Define the vertical direction of the rangemeter in the body frame.
    uz = torch.tensor([0.0, 0.0, 1.0]).to(q) 
    
    # Rotate the direction to the world-frame 
    uw = quat_rotate(q, uz)
    
    # Retrieve the vertical position component 
    return -torch.abs(ranges*uw[..., 2:3]) 
    
