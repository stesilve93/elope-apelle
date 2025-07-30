
import cv2
import numpy as np 
import torch
import torch.nn.functional as F

from PIL import Image
from pathlib import Path

from numba import njit

from elope.evflow import EVFlowNet, load_model, pad_image_evflow, unpad_image_evflow
from elope.utils import LOGGER

@njit
def encode_events(events: np.ndarray, t_end: float, integration_window: float) -> tuple:

    # Define the time bins vector
    time_bins = np.arange(0, t_end, integration_window)

    # Define the event array sizes
    H, W, T = 200, 200, len(time_bins)

    # Lets pack the events every TBD-us together
    events_count = np.zeros((T, H, W, 2))
    events_stamp = np.zeros((T, H, W, 2))

    # Store the events happening before each time
    tidx = 0
    for ev in events:   
        
        t = ev[2]
        
        if t > time_bins[tidx]:
            tidx += 1 
        
        if tidx >= T: 
            break
        
        # Retrieve the event channel
        c = 0 if ev[-1] > 0 else 1
        
        # Retrieve the coordinates
        xi, yi = int(ev[0]), int(ev[1])
        
        events_count[tidx, yi, xi, c] += 1 
        
        
        events_stamp[tidx, yi, xi, c] = t
    
    return events_count, events_stamp, time_bins

def flow_to_rgb(flow: np.ndarray) -> Image:
    """
    Converts optical flow (H, W, 2) to RGB image (H, W, 3) using HSV mapping.
    flow: np.ndarray or torch.Tensor, last dim is (u, v)
    """
    
    if isinstance(flow, torch.Tensor):
        flow = flow.cpu().numpy()

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # Compute magnitude and angle
    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)  # range [-pi, pi]
    
    # Normalize angle to [0, 1] → map to hue [0, 180]
    hue = (angle + np.pi) / (2 * np.pi)  # [0, 1]
    hue = (hue * 179).astype(np.uint8)

    # Normalize magnitude to [0, 255] for value
    mag_norm = np.clip(magnitude / (np.percentile(magnitude, 99) + 1e-6), 0, 1)
    value = (mag_norm * 255).astype(np.uint8)

    # Saturation at max
    saturation = np.ones_like(value, dtype=np.uint8) * 255

    img_hsv = np.stack((hue, saturation, value), axis=-1)  # (H, W, 3)
    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    
    return Image.fromarray(img_rgb)

def events_stamp_to_image(
    events: np.ndarray, 
    polarity: str, 
    grayscale: bool=False, 
    bg_white: bool=False,
) -> Image: 
    
    # Check only one polarity is selected
    assert polarity in ("positive", "negative")
    disp_pos = polarity == "positive"
    
    # Select the desired event channel
    events = events[..., 0] if disp_pos else events[..., 1]
    
    if grayscale: 
        
        img = 1 - events if bg_white else events
        img = (255 * img).astype(np.uint8)
        return Image.fromarray(img)
            
    H, W = events.shape[:2] 
    
    if bg_white: 

        img = np.ones((H, W, 3), dtype=np.float32)
        img[..., 2 if disp_pos else 0] -= events
        img[..., 1] -= events

    else: 
        img = np.zeros((H, W, 3), dtype=np.float32)
        img[..., 0 if disp_pos else 2] = events

    img = (255 * img).astype(np.uint8)
    return Image.fromarray(img)

def collect_events(
    events_count: np.ndarray,
    events_stamp: np.ndarray,
    times: np.ndarray, 
    stack: int, 
) -> tuple:
    
    
    assert len(events_count) == len(events_stamp)
    
    stack_count = []
    stack_stamp = []
    
    n = 0
    while n < len(events_count):     
        
        m = min(n+stack, len(events_count)-1)
        if n == m: 
            break 
        
        ts, te = times[n], times[m]
        
        img_count = events_count[n:m].sum(axis=0)
        img_stamp = events_stamp[n:m].max(axis=0)
        
        mask = img_stamp > 0
        img_stamp[mask] = (img_stamp[mask] - ts)/(te-ts)
        
        stack_count.append(img_count)
        stack_stamp.append(img_stamp)
        
        n += stack    

    return np.stack(stack_count), np.stack(stack_stamp), 

def frames_to_gif(filename: str, frames: list | tuple, loop=0, **kwargs): 
    
    if isinstance(frames, tuple) and isinstance(frames[0], list):
        
        assert len(frames) <= 3
        
        # We display two gifs together 
        outframes = []
        for fs in zip(*frames):
            
            f1, f2 = fs[:2]
            f1 = f1.convert("RGBA")
            f2 = f2.convert("RGBA")
            
            w = f1.width + f2.width
            h = max(f1.height, f2.height)
            
            if len(fs) == 3:
                f3 = fs[-1]
                
                w = w + f3.width 
                h = max(h, f3.width)
            
            f = Image.new("RGBA", (w, h))
            
            f.paste(f1, (0,0))
            f.paste(f2, (f1.width, 0))
            
            if len(fs) == 3: 
                f.paste(f3, (f1.width + f2.width, 0))
            
            outframes.append(f) 
        
        frames = outframes        
    
    frames[0].save(
        filename, 
        save_all=True, 
        append_images=frames[1:], 
        loop=loop,
        **kwargs
    )
    
# Retrieve the sequence path
SEQUENCE_PATH = Path("elope_data") / "train"

# Path where to store the data
SAVE_PATH = Path("sequence_events")

# Event integration window in microseconds
INTEGRATION_WINDOW = 1e4

# Maximum integration time
INTEGRATION_TIME = 1e6

# True if we should concatenate also the results from evflownet
USE_EVFLOW = True

# Get how many event tensors we need to stack
STACK = int(INTEGRATION_TIME / INTEGRATION_WINDOW)

# Ensure the path exists
SAVE_PATH.mkdir(exist_ok=True, parents=True)

if USE_EVFLOW:
    
    # Device configuration 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    LOGGER.info(f"Using device: {device}")

    # Load the EVFlowNet network 
    model = EVFlowNet(batch_norm=True)
    data = torch.load("weights/evflownet/evflownet.pth") 
    model.load_state_dict(data)
    LOGGER.info(f"EVFlowNet initialized.")
    
    model.eval() 
    model = model.to(device)

sequences = [str(i).zfill(4) for i in range(28)]
for SEQUENCE_ID in sequences: 
        
    # Load the sequence data
    data = np.load(SEQUENCE_PATH / (SEQUENCE_ID + ".npz"))

    # Retrieve the event tensor
    events = data["events"]
    events = np.column_stack([
        events['x'], events['y'], events['t'].astype(int), events['p'].astype(int)
    ])

    # Sort the events by time 
    events = events[np.argsort(events[:, 2])]

    # Ending event time
    t_end = 1e6*data["range_meter"][-1, 0]

    # Create the event count tensor 
    events_count, events_stamp, times = encode_events(events, t_end, INTEGRATION_WINDOW)

    # Stack the event slicing according to the desired time
    e_count, e_stamp = collect_events(events_count, events_stamp, times, STACK)
            
    frames = [
        [events_stamp_to_image(e, p, grayscale=False, bg_white=True) for e in e_stamp] 
        for p in ("positive", "negative")
    ]
    
    if USE_EVFLOW: 
        # Generate the prediction from the network
        img_flow = []
        
        for k in range(len(e_count)): 
            
            # Create the tensor for the stacking 
            event_img = np.vstack(
                (e_count[k].transpose(2, 0, 1), e_stamp[k].transpose(2, 0, 1))
            ).astype(np.float32)
            
            # Move to PyTorch and upsample to 256x256
            x = torch.from_numpy(event_img).to(device).unsqueeze(0)
            x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
            
            # Run the network inference
            with torch.no_grad():
                output = model(x)
            
            # Downsample the image to the original resolution
            flow = output['flow3']   
            flow = F.interpolate(flow, size=(200, 200), mode="bilinear", align_corners=False)     

            # Transform back to NumPy and adjust the dimensions
            flow = flow.detach().squeeze().cpu().numpy()
            flow = flow.transpose(1, 2, 0)
            flow = np.flip(flow, 2)
            
            img_flow.append(flow_to_rgb(flow))

        frames.append(img_flow)

    frames_to_gif(SAVE_PATH / f"{SEQUENCE_ID}.gif", tuple(frames), duration=3, loop=0)