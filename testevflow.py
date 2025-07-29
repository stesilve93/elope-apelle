
import cv2 
import numpy as np
import torch 
import torch.nn.functional as F

from elope.datasets import FixedSequenceLoader, VariableSequenceLoader
from elope.evflow import EVFlowNet, load_model, pad_image_evflow, unpad_image_evflow
from elope.utils import LOGGER

# Device configuration 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
LOGGER.info(f"Using device: {device}")

model = EVFlowNet(batch_norm=True)
data = torch.load("weights/evflownet/evflownet.pth") 
model.load_state_dict(data)
LOGGER.info(f"EVFlowNet initialized.")

model.eval()
model = model.to(device)

seq = FixedSequenceLoader(
    "elope_data/train", 
    time_step=1.0, 
    event_integration_window=1e6, 
    event_encoder_method="hybrid", 
    event_clamp=-1, 
    event_H=200, 
    event_W=200, 
    event_T=3, 
    sequence_len=2, 
    sequence_pad="static"
)

seq.load_sequence('0023')

from PIL import Image
def image_from_count(event_count: np.ndarray) -> Image: 
    
    if isinstance(event_count, torch.Tensor):
        event_count = event_count.cpu().numpy()
    
    event_count = 255.0 * (event_count / event_count.max())
    img = Image.fromarray(event_count.astype('uint8'))
    return img

def image_from_stamp(event_stamp: np.ndarray) -> Image:
    
    if isinstance(event_stamp, torch.Tensor):
        event_stamp = event_stamp.cpu().numpy()
    
    event_stamp = 255.0 * event_stamp
    img = Image.fromarray(event_stamp.astype('uint8'))
    return img

def flow_to_rgb(flow):
    """
    Converts optical flow (H, W, 2) to RGB image (H, W, 3) using HSV mapping.
    flow: np.ndarray or torch.Tensor, last dim is (u, v)
    """
    if isinstance(flow, torch.Tensor):
        flow = flow.cpu().numpy()

    h, w, _ = flow.shape
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

    hsv = np.stack((hue, saturation, value), axis=-1)  # (H, W, 3)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb


# Retrieve the data
flows  = []
stamps = []

for k in range(len(seq)): 
    
    data_k = seq.get_data_at_index(k)

    # Unpack and move to device after adding the batch dimension 
    events = data_k['events'][-1] # (2, C, H, W)

    count_pos = events[0, 0]
    stamp_pos = events[0, 2]

    count_neg = events[1, 0]
    stamp_neg = events[1, 2]

    # Stack the images 
    event_img = torch.stack([count_pos, count_neg, stamp_pos, stamp_neg], dim=0)

    # Apply zero-padding to match the 256x256 resolution
    event_img_pad = pad_image_evflow(event_img, seq.H, seq.W)
    event_img_pad = event_img_pad.unsqueeze(0).to(device)

    # Run inference on the input
    output = model(event_img_pad)

    # Lets check what the fuck we get 
    flow = output['flow3'].detach()
    flow = unpad_image_evflow(flow, seq.H, seq.W)
    flow = flow.squeeze().cpu().numpy()

    flow = np.transpose(flow, (1, 2, 0))
    flow = np.flip(flow, 2)

    flows.append(flow_to_rgb(flow))
    stamps.append(image_from_stamp(stamp_pos))

frames_flow, frames_stamp = [], []
for k in range(len(seq)):
    frames_flow.append(Image.fromarray(flows[k]))
    frames_stamp.append(stamps[k])
    

def stack_frames(frames1, frames2, savepath, **kwargs): 
    
    outframes = []
    for f1, f2 in zip(frames1, frames2): 
        f1 = f1.convert("RGBA")
        f2 = f2.convert("RGBA")
        w = f1.width + f2.width
        h = max(f1.height, f2.height)
        
        f = Image.new("RGBA", (w, h))
        f.paste(f1, (0,0))
        f.paste(f2, (f1.width, 0))
        outframes.append(f) 
    
    outframes[0].save(
        savepath, 
        save_all=True, 
        append_images=outframes[1:], 
        **kwargs
    )

stack_frames(frames_flow, frames_stamp, "testflow_0023.gif", duration=1, loop=0)
    
# frames_flow[0].save(
#     f"test_flow_0023.gif", 
#     save_all=True, 
#     append_images=frames_flow[1:], 
#     duration=1, 
#     loop=0
# )

# frames_stamp[0].save(
#     f"test_stamp_0023.gif", 
#     save_all=True, 
#     append_images=frames_stamp[1:], 
#     duration=1, 
#     loop=0
# )

# img_stamp_pos = image_from_stamp(stamp_pos)
# img_stamp_pos.save("stamp_pos.png")

# img_stamp_neg = image_from_stamp(stamp_neg)
# img_stamp_neg.save("stamp_neg.png")

# img_count_neg = image_from_count(count_neg)
# img_count_neg.save("count_neg.png")

# img_count_neg = image_from_count(count_neg)
# img_count_neg.save("count_neg.png")

# img_flow = flow_to_rgb(flow)
# cv2.imwrite("testflow.png", img_flow)


import flow_vis
flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)

cv2.imwrite("test.png", flow_color)