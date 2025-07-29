
import torch 
import torch.nn.functional as F 
import torchvision.transforms.functional as Fv


def pad_image_evflow(img: torch.Tensor, H: int, W: int) -> torch.Tensor: 
    
    # Pad the image to match EVFlowNet resolution 
    pad_h = 256 - H 
    pad_w = 256 - W 
    
    pad_l = pad_w // 2 
    pad_r = pad_w - pad_l
    pad_t = pad_h // 2 
    pad_b = pad_h - pad_t 
    
    img_pad = F.pad(img, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=0)
    return img_pad
    

def unpad_image_evflow(img: torch.Tensor, H: int, W: int) -> torch.Tensor: 
    
    # Unpad the image to the original resolution before the EVFlowNet application 
    return Fv.center_crop(img, (H, W))
    

