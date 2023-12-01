#--------------------Our Implementation--------------------------#
import math
import torch
import torch.nn.functional as nF
import numpy as np
import torchvision.transforms.functional as F


def extract_pitch_from_camera_origins(camera_origins):
    if len(camera_origins.shape) == 1:
        camera_origins = camera_origins.unsqueeze(0)
    radius = torch.norm(camera_origins, dim=1, keepdim=True) 

    y = camera_origins[:, 1:2]
    phi = torch.acos(y / radius)
    v = (1 - torch.cos(phi)) / 2
    vertical_mean = v * math.pi
    return (vertical_mean - np.pi/2)

def extract_RO_from_cam2world(cam2world):
    R = cam2world[:, :3, :3] 
    O = cam2world[:, :3, 3]  
    return R, O

def extract_angles_from_cam2world(cam2world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    R, O = extract_RO_from_cam2world(cam2world) 

    r11 = R[:, 0, 0]
    r21 = R[:, 1, 0]
    r13 = R[:, 0, 2]

    yaw = torch.atan2(-r13, torch.sqrt(r11**2 + r21**2))
    pitch = extract_pitch_from_camera_origins(O)

    # Applying rounding and pitch correction for entire batch
    yaw = torch.round(yaw, decimals=2)
    pitch = torch.round(pitch, decimals=2)
    pitch = -pitch

    return pitch, yaw


def prep_input_tensor(image: torch.Tensor):
    """Preparing a Torch Tensor as input to L2CS-Net."""
    image = (image.float() * 127.5 + 128).clamp(0, 255)
    device = image.device
    image = nF.interpolate(image, size=(448, 448), mode='bilinear', align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    image = image.float() / 255.0

    image = (image - mean) / std

    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    return image
#----------------------------------------------------------------#