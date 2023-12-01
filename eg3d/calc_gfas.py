# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import mrcfile


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator

#-------------------------Our Implementation-------------------------#
import torch
import torch.nn.functional as F
import os
from gaze_utils.estimator import GazeEstimator
from gaze_utils.utils import extract_angles_from_cam2world
#--------------------------------------------------------------------#

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#-------------------------Our Implementation-------------------------#
def prep_input_tensor(image: torch.Tensor):
    """Preparing a Torch Tensor as input to L2CS-Net."""
    image = (image.float() * 127.5 + 128).clamp(0, 255)
    device = image.device
    image = F.interpolate(image, size=(448, 448), mode='bilinear', align_corners=False)

    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    # Scale image
    image = image.float() / 255.0

    image = (image - mean) / std

    # Add dimension if not in batch mode
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    return image

def generate_and_estimate(G, z, camera_params, conditioning_params, truncation_psi, truncation_cutoff, gaze_estimator, device, face_pitch, face_yaw):
    ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
    img = G.synthesis(ws, camera_params)['image']
    img_prep = prep_input_tensor(img)
    img_cp = img_prep.clone()
    img_cp = (img_cp.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    gaze_pitch, gaze_yaw = gaze_estimator.estimate(img_prep)
    gaze_loss = F.mse_loss(torch.tensor([gaze_pitch, gaze_yaw], device=device), torch.tensor([face_pitch, face_yaw], device=device))
    return img, gaze_loss.item(), gaze_pitch, gaze_yaw

def convert_image(img_tensor):
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    img_cp = img_tensor.clone()
    img_cp = (img_cp.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return img_cp

def to_pil_image(img_tensor):
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)  
    img_cp = img_tensor.clone()
    img_cp = (img_cp.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return PIL.Image.fromarray(img_cp[0].cpu().numpy(), 'RGB')
#--------------------------------------------------------------------#


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', default='0-1023')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    class_idx: Optional[int],
    reload_modules: bool,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """
    
    print('Loading networks...')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G1 = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    
#-------------------------Our Implementation-----------------------
# 
# --#
    gaze_estimator = GazeEstimator(device=device)

    total_gaze_loss = 0  # Initialize total gaze loss
    num_seeds = len(seeds)
#--------------------------------------------------------------------#

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G1.z_dim)).to(device)

        imgs = []
        angle_p = -0.2
        i=0
        for angle_y, angle_p in [(.4, angle_p), (0, angle_p), (-.4, angle_p)]:
            cam_pivot = torch.tensor(G1.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
            cam_radius = G1.rendering_kwargs.get('avg_camera_radius', 2.7)
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
            conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

#--------------------------------Our Implementation------------------------------#
            cam2world_pose = camera_params[:, :16].reshape(-1, 4, 4)
            face_pitch, face_yaw = extract_angles_from_cam2world(cam2world_pose)

            img, gaze_loss, gaze_pitch, gaze_yaw = generate_and_estimate(G1, z, camera_params, conditioning_params, truncation_psi, truncation_cutoff, gaze_estimator, device, face_pitch, face_yaw)
            
            # Accumulate gaze losses
            total_gaze_loss += total_gaze_loss

            gaze_pitch = gaze_pitch.cpu().item()
            gaze_yaw = gaze_yaw.cpu().item()
            
            face_pitch = face_pitch.cpu().item()
            face_yaw = face_yaw.cpu().item()
            i += 1
            
        print(f'[Progress] Seed: {seed_idx}/{num_seeds}')

        # Calculate average gaze loss for each network
    avg_gaze_loss = total_gaze_loss / num_seeds

    # Print the average gaze losses
    print(f'GFAS: {avg_gaze_loss}')
#--------------------------------------------------------------------------------#

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
