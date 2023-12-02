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
import PIL.Image
import datetime

import numpy as np
import torch


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from tqdm import tqdm
from deepface import DeepFace

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--num-gen', 'num_gen', type=int, help='Number of samples to generate', default=1024, required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def calc_id(
    network_pkl: str,
    num_gen: int,
    outdir: str,
    truncation_psi: float,
    truncation_cutoff: int,
    fov_deg: float,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python calc_arcface.py --network=/root/volume/models/pretrained/ffhq512-128.pkl --outdir=output 
    python calc_arcface.py --network=/root/volume/kinam/eg3d/eg3d/~/training-runs/00043-ffhq-140-gpus2-batch12-gamma1/network-snapshot-000100.pkl --outdir=output
    """
    model_name = 'ArcFace'
    distance_metric = 'cosine'
    
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # Make directorys
    os.makedirs(outdir, exist_ok=True)
    trial_suffix = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    trial_dir = os.path.join(outdir, trial_suffix)
    os.makedirs(trial_dir, exist_ok=True)
    arcface_dir = os.path.join(trial_dir, 'arcface')
    os.makedirs(arcface_dir, exist_ok=True)

    # Generate images.
    #prog_bar = mmcv.ProgressBar(num_gen)
    
    
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    
    scores = .0
    for sample_idx in tqdm(range(num_gen)):
        print(f"[Progress] {sample_idx}/{num_gen}")
        z = torch.randn(1, G.z_dim).to(device)
        img_paths = []
        for angle_idx in range(2):
            angles = (np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4))
            cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
            cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angles[0], np.pi/2 + angles[1], cam_pivot, radius=cam_radius, device=device)
            conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            img = G.synthesis(ws, camera_params)['image']

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img_path = f'{outdir}/{trial_suffix}/arcface/sample{sample_idx:04d}_{angle_idx}.png'
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(img_path)
            
            img_paths.append(img_path)
        
    #----------------------Our Implementation----------------------#
        obj = DeepFace.verify(img_paths[0], img_paths[1], 
                              enforce_detection=False,
                              model_name=model_name, 
                              distance_metric=distance_metric)
        scores += obj["distance"]
    
        #prog_bar.update()
    similarity = scores / num_gen
    
    print(f"ID: {similarity}")
    #--------------------------------------------------------------#

#----------------------------------------------------------------------------

if __name__ == "__main__":
    calc_id() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
