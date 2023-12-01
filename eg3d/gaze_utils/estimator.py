import torch
import numpy as np
import torch.nn as nn

import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir))

from model import getArch


class GazeEstimator:

    def __init__(self, device):
        self.device = device
        self.arch = "ResNet50"
        self.weight_path = "/root/volume/models/L2CSNet_gaze360.pkl"
        if not os.path.exists(self.weight_path):
            raise FileNotFoundError("The weight path does not exist.")
#------------From L2CS-Net Code----------------#
        self.model = getArch(self.arch, 90)
        self.model.load_state_dict(torch.load(self.weight_path, map_location=device))
        self.model.to(self.device)
        self.model.eval()
        self.softmax = nn.Softmax(dim=1)
        self.idx_tensor = [idx for idx in range(90)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor).to(self.device)

    def estimate(self, image: torch.Tensor):
        gaze_pitch, gaze_yaw = self.model(image)
        pitch_predicted = self.softmax(gaze_pitch)
        yaw_predicted = self.softmax(gaze_yaw)
        
        pitch_predicted = torch.sum(pitch_predicted.data * self.idx_tensor, dim=1) * 4 - 180
        yaw_predicted = torch.sum(yaw_predicted.data * self.idx_tensor, dim=1) * 4 - 180

        pitch_predicted = pitch_predicted * torch.tensor(np.pi / 180.0).to(pitch_predicted.device)
        yaw_predicted = yaw_predicted * torch.tensor(np.pi / 180.0).to(yaw_predicted.device)
        pitch_predicted, yaw_predicted = yaw_predicted, pitch_predicted
        
        return pitch_predicted, yaw_predicted

#-------------------------------------------------------#