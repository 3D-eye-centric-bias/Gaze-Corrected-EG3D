import sys
import os
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from gaze_utils.estimator import GazeEstimator

gazeestimator = GazeEstimator()
print(gazeestimator)