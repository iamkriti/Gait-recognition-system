import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.misc import imresize, imread
from human_pose_nn import HumanPoseIRNetwork
from gait_nn import GaitNetwork
# Initialize computational graphs of both sub-networks
net_pose = HumanPoseIRNetwork()
net_gait = GaitNetwork(recurrent_unit = 'GRU', rnn_layers = 2)

# Load pre-trained models
net_pose.restore('models/MPII+LSP.ckpt')
net_gait.restore('models/M+L-GRU-2.ckpt')

vid=imread('images/demo.jpg')
vid = imresize(vid, [299, 299])
video_frames = np.expand_dims(vid, 0)

# Create features from input frames in shape (TIME, HEIGHT, WIDTH, CHANNELS)
spatial_features = net_pose.feed_forward_features(video_frames)

# Process spatial features and generate identification vector
identification_vector = net_gait.feed_forward(spatial_features)

print(identification_vector)