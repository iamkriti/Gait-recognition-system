# Gait Recognition System

## Introduction
This Gait Recognition System leverages the power of computer vision and deep learning to track and analyze human movements by setting anchor points on the joints of the body. Implemented using Python, OpenCV, and TensorFlow, the system inputs raw RGB video frames of a pedestrian and outputs a unique gait descriptor as an identification vector. This project makes use of the HumanPose Estimator dataset for training and validation.

![Pose Estimation](https://github.com/samarsingh007/Gait-Recognition-System/blob/master/Results/Screenshot%20(37).png)

## Architecture Overview
The system architecture comprises two main sub-networks connected in cascade: HumanPoseNN and GaitNN.

- **HumanPoseNN**: This sub-network takes raw video frames as input and outputs spatial features describing the pose of the pedestrian. It serves as the foundation for extracting pose descriptors, which can also be utilized independently for 2D pose estimation tasks.

- **GaitNN**: The second sub-network processes the spatial features from HumanPoseNN using a residual convolutional network to produce one-dimensional pose descriptors. Temporal features are extracted via LSTM or GRU cells, aggregated through Average temporal pooling, resulting in one-dimensional identification vectors with high discriminatory properties. These vectors are linearly separable, allowing for classification with linear SVM or similar classifiers.

## Technologies Used
- Python
- OpenCV for image processing
- TensorFlow for building and training the neural network models
- HumanPose Estimator dataset for training and validation

## Getting Started
These instructions will guide you through setting up the project on your local machine for development and testing.

### Prerequisites
Ensure you have Python installed along with OpenCV and TensorFlow. You can install them using pip:

```bash
pip install opencv-python tensorflow
