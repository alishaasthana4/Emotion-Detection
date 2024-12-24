# Emotion Detection using FER-2013 and Transfer Learning (VGG16)

## Overview
This repository contains an Emotion Detection project leveraging the FER-2013 dataset and deep learning techniques. The implementation uses a Convolutional Neural Network (CNN) combined with transfer learning via the VGG16 pre-trained model. The goal is to classify facial expressions into distinct emotional categories such as happiness, sadness, anger, surprise, etc.

## Dataset
### FER-2013
The FER-2013 (Facial Expression Recognition 2013) dataset consists of grayscale images of faces, each 48x48 pixels, labeled with one of seven emotion categories:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Architecture
The model employs a two-stage process:

1. **Baseline CNN:**
   - A simple convolutional neural network to establish a baseline for performance.
   
2. **Transfer Learning with VGG16:**
   - VGG16, a pre-trained model from ImageNet, is used to enhance feature extraction capabilities.
   - The fully connected layers are replaced with custom layers tailored for emotion classification.

### CNN Architecture for Emotion Detection
The CNN used in this project includes the following key layers:
- **Convolutional Layers:** Extract spatial features from input images using filters.
- **MaxPooling Layers:** Downsample feature maps to reduce dimensionality and retain important features.
- **Flatten Layer:** Converts 2D feature maps into a 1D vector.
- **Fully Connected Layers:** Perform high-level reasoning to classify emotions.
- **Dropout Layers:** Prevent overfitting by randomly disabling neurons during training.

The baseline architecture ensures efficient feature extraction and robust learning for emotion recognition tasks.

### ResNet Performance
ResNet, another pre-trained model, was evaluated alongside VGG16 and the custom CNN. It outperformed both, achieving higher accuracy and better generalization. The residual connections in ResNet enabled efficient gradient flow, making it particularly effective for this task.



