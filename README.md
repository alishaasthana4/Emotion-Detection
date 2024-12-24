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

### Model Training
1. **Data Preprocessing:**
   - Load and normalize images from the FER-2013 dataset.
   - Augment data using techniques such as rotation, flipping, and zooming.

2. **Building Models:**
   - Baseline CNN: Custom layers with Conv2D, MaxPooling2D, Flatten, Dense, and Dropout.
   - VGG16 Transfer Learning:
     - Load the VGG16 model with `include_top=False`.
     - Add GlobalAveragePooling2D, Dense, and Dropout layers for fine-tuning.

3. **Training:**
   - Compile models using an optimizer like Adam with a learning rate of 0.0001.
   - Train the models with categorical cross-entropy loss and monitor validation accuracy.

### Evaluation
- Evaluate the models on a separate validation set.
- Compare the performance of the baseline CNN, VGG16, and ResNet approaches.
- ResNet outperformed the other models, showing the highest accuracy and better generalization. Below is the classification report for the ResNet model:

```
Classification Report:
               precision    recall  f1-score   support

       angry       0.54      0.57      0.55       958
     disgust       0.57      0.61      0.59       111
        fear       0.43      0.37      0.40      1024
       happy       0.89      0.79      0.84      1774
     neutral       0.53      0.63      0.58      1233
         sad       0.55      0.39      0.46      1247
    surprise       0.56      0.84      0.67       831

    accuracy                           0.61      7178
   macro avg       0.58      0.60      0.58      7178
weighted avg       0.61      0.61      0.60      7178
```

- Generate a confusion matrix and classification report to assess performance metrics.



