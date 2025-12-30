# DeepFER: Facial Emotion Recognition Using Deep Learning

## 🎯 Project Overview

DeepFER is an advanced facial emotion recognition system that leverages deep learning techniques to classify human emotions from facial expressions. The system uses a custom Convolutional Neural Network (CNN) architecture to identify seven distinct emotional states with real-time inference capabilities.

**Project Goal**: Develop an advanced and efficient CNN-based system to classify 7 facial emotion classes (angry, sad, happy, fear, neutral, disgust, surprise) with real-time capabilities.

### Key Features
- **Multi-class Classification**: Recognizes 7 emotion categories (angry, disgust, fear, happy, neutral, sad, surprise)
- **Real-time Inference**: Interactive web interface using Gradio for live emotion detection
- **Robust Architecture**: Custom CNN with batch normalization, dropout, and advanced regularization
- **Data Augmentation**: Enhanced training with image transformations for improved generalization
- **Professional Training Pipeline**: Includes callbacks for early stopping, learning rate scheduling, and model checkpointing
- **Comprehensive Visualization**: Training metrics, class distribution, and performance plots
- **Model Persistence**: Saves models in both Keras (.keras) and H5 (.h5) formats

## 📊 Dataset

The project utilizes the **Face Emotion Recognition Dataset** with the following structure:

```
Face Emotion Recognition Dataset/
├── train/
│   ├── angry/     (3,995 images)
│   ├── disgust/   (436 images)
│   ├── fear/      (4,097 images)
│   ├── happy/     (7,215 images)
│   ├── neutral/   (4,965 images)
│   ├── sad/       (4,830 images)
│   └── surprise/  (3,171 images)
└── validation/
    ├── angry/     (958 images)
    ├── disgust/   (111 images)
    ├── fear/      (1,024 images)
    ├── happy/     (1,774 images)
    ├── neutral/   (1,233 images)
    ├── sad/       (1,247 images)
    └── surprise/  (831 images)
```

**Dataset Statistics:**
- **Training Images**: 28,821
- **Validation Images**: 7,066
- **Image Size**: 48×48 pixels (grayscale)
- **Total Classes**: 7 emotions

## 🏗️ Model Architecture

### Custom CNN Architecture
The DeepFER model features a sophisticated CNN architecture optimized for emotion recognition with **1.72M parameters**:

#### Architecture Overview
- **Input Layer**: 48×48×1 grayscale images
- **Convolutional Blocks**: 4 progressive blocks with increasing filters (64→128→256→512)
- **Dense Layers**: Two fully connected layers for classification
- **Output Layer**: 7-class softmax for emotion prediction

#### Key Architecture Features
- **Progressive Feature Extraction**: 4 convolutional blocks with increasing filter sizes (64→128→256→512)
- **Batch Normalization**: Applied after each convolutional and dense layer for stable training
- **Dropout Regularization**: 25% dropout rate to prevent overfitting
- **Dense Classification Head**: Two fully connected layers (256→128→7) for final classification
- **ReLU Activation**: Used throughout the network for non-linearity
- **Max Pooling**: 2×2 pooling for spatial dimension reduction


