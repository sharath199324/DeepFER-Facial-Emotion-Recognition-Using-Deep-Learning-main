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

#### Model Summary
```
Total params: 1,720,327
Trainable params: 1,717,639
Non-trainable params: 2,688
```

## 🚀 Performance & Results

### Training Results
- **Final Training Accuracy**: ~59.1%
- **Final Validation Accuracy**: ~57.6%
- **Training Epochs**: 20
- **Best Validation Loss**: 1.0311 (Epoch 18)
- **Best Validation Accuracy**: 61.14% (Epoch 18)

### Training Progress
The model showed consistent improvement throughout training:
- **Epoch 1**: Training Acc: 22.06%, Validation Acc: 14.24%
- **Epoch 10**: Training Acc: 53.44%, Validation Acc: 53.82%
- **Epoch 18**: Training Acc: 58.38%, Validation Acc: 61.14% (Best)
- **Epoch 20**: Training Acc: 59.10%, Validation Acc: 57.64% (Final)

### Model Performance Characteristics
- **Convergence**: Model converged around epoch 15-18
- **Overfitting**: Minimal overfitting observed (training and validation accuracy close)
- **Stability**: Consistent performance with proper regularization
- **Class Balance**: Handles imbalanced dataset reasonably well

### Visualization Features
The project includes comprehensive visualizations:
- **Training History**: Accuracy and loss curves over epochs
- **Class Distribution**: Bar plots showing dataset distribution
- **Model Architecture**: Visual representation of CNN layers
- **Performance Metrics**: Real-time training progress monitoring

## 🛠️ Installation & Setup

### Prerequisites
```bash
Python 3.8+
TensorFlow 2.x
OpenCV
scikit-learn
matplotlib
seaborn
gradio
PIL (Pillow)
```

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/sharath199324/DeepFER-Facial-Emotion-Recognition-Using-Deep-Learning-main.git
   ```

2. **Install dependencies**
   ```bash
   pip install tensorflow opencv-python scikit-learn matplotlib seaborn gradio pillow numpy
   ```

3. **Download the dataset**
   - Place the Face Emotion Recognition Dataset in the project directory
   - Ensure the folder structure matches the format shown above

## 📁 Project Structure

```
DeepFER-Facial-Emotion-Recognition/
├── main.ipynb                          # Main training notebook with complete implementation
├── Custom_CNN_model.keras              # Trained model (Keras format)
├── Custom_CNN_model.h5                 # Trained model (H5 format)
├── Face Emotion Recognition Dataset/   # Dataset directory
│   ├── train/                         # Training images (28,821 images)
│   │   ├── angry/                     # Angry emotion images
│   │   ├── disgust/                   # Disgust emotion images
│   │   ├── fear/                      # Fear emotion images
│   │   ├── happy/                     # Happy emotion images
│   │   ├── neutral/                   # Neutral emotion images
│   │   ├── sad/                       # Sad emotion images
│   │   └── surprise/                  # Surprise emotion images
│   └── validation/                    # Validation images (7,066 images)
│       ├── angry/                     # Angry emotion images
│       ├── disgust/                   # Disgust emotion images
│       ├── fear/                      # Fear emotion images
│       ├── happy/                     # Happy emotion images
│       ├── neutral/                   # Neutral emotion images
│       ├── sad/                       # Sad emotion images
│       └── surprise/                  # Surprise emotion images
└── README.md                          # Project documentation
```

## 🔬 Technical Details

### Data Preprocessing
- **Image Resizing**: All images resized to 48×48 pixels
- **Grayscale Conversion**: RGB images converted to grayscale (single channel)
- **Normalization**: Pixel values scaled to [0, 1] range
- **Categorical Encoding**: One-hot encoding for emotion labels (7 classes)

### Data Augmentation
Real-time augmentation applied during training using ImageDataGenerator:
- **Rotation range**: ±20° random rotation
- **Width/Height shift**: ±0.2 random translation
- **Shear range**: 0.2 shear transformation
- **Zoom range**: 0.2 random zoom
- **Horizontal flip**: Random horizontal flipping
- **Fill mode**: 'nearest' for boundary pixels

### Model Training Configuration
- **Optimizer**: Adam with learning_rate=0.001
- **Loss Function**: Categorical crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 128
- **Epochs**: 20 (with early stopping)

### Advanced Training Features
- **Early Stopping**: Monitors validation loss with patience=3
- **Model Checkpointing**: Saves best model based on validation loss
- **Learning Rate Reduction**: ReduceLROnPlateau with factor=0.2, patience=3
- **Validation Split**: Separate validation set for unbiased evaluation

## 📈 Future Enhancements

- [ ] **Transfer Learning**: Implement pre-trained models (ResNet, EfficientNet, Vision Transformer)
- [ ] **Advanced Evaluation**: Add confusion matrix, classification reports, per-class metrics
- [ ] **Model Interpretability**: Integrate Grad-CAM and LIME for explainable AI
- [ ] **Cross-validation**: Implement k-fold validation for robust evaluation
- [ ] **Hyperparameter Optimization**: Add automated hyperparameter tuning
- [ ] **Ensemble Methods**: Combine multiple models for improved accuracy
- [ ] **Real-time Video**: Extend to real-time video emotion recognition
- [ ] **Mobile Deployment**: Optimize for mobile and edge devices

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 🙏 Acknowledgments

- Face Emotion Recognition Dataset contributors
- TensorFlow and Keras development teams
- Gradio team for the intuitive web interface framework
- Open source community for various tools and libraries


