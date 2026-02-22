# 🔢 Digit Recognizer: Handwritten Digit Classification

**Goal:** Recognize and classify handwritten digits (0-9) using Deep Learning.<br>
**Challenge Source:** Kaggle - Digit Recognizer Competition

## 📌 Project Overview
This project implements a **Convolutional Neural Network (CNN)** to classify handwritten digits from the MNIST dataset. The model learns to recognize digits by training on 42,000 labeled examples and then predicts the digit value from pixel data.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, TensorFlow/Keras, Matplotlib
* **Model:** Convolutional Neural Network (CNN)
* **Framework:** TensorFlow Keras Sequential API

## 📊 The Data Process (My Approach)

### 1. Data Loading & Exploration
The MNIST dataset contains flattened handwritten digit images stored in CSV format.
* **Dataset Size:** 42,000 training images
* **Image Dimensions:** Each image is **28×28 pixels** (784 pixel values when flattened)
* **Pixel Values:** Range from 0 to 255 (grayscale intensity, 0 = black, 255 = white)
* **Labels:** 10 classes (digits 0-9)

### 2. Data Preprocessing
Raw pixel data needs transformation before feeding to the neural network:
* **Normalization:** Scaled pixel values from [0, 255] to [0, 1] by dividing by 255. This standardization helps the model converge faster and improves numerical stability.
* **Reshaping:** Converted the flattened 784-pixel vectors back into **28×28×1 images** (height, width, color channels). The extra dimension represents grayscale (single color channel).
* **Train-Validation Split:** Used 20% of the training data for validation during training to monitor for overfitting.

### 3. Model Architecture
Built a multi-layer Convolutional Neural Network optimized for image classification:

```
Layer 1: Conv2D (32 filters, 3×3 kernel, ReLU activation)
         ↓ Captures basic features (edges, corners)
Layer 2: MaxPooling2D (2×2)
         ↓ Reduces spatial dimensions
Layer 3: Conv2D (64 filters, 3×3 kernel, ReLU activation)
         ↓ Learns more complex patterns
Layer 4: MaxPooling2D (2×2)
         ↓ Further reduces dimensions
Layer 5: Conv2D (64 filters, 3×3 kernel, ReLU activation)
         ↓ Detects higher-level features
Layer 6: Flatten
         ↓ Converts 2D feature maps to 1D vector
Layer 7: Dense (64 neurons, ReLU activation)
         ↓ Learns digit-specific patterns
Layer 8: Dense (10 neurons, Softmax activation)
         ↓ Outputs probability for each digit (0-9)
```

**Design Rationale:**
* **Convolutional Layers:** Automatically learn spatial features from pixel neighborhoods, recognizing patterns like curves and lines that define digits.
* **Pooling Layers:** Reduce computational load and provide translation invariance (model recognizes digits regardless of slight position shifts).
* **ReLU Activation:** Introduces non-linearity, allowing the network to learn complex digit shapes.
* **Softmax Output:** Ensures normalized probabilities that sum to 1 across 10 digit classes.

## ⚙️ Model Training & Results

### Training Configuration
* **Optimizer:** Adam (adaptive learning rate, efficient for CNNs)
* **Loss Function:** Sparse Categorical Crossentropy (suitable for multi-class classification)
* **Metrics:** Accuracy
* **Epochs:** 5
* **Batch Size:** Default (32)
* **Validation Split:** 20%

### Performance Metrics
During training, the model tracks:
* **Training Accuracy:** Accuracy on the 80% training subset
* **Validation Accuracy:** Accuracy on the 20% validation subset
* **Training Loss:** Measures prediction error on training data
* **Validation Loss:** Measures prediction error on unseen validation data

The accuracy and loss plots show the model's learning trajectory, with convergence indicating successful training.

## 🚀 How to Run
1. Clone this repository.
2. Ensure you have the `train.csv` file from the Kaggle Digit Recognizer competition.
3. Place the CSV file at `/kaggle/input/competitions/digit-recognizer/train.csv` or update the file path in the notebook.
4. Run the Jupyter Notebook cell by cell to see data loading, preprocessing, model building, and training.

## 📈 Key Learnings
* **Image Classification Fundamentals:** Understanding how CNNs extract spatial features from pixel data.
* **Convolutional Layers:** Each layer learns increasingly abstract representations (edges → shapes → digits).
* **Normalization Importance:** Scaling pixel values dramatically improves training speed and stability.
* **Deep Learning Workflow:** Data preparation, model architecture design, training, and evaluation.

## 🔮 Potential Improvements
* **Data Augmentation:** Rotate, shift, or scale training images to teach the model robustness.
* **More Epochs:** Train longer to achieve higher accuracy (5 epochs is conservative).
* **Regularization:** Add dropout layers or L2 regularization to prevent overfitting on larger datasets.
* **Batch Normalization:** Normalize inputs to each layer for more stable training.
* **Model Ensemble:** Combine multiple trained models for improved predictions.
