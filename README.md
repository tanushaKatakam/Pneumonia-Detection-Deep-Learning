# Pneumonia Detection from Chest X-Ray Images using Deep Learning

An end-to-end Deep Learning project that automatically detects **Pneumonia** from Chest X-ray images using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

The project covers the complete machine learning workflow including dataset exploration, preprocessing, model training, evaluation, visualization, and prediction on new X-ray images.

---

# Project Overview

Pneumonia is a serious lung infection that can be diagnosed using Chest X-ray images. Manual diagnosis can be time-consuming and depends on clinical expertise.

This project develops a CNN-based image classification model capable of identifying whether a Chest X-ray belongs to:

- NORMAL
- PNEUMONIA

The system also allows prediction on any user-provided Chest X-ray image.

---

# Dataset

Dataset: Chest X-ray Images (Pneumonia)

Directory Structure

dataset/
└──  chest_xray/

├──  train/

├──  val/

└──  test/

Classes

- NORMAL
- PNEUMONIA

---

# Features

- Custom CNN Architecture
- Data Augmentation
- L2 Regularization
- Dropout for Overfitting Reduction
- Early Stopping
- Learning Rate Scheduling
- Model Checkpointing
- Confusion Matrix
- Classification Report
- Prediction on Custom Images
- Training History Visualization

---

# Project Structure

```
PneumoniaDetectionProject
│
├── dataset/
│
├── notebooks/
│   └── Training_analysis.ipynb
│
├── outputs/
│   ├── accuracy.png
│   ├── loss.png
│   ├── confusion_matrix.png
│   ├── class_distribution.png
│   ├── sample_images.png
│   └── training_history.csv
│
├── saved_models/
│   └── pneumonia_cnn.keras
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── requirements.txt
└── README.md
```

---

# Model Architecture

Input Image (128×128×3)

↓

Conv2D (32) + MaxPooling

↓

Conv2D (64) + MaxPooling

↓

Conv2D (128) + MaxPooling

↓

Flatten

↓

Dense (128)

↓

Dropout (0.6)

↓

Output (Sigmoid)

---

# Regularization Techniques

To improve generalization and reduce overfitting, the model uses:

- Data Augmentation
- Dropout (0.6)
- L2 Regularization
- Early Stopping
- ReduceLROnPlateau

---

# Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **86.70%** |
| Precision | **89.06%** |
| Recall | **89.74%** |
| F1 Score | **89%** |

---

# Confusion Matrix

| | Predicted Normal | Predicted Pneumonia |
|---|---:|---:|
| Actual Normal | 191 | 43 |
| Actual Pneumonia | 40 | 350 |

---

# Training Visualizations

The project generates:

- Training vs Validation Accuracy
- Training vs Validation Loss
- Confusion Matrix
- Class Distribution
- Sample Chest X-rays

All plots are automatically saved inside the **outputs/** folder.

---

# Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python src/train.py
```

### Evaluate Model

```bash
python src/evaluate.py
```

### Predict on a New Image

```bash
python src/predict.py
```

Enter the image path when prompted.

Example:

```
../dataset/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg
```

---

# Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

# Future Improvements

- Transfer Learning (EfficientNet / ResNet)
- Hyperparameter Optimization
- Explainable AI using Grad-CAM
- Flask or Streamlit Web Application
- Model Deployment

---
