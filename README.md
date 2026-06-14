# 🌿 Plant Disease Detection System

> A high-accuracy Computer Vision pipeline built using **TensorFlow/Keras** to automate the detection and classification of plant leaf diseases. Utilizing transfer learning with a pre-trained **MobileNetV2** architecture, the model processes leaf images to classify them into 38 distinct health and disease categories, achieving a peak validation accuracy of **97.38%**.

---

## 📋 Core Architectural Pipeline

The project implements an end-to-end deep learning training pipeline engineered to extract deep visual features and perform multi-class classification:

* 🖼️ **Data Ingestion & Augmentation:** Loads images from the **New Plant Diseases Dataset** (38 classes) and applies real-time geometric augmentations (40° rotations, shifts, shears, zooms, and horizontal flips) via `ImageDataGenerator` to mitigate overfitting.
* 📐 **Feature Extraction (Phase 1):** Instantiates a pre-trained **MobileNetV2** backbone (weights from ImageNet), freezes its base layers, and appends a custom classification head (`GlobalAveragePooling2D` -> `Dense(256, ReLU)` -> `Dropout(0.5)` -> `Dense(38, Softmax)`).
* 🔧 **Selective Fine-Tuning (Phase 2):** Unfreezes the upper layers of the MobileNetV2 backbone, keeping the early foundational layers frozen, and runs low-learning-rate (`1e-5`) backpropagation to optimize top-level contextual feature maps.

---

## ✨ Features

* **38-Class Deep Diagnostic Scope:** Covers a massive variety of crops (including Tomato, Potato, Apple, Grape, Pepper, etc.) mapping both healthy states and specific bacterial, fungal, or viral infections.
* **Two-Stage Strategic Training:** Maximizes accuracy convergence by combining efficient feature extraction with precise neural network fine-tuning.
* **Production-Minded Optimization:** Uses MobileNetV2 to keep the model lightweight and computationally efficient, making it ideal for future deployment on mobile or edge devices.

---

## 🛠️ Tech Stack & Environment

* **Deep Learning Framework:** TensorFlow 2.x / Keras
* **Hardware Accelerator:** NVIDIA Tesla T4 GPU (Kaggle Environment)
* **Data Processing & Analytics:** NumPy, Pandas
* **Visualization Engine:** Matplotlib

---

## 📊 Training Performance Summary

The network was trained systematically over a dual-phase execution scheme:

* **Phase 1 (Feature Extraction):** 10 Epochs with frozen base layers. Validation accuracy successfully converged from `47.86%` up to `86.88%`.
* **Phase 2 (Fine-Tuning):** 10 Epochs with the top 30 layers of the backbone unfrozen. The model reached its flagship metric, concluding with a **97.38% validation accuracy** and a validation loss of `0.0789`.

> 📈 *Detailed training/validation loss and accuracy charts are rendered interactively inside the Jupyter Notebook.*

---

## 📂 Project Directory Structure

```text
Plant-Disease-Detection-system/
│
├── plant-disease-detection (1).ipynb   # Comprehensive training notebook & pipeline pipeline
├── image_2b84e2.png                    # Exported performance & convergence curves
└── README.md                           # Documentation page
