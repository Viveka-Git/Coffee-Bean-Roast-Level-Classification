# ☕ Coffee Bean Roast Level Classification

### Using Custom CNN and MobileNetV2 (Comparative Study)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Project Overview

Coffee roast level grading is traditionally performed by human experts, which introduces subjectivity and inconsistency in large-scale production. This project presents an **automated computer vision system** for **coffee bean roast level classification** using **deep learning**.

We classify coffee beans into **four roast levels**:

* 🟢 Green
* ☕ Light
* ☕☕ Medium
* ☕☕☕ Dark

A **comparative study** is conducted between:

* A **Custom Convolutional Neural Network (CNN)**
* A **Transfer Learning model using MobileNetV2**

Both models are evaluated on unseen test data under realistic conditions.

---

## Objectives

* Automate coffee bean roast classification using images
* Reduce subjectivity in roast grading
* Evaluate robustness under real-world noise and lighting variations
* Compare **custom CNN vs MobileNetV2**
* Demonstrate real-time classification with confidence visualization

---

## 📁 Dataset Structure

```text
/train
 ├── /Dark
 ├── /Green
 ├── /Light
 └── /Medium

/test
 ├── /Dark
 ├── /Green
 ├── /Light
 └── /Medium
```

Each folder contains labeled coffee bean images corresponding to roast levels.

---

## Preprocessing & Segmentation

* Image resizing to **150 × 150 × 3**
* Normalization to **[0,1]**
* ROI segmentation using:

  * Grayscale thresholding
  * Morphological operations
* Feature focus on **color intensity and texture**
* Data augmentation to simulate real-world conditions:

  * Brightness variation
  * Noise
  * Blurring

---

## Models Used

### 1️⃣ Custom CNN Architecture

| Layer           | Purpose                                    |
| --------------- | ------------------------------------------ |
| Conv2D (32)     | Edge & color gradient extraction           |
| Conv2D (64)     | Texture & crack patterns                   |
| Conv2D (128)    | Roast intensity & surface features         |
| Dense + Dropout | Non-linear classification & regularization |
| Softmax         | 4-class probability output                 |

**Training Details:**

* Optimizer: Adam
* Loss: Categorical Crossentropy
* Epochs: 10
* Input Shape: (150, 150, 3)

---

### 2️⃣ MobileNetV2 (Transfer Learning)

* Pre-trained on **ImageNet**
* Lightweight & efficient
* Global Average Pooling + Dense layers
* Fine-tuned for 4-class classification

**Why MobileNetV2?**

* Faster inference
* Fewer parameters
* Suitable for mobile & embedded deployment

---

## Evaluation Results

### Custom CNN Performance

| Class        | Precision | Recall | F1-Score |
| ------------ | --------- | ------ | -------- |
| Dark         | 0.98      | 0.95   | 0.96     |
| Green        | 1.00      | 1.00   | 1.00     |
| Light        | 1.00      | 0.99   | 0.99     |
| Medium       | 0.94      | 0.98   | 0.96     |
| **Accuracy** |           |        | **0.98** |

---

### MobileNetV2 Performance

| Class        | Precision | Recall | F1-Score |
| ------------ | --------- | ------ | -------- |
| Dark         | 1.00      | 0.99   | 0.99     |
| Green        | 0.99      | 1.00   | 1.00     |
| Light        | 1.00      | 0.93   | 0.96     |
| Medium       | 0.93      | 1.00   | 0.97     |
| **Accuracy** |           |        | **0.98** |

---

## Comparative Analysis

| Aspect           | Custom CNN    | MobileNetV2   |
| ---------------- | ------------- | ------------- |
| Accuracy         | 98%           | 98%           |
| Model Size       | Larger        | Lightweight   |
| Training Time    | Higher        | Faster        |
| Feature Learning | Task-specific | Pre-trained   |
| Deployment       | Server        | Mobile / Edge |

➡ **Conclusion:**
Both models perform equally well, but **MobileNetV2 is preferred for deployment**, while **Custom CNN provides better architectural interpretability**.

---

## Interactive Demo

A real-time demo:

* Randomly selects a test image
* Predicts roast level with confidence
* Displays real-world usage of the roast type

Implemented using **Matplotlib + ipywidgets**

---

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy, Pandas
* Matplotlib, Seaborn
* Google Colab

---

## Key Contributions

* Light-invariant coffee roast classification
* ROI-based segmentation
* Robust evaluation under noise
* Comparative deep learning study
* Deployment-ready architecture

---

## License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.

---

## Acknowledgments

* Open-source deep learning community
* TensorFlow & Keras documentation
* Academic references on computer vision and CNNs

---

## Contact

For academic discussion or collaboration:

> **VIVEKA S**
> *Big Data Analytics Core Project*

---
