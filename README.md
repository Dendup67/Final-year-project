# Drowsiness Detection System Using Deep Learning

Welcome to the repository for my Final Year Project: **Drowsiness Detection System using Deep Learning and the Bhutanese Drowsiness Dataset**. This project aims to provide an effective, deployable, and accurate solution to detect drowsiness using a custom private dataset and a convolutional neural network (CNN).

---

## Table of Contents

- [Abstract](#abstract)
- [Features](#features)
- [Bhutanese Drowsiness Dataset](#bhutanese-drowsiness-dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [License](#license)


---

## Abstract

Drowsiness is a critical factor in many accidents, yet detection systems are often lacking, especially in Bhutan. This project presents a CNN-based drowsiness detection system, utilizing the first Bhutanese Drowsiness Dataset with 4000 labeled images. The proposed model achieved 99.9% accuracy on training and 93.25% on testing, evaluated using standard metrics. The system is realistic, easy to deploy, and highly effective.

---

## Features

- **Custom Dataset:** 4000 images from 50 Bhutanese participants, labeled as "Awake" or "Drowsy".
- **Deep Learning (CNN):** VGG-like architecture with four convolutional blocks, batch normalization, max pooling, and dropout.
- **Image Preprocessing:** Data augmentation (saturation, brightness, flipping, rotation, etc.) to generate robust training data.
- **High Accuracy:** 99.9% (train), 92.75% (validation), 93.25% (test).
- **Evaluation Metrics:** Confusion matrix, precision, recall, and F1-score.
- **Easy Deployment:** Model saved in `.h5` format; ready for integration with web/mobile apps.

---

## Bhutanese Drowsiness Dataset

- **Images:** 4000 total, equally split between "Awake" and "Drowsy".
- **Sources:** Video frames and images from 50 young Bhutanese individuals.
- **Preprocessing:** Frames extracted every 10th frame, duplicates removed, and augmented for diversity.
- **Split:** 70% training, 20% validation, 10% testing.

---

## Methodology

1. **Data Collection:** Gathered videos and images of participants in awake and drowsy states.
2. **Image Preprocessing:** Augmentation to address overfitting and enrich dataset.
3. **Model Architecture:** CNN inspired by VGGNet (four convolutional blocks, batch normalization, max pooling, dropout, ReLU activation, and sigmoid output).
4. **Training:** Used Adam optimizer, binary cross-entropy loss, ReduceLROnPlateau, and early stopping to optimize training.
5. **Evaluation:** Used confusion matrix and classification report (precision, recall, F1-score).

---

## Results

| Model Configuration         | Train Acc. | Val. Acc. | Test Acc. |
|----------------------------|------------|-----------|-----------|
| Baseline (11 epochs)       | 97.66%     | 81.25%    | 85.75%    |
| Baseline (100 epochs)      | 99.93%     | 81.88%    | 91.25%    |
| + Dropout                  | 99.93%     | 80.26%    | 78.25%    |
| + Early stopping           | 97.5%      | 67.87%    | 79.75%    |
| **Finalized Model**        | **99.90%** | **92.75%**| **93.25%**|

**Classification Report:**

| Class   | Precision | Recall | F1-score |
|---------|-----------|--------|----------|
| Awake   | 0.88      | 1.00   | 0.94     |
| Drowsy  | 1.00      | 0.86   | 0.93     |
| Weighted Avg | 0.94 | 0.93   | 0.93     |

---

## Tech Stack

- **Language:** Python
- **Libraries:** TensorFlow, Keras, OpenCV, NumPy, scikit-learn, Matplotlib
- **Hardware:** Intel Core i5, 8GB RAM, Nvidia GeForce 1650Ti GPU

---

## Scientific Paper
- **Read More:** [Paper](https://ddc-dictionary.en.softonic.com/android](https://github.com/Dendup67/Final-year-project/blob/main/Scientific%20paper_Group3_FYP2021.pdf)

## License

This project is licensed under the [MIT License](LICENSE).

---

