# Multi-Digit Spatial Grid Detector

A custom PyTorch-based object detection model designed to simultaneously localize and classify multiple digits within a single image. This project implements a lightweight convolutional neural network (CNN) with a fixed-grid detection head, inspired by single-shot detection architectures.

##  Project Overview

The model processes images containing multiple digits (classes 0-9) and predicts bounding boxes and class labels for each object. It divides the input image into an 8x11 spatial grid, where each cell is responsible for predicting objectness confidence, bounding box offsets (x, y, w, h), and class probabilities.

##  Key Features

*   **Custom Grid Detection Head:** Utilizes a custom CNN backbone followed by an `AdaptiveAvgPool2d` layer to map extracted features to a fixed 8x11 spatial grid.
*   **Dynamic Target Building:** Automatically extracts ground-truth bounding boxes directly from boolean segmentation masks and encodes them into dense grid targets.
*   **Multi-Part Loss Function:** Dynamically balances three distinct loss components during training:
    *   *Binary Cross-Entropy (BCE)* with positive class weighting for Objectness.
    *   *Smooth L1 Loss* for Bounding Box Regression.
    *   *Cross-Entropy Loss* for Multi-Class Classification.
*   **Robust Prediction Decoding:** Employs Torchvision's Non-Maximum Suppression (NMS) to effectively filter out redundant, overlapping bounding box predictions based on an IoU threshold.
*   **Interactive Visualization Dashboard:** Features a custom Matplotlib-based viewer that allows users to seamlessly navigate through test images using keyboard controls, comparing ground-truth bounding boxes side-by-side with the model's real-time predictions and confidence scores.

##  Repository Structure

*   `m2nist_grid_detector.py`: The core script containing the model architecture (`GridDetector`), training loop, target building logic, prediction decoding, and the interactive visualization tool.
*   `dataset/imagedata.npy`: The dataset of input images *(must be provided in the working directory)*.
*   `dataset/groundtruth.npy`: The corresponding dataset of boolean segmentation masks *(must be provided in the working directory)*.

##  Technologies Used

*   **Python 3**
*   **Deep Learning:** PyTorch, Torchvision (`nms`)
*   **Data Manipulation:** NumPy, Scikit-learn (`train_test_split`)
*   **Visualization:** Matplotlib

##  Usage

### 1. Prerequisites
Ensure you have the necessary libraries installed (`torch`, `torchvision`, `numpy`, `scikit-learn`, `matplotlib`)

### 2. Download the Dataset
The `.npy` dataset and ground truth masks are too large to be hosted directly in this repository. You will need to download them from Kaggle before running the model:

1. Download the dataset from Kaggle: [Multi-MNIST (M2NIST) Dataset](https://www.kaggle.com/datasets/farhanhubble/multimnistm2nist)
2. Create a folder named `dataset/` in the root directory of this repository.
3. Place the downloaded files (`imagedata.npy` and `segmented.npy`) inside the `dataset/` folder.
4. *Note: The script expects the ground truth file to be named `groundtruth.npy`. You can either rename `segmented.npy` to `groundtruth.npy`, or update the file path in `m2nist_grid_detector.py`.*

### 3. Training & Evaluation
To train the model from scratch and automatically launch the evaluation dashboard, run:
```bash
python m2nist_grid_detector.py
```

### 4. Interactive Viewer Controls
* Once the specified training epochs are complete, the interactive viewer will launch automatically.
* Right Arrow / Down Arrow: Advance to the next image.
* Left Arrow / Up Arrow: Return to the previous image.
