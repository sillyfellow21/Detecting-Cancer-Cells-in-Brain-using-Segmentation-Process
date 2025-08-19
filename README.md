# Detecting Cancer Cells in Brain using Segmentation with U-Net

This repository contains an end-to-end pipeline for **3D brain cell segmentation** using multi-parametric MRI volumes. It is based on the well-known BraTS (Brain Tumor Segmentation) challenge data and implements a **3D U-Net** to segment glioma sub-regions (edema, enhancing tumor, and necrotic core) from stacked MRI modalities. The included Jupyter notebook demonstrates the following steps:
## Features
- **3D U-Net Model**: A convolutional neural network (CNN) architecture for semantic segmentation in 3D images.
- **BraTS Dataset**: Multi-modal MRI data, including FLAIR, T1c, T2, and segmented labels.
- **Segmentation**: Automatic segmentation of tumor sub-regions, including edema, necrotic core, and enhancing tumor.
- **Custom Loss Functions**: Combination of Dice loss and categorical focal loss for improved training performance.

- ## Libraries Used
- **TensorFlow**: A machine learning framework for building and training deep learning models.
- **Keras**: High-level neural networks API, running on top of TensorFlow.
- **NumPy**: For numerical operations, including data manipulation.
- **Scikit-learn**: For preprocessing and metric calculations.
- **Matplotlib**: For plotting and visualizing training results and outputs.
- **Nibabel**: For reading and processing MRI data formats such as `.nii.gz`.
- **Segmentation Models 3D**: For 3D segmentation loss functions and metrics.
 ## Requirements

- Python 3.7+
- Libraries: numpy, nibabel, tensorflow, keras, matplotlib, tifffile, scikit-learn, splitfolders

  
## Setup

1. Download BraTS 2021 dataset to `C:\Users\20101309\Desktop\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\`.
2. Preprocess data to 128x128x128 format in `E:\BRATS 2021\input_data_128\`.

## Usage

Run the notebook `Cancer_cell_segmentation.ipynb` in Jupyter.

- Loads and normalizes images/masks.
- Predicts on validation images (e.g., 402, 34, 247).
- Saves predictions to `C:\Users\20101309\Desktop\outputs\`.

Example prediction plot:

![Prediction Example](outputs/402_pred.png)

## Results

Predictions for slices shown in notebook. Model evaluates on validation set.

## License

MIT


Did this project in Spring 2024 for my CSE438 course with my teammates : Md Ashiq Ul Islam Sajid & Rumaysa Mumtahana





