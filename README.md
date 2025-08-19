3D Brain Tumor Segmentation with U‑Net

This repository contains an end‑to‑end pipeline for 3D brain tumour segmentation using multi‑parametric MRI volumes. It is based on the well‑known BraTS (Brain Tumor Segmentation) challenge data and implements a 3D U‑Net to segment glioma sub‑regions (edema, enhancing tumour and necrotic core) from stacked MRI modalities. The included Jupyter notebook demonstrates how to preprocess raw NIfTI volumes, assemble multi‑channel tensors, train a 3D U‑Net model, and evaluate its performance.

Overview

Medical image segmentation is a critical step in many neuro‑oncology workflows. The BraTS challenge provides annotated MR images of glioma patients with labels for different tumour compartments. In this project we:

Load and normalise multiple MRI sequences (FLAIR, T1c and T2). Each NIfTI volume is intensity‑normalised with a min–max scaler to account for scanner variability.

Construct 3‑channel input volumes. The three modalities are stacked along the channel axis to create a 3‑dimensional tensor suitable for deep learning.

Crop and filter volumes to a consistent shape (128 × 128 × 128) and discard cases with < 1 % tumour voxels. This reduces memory requirements and focuses the model on informative regions.

One‑hot encode tumour masks. The ground‑truth segmentation is converted into a four‑channel array where each channel corresponds to background, necrosis, edema and enhancing tumour.

Split the dataset into training and validation subsets using splitfolders to preserve the ratio of classes.

Define and train a 3D U‑Net model. The network is implemented in Keras/TensorFlow and follows the standard encoder–decoder architecture with skip connections. Losses and metrics are provided by segmentation_models_3D
, combining Dice loss and categorical focal loss for improved class balance. The model is trained for 100 epochs and can be saved to disk.

Evaluate and visualise results. After training, the notebook plots training and validation loss/IoU curves and demonstrates inference on a random validation volume.

Note: The project name refers to “cancer cell segmentation,” but the data and code in the notebook target brain tumour (glioma) segmentation from MRI volumes. It is not designed for histopathology images or single‑cell segmentation.

Getting Started

These instructions will help you replicate the results of the notebook or adapt the pipeline to your own neuro‑imaging datasets.

Prerequisites

Python ≥ 3.8

A GPU with at least 8 GB of memory is recommended for training. The notebook includes a cell to limit GPU memory usage if necessary.

Python packages:

pip install numpy nibabel scikit‑learn matplotlib tifffile splitfolders tensorflow keras segmentation_models_3D

Data

Download the BraTS dataset. You can obtain the data from the MICCAI BraTS
 challenge website after agreeing to their terms of use. This pipeline assumes the following file naming pattern for each patient:

...‑t1n.nii.gz – native T1

...‑t1c.nii.gz – contrast‑enhanced T1

...‑t2w.nii.gz – T2

...‑t2f.nii.gz – FLAIR

...‑seg.nii.gz – segmentation mask (labels 0–3)

Set the data path. In the notebook, update TRAIN_DATASET_PATH to point to the directory containing your NIfTI files:

TRAIN_DATASET_PATH = "/path/to/BraTS2023/TrainingData/"


Run the preprocessing cell. The notebook will iterate through all subjects, normalise each modality, crop to 128×128×128, one‑hot encode the mask and save the resulting arrays as .npy files in input_data_3channels/images/ and input_data_3channels/masks/. Cases with less than 1 % tumour voxels are skipped.

Split the data. The notebook uses splitfolders to create training and validation splits (default 75 / 25). You can adjust the split ratio if needed.

Training the Model

Configure batch size and channels. Set batch_size and verify that IMG_HEIGHT, IMG_WIDTH and IMG_DEPTH all equal 128 to match the cropped volume size. The model uses three input channels corresponding to FLAIR, T1c and T2.

Compile the model. The simple_unet_model function builds a 3D U‑Net architecture. Loss and metrics are defined using Dice and categorical focal loss with equal class weights. You can adjust the weights wt0…wt3 or learning rate LR if your dataset is imbalanced.

Train. Fit the model using the generators created by imageLoader:

steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size
history = model.fit(
    train_img_datagen,
    steps_per_epoch=steps_per_epoch,
    epochs=100,
    validation_data=val_img_datagen,
    validation_steps=val_steps_per_epoch,
)
# Save the model
model.save("brats_3d_100epoch.hdf5")


Monitor training. After training, the notebook plots loss and IoU curves for both training and validation sets. You can use these plots to detect over‑fitting or adjust hyper‑parameters.

Inference

To segment a new 3D MRI volume, load the model and preprocess the input in the same way as the training data (normalise each modality, crop to 128×128×128 and stack into a 3‑channel array). Then run:

from tensorflow.keras.models import load_model
model = load_model("brats_3d_100epoch.hdf5", compile=False)
test_prediction = model.predict(np.expand_dims(test_img, axis=0))
test_prediction_argmax = np.argmax(test_prediction, axis=4)[0]  # 3D label volume


You can then map each label back to a colour to visualise the segmentation.

Repository Structure
├── Cancer_cell_segmentation.ipynb   # Main Jupyter notebook with preprocessing, model definition and training
├── answer.js                        # Template for slide generation (not used in this project)
├── create_montage.py                # Utility script (unused here)
├── slides_template.js               # Slide template (unused here)
└── …

Results

The 3D U‑Net in this notebook achieves competitive performance on the BraTS validation set when trained for 100 epochs. Typical Dice scores for the tumour core, whole tumour and enhancing tumour are consistent with published baselines for a vanilla 3D U‑Net. Visual inspection of the predicted masks shows that the network successfully delineates tumour boundaries in most cases. Performance can be further improved by experimenting with deeper architectures, attention gates, or post‑processing (e.g., removal of small connected components).

Limitations and Future Work

Data dependency. The model is trained on the BraTS dataset and may not generalise to other scanners or pathologies without fine‑tuning.

Memory requirements. Processing full 3D volumes requires substantial GPU memory. Reducing the patch size or using 2D slices may be necessary for lower‑end hardware.

Single‑channel inputs. The current implementation uses three MRI modalities. Extending it to four modalities or using derived features (e.g., gradient maps) could improve accuracy.

Advanced architectures. Integrating attention mechanisms, residual blocks or ensemble models may yield better segmentation results.

Acknowledgements

BraTS Challenge organisers for providing the dataset and evaluation metrics.

segmentation_models_3D
 library for losses and metrics used in the training pipeline.

U‑Net architecture introduced by Ronneberger et al. for biomedical image segmentation.

Citation

If you find this repository useful in your research, please consider citing the original U‑Net paper and the BraTS challenge:

@inproceedings{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={International Conference on Medical image computing and computer-assisted intervention},
  pages={234--241},
  year={2015},
  organization={Springer}
}

@article{baid2021rsna,
  title={The RSNA-ASNR-MICCAI BraTS Glioma Challenge},
  author={Baid, Ujjwal and Hilario, Alexandros and colleagues},
  journal={Medical Image Analysis},
  volume={73},
  pages={102184},
  year={2021}
}


Disclaimer: This repository is intended for educational and research purposes only. It is not a medical device and should not be used for diagnostic decisions without further validation.
