# BSCS23124_A_0-
ANN 
# README.md

# Assignment 02: ResNet and U-Net Fine-Tuning

## Student Information

| Field | Details |
|---|---|
| **Name** | Syeda Ayesha Gillani |
| **Roll No.** | BSCS23124 |
| **Course** | ANN & Deep Learning |
| **Assignment** | 02 - ResNet and U-Net |

---

## Project Overview

This project focuses on applying transfer learning and fine-tuning techniques to two well known convolutional neural network architectures.

| Model | Task | Description |
|---|---|---|
| **ResNet-50** | Image Classification | Identifying the class of a flower from an image |
| **U-Net** | Image Segmentation | Identifying which pixels belong to a pet |

The goal is to adapt models pretrained on ImageNet to small custom datasets using transfer learning.

---

## Datasets

Two datasets were used in this assignment.

### Dataset 1: Flower Classification

| Field | Details |
|---|---|
| **Source** | Oxford Flowers-102 |
| **Download Link** | https://www.robots.ox.ac.uk/~vgg/data/flowers/102/ |
| **Files** | `102flowers.tgz`, `imagelabels.mat` |
| **Size Used** | 5 classes x 40 images = 200 images |
| **Used By** | ResNet-50 |

### Dataset 2: Pet Segmentation

| Field | Details |
|---|---|
| **Source** | Oxford-IIIT Pet |
| **Download Link** | https://www.robots.ox.ac.uk/~vgg/data/pets/ |
| **Files** | `images.tar.gz`, `annotations.tar.gz` |
| **Size Used** | 150 image and mask pairs |
| **Used By** | U-Net |

For the pet masks, the original trimap convention is kept. Pixel value 1 is pet, 2 is background, and 3 is border.

---

## Required Folder Structure

The datasets should be organized on Google Drive as follows.

```text
MyDrive/
└── Ann_DL_Assignment/
    ├── Flowers/                     <- classification dataset (ImageFolder format)
    │   ├── flower_class_000/  *.jpg
    │   ├── flower_class_001/  *.jpg
    │   ├── flower_class_002/  *.jpg
    │   ├── flower_class_003/  *.jpg
    │   └── flower_class_004/  *.jpg
    ├── Pets/                        <- segmentation dataset
    │   ├── images/   *.jpg
    │   └── masks/    *.png          <- same file name as the image
    └── checkpoints/                 <- trained model weights
```

---

## Environment Setup

This project was developed and executed in Google Colab.

### Requirements

- Google Colab with T4 GPU runtime
- PyTorch
- Torchvision
- Segmentation Models PyTorch (SMP)
- Scikit-learn
- Matplotlib
- NumPy
- Pandas
- PIL

### Installation

The following command installs the segmentation library.

```bash
pip install -q segmentation-models-pytorch
```

---

## Project Workflow

### Part A: ResNet-50 Fine-Tuning (Classification)

| Step | Description |
|---|---|
| Step 1 | Environment setup, imports, and reproducibility seeds |
| Step 2 | Mount Google Drive and set dataset paths |
| Step 3 | Explore the flower dataset |
| Step 4 | Preprocessing, augmentation, and train/validation/test split |
| Step 5 | ResNet-50 architecture and loading pretrained model |
| Step 6 | Training utilities |
| Step 7 | Phase 1 feature extraction with frozen backbone |
| Step 8 | Phase 2 fine-tuning by unfreezing layer4 |
| Step 9 | Training curves |
| Step 10 | Evaluation on the test set |

### Part B: U-Net Fine-Tuning (Segmentation)

| Step | Description |
|---|---|
| Step 11 | Explore the pet segmentation dataset |
| Step 12 | Joint image and mask preprocessing, split, and loaders |
| Step 13 | U-Net architecture and model creation |
| Step 14 | Loss function and metrics |
| Step 15 | Phase 1 train the decoder with frozen encoder |
| Step 16 | Phase 2 fine-tune the encoder |
| Step 17 | Evaluation and visualization |

---

## Implementation Details

### Image Preprocessing

All images were resized to 224x224 pixels for ResNet-50.

| Pipeline | Transformations |
|---|---|
| **Training** | Resize, RandomHorizontalFlip, RandomRotation(15), ColorJitter, ToTensor, Normalize |
| **Evaluation** | Resize, ToTensor, Normalize |

Normalization used ImageNet mean and standard deviation values.

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.225, 0.224, 0.225]
```

### Data Splitting

The flower dataset was split using a stratified 70/15/15 approach.

| Set | Percentage | Number of Images |
|---|---|---|
| Training | 70% | 140 |
| Validation | 15% | 30 |
| Test | 15% | 30 |

---

## Results

### Flower Classification Dataset

**Class Distribution**

The dataset contains 5 classes with 40 images each. The dataset is perfectly balanced.

```text
Number of classes: 5
Total images      : 200
  flower_class_000: 40
  flower_class_001: 40
  flower_class_002: 40
  flower_class_003: 40
  flower_class_004: 40
```

**Data Split Output**

```text
Classes: ['flower_class_000', 'flower_class_001', 'flower_class_002', 'flower_class_003', 'flower_class_004']
Train/Val/Test: 140 30 30
```

### Pet Segmentation Dataset

**Data Preparation Output**

```text
Found 0 valid image/mask pairs
Copied 0 images and 0 masks

Final structure
Pets/images: 0
Pets/masks : 0
```

The data preparation script failed to find valid image and mask pairs. This indicates a problem with file paths or filename matching logic. The U-Net training could not proceed due to this issue.

---

## Issues Encountered

### 1. Segmentation Dataset Preparation Failure

The script that matches pet images with their corresponding masks found zero pairs. This could be due to:

- Incorrect path to raw image or mask directories
- Mismatch in filenames between images and masks
- Different naming conventions than expected

**Solution:** Verify that `RAW_PET_IMG` and `RAW_PET_MASK` paths are correct and that image files like `pet_name.jpg` match mask files like `pet_name.png`.

### 2. Incomplete Model Training

The ResNet-50 and U-Net training, evaluation, and reporting tasks were not executed in the provided notebook. Results for model accuracy, loss curves, confusion matrices, and segmentation metrics are not available.

---

## Key Learnings

1. **Transfer Learning:** Using pretrained models saves time and resources compared to training from scratch.
2. **Data Augmentation:** Random flips, rotations, and color jitter help prevent overfitting on small datasets.
3. **Stratified Splitting:** Ensures each data subset has the same class proportions as the original dataset.
4. **Two-Phase Fine-Tuning:** Training with a frozen backbone first and then unfreezing layers helps the model adapt gradually.
5. **Importance of Data Verification:** Visualizing and counting data before training catches problems early.

---

## File Structure

```text
BSCS_23124_Assignment_02.ipynb    <- Main Jupyter Notebook
README.md                         <- Project documentation
```

---

## How to Run

1. Open the notebook in Google Colab.
2. Set runtime to T4 GPU.
3. Run the installation cell to install SMP.
4. Mount Google Drive when prompted.
5. Ensure datasets are uploaded to the correct paths on Google Drive.
6. Run all cells in order.

---

## Contact

For questions or clarifications, contact:

**Syeda Ayesha Gillani**
Roll No. BSCS23124
