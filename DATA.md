# Dataset Documentation (DATA.md)

This document outlines the dataset sources, versioning, patient-wise data splitting strategy, preprocessing pipelines, and reproduction instructions for the **Chest X-ray Multi-Label Classification** project.

---

## 1. Official Dataset Source & Version
- **Dataset Name:** NIH Chest X-ray14 Dataset
- **Official Source URL:** [NIH Clinical Center Chest X-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- **Dataset Version:** 2017 Release (`Data_Entry_2017.csv`)
- **Description:** The official dataset consists of 112,120 frontal-view X-ray images from 30,805 unique patients, labeled across 14 common thoracic pathologies (e.g., Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, etc.).

---

## 2. Data Splits (Patient-Wise Split)
To strictly prevent data leakage (ensuring that images belonging to the same patient do not cross over between training, validation, and testing sets), the dataset is split at the **patient level**:
- **Training Set (`train.csv`):** 75,860 images
- **Validation Set (`val.csv`):** 10,664 images
- **Official Test Set (`test.csv`):** 25,596 images (derived strictly from the official NIH test list `test_list.txt`)

---

## 3. Data Preprocessing Pipeline (Blocks A & B)
The data preprocessing and splitting mechanism operates through the following stages:

- **Block A (Data Preprocessing):**
  - **Inputs:** Kaggle dataset images, `Data_Entry_2017.csv`, and `test_list_NIH.txt`.
  - **Tasks:** Parse and transform labels into 14 binary columns ($0/1$), tag official test images, and record image paths.
  - **Output:** `meta.csv` (112,120 rows containing image file names, patient IDs, official split flags, file paths, and 14 binary labels).

- **Block B (Data Splitting):**
  - **Input:** `meta.csv`.
  - **Task:** Perform a strict patient-level split (*Patient-wise split*) to prevent data leakage.
  - **Outputs:** `train.csv` (75,860 images), `val.csv` (10,664 images), and `test.csv` (25,596 images).

- **Image Transformation:**
  - All input images are resized to $224 \times 224$ pixels to match the input dimensions required by DenseNet-121.
  - Pixel intensities are normalized using standard ImageNet mean and standard deviation values.

---

## 4. Reproduction Scripts
To reproduce the data preprocessing and splitting pipelines from the raw NIH dataset, run the following commands sequentially:

```bash
# Step 1: Generate metadata and format labels (Block A)
python scripts/data_preprocessing.py --input_dir /path/to/raw/images --output data/meta.csv

# Step 2: Execute patient-wise splitting into train, validation, and test sets (Block B)
python scripts/patient_split.py --meta_file data/meta.csv --output_dir data/splits/

5. Processed Dataset Download Link
If you wish to use our pre-processed dataset splits directly without running the preprocessing scripts from scratch, you can download the processed CSV files and metadata via the following link:

Processed Dataset Link: https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized