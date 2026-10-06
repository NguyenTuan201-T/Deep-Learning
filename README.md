# Chest X-ray Multi-Label Classification Project

A multi-label classification project for chest X-ray images using DenseNet-121[cite: 3]. The system takes a chest X-ray image as input and answers 14 binary questions (whether the image exhibits a specific pathology or not)[cite: 3].

## Overview of the Processing Pipeline
- **Input:** A chest X-ray image ($224 \times 224$ pixels)[cite: 3].
- **Feature Extraction:** DenseNet-121 outputs 14 raw logits ($z$).
- **Temperature Scaling:** Divide by the learned temperature parameter ($T$) for each specific pathology (handled by Block E).
- **Probability Computation:** Compute probabilities $p = \text{sigmoid}(z / T)$.
- **Thresholding:** Compare against the optimized decision threshold ($t$) for each pathology (handled by Block D).
- **Output:** 14 binary decisions along with 14 independent probabilities (each pathology has its own probability; they do not sum to 1)[cite: 3].
- *Note:* The temperature scaling ($T$) and decision thresholds ($t$) are not intrinsic parts of the neural network architecture; they are optimized post-training using the validation set (implemented in Blocks D and E).

## Pipeline Architecture across 5 Blocks

- **Block A (Data Preprocessing):**
  - **Inputs:** Kaggle dataset images, `Data_Entry_2017.csv`, and `test_list_NIH.txt`[cite: 3].
  - **Tasks:** Parse and transform labels into 14 binary columns ($0/1$), tag official test images, and record image paths[cite: 3].
  - **Output:** `meta.csv` (112,120 rows containing image file names, patient IDs, official split flags, file paths, and 14 binary labels)[cite: 3].

- **Block B (Data Splitting):**
  - **Input:** `meta.csv`[cite: 3].
  - **Task:** Perform a strict patient-level split (*Patient-wise split*) to prevent data leakage[cite: 3].
  - **Outputs:** `train.csv` (75,860 images), `val.csv` (10,664 images), and `test.csv` (25,596 images)[cite: 3].

- **Block C (Model Training):**
  - **Inputs:** `train.csv`, `val.csv`, and raw image files.
  - **Task:** Train DenseNet-121 and select the best epoch based on validation AUROC[cite: 3].
  - **Outputs:** `best.pt` model weights and 4 numpy arrays (`logits_val.npy`, `labels_val.npy`, `logits_test.npy`, `labels_test.npy`)[cite: 3].

- **Block D (Decision Threshold Tuning):**
  - **Inputs:** Validation logits and labels (subsequently evaluated on the test set).
  - **Tasks:** Optimize individual decision thresholds ($t$) per pathology, calculate AUROC, AP, F1-scores, and bootstrap confidence intervals[cite: 3].
  - **Output:** Excel spreadsheet summarizing performance metrics per pathology.

- **Block E (Probability Calibration):**
  - **Inputs:** The 4 logits/labels arrays generated from Block C.
  - **Tasks:** Learn temperature scaling coefficients ($T$) per pathology, compute Expected Calibration Error (ECE), Brier scores, and plot reliability diagrams.
  - **Outputs:** Excel spreadsheets and calibration PNG plots.
