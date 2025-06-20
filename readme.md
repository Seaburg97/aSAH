# ASAH Patient Outcome Prediction Model

> **Note**: This repository is under active development. While the core functionality is complete, documentation and code comments are being improved. Additional features and detailed documentation will be added incrementally.

This repository contains an implementation of deep learning models for predicting modified Rankin Scale (mRS) scores in patients with aneurysmal subarachnoid hemorrhage (ASAH). The model architecture refer to the approach presented in:

> Liu Y, Yu Y, Ouyang J, et al. Prediction of ischemic stroke functional outcomes from acute-phase noncontrast CT and clinical information. Radiology 2024;313:e240137.

## Model Overview

We provide **five main models** from the manuscript, along with **two alternative fusion variants** from the supplementary material:(models sees in https://drive.google.com/drive/folders/1h4DCQZDT3-uWbUz6fDorrNToGVo23zg2?usp=drive_link):

### 🔹 Main Models
1. **Pre-operative Model**: A CBAM-enhanced ResNet50 architecture for predicting mRS scores using pre-operative CT scans
2. **Post-operative Model**: Similar CBAM-ResNet50 architecture for post-operative outcome prediction
3. **Stacking Imaging Model**: An SVR-based model that combines predictions from both pre-operative and post-operative models
4. **Clinical Model**: An SVR-based model utilizing clinical features for outcome prediction
5. **Fusion Model**: A  SVR-based model that integrates both Stacking imaging and clinical models predictions

### 🔹 Supplementary Models
- `Fusion-Alt1 Model` *(A  SVR-based model that integrates both pre-operative and clinical models predictions)*
- `Fusion-Alt2 Model` *(A  SVR-based model that integrates both post-operative and clinical models predictions)*

Each model has been saved as a separate folder containing:
- Pre-trained model weights
- Python scripts for inference
 🔗 **Note**: Due to file size limits, the `Pre-operative` and `Post-operative` models are hosted on Google Drive. Download links are provided in their respective folders.

---

## 🛠️ How to Use

Each model is self-contained and can be run independently by modifying only the input path.

**Steps:**
1. Navigate to the desired model folder (e.g., `Fusion_model/`)
2. Open the provided Python script (e.g., `test.py`)
3. Modify the `data_path` or `model_path` to your local file structure
4. Run the script to obtain predictions

> 💡 You do **not** need to re-train the models. All weights are pre-loaded and ready for evaluation.

---
## Requirements

- PyTorch
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- TorchIO
...(sees in Python script)

### Model Evaluation

```bash
python fusion_test.py --model [path_to_model] --data [path_to_test_data]
```

## Model Evaluation

The implementation includes comprehensive evaluation tools for:
- ROC curve analysis
- MAE (Mean Absolute Error) calculation
- Accuracy metrics for different mRS thresholds
- Model comparison and statistical analysis

## Performance Metrics

The models are evaluated using:
- Mean Absolute Error (MAE)
- Area Under the Curve (AUC)
- Sensitivity and Specificity
- Overall Accuracy

---

## 🚧 Project Roadmap

This repository is part of an ongoing research project. In the next stage, we aim to:

- Integrate the five main models into a unified **end-to-end pipeline**
- Simplify model usage and reduce code redundancy
- Support batched predictions and clinical UI integration
- Release additional tools for visualization and interpretability (e.g., SHAP, Grad-CAM)

---
