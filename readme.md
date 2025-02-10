# ASAH Patient Outcome Prediction Model

> **Note**: This repository is under active development. While the core functionality is complete, documentation and code comments are being improved. Additional features and detailed documentation will be added incrementally.

This repository contains an implementation of deep learning models for predicting modified Rankin Scale (mRS) scores in patients with aneurysmal subarachnoid hemorrhage (ASAH). The model architecture refer to the approach presented in:

> Liu Y, Yu Y, Ouyang J, et al. Prediction of ischemic stroke functional outcomes from acute-phase noncontrast CT and clinical information. Radiology 2024;313:e240137.

## Model Overview

The project consists of five main models(models sees in https://drive.google.com/drive/folders/1h4DCQZDT3-uWbUz6fDorrNToGVo23zg2?usp=drive_link):

1. **Pre-operative Model**: A CBAM-enhanced ResNet50 architecture for predicting mRS scores using pre-operative CT scans
2. **Post-operative Model**: Similar CBAM-ResNet50 architecture for post-operative outcome prediction
3. **Stacking Imaging Model**: An SVR-based model that combines predictions from both pre-operative and post-operative models
4. **Clinical Model**: An SVR-based model utilizing clinical features for outcome prediction
5. **Fusion Model**: A comprehensive SVR-based model that integrates both imaging and clinical predictions

### Model Architecture Details

#### CBAM ResNet50 (Pre-operative & Post-operative Models)
- Incorporates Channel and Spatial Attention Mechanisms
- Adapted 3D convolutional layers for CT image processing
- Enhanced feature extraction through attention modules
- Dropout layers for preventing overfitting

#### SVR-based Models (Stacking, Clinical & Fusion)
- Support Vector Regression based architecture
- Optimized feature selection and weighting
- Cross-validation based training approach
- Comprehensive performance evaluation

## Key Components

- `CBAM_resnet3D.py`: Implementation of the CBAM-enhanced ResNet50 architecture
- `fusion.py`: SVR-based stacking model implementation
- `fusion_test.py`: Testing pipeline for the stacking model
- `regression_train_test.py`: Training and evaluation scripts for individual models
- `SVR.py`: Implementation of Clinical and Fusion models
- `ROCandACC.py`: Performance evaluation and visualization tools
- `calculate.py`: Evaluation metrics and loss functions

## Features

- Comprehensive mRS score prediction for ASAH patients
- Integration of pre and post-operative CT scan analysis
- Clinical feature analysis and integration
- Model stacking and fusion for improved prediction accuracy
- Detailed performance metrics and visualization tools
- Support for both individual and ensemble model evaluation

## Requirements

- PyTorch
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- TorchIO
- SHAP (for model interpretation)

## Usage

### Training Individual Models

```bash
python regression_train_test.py --train_folder [path_to_training_data] --model CBAM --train_mode [pre/post]
```

### Training Stacking Model

```bash
python fusion.py --train_path [path_to_training_data] --test_path [path_to_test_data]
```

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
- SHAP value analysis for model interpretation

## Performance Metrics

The models are evaluated using:
- Mean Absolute Error (MAE)
- Area Under the Curve (AUC)
- Sensitivity and Specificity
- Overall Accuracy
- Confusion Matrix Analysis
