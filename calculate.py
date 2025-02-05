# --coding:utf-8--
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report
import torch
import torch.nn as nn

class RegressionWithBinaryMetrics:
    def __init__(self, threshold=2.5):
        self.threshold = threshold
        self.mae_loss = nn.L1Loss()

    def get_binary_predictions(self, continuous_preds):
        return (continuous_preds > self.threshold).float()

    def get_rounded_predictions(self, continuous_preds):
        return torch.round(continuous_preds)

    def compute_metrics(self, true_labels, continuous_preds):
        true_labels = true_labels.squeeze()
        continuous_preds = continuous_preds.squeeze()
        # Regression
        mae = self.mae_loss(continuous_preds, true_labels).item()

  
        binary_labels = (true_labels > self.threshold).float()
        binary_preds = self.get_binary_predictions(continuous_preds)

        # classfication
        accuracy = accuracy_score(binary_labels.cpu(), binary_preds.cpu())
        precision = precision_score(binary_labels.cpu(), binary_preds.cpu(), zero_division=0)
        recall = recall_score(binary_labels.cpu(), binary_preds.cpu(), zero_division=0)
        f1 = f1_score(binary_labels.cpu(), binary_preds.cpu(), zero_division=0)

    
        try:
            auc = roc_auc_score(binary_labels.cpu(), continuous_preds.cpu())
        except:
            auc = 0.5


        tn, fp, fn, tp = confusion_matrix(binary_labels.cpu(), binary_preds.cpu()).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return {
            'mae': mae,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity
        }
