import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os
from typing import Tuple, Optional, Dict, List
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import joblib
from pathlib import Path
import numpy as np

# SVRStacking class definition is required for joblib to load the model
class SVRStacking:
    def __init__(self, n_splits: int = 5, kernel: str = 'rbf', C: float = 1.0,
                 pos_weight: float = 2.0, output_dir: str = 'output'):
        self.meta_model = SVR(kernel=kernel, C=C)
        self.scaler = StandardScaler()
        self.n_splits = n_splits
        self.pos_weight = pos_weight
        self.feature_names = ['Pre-operative model', 'Post-operative model']
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shap_explainer = None
        
    def _init_shap_explainer(self, X: np.ndarray, n_samples: int = 100) -> None:
        if self.shap_explainer is None:
            if len(X) > n_samples:
                background_data = X[np.random.choice(len(X), n_samples, replace=False)]
            else:
                background_data = X
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.shap_explainer = shap.KernelExplainer(
                        self.meta_model.predict, 
                        background_data,
                        link="identity"
                    )
            except Exception as e:
                print(f"SHAP explainer initialization failed: {str(e)}")
                self.shap_explainer = None

    def calculate_shap_values(self, X: np.ndarray, 
                            batch_size: int = 100) -> Tuple[np.ndarray, np.ndarray, Dict]:
        try:
            self._init_shap_explainer(X)
            if self.shap_explainer is None:
                return np.zeros((len(X), X.shape[1])), np.zeros(X.shape[1]), {}

            all_shap_values = []
            for i in tqdm(range(0, len(X), batch_size), desc="Computing SHAP values"):
                batch = X[i:i + batch_size]
                batch_shap_values = self.shap_explainer.shap_values(batch)
                all_shap_values.append(batch_shap_values)

            shap_values = np.vstack(all_shap_values)
            
            abs_shap_values = np.abs(shap_values)
            feature_importance = abs_shap_values.mean(axis=0)
            
            importance_metrics = {
                'mean_abs_shap': feature_importance,
                'max_abs_shap': abs_shap_values.max(axis=0),
                'std_shap': shap_values.std(axis=0),
                'relative_importance': feature_importance / feature_importance.sum()
            }
            
            return shap_values, feature_importance, importance_metrics
            
        except Exception as e:
            print(f"SHAP calculation failed: {str(e)}")
            return np.zeros((len(X), X.shape[1])), np.zeros(X.shape[1]), {}

    def plot_shap_analysis(self, X: np.ndarray, shap_values: np.ndarray) -> None:
        try:
            plots_dir = self.output_dir / 'shap_plots'
            plots_dir.mkdir(exist_ok=True)
            
            X_original = self.scaler.inverse_transform(X)
            
            plt.figure(figsize=(12, 6))
            feature_order = [1, 0]
            reordered_shap_values = shap_values[:, feature_order]
            reordered_X = X_original[:, feature_order]
            reordered_feature_names = [self.feature_names[i] for i in feature_order]
            
            feature_importance = np.abs(shap_values).mean(axis=0)
            relative_importance = feature_importance / feature_importance.sum()
            reordered_importance = relative_importance[feature_order]
            
            ax = plt.gca()
            shap.summary_plot(reordered_shap_values, reordered_X,
                            feature_names=reordered_feature_names,
                            show=False)
            
            for idx, imp in enumerate(reordered_importance):
                y_pos = len(reordered_feature_names) - 1 - idx
                text = f'Importance: {imp:.3f}'
                plt.text(0.4, y_pos + 0.25, text,
                        transform=ax.get_yaxis_transform(),
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', pad=0.5),
                        ha='left', va='bottom',
                        fontsize=9)
            
            plt.subplots_adjust(top=0.9)
            plt.tight_layout()
            plt.savefig(plots_dir / 'shap_summary.png', bbox_inches='tight', dpi=300)
            plt.close()
            
            for i, feature in enumerate(self.feature_names):
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(
                    i, shap_values, X_original,
                    feature_names=self.feature_names,
                    show=False
                )
                plt.xlim(-0.2, 6.2)
                plt.xlabel(f'{feature} Prediction Score (0-6)')
                plt.ylabel(f'SHAP value impact on final prediction')
                plt.title(f'Impact of {feature} Predictions on Model Output')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(plots_dir / f'shap_dependence_{feature}.png', dpi=300)
                plt.close()
            
            plt.figure(figsize=(10, 6))
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': feature_importance
            })
            sns.barplot(data=importance_df, x='Feature', y='Importance')
            plt.title('SHAP Feature Importance')
            plt.tight_layout()
            plt.savefig(plots_dir / 'feature_importance.png')
            plt.close()
            
        except Exception as e:
            print(f"SHAP visualization failed: {str(e)}")

    def get_cv_predictions(self, pre_op_preds: np.ndarray, post_op_preds: np.ndarray, 
                          y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cv_pred1 = np.zeros_like(pre_op_preds)
        cv_pred2 = np.zeros_like(post_op_preds)
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=1)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(pre_op_preds)):
            weights = np.ones_like(y[train_idx])
            weights[y[train_idx] > 2.5] = self.pos_weight
            
            fold_scaler = StandardScaler()
            
            pre_train = pre_op_preds[train_idx].reshape(-1, 1)
            pre_val = pre_op_preds[val_idx].reshape(-1, 1)
            pre_train = fold_scaler.fit_transform(pre_train)
            pre_val = fold_scaler.transform(pre_val)
            fold_model1 = SVR(kernel='rbf', C=1.0)
            fold_model1.fit(pre_train, y[train_idx], sample_weight=weights)
            cv_pred1[val_idx] = fold_model1.predict(pre_val)
            
            post_train = post_op_preds[train_idx].reshape(-1, 1)
            post_val = post_op_preds[val_idx].reshape(-1, 1)
            post_train = fold_scaler.fit_transform(post_train)
            post_val = fold_scaler.transform(post_val)
            fold_model2 = SVR(kernel='rbf', C=1.0)
            fold_model2.fit(post_train, y[train_idx], sample_weight=weights)
            cv_pred2[val_idx] = fold_model2.predict(post_val)
            
        return cv_pred1, cv_pred2
    
    def fit(self, pre_op_preds: np.ndarray, post_op_preds: np.ndarray, 
            y: np.ndarray) -> Tuple[dict, np.ndarray]:
        cv_pred1, cv_pred2 = self.get_cv_predictions(pre_op_preds, post_op_preds, y)
        
        X_meta = np.column_stack([pre_op_preds, post_op_preds])
        X_meta = self.scaler.fit_transform(X_meta)
        
        weights = np.ones_like(y)
        weights[y > 2.5] = self.pos_weight
        
        self.meta_model.fit(X_meta, y, sample_weight=weights)
        
        shap_values, feature_importance, importance_metrics = self.calculate_shap_values(X_meta)
        
        self.plot_shap_analysis(X_meta, shap_values)
        
        final_pred = self.meta_model.predict(X_meta)
        final_pred = np.clip(final_pred, 0, 6)
        
        mse = mean_squared_error(y, final_pred)
        mae = mean_absolute_error(y, final_pred)
        r2 = r2_score(y, final_pred)
        
        metrics = {
            'pre_op_importance': feature_importance[0],
            'post_op_importance': feature_importance[1],
            'importance_metrics': importance_metrics,
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
        
        self.save_model()
        
        return metrics, final_pred
    
    def predict(self, pre_op_preds_test: np.ndarray, post_op_preds_test: np.ndarray) -> np.ndarray:
        X_meta_test = np.column_stack([pre_op_preds_test, post_op_preds_test])
        X_meta_test = self.scaler.transform(X_meta_test)
        predictions = self.meta_model.predict(X_meta_test)
        return np.clip(predictions, 0, 6)
    
    def save_model(self, filename: str = 'model.joblib') -> None:
        model_path = self.output_dir / filename
        joblib.dump(self, model_path)
        print(f"Model saved to: {model_path}")
    
    @classmethod
    def load_model(cls, filename: str) -> 'SVRStacking':
        return joblib.load(filename)


# Simple prediction functions
def load_model(model_path: str):
    """Load saved model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    print(f"Model loaded successfully: {model_path}")
    return model


def predict_single(model, pre_op_pred: float, post_op_pred: float) -> float:
    """
    Single sample prediction
    
    Args:
        model: Loaded model object
        pre_op_pred: Pre-operative model prediction
        post_op_pred: Post-operative model prediction
        
    Returns:
        Prediction result (0-6 range)
    """
    pre_op_array = np.array([pre_op_pred])
    post_op_array = np.array([post_op_pred])
    result = model.predict(pre_op_array, post_op_array)
    return float(result[0])


def quick_predict(model_path: str, pre_op_pred: float, post_op_pred: float) -> float:
    """
    Quick prediction - load model and predict in one call
    
    Args:
        model_path: Path to model file
        pre_op_pred: Pre-operative model prediction
        post_op_pred: Post-operative model prediction
        
    Returns:
        Prediction result
    """
    model = load_model(model_path)
    return predict_single(model, pre_op_pred, post_op_pred)


# Usage example
if __name__ == "__main__":
    # Set model path (change to your actual path)
    model_path = "/content/stacking_imaging_model.joblib"
    
    try:
        # Method 1: Quick prediction (for single use)
        result = quick_predict(model_path, 3.46, 1.36)
        print(f"Quick prediction  -> Result: {result:.4f}")
        
        # Method 2: Load once, predict multiple times (recommended for multiple predictions)
        model = load_model(model_path)
        result1 = predict_single(model, 1.8, 2.3)
        result2 = predict_single(model, 3.06, 2.74)
        print(f"Prediction 1 --> Result: {result1:.4f}")
        print(f"Prediction 2 --> Result: {result2:.4f}")
            
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found")
        print("Please ensure the model file path is correct")
    except Exception as e:
        print(f"Error occurred: {e}")


"""
Simple usage template:

# Quick single prediction
result = quick_predict("your_model_path.joblib", 2.5, 3.2)

# Multiple predictions (recommended)
model = load_model("your_model_path.joblib")
result1 = predict_single(model, 2.5, 3.2)
result2 = predict_single(model, 1.8, 2.1)"""