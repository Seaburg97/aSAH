import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import argparse
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SVRStacking:
    """Simplified version of SVR stacking model for prediction only"""
    def __init__(self):
        self.meta_model = None
        self.scaler = None
        self.feature_names = ['Pre-operative', 'Post-operative']
    
    def predict(self, pre_op_preds_test: np.ndarray, post_op_preds_test: np.ndarray) -> np.ndarray:
        """Make predictions for new samples"""
        X_meta_test = np.column_stack([pre_op_preds_test, post_op_preds_test])
        X_meta_test = self.scaler.transform(X_meta_test)
        predictions = self.meta_model.predict(X_meta_test)
        return np.clip(predictions, 0, 6)
    
    @classmethod
    def load_model(cls, filename: str) -> 'SVRStacking':
        """Load model from file"""
        return joblib.load(filename)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate performance evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Binary classification metrics (threshold = 2.5)
    y_binary = (y_true > 2.5).astype(int)
    y_pred_binary = (y_pred > 2.5).astype(int)
    
    auc = roc_auc_score(y_binary, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_binary, y_pred_binary).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    accuracy = np.mean(np.round(y_pred) == y_true)
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy
    }

def compare_predictions(y_true: np.ndarray, pred_pre: np.ndarray, 
                       pred_post: np.ndarray, pred_stacking: np.ndarray):
    """Compare performance of different prediction methods"""
    print("\n=== Individual Model Evaluation Metrics ===")
    
    # Pre-operative metrics
    metrics_pre = calculate_metrics(y_true, pred_pre)
    print("\nPre-operative Model Metrics:")
    print(f"MSE: {metrics_pre['mse']:.4f}")
    print(f"MAE: {metrics_pre['mae']:.4f}")
    print(f"R-squared: {metrics_pre['r2']:.4f}")
    print(f"AUC: {metrics_pre['auc']:.4f}")
    print(f"Sensitivity: {metrics_pre['sensitivity']:.4f}")
    print(f"Specificity: {metrics_pre['specificity']:.4f}")
    print(f"Accuracy: {metrics_pre['accuracy']:.4f}")
    
    # Post-operative metrics
    metrics_post = calculate_metrics(y_true, pred_post)
    print("\nPost-operative Model Metrics:")
    print(f"MSE: {metrics_post['mse']:.4f}")
    print(f"MAE: {metrics_post['mae']:.4f}")
    print(f"R-squared: {metrics_post['r2']:.4f}")
    print(f"AUC: {metrics_post['auc']:.4f}")
    print(f"Sensitivity: {metrics_post['sensitivity']:.4f}")
    print(f"Specificity: {metrics_post['specificity']:.4f}")
    print(f"Accuracy: {metrics_post['accuracy']:.4f}")
    
    # Stacking metrics
    metrics_stacking = calculate_metrics(y_true, pred_stacking)
    print("\nStacking Model Metrics:")
    print(f"MSE: {metrics_stacking['mse']:.4f}")
    print(f"MAE: {metrics_stacking['mae']:.4f}")
    print(f"R-squared: {metrics_stacking['r2']:.4f}")
    print(f"AUC: {metrics_stacking['auc']:.4f}")
    print(f"Sensitivity: {metrics_stacking['sensitivity']:.4f}")
    print(f"Specificity: {metrics_stacking['specificity']:.4f}")
    print(f"Accuracy: {metrics_stacking['accuracy']:.4f}")
    
    # Calculate MAE t-tests
    errors_pre = np.abs(y_true - pred_pre)
    errors_post = np.abs(y_true - pred_post)
    errors_stacking = np.abs(y_true - pred_stacking)
    
    # Paired t-tests
    t_post_pre, p_post_pre = stats.ttest_rel(errors_post, errors_pre)
    t_post_stack, p_post_stack = stats.ttest_rel(errors_post, errors_stacking)
    t_pre_stack, p_pre_stack = stats.ttest_rel(errors_pre, errors_stacking)
    
    print("\n=== MAE Comparison and Statistical Tests ===")
    print("\nMAE Comparison:")
    print(f"Pre-operative: {metrics_pre['mae']:.4f}")
    print(f"Post-operative: {metrics_post['mae']:.4f}")
    print(f"Stacking: {metrics_stacking['mae']:.4f}")
    
    print("\nPaired t-test Results for MAE:")
    print(f"Post-op vs Pre-op: t={t_post_pre:.4f}, p={p_post_pre:.6f}")
    print(f"Post-op vs Stacking: t={t_post_stack:.4f}, p={p_post_stack:.6f}")
    print(f"Pre-op vs Stacking: t={t_pre_stack:.4f}, p={p_pre_stack:.6f}")

def predict_new_data(model_path: str, data_path: str, save_predictions: bool = True) -> np.ndarray:
    """Make predictions on new data using trained model"""
    try:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        print(f"Loading model from: {model_path}")
        model = SVRStacking.load_model(model_path)
        
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        
        required_columns = ['Prediction_pre', 'Prediction_post']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        pre_op_preds = df['Prediction_pre'].values
        post_op_preds = df['Prediction_post'].values
        
        print("Generating predictions...")
        predictions = model.predict(pre_op_preds, post_op_preds)
        
        if save_predictions:
            df['stacking_pred'] = predictions
            df.to_csv(data_path, index=False)
            print(f"Predictions saved to: {data_path}")
        
        if 'True_Label' in df.columns:
            compare_predictions(
                df['True_Label'].values,
                df['Prediction_pre'].values,
                df['Prediction_post'].values,
                predictions
            )
        
        return predictions
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained SVR stacking model')
    parser.add_argument('--model', type=str, default='model.joblib', 
                       help='Path to model file (.joblib)')
    parser.add_argument('--data', type=str, default='results.csv', 
                       help='Path to data file for prediction (.csv)')
    parser.add_argument('--no-save', action='store_true', help='Do not save predictions to original file')
    
    args = parser.parse_args()
    
    try:
        predictions = predict_new_data(
            model_path=args.model,
            data_path=args.data,
            save_predictions=not args.no_save
        )
        print("\nPrediction completed!")
        
    except Exception as e:
        print(f"\nPrediction failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
