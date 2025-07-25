import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
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

class GBDTStacking:
    def __init__(self, n_splits: int = 5, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, min_samples_split: int = 2, min_samples_leaf: int = 1,
                 subsample: float = 1.0, pos_weight: float = 2.0,
                 output_dir: str = 'output'):
        """
        Initialize stacking GBDT model.
        
        Args:
            n_splits: Number of cross-validation folds
            n_estimators: Number of boosting stages
            learning_rate: Learning rate (shrinkage parameter)
            max_depth: Maximum depth of individual regression estimators
            min_samples_split: Minimum samples required to split internal node
            min_samples_leaf: Minimum samples required at leaf node
            subsample: Fraction of samples for fitting base learners
            pos_weight: Weight for positive samples
            output_dir: Output directory path
        """
        self.meta_model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.n_splits = n_splits
        self.pos_weight = pos_weight
        self.feature_names = ['Pre-operative model', 'Post-operative model']
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shap_explainer = None
        
    def _init_shap_explainer(self, X: np.ndarray, n_samples: int = 100) -> None:
        """Initialize SHAP explainer."""
        if self.shap_explainer is None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.shap_explainer = shap.TreeExplainer(self.meta_model)
            except Exception as e:
                print(f"SHAP explainer initialization failed: {str(e)}")
                self.shap_explainer = None

    def calculate_shap_values(self, X: np.ndarray, 
                            batch_size: int = 100) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Calculate SHAP values.
        
        Args:
            X: Input feature matrix
            batch_size: Samples per batch
            
        Returns:
            shap_values: SHAP values array
            feature_importance: Feature importance array
            importance_metrics: Importance metrics dictionary
        """
        try:
            self._init_shap_explainer(X)
            if self.shap_explainer is None:
                return np.zeros((len(X), X.shape[1])), np.zeros(X.shape[1]), {}

            shap_values = self.shap_explainer.shap_values(X)
            
            abs_shap_values = np.abs(shap_values)
            feature_importance = abs_shap_values.mean(axis=0)
            
            tree_importance = self.meta_model.feature_importances_
            
            importance_metrics = {
                'mean_abs_shap': feature_importance,
                'max_abs_shap': abs_shap_values.max(axis=0),
                'std_shap': shap_values.std(axis=0),
                'relative_importance': feature_importance / feature_importance.sum(),
                'tree_importance': tree_importance,
                'tree_relative_importance': tree_importance / tree_importance.sum()
            }
            
            return shap_values, feature_importance, importance_metrics
            
        except Exception as e:
            print(f"SHAP value calculation failed: {str(e)}")
            return np.zeros((len(X), X.shape[1])), np.zeros(X.shape[1]), {}

    def plot_shap_analysis(self, X: np.ndarray, shap_values: np.ndarray) -> None:
        """Generate SHAP analysis visualizations."""
        try:
            plots_dir = self.output_dir / 'shap_plots'
            plots_dir.mkdir(exist_ok=True)
            
            X_original = self.scaler.inverse_transform(X)
            
            # Modified summary plot with importance annotations
            plt.figure(figsize=(12, 6))
            feature_order = [1, 0]
            reordered_shap_values = shap_values[:, feature_order]
            reordered_X = X_original[:, feature_order]
            reordered_feature_names = [self.feature_names[i] for i in feature_order]
            
            # Calculate feature importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            relative_importance = feature_importance / feature_importance.sum()
            reordered_importance = relative_importance[feature_order]
            
            ax = plt.gca()
            shap.summary_plot(reordered_shap_values, reordered_X,
                            feature_names=reordered_feature_names,
                            show=False)
            
            # Add importance annotations
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
            
            # Dependence plots
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
            
            # Feature importance comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # SHAP importance
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': feature_importance
            })
            sns.barplot(data=importance_df, x='Feature', y='Importance', ax=ax1)
            ax1.set_title('SHAP Feature Importance')
            
            # Tree-based importance
            tree_importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.meta_model.feature_importances_
            })
            sns.barplot(data=tree_importance_df, x='Feature', y='Importance', ax=ax2)
            ax2.set_title('GBDT Feature Importance')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'feature_importance_comparison.png')
            plt.close()
            
            # Plot training progress (unique to GBDT)
            plt.figure(figsize=(10, 6))
            plt.plot(self.meta_model.train_score_, label='Training Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('GBDT Training Progress')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(plots_dir / 'gbdt_training_progress.png')
            plt.close()
            
        except Exception as e:
            print(f"SHAP visualization generation failed: {str(e)}")

    def get_cv_predictions(self, pre_op_preds: np.ndarray, post_op_preds: np.ndarray, 
                          y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get base model predictions using cross-validation."""
        cv_pred1 = np.zeros_like(pre_op_preds)
        cv_pred2 = np.zeros_like(post_op_preds)
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=1)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(pre_op_preds)):
            weights = np.ones_like(y[train_idx])
            weights[y[train_idx] > 2.5] = self.pos_weight
            
            fold_scaler = StandardScaler()
            
            # Pre-operative model
            pre_train = pre_op_preds[train_idx].reshape(-1, 1)
            pre_val = pre_op_preds[val_idx].reshape(-1, 1)
            pre_train = fold_scaler.fit_transform(pre_train)
            pre_val = fold_scaler.transform(pre_val)
            fold_model1 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, 
                                                   max_depth=3, random_state=42)
            fold_model1.fit(pre_train, y[train_idx], sample_weight=weights)
            cv_pred1[val_idx] = fold_model1.predict(pre_val)
            
            # Post-operative model
            post_train = post_op_preds[train_idx].reshape(-1, 1)
            post_val = post_op_preds[val_idx].reshape(-1, 1)
            post_train = fold_scaler.fit_transform(post_train)
            post_val = fold_scaler.transform(post_val)
            fold_model2 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, 
                                                   max_depth=3, random_state=42)
            fold_model2.fit(post_train, y[train_idx], sample_weight=weights)
            cv_pred2[val_idx] = fold_model2.predict(post_val)
            
        return cv_pred1, cv_pred2
    
    def fit(self, pre_op_preds: np.ndarray, post_op_preds: np.ndarray, 
            y: np.ndarray) -> Tuple[dict, np.ndarray]:
        """Train stacking model and calculate SHAP values."""
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
        """Make predictions on new samples."""
        X_meta_test = np.column_stack([pre_op_preds_test, post_op_preds_test])
        X_meta_test = self.scaler.transform(X_meta_test)
        predictions = self.meta_model.predict(X_meta_test)
        return np.clip(predictions, 0, 6)
    
    def save_model(self, filename: str = 'gbdt_model.joblib') -> None:
        """Save model to file."""
        model_path = self.output_dir / filename
        joblib.dump(self, model_path)
        print(f"Model saved to: {model_path}")
    
    @classmethod
    def load_model(cls, filename: str) -> 'GBDTStacking':
        """Load model from file."""
        return joblib.load(filename)

def load_data(train_path: str, test_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load training data and optional test data."""
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data file not found: {train_path}")
    
    df_train = pd.read_csv(train_path)
    
    df_test = None
    if test_path:
        if os.path.exists(test_path):
            df_test = pd.read_csv(test_path)
        else:
            print(f"Warning: Test file not found: {test_path}")
    
    return df_train, df_test

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate performance evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    y_binary = (y_true > 2.5).astype(int)
    y_pred_binary = (y_pred > 2.5).astype(int)
    
    auc = roc_auc_score(y_binary, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_binary, y_pred_binary).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

def print_metrics(metrics: dict, dataset_name: str = ""):
    """Print performance evaluation metrics."""
    print(f"\n{dataset_name} Performance Evaluation:")
    print(f"MSE (Mean Squared Error): {metrics['mse']:.4f}")
    print(f"MAE (Mean Absolute Error): {metrics['mae']:.4f}")
    print(f"R-squared Score: {metrics['r2']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")

def main(train_path: str, test_path: Optional[str] = None, output_dir: str = 'output'):
    """
    Main function to run stacking model.
    
    Args:
        train_path: Training data path
        test_path: Test data path (optional)
        output_dir: Output directory
    """
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print("Loading data...")
        df_train, df_test = load_data(train_path, test_path)
        
        # Extract features and labels
        pre_op_preds = df_train['Prediction_pre'].values
        post_op_preds = df_train['Prediction_post'].values
        y = df_train['True_Label'].values
        
        # Train model
        print("\nTraining GBDT stacking model...")
        stacking = GBDTStacking(output_dir=output_dir)
        metrics, train_predictions = stacking.fit(pre_op_preds, post_op_preds, y)
        
        # Add predictions to original training CSV
        df_orig_train = pd.read_csv(train_path)
        df_orig_train['gbdt_stacking_pred'] = train_predictions
        df_orig_train.to_csv(train_path, index=False)
        print(f"\nTraining predictions added to original file: '{train_path}'")
        
        # Print feature importance and SHAP analysis results
        print("\nSHAP Feature Importance Analysis:")
        print(f"Pre-operative model importance: {metrics['pre_op_importance']:.4f}")
        print(f"Post-operative model importance: {metrics['post_op_importance']:.4f}")
        
        # Print detailed importance metrics
        importance_metrics = metrics['importance_metrics']
        print("\nDetailed Feature Importance Metrics:")
        print("SHAP Relative Importance:")
        for feat, imp in zip(stacking.feature_names, 
                           importance_metrics['relative_importance']):
            print(f"{feat}: {imp:.4f}")
        
        print("\nTree Model Relative Importance:")
        for feat, imp in zip(stacking.feature_names, 
                           importance_metrics['tree_relative_importance']):
            print(f"{feat}: {imp:.4f}")
        
        print("\nSHAP Statistics:")
        print("Maximum Absolute SHAP Values:")
        for feat, max_shap in zip(stacking.feature_names, 
                                importance_metrics['max_abs_shap']):
            print(f"{feat}: {max_shap:.4f}")
        
        print("\nSHAP Value Standard Deviation:")
        for feat, std_shap in zip(stacking.feature_names, 
                                importance_metrics['std_shap']):
            print(f"{feat}: {std_shap:.4f}")
        
        # Calculate and print training set metrics
        train_metrics = calculate_metrics(y, train_predictions)
        print_metrics(train_metrics, "Training Set")
        
        # Process test data if available
        if df_test is not None:
            print("\nProcessing test set...")
            pre_op_test = df_test['Prediction_pre'].values
            post_op_test = df_test['Prediction_post'].values
            test_predictions = stacking.predict(pre_op_test, post_op_test)
            
            # Add predictions to original test CSV
            df_orig_test = pd.read_csv(test_path)
            df_orig_test['gbdt_stacking_pred'] = test_predictions
            df_orig_test.to_csv(test_path, index=False)
            print(f"\nTest predictions added to original file: '{test_path}'")
            
            # Calculate test metrics if labels available
            if 'True_Label' in df_test.columns:
                test_metrics = calculate_metrics(df_test['True_Label'].values, 
                                              test_predictions)
                print_metrics(test_metrics, "Test Set")
            else:
                print("\nWarning: No true labels in test set, cannot calculate performance metrics")
        
        print(f"\nSHAP analysis visualizations saved to '{output_path}/shap_plots' directory")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    # Usage example
    train_path = '/home/yinpengzhan/DL_project/merge/merged_yjsresults.csv'
    test_path = '/home/yinpengzhan/DL_project/merge/test_predictions.csv'
    output_dir = '/home/yinpengzhan/DL_project/merge/'
    
    main(train_path, test_path, output_dir)