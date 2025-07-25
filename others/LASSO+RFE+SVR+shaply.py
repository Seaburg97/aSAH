import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
import matplotlib
from sklearn.inspection import permutation_importance
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import joblib
import os

class EnsembleRegressorAnalyzer:
    def __init__(self, model_type='RandomForest', enable_shap=False):
        """Initialize regression analyzer
        
        Args:
            model_type: Model type - 'RandomForest', 'GradientBoosting', 'SVR'
            enable_shap: Whether to enable SHAP analysis
        """
        self.model_type = model_type
        self.enable_shap = enable_shap
        self.model = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.selected_features = None
        self.lasso_selected_features = None
        self.rfe_selected_features = None
        self.additional_features = []
        self.all_used_features = []
        self.excluded_columns = [0, 1, 2, 3, 7, 8, 12,13,14, 21, 22, 23, 24, 25]
        self.prediction_results = []
        self.mrs_threshold = 2.5
        self.original_data = None
        self.lasso_model = None
        self.lasso_cv_scores = None
        self.rfe_model = None
        self.rfe_cv_scores = None
        self.used_permutation_importance = False
        
        # File output management
        self.output_base_dir = 'output'
        self.create_output_directories()
        
        # SHAP related (initialize only when enabled)
        if self.enable_shap:
            try:
                import shap
                self.shap = shap
                self.shap_explainer = None
                self.shap_values = None
            except ImportError:
                print("Warning: SHAP library not installed, SHAP analysis disabled")
                self.enable_shap = False
        
    def create_output_directories(self):
        """Create output directory structure"""
        model_name = self.model_type.lower()
        
        self.model_dir = Path(self.output_base_dir) / f"{model_name}_lasso_rfe"
        self.plots_dir = self.model_dir / "plots"
        self.data_dir = self.model_dir / "data"
        self.analysis_dir = self.model_dir / "analysis"
        
        for directory in [self.model_dir, self.plots_dir, self.data_dir, self.analysis_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        if self.enable_shap:
            self.shap_dir = self.analysis_dir / "shap"
            self.shap_dir.mkdir(parents=True, exist_ok=True)
    
    def add_additional_features(self, feature_indices):
        """Add additional feature column indices"""
        self.additional_features = feature_indices
        
    def select_features_by_lasso(self, X_scaled, y):
        """Feature selection using LassoCV, keeping all non-zero coefficient features"""
        print("\nPerforming feature selection using LassoCV...")
        
        alphas = np.logspace(-2, 1, 100)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        lasso_cv = LassoCV(alphas=alphas, cv=cv, random_state=42, max_iter=1000)
        lasso_cv.fit(X_scaled, y)
        
        best_alpha_idx = np.where(lasso_cv.alphas_ == lasso_cv.alpha_)[0][0]
        best_mse = lasso_cv.mse_path_.mean(axis=1)[best_alpha_idx]
        
        self.lasso_model = lasso_cv
        self.lasso_cv_scores = {
            'alphas': lasso_cv.alphas_,
            'mse_path': lasso_cv.mse_path_,
            'best_alpha': lasso_cv.alpha_,
            'best_score': best_mse
        }
        
        # Keep all non-zero coefficient features
        nonzero_mask = np.abs(lasso_cv.coef_) > 1e-10
        selected_features = [self.feature_names[i] for i in range(len(nonzero_mask)) if nonzero_mask[i]]
        
        print(f"\nLassoCV Results:")
        print(f"  - Optimal Alpha: {lasso_cv.alpha_:.6f}")
        print(f"  - Cross-validation MSE: {best_mse:.4f}")
        print(f"  - R² Score: {lasso_cv.score(X_scaled, y):.4f}")
        print(f"  - Non-zero coefficient features: {len(selected_features)}")
        print(f"  - Zero coefficient features: {len(self.feature_names) - len(selected_features)}")
        
        return selected_features
    
    def select_features_by_rfe(self, X_lasso, y, feature_names_lasso):
        """Further feature selection using RFE cross-validation"""
        print("\nPerforming further feature selection using RFE cross-validation...")
        
        if self.model_type == 'RandomForest':
            base_estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        elif self.model_type == 'GradientBoosting':
            base_estimator = GradientBoostingRegressor(n_estimators=50, random_state=42)
        elif self.model_type == 'SVR':
            # For SVR, use permutation importance with RBF kernel
            from sklearn.inspection import permutation_importance
            from sklearn.model_selection import cross_val_score
            
            print("  Note: For SVR, using RBF kernel with permutation importance for feature selection")
            svr_rbf = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
            svr_rbf.fit(X_lasso, y)
            
            print("  Calculating permutation importance...")
            perm_importance = permutation_importance(
                svr_rbf, X_lasso, y, 
                n_repeats=30,
                random_state=42,
                scoring='neg_mean_squared_error'
            )
            
            importance_mean = perm_importance.importances_mean
            importance_std = perm_importance.importances_std
            
            # Select features with importance significantly > 0
            threshold = 0
            significant_features = importance_mean - 1.96 * importance_std > threshold
            
            selected_features = [feature_names_lasso[i] for i in range(len(feature_names_lasso)) if significant_features[i]]
            
            if len(selected_features) == 0:
                best_idx = np.argmax(importance_mean)
                selected_features = [feature_names_lasso[best_idx]]
            
            # Evaluate performance with different feature counts
            print("  Evaluating performance with different feature counts...")
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            
            sorted_indices = np.argsort(importance_mean)[::-1]
            
            cv_scores = []
            n_features_range = range(1, len(feature_names_lasso) + 1)
            
            for n_features in n_features_range:
                top_n_indices = sorted_indices[:n_features]
                X_subset = X_lasso[:, top_n_indices]
                
                scores = cross_val_score(svr_rbf, X_subset, y, cv=cv, 
                                    scoring='neg_mean_squared_error', n_jobs=-1)
                cv_scores.append(-scores.mean())
                
                if n_features % 5 == 0 or n_features == len(feature_names_lasso):
                    print(f"    Testing {n_features} features: MSE = {-scores.mean():.4f}")
            
            cv_scores = np.array(cv_scores)
            optimal_n_features = np.argmin(cv_scores) + 1
            
            optimal_indices = sorted_indices[:optimal_n_features]
            selected_features = [feature_names_lasso[i] for i in optimal_indices]
            
            print(f"\nPermutation importance feature selection results:")
            print(f"  - Features from LASSO: {len(feature_names_lasso)}")
            print(f"  - Optimal feature count: {optimal_n_features}")
            print(f"  - Optimal feature set MSE: {cv_scores[optimal_n_features-1]:.4f}")
            
            print("\nTop 10 features by permutation importance:")
            for i in range(min(10, len(sorted_indices))):
                idx = sorted_indices[i]
                print(f"  {feature_names_lasso[idx]}: {importance_mean[idx]:.4f} ± {importance_std[idx]:.4f}")
            
            self.rfe_model = None
            self.used_permutation_importance = True
            self.rfe_cv_scores = {
                'cv_scores': cv_scores,
                'n_features': np.arange(1, len(feature_names_lasso) + 1),
                'optimal_features': optimal_n_features,
                'support': np.isin(range(len(feature_names_lasso)), optimal_indices),
                'ranking': np.argsort(np.argsort(-importance_mean)) + 1,
                'importance_mean': importance_mean,
                'importance_std': importance_std
            }
            
            return selected_features
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # For tree models, continue with RFECV
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        try:
            rfecv = RFECV(estimator=base_estimator, 
                        cv=cv, 
                        scoring='neg_mean_squared_error',
                        min_features_to_select=1,
                        n_jobs=-1,
                        verbose=0)
            
            rfecv.fit(X_lasso, y)
            
        except Exception as e:
            print(f"  RFE failed: {str(e)}")
            print("  Using all LASSO-selected features")
            self.rfe_model = None
            self.rfe_cv_scores = None
            return feature_names_lasso
        
        self.rfe_model = rfecv
        
        if hasattr(rfecv, 'cv_results_'):
            mean_test_scores = rfecv.cv_results_['mean_test_score']
        elif hasattr(rfecv, 'grid_scores_'):
            mean_test_scores = rfecv.grid_scores_
        else:
            mean_test_scores = np.zeros(len(feature_names_lasso))
            print("Warning: Unable to retrieve RFE cross-validation scores")
        
        self.rfe_cv_scores = {
            'cv_scores': mean_test_scores,
            'n_features': np.arange(1, len(feature_names_lasso) + 1),
            'optimal_features': rfecv.n_features_,
            'support': rfecv.support_,
            'ranking': rfecv.ranking_
        }
        
        selected_features = [feature_names_lasso[i] for i in range(len(feature_names_lasso)) if rfecv.support_[i]]
        
        print(f"\nRFE cross-validation results:")
        print(f"  - Features from LASSO: {len(feature_names_lasso)}")
        print(f"  - RFE optimal features: {rfecv.n_features_}")
        if len(mean_test_scores) > 0 and rfecv.n_features_ > 0:
            print(f"  - Optimal feature set CV score: {mean_test_scores[rfecv.n_features_-1]:.4f}")
        
        return selected_features
        
    def process_data(self, data, is_training=True):
        """Process dataset including encoding and standardization"""
        self.patient_ids = data.iloc[:, 1].values
        data_copy = data.copy()
        
        if is_training:
            self.original_data = data_copy.copy()
        
        columns_to_drop = data_copy.columns[self.excluded_columns].tolist()
        X = data_copy.drop(columns_to_drop + ['mRS'], axis=1)
        y = data_copy['mRS']
        
        if is_training:
            # Encode categorical features
            for column in X.columns:
                if X[column].dtype == 'object':
                    self.encoders[column] = LabelEncoder()
                    X[column] = self.encoders[column].fit_transform(X[column].astype(str))
            
            self.feature_names = list(X.columns)
            X_scaled = self.scaler.fit_transform(X)
            
            # LASSO feature selection
            self.lasso_selected_features = self.select_features_by_lasso(X_scaled, y)
            
            lasso_feature_indices = [i for i, feature in enumerate(self.feature_names) if feature in self.lasso_selected_features]
            X_lasso = X_scaled[:, lasso_feature_indices]
            
            # RFE feature selection
            self.rfe_selected_features = self.select_features_by_rfe(X_lasso, y, self.lasso_selected_features)
            
            if self.rfe_selected_features is None:
                self.rfe_selected_features = self.lasso_selected_features
                
            self.selected_features = self.rfe_selected_features
            self.all_used_features = self.selected_features.copy()
            
            # Handle additional features
            if self.additional_features:
                for idx in self.additional_features:
                    if 0 <= idx < len(data.columns):
                        col_name = data.columns[idx]
                        
                        if col_name not in self.all_used_features:
                            self.all_used_features.append(col_name)
                            
                            feature_data = self.original_data[col_name].copy()
                            
                            if feature_data.dtype == 'object':
                                self.encoders[col_name] = LabelEncoder()
                                self.original_data[col_name] = self.encoders[col_name].fit_transform(feature_data.astype(str))
            
            # Create final feature matrix
            if self.additional_features:
                X_final = np.zeros((X_scaled.shape[0], len(self.all_used_features)))
                
                for i, feature in enumerate(self.all_used_features):
                    if feature in self.feature_names:
                        feature_idx = self.feature_names.index(feature)
                        X_final[:, i] = X_scaled[:, feature_idx]
                    else:
                        feature_data = self.original_data[feature].values
                        feature_data = (feature_data - np.mean(feature_data)) / np.std(feature_data)
                        X_final[:, i] = feature_data
            else:
                selected_indices = [i for i, feature in enumerate(self.feature_names) if feature in self.selected_features]
                X_final = X_scaled[:, selected_indices]
            
            print(f"\nFeature selection results:")
            print(f"LASSO selected features: {len(self.lasso_selected_features)}")
            print(f"RFE selected features: {len(self.rfe_selected_features)}")
            print(f"Additional specified features: {len(self.all_used_features) - len(self.selected_features)}")
            print(f"Total features used: {len(self.all_used_features)}")
            
            # Print LASSO features and coefficients
            if self.lasso_model is not None:
                print(f"\nLASSO selected features and coefficients:")
                for feature in self.lasso_selected_features[:10]:
                    if feature in self.feature_names:
                        feature_idx = self.feature_names.index(feature)
                        coef = self.lasso_model.coef_[feature_idx]
                        print(f"  {feature}: {coef:.6f}")
                if len(self.lasso_selected_features) > 10:
                    print(f"  ... and {len(self.lasso_selected_features) - 10} more features")
            
            # Print final selected features
            if self.rfe_selected_features:
                if self.model_type == 'SVR' and hasattr(self, 'used_permutation_importance') and self.used_permutation_importance:
                    print(f"\nFinal features selected by permutation importance:")
                else:
                    print(f"\nFinal features selected by RFE:")
                
                for feature in self.rfe_selected_features:
                    print(f"  - {feature}")
            else:
                print(f"\nSecond feature selection step not performed, using all LASSO features")
            
            return X_final, y
            
        else:
            # Process test data
            for column in X.columns:
                if column in self.encoders:
                    try:
                        X[column] = self.encoders[column].transform(X[column].astype(str))
                    except ValueError as e:
                        print(f"Warning: New categories found in column {column}")
                        X[column] = X[column].map(lambda x: -1 if x not in self.encoders[column].classes_ else self.encoders[column].transform([x])[0])
            
            X_scaled = self.scaler.transform(X)
            
            X_final = np.zeros((X_scaled.shape[0], len(self.all_used_features)))
            
            for i, feature in enumerate(self.all_used_features):
                if feature in self.feature_names:
                    feature_idx = self.feature_names.index(feature)
                    X_final[:, i] = X_scaled[:, feature_idx]
                else:
                    feature_data = data_copy[feature].values
                    
                    if feature in self.encoders:
                        try:
                            feature_data = self.encoders[feature].transform(feature_data.astype(str))
                        except ValueError:
                            feature_data = np.array([-1] * len(feature_data))
                    
                    feature_mean = np.mean(self.original_data[feature].values)
                    feature_std = np.std(self.original_data[feature].values)
                    feature_data = (feature_data - feature_mean) / feature_std
                    X_final[:, i] = feature_data
            
            return X_final, y

    def plot_rfe_cv_results(self, dpi=300):
        """Plot RFE cross-validation results"""
        if self.rfe_cv_scores is None:
            print("Warning: No RFE results found, skipping RFE plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        n_features = self.rfe_cv_scores['n_features']
        cv_scores = self.rfe_cv_scores['cv_scores']
        optimal_features = self.rfe_cv_scores['optimal_features']
        
        if cv_scores.ndim == 1:
            mean_scores = cv_scores
            std_scores = np.zeros_like(mean_scores)
        else:
            mean_scores = cv_scores.mean(axis=1)
            std_scores = cv_scores.std(axis=1)
        
        # Convert negative scores to positive for better visualization
        if mean_scores.mean() < 0:
            mean_scores = -mean_scores
            ylabel = 'Mean Squared Error'
        else:
            ylabel = 'Cross-validation Score'
        
        plt.plot(n_features, mean_scores, 'b-', linewidth=2, label='Mean Score')
        
        if std_scores.sum() > 0:
            plt.fill_between(n_features, mean_scores - std_scores, mean_scores + std_scores, 
                            alpha=0.2, color='blue', label='±1 std')
        
        if optimal_features <= len(mean_scores):
            optimal_idx = optimal_features - 1
            plt.plot(optimal_features, mean_scores[optimal_idx], 'ro', markersize=10, 
                    label=f'Optimal: {optimal_features} features')
        
        plt.xlabel('Number of Features', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        
        if self.model_type == 'SVR' and 'importance_mean' in self.rfe_cv_scores:
            plt.title('SVR Permutation Importance Feature Selection Results', fontsize=16)
        else:
            plt.title('RFE Cross-Validation Results', fontsize=16)
        
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(n_features[::max(1, len(n_features)//10)])
        plt.tight_layout()
        
        save_path = self.plots_dir / "rfe_cv_results.png"
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"\nFeature selection results plot saved to: {save_path}")
        
        if self.model_type == 'SVR' and 'importance_mean' in self.rfe_cv_scores:
            self.plot_permutation_importance(dpi)

    def plot_permutation_importance(self, dpi=300):
        """Plot SVR permutation importance"""
        if 'importance_mean' not in self.rfe_cv_scores:
            return
        
        plt.figure(figsize=(12, 8))
        
        importance_mean = self.rfe_cv_scores['importance_mean']
        importance_std = self.rfe_cv_scores['importance_std']
        feature_names = self.lasso_selected_features
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_mean,
            'Std': importance_std
        }).sort_values('Importance', ascending=True)
        
        top_n = min(20, len(importance_df))
        plot_df = importance_df.tail(top_n)
        
        y_pos = np.arange(len(plot_df))
        plt.barh(y_pos, plot_df['Importance'], xerr=plot_df['Std'], 
                color='skyblue', alpha=0.7, capsize=5)
        
        plt.yticks(y_pos, plot_df['Feature'])
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        plt.xlabel('Permutation Importance', fontsize=14)
        plt.title('SVR Permutation Feature Importance (RBF Kernel)', fontsize=16)
        plt.tight_layout()
        
        save_path = self.plots_dir / "svr_permutation_importance.png"
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"SVR permutation importance plot saved to: {save_path}")
        
    def init_shap_explainer(self, X: np.ndarray, n_samples: int = 100) -> None:
        """Initialize SHAP explainer"""
        if not self.enable_shap:
            return
            
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                if self.model_type in ['RandomForest', 'GradientBoosting']:
                    self.shap_explainer = self.shap.TreeExplainer(self.model)
                elif self.model_type == 'SVR':
                    background_idx = np.random.choice(X.shape[0], min(n_samples, X.shape[0]), replace=False)
                    background_data = X[background_idx]
                    self.shap_explainer = self.shap.KernelExplainer(self.model.predict, background_data)
                else:
                    raise ValueError(f"Unsupported model type: {self.model_type}")
                    
        except Exception as e:
            print(f"SHAP explainer initialization failed: {str(e)}")
            self.shap_explainer = None

    def calculate_shap_values(self, X: np.ndarray, batch_size: int = 100) -> tuple:
        """Calculate SHAP values"""
        if not self.enable_shap:
            return np.zeros((len(X), X.shape[1])), np.zeros(X.shape[1]), {}
            
        try:
            if self.shap_explainer is None:
                self.init_shap_explainer(X)
            
            if self.shap_explainer is None:
                return np.zeros((len(X), X.shape[1])), np.zeros(X.shape[1]), {}
            
            print(f"\nCalculating SHAP values for {self.model_type} model...")
            
            if self.model_type in ['RandomForest', 'GradientBoosting']:
                shap_values = self.shap_explainer.shap_values(X)
            elif self.model_type == 'SVR':
                # Batch processing for SVR
                n_samples = X.shape[0]
                shap_values_list = []
                
                for i in range(0, n_samples, batch_size):
                    end_idx = min(i + batch_size, n_samples)
                    batch_X = X[i:end_idx]
                    batch_shap = self.shap_explainer.shap_values(batch_X)
                    shap_values_list.append(batch_shap)
                    print(f"  Progress: {end_idx}/{n_samples}")
                
                shap_values = np.vstack(shap_values_list)
            
            self.shap_values = shap_values
            
            abs_shap_values = np.abs(shap_values)
            feature_importance = abs_shap_values.mean(axis=0)
            
            if hasattr(self.model, 'feature_importances_'):
                model_importance = self.model.feature_importances_
            else:
                model_importance = None
            
            importance_metrics = {
                'mean_abs_shap': feature_importance,
                'max_abs_shap': abs_shap_values.max(axis=0),
                'std_shap': shap_values.std(axis=0),
                'relative_importance': feature_importance / feature_importance.sum() if feature_importance.sum() > 0 else feature_importance,
                'model_importance': model_importance
            }
            
            if model_importance is not None:
                importance_metrics['model_relative_importance'] = model_importance / model_importance.sum() if model_importance.sum() > 0 else model_importance
            
            return shap_values, feature_importance, importance_metrics
            
        except Exception as e:
            print(f"SHAP value calculation failed: {str(e)}")
            return np.zeros((len(X), X.shape[1])), np.zeros(X.shape[1]), {}

    def plot_shap_analysis(self, X: np.ndarray, shap_values: np.ndarray) -> None:
        """Generate SHAP analysis visualizations"""
        if not self.enable_shap:
            return
            
        try:
            model_prefix = self.model_type.lower()
            
            # 1. SHAP summary plot
            plt.figure(figsize=(12, 8))
            self.shap.summary_plot(shap_values, X, feature_names=self.all_used_features, show=False)
            plt.title(f'{self.model_type} - SHAP Summary Plot')
            plt.tight_layout()
            plt.savefig(self.shap_dir / f'{model_prefix}_shap_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. SHAP feature importance bar plot
            plt.figure(figsize=(10, 8))
            self.shap.summary_plot(shap_values, X, feature_names=self.all_used_features, 
                            plot_type="bar", show=False)
            plt.title(f'{self.model_type} - SHAP Feature Importance')
            plt.tight_layout()
            plt.savefig(self.shap_dir / f'{model_prefix}_shap_bar.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. SHAP dependence plots for top 5 features
            feature_importance = np.abs(shap_values).mean(axis=0)
            top_features_idx = np.argsort(feature_importance)[-5:][::-1]
            
            for idx in top_features_idx:
                plt.figure(figsize=(8, 6))
                self.shap.dependence_plot(idx, shap_values, X, 
                                   feature_names=self.all_used_features, 
                                   show=False)
                plt.title(f'{self.model_type} - SHAP Dependence: {self.all_used_features[idx]}')
                plt.tight_layout()
                safe_feature_name = self.all_used_features[idx].replace('/', '_').replace(' ', '_')
                plt.savefig(self.shap_dir / f'{model_prefix}_shap_dependence_{safe_feature_name}.png', 
                          dpi=300, bbox_inches='tight')
                plt.close()
            
            # 4. SHAP force plots for first 3 samples
            if hasattr(self.shap_explainer, 'expected_value'):
                expected_value = self.shap_explainer.expected_value
            else:
                expected_value = self.model.predict(X).mean()
            
            for i in range(min(3, X.shape[0])):
                plt.figure(figsize=(20, 3))
                self.shap.force_plot(expected_value, shap_values[i], X[i], 
                              feature_names=self.all_used_features, 
                              matplotlib=True, show=False)
                plt.title(f'{self.model_type} - SHAP Force Plot for Sample {i+1}')
                plt.tight_layout()
                plt.savefig(self.shap_dir / f'{model_prefix}_shap_force_sample_{i+1}.png', 
                          dpi=300, bbox_inches='tight')
                plt.close()
            
            # 5. Feature importance comparison (SHAP vs built-in)
            if hasattr(self.model, 'feature_importances_'):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                shap_importance_df = pd.DataFrame({
                    'Feature': self.all_used_features,
                    'Importance': feature_importance
                }).sort_values('Importance', ascending=True).tail(20)
                
                ax1.barh(shap_importance_df['Feature'], shap_importance_df['Importance'])
                ax1.set_xlabel('SHAP Importance')
                ax1.set_title(f'{self.model_type} - SHAP Feature Importance')
                
                model_importance_df = pd.DataFrame({
                    'Feature': self.all_used_features,
                    'Importance': self.model.feature_importances_
                }).sort_values('Importance', ascending=True).tail(20)
                
                ax2.barh(model_importance_df['Feature'], model_importance_df['Importance'])
                ax2.set_xlabel('Model Feature Importance')
                ax2.set_title(f'{self.model_type} - Built-in Feature Importance')
                
                plt.tight_layout()
                plt.savefig(self.shap_dir / f'{model_prefix}_importance_comparison.png', 
                          dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"\nSHAP analysis visualizations saved to: {self.shap_dir}")
            
        except Exception as e:
            print(f"SHAP visualization generation failed: {str(e)}")

    def save_shap_values(self, X: np.ndarray, shap_values: np.ndarray, 
                        filename: str = 'shap_analysis.csv') -> None:
        """Save SHAP values to CSV file"""
        if not self.enable_shap:
            return
            
        try:
            shap_df = pd.DataFrame(X, columns=[f'{feat}_value' for feat in self.all_used_features])
            
            for i, feat in enumerate(self.all_used_features):
                shap_df[f'{feat}_shap'] = shap_values[:, i]
            
            predictions = self.model.predict(X)
            shap_df['prediction'] = predictions
            
            output_path = self.data_dir / filename
            shap_df.to_csv(output_path, index=False)
            print(f"\nSHAP values saved to: {output_path}")
            
        except Exception as e:
            print(f"Failed to save SHAP values: {str(e)}")

    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        y_pred_clipped = np.clip(y_pred, 0, 6)
        
        mae = mean_absolute_error(y_true, y_pred_clipped)
        acc = accuracy_score(y_true.round(), y_pred_clipped.round())
        
        y_true_binary = (y_true > self.mrs_threshold).astype(int)
        y_pred_binary = (y_pred_clipped > self.mrs_threshold).astype(int)
        
        auc = roc_auc_score(y_true_binary, y_pred_clipped)
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'mae': mae,
            'acc': acc,
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity
        }

    def store_predictions(self, dataset_name, patient_ids, y_true, y_pred):
        """Store prediction results"""
        y_pred_clipped = np.clip(y_pred, 0, 6)
        
        for pid, true_val, pred_val in zip(patient_ids, y_true, y_pred_clipped):
            self.prediction_results.append({
                'Dataset': dataset_name,
                'Patient_ID': pid,
                'True_Value': true_val,
                'Predicted_Value': pred_val,
                'Binary_True': 1 if true_val > self.mrs_threshold else 0,
                'Binary_Pred': 1 if pred_val > self.mrs_threshold else 0
            })

    def save_predictions_to_csv(self, filename='prediction_results.csv'):
        """Save all prediction results to CSV"""
        df = pd.DataFrame(self.prediction_results)
        output_path = self.data_dir / filename
        df.to_csv(output_path, index=False)
        print(f"\nPrediction results saved to: {output_path}")
    
    def plot_lasso_cv_results(self, dpi=300):
        """Plot LassoCV cross-validation results"""
        if self.lasso_cv_scores is None:
            print("Error: No LassoCV results found")
            return
        
        plt.figure(figsize=(12, 6))
        
        alphas = self.lasso_cv_scores['alphas']
        mse_path = self.lasso_cv_scores['mse_path']
        best_alpha = self.lasso_cv_scores['best_alpha']
        
        mean_mse = mse_path.mean(axis=1)
        std_mse = mse_path.std(axis=1)
        
        plt.semilogx(alphas, mean_mse, 'b-', linewidth=2, label='Mean MSE')
        plt.fill_between(alphas, mean_mse - std_mse, mean_mse + std_mse, 
                        alpha=0.2, color='blue', label='±1 std')
        
        best_idx = np.where(alphas == best_alpha)[0][0]
        plt.plot(best_alpha, mean_mse[best_idx], 'ro', markersize=10, 
                label=f'Best alpha = {best_alpha:.6f}')
        
        plt.xlabel('Alpha (Regularization Strength)', fontsize=14)
        plt.ylabel('Mean Squared Error', fontsize=14)
        plt.title('LassoCV: Cross-Validation Results', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.plots_dir / "lasso_cv_results.png"
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"\nLassoCV results plot saved to: {save_path}")
    
    def plot_lasso_coefficients(self, dpi=300):
        """Plot Lasso coefficient bar chart"""
        if self.lasso_model is None or self.feature_names is None:
            print("Error: No Lasso model or feature names found")
            return
        
        coef_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.lasso_model.coef_,
            'Selected': np.isin(self.feature_names, self.lasso_selected_features)
        })
        
        coef_df['AbsCoef'] = np.abs(coef_df['Coefficient'])
        coef_df = coef_df.sort_values('AbsCoef', ascending=True)
        
        plt.figure(figsize=(12, 8))
        
        plot_df = coef_df.tail(min(25, len(coef_df)))
        colors = ['green' if selected else 'gray' for selected in plot_df['Selected']]
        
        bars = plt.barh(plot_df['Feature'], plot_df['Coefficient'], color=colors, alpha=0.7)
        
        # Add coefficient values
        for i, bar in enumerate(bars):
            width = bar.get_width()
            if width != 0:
                label_x = width + 0.01 if width > 0 else width - 0.01
                align = 'left' if width > 0 else 'right'
                plt.text(label_x, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}',
                        va='center', ha=align, fontsize=8)
        
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        threshold = 1e-10
        plt.axvline(x=threshold, color='red', linestyle='--', linewidth=1.0, label=f'Nonzero threshold = {threshold:.1e}')
        plt.axvline(x=-threshold, color='red', linestyle='--', linewidth=1.0)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Selected Features'),
            Patch(facecolor='gray', alpha=0.7, label='Unselected Features')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.title(f'LASSO Coefficient Analysis (α = {self.lasso_model.alpha_:.6f})', fontsize=20)
        plt.xlabel('Coefficient Value', fontsize=16)
        plt.ylabel('Features', fontsize=16)
        plt.tight_layout()
        
        save_path = self.plots_dir / "lasso_coefficients.png"
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"\nLasso coefficient analysis plot saved to: {save_path}")
        
        csv_path = self.data_dir / "lasso_coefficients.csv"
        coef_df['Optimal_Alpha'] = self.lasso_model.alpha_
        coef_df.to_csv(csv_path, index=False)
        print(f"Lasso coefficient data saved to: {csv_path}")
    
    def plot_lasso_regularization_path(self, X, y, n_alphas=100, dpi=300):
        """Plot Lasso regularization path"""
        from sklearn.linear_model import lasso_path

        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        alphas, coefs, _ = lasso_path(X_array, y, n_alphas=n_alphas, eps=1e-5)

        plt.figure(figsize=(14, 8))

        for i, feature in enumerate(self.feature_names):
            plt.plot(-np.log10(alphas), coefs[i], label=feature, linewidth=1.5)
        
        if self.lasso_model is not None:
            plt.axvline(-np.log10(self.lasso_model.alpha_), color='black', linestyle='--', linewidth=2)
            plt.text(-np.log10(self.lasso_model.alpha_) + 0.1, plt.ylim()[1] * 0.9, 
                    f'Optimal α={self.lasso_model.alpha_:.6f}', rotation=90, va='top', fontsize=12)

        plt.title(f'LASSO Regularization Path', fontsize=20)
        plt.xlabel('-log(alpha)', fontsize=16)
        plt.ylabel('Feature Coefficients', fontsize=16)
        plt.legend(loc='best', fontsize=8, ncol=2, frameon=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.plots_dir / "lasso_regularization_path.png"
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        print(f"\nLasso regularization path plot saved to: {save_path}")

    def train_model(self, train_data, params=None):
        """Train regression model"""
        X_train, y_train = self.process_data(train_data, is_training=True)
        
        # Initialize model based on type
        if self.model_type == 'RandomForest':
            if params is None:
                params = {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': 42
                }
            self.model = RandomForestRegressor(**params)
        elif self.model_type == 'GradientBoosting':
            if params is None:
                params = {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'random_state': 42
                }
            self.model = GradientBoostingRegressor(**params)
        elif self.model_type == 'SVR':
            if params is None:
                params = {'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1, 'gamma': 'scale'}
            self.model = SVR(**params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_train)
        y_pred_clipped = np.clip(y_pred, 0, 6)
        
        self.store_predictions('Train', self.patient_ids, y_train, y_pred_clipped)
        
        train_metrics = self.calculate_metrics(y_train, y_pred_clipped)
        
        # Calculate SHAP values if enabled
        if self.enable_shap:
            shap_values, feature_importance, importance_metrics = self.calculate_shap_values(X_train)
            self.plot_shap_analysis(X_train, shap_values)
            self.save_shap_values(X_train, shap_values, 
                                 filename=f'{self.model_type.lower()}_lasso_rfe_shap_values.csv')
            train_metrics['shap_feature_importance'] = feature_importance
            train_metrics['shap_importance_metrics'] = importance_metrics
        
        return train_metrics

    def calculate_feature_importance(self, X, y, n_repeats=30):
        """Calculate feature importance"""
        if self.model_type in ['RandomForest', 'GradientBoosting']:
            importance_scores = self.model.feature_importances_
            importance_stds = np.zeros(len(importance_scores))
        else:
            # Permutation importance for SVR
            baseline_pred = np.clip(self.model.predict(X), 0, 6)
            baseline_mae = mean_absolute_error(y, baseline_pred)
            importance_scores = []
            importance_stds = []
            
            for feature_idx in range(X.shape[1]):
                scores = []
                for _ in range(n_repeats):
                    X_permuted = X.copy()
                    X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])
                    permuted_pred = np.clip(self.model.predict(X_permuted), 0, 6)
                    permuted_score = mean_absolute_error(y, permuted_pred)
                    scores.append(permuted_score - baseline_mae)
                
                importance_scores.append(np.mean(scores))
                importance_stds.append(np.std(scores))
        
        feature_sources = []
        for feature in self.all_used_features:
            if self.rfe_selected_features and feature in self.lasso_selected_features and feature in self.rfe_selected_features:
                feature_sources.append('LASSO+RFE selected')
            elif feature in self.selected_features:
                feature_sources.append('RFE selected' if self.rfe_selected_features else 'LASSO selected')
            else:
                feature_sources.append('Additionally specified')
                
        importance_df = pd.DataFrame({
            'Feature': self.all_used_features,
            'Importance': importance_scores,
            'Std': importance_stds,
            'Source': feature_sources
        })
        
        if self.enable_shap and hasattr(self, 'shap_values') and self.shap_values is not None:
            shap_importance = np.abs(self.shap_values).mean(axis=0)
            importance_df['SHAP_Importance'] = shap_importance
        
        return importance_df.sort_values('Importance', ascending=False)

    def plot_feature_importance(self, importance_df, dpi=300):
        """Create feature importance analysis plot"""
        plt.style.use('default')
        sns.set_style("whitegrid")
        
        plt.figure(figsize=(12, 8))
        
        importance_df = importance_df.sort_values('Importance', ascending=True)
        y_pos = np.arange(len(importance_df))
        
        ax = plt.gca()
        
        colors = []
        for i, source in enumerate(importance_df['Source']):
            if 'LASSO+RFE selected' in source:
                colors.append('green' if importance_df['Importance'].iloc[i] > 0 else 'darkblue')
            elif 'RFE selected' in source:
                colors.append('lightgreen' if importance_df['Importance'].iloc[i] > 0 else 'lightblue')
            else:
                colors.append('orange' if importance_df['Importance'].iloc[i] > 0 else 'purple')
        
        bars = ax.barh(y_pos, importance_df['Importance'], color=colors, alpha=0.7)
        
        if importance_df['Std'].sum() > 0:
            ax.errorbar(importance_df['Importance'], y_pos,
                    xerr=importance_df['Std'],
                    fmt='none', color='black',
                    capsize=8, alpha=0.8,
                    capthick=2, elinewidth=4)
        
        ax.set_yticks(y_pos)
        feature_labels = [feat for feat in importance_df['Feature']]
        ax.set_yticklabels(feature_labels, fontsize=12)
        
        x_min, x_max = ax.get_xlim()
        x_range = x_max - x_min
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            value = importance_df['Importance'].iloc[i]
            std = importance_df['Std'].iloc[i]
            
            if std > 0:  # SVR
                if value >= 0:
                    x_pos = width + std + x_range * 0.02
                else:
                    x_pos = width - std - x_range * 0.02
            else:  # Tree models
                if value >= 0:
                    x_pos = width + x_range * 0.02
                else:
                    x_pos = width - x_range * 0.02
            
            ax.text(x_pos,
                    bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}',
                    va='center',
                    ha='left' if value >= 0 else 'right',
                    color='black',
                    fontsize=10)
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        model_name = {
            'RandomForest': 'Random Forest',
            'GradientBoosting': 'Gradient Boosting',
            'SVR': 'SVR'
        }.get(self.model_type, self.model_type)
        
        ax.set_title(f'{model_name} Feature Importance Analysis\n(LASSO + RFE Feature Selection)', 
                    pad=20, size=18)
        
        ax.tick_params(axis='x', labelsize=12)
        plt.margins(x=0.3)
        plt.tight_layout()
        
        save_path = self.plots_dir / "feature_importance.png"
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"\n{model_name} feature importance analysis plot saved to: {save_path}")

    def external_validate(self, external_data, dataset_name):
        """Validate on external dataset"""
        X_test, y_test = self.process_data(external_data, is_training=False)
        y_pred = self.model.predict(X_test)
        y_pred_clipped = np.clip(y_pred, 0, 6)
        
        self.store_predictions(dataset_name, self.patient_ids, y_test, y_pred_clipped)
        metrics = self.calculate_metrics(y_test, y_pred_clipped)
        
        return metrics, y_pred_clipped

def run_model_analysis(model_type='RandomForest', enable_shap=False):
    """Run analysis for specific model type"""
    try:
        # Load datasets
        data_A = pd.read_csv('/home/yinpengzhan/DL_project/dataexcel/yjs.csv')
        data_B = pd.read_csv('/home/yinpengzhan/DL_project/dataexcel/ay.csv')
        data_C = pd.read_csv('/home/yinpengzhan/DL_project/dataexcel/tl.csv')
        data_D = pd.read_csv('/home/yinpengzhan/DL_project/dataexcel/fy.csv')
        data_E = pd.read_csv('/home/yinpengzhan/DL_project/dataexcel/testallWithOutCF.csv')
        
        analyzer = EnsembleRegressorAnalyzer(model_type=model_type, enable_shap=enable_shap)
        
        # Add additional features (e.g., column 23)
        analyzer.add_additional_features([23])
        
        print(f"\n{'='*50}")
        print(f"Starting {model_type} model analysis")
        print(f"Using LASSO + RFE for feature selection")
        print(f"SHAP analysis: {'Enabled' if enable_shap else 'Disabled'}")
        print(f"Output directory: {analyzer.model_dir}")
        print(f"{'='*50}")
        
        print("\nTraining on dataset A...")
        train_metrics = analyzer.train_model(data_A)
        print("\nTraining set evaluation metrics:")
        for metric, value in train_metrics.items():
            if metric not in ['shap_feature_importance', 'shap_importance_metrics']:
                print(f"{metric}: {value:.4f}")
        
        # Print SHAP feature importance if enabled
        if enable_shap and 'shap_feature_importance' in train_metrics:
            print("\nSHAP feature importance:")
            shap_importance = train_metrics['shap_feature_importance']
            for i, feature in enumerate(analyzer.all_used_features):
                print(f"{feature}: {shap_importance[i]:.4f}")
        
        X_train, y_train = analyzer.process_data(data_A, is_training=False)
        
        print("\nLASSO selected features:")
        for feature in analyzer.lasso_selected_features[:10]:
            print(f"- {feature}")
        if len(analyzer.lasso_selected_features) > 10:
            print(f"... and {len(analyzer.lasso_selected_features) - 10} more features")
        
        if analyzer.rfe_selected_features:
            if analyzer.model_type == 'SVR' and hasattr(analyzer, 'used_permutation_importance') and analyzer.used_permutation_importance:
                print("\nFinal features selected by permutation importance:")
            else:
                print("\nFinal features selected by RFE:")
            
            for feature in analyzer.rfe_selected_features:
                print(f"- {feature}")
        else:
            print("\nSecond feature selection step not performed, using all LASSO features")
            
        print("\nAdditionally specified features:")
        for feature in analyzer.all_used_features:
            if feature not in analyzer.selected_features:
                print(f"- {feature}")
        
        # Calculate feature importance
        importance_df = analyzer.calculate_feature_importance(X_train, y_train)
        print("\nFeature importance:")
        print(importance_df)
        
        importance_csv_path = analyzer.data_dir / "feature_importance.csv"
        importance_df.to_csv(importance_csv_path, index=False)
        print(f"Feature importance data saved to: {importance_csv_path}")
        
        # Create plots
        analyzer.plot_feature_importance(importance_df, dpi=300)
        analyzer.plot_lasso_cv_results(dpi=300)
        analyzer.plot_rfe_cv_results(dpi=300)
        analyzer.plot_lasso_coefficients(dpi=300)
        
        # Create Lasso regularization path plot
        columns_to_drop = data_A.columns[analyzer.excluded_columns].tolist()
        X_for_path = data_A.drop(columns_to_drop + ['mRS'], axis=1)
        y_for_path = data_A['mRS']
        
        for column in X_for_path.columns:
            if X_for_path[column].dtype == 'object':
                if column in analyzer.encoders:
                    X_for_path[column] = analyzer.encoders[column].transform(X_for_path[column].astype(str))
                else:
                    le = LabelEncoder()
                    X_for_path[column] = le.fit_transform(X_for_path[column].astype(str))
        
        X_for_path_scaled = analyzer.scaler.transform(X_for_path)
        analyzer.plot_lasso_regularization_path(X_for_path_scaled, y_for_path, dpi=300)
        
        # External validation
        external_datasets = {
            'Test_B': data_B,
            'Test_C': data_C,
            'Test_D': data_D,
            'Test_E': data_E
        }
        
        print("\nExternal validation results:")
        for name, data in external_datasets.items():
            metrics, predictions = analyzer.external_validate(data, name)
            print(f"\nDataset {name}:")
            print("Evaluation metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        
        analyzer.save_predictions_to_csv('prediction_results.csv')
        
        # Save feature selection results
        feature_selection_df = pd.DataFrame({
            'Feature': analyzer.feature_names,
            'LASSO_Selected': [f in analyzer.lasso_selected_features for f in analyzer.feature_names],
            'RFE_Selected': [f in analyzer.rfe_selected_features if analyzer.rfe_selected_features else f in analyzer.lasso_selected_features for f in analyzer.feature_names],
            'Final_Used': [f in analyzer.all_used_features for f in analyzer.feature_names]
        })
        feature_selection_path = analyzer.data_dir / "feature_selection_results.csv"
        feature_selection_df.to_csv(feature_selection_path, index=False)
        print(f"\nFeature selection results saved to: {feature_selection_path}")
        
        print(f"\nAll analysis results saved to: {analyzer.model_dir}")
        print(f"- Plots saved in: {analyzer.plots_dir}")
        print(f"- Data saved in: {analyzer.data_dir}")
        print(f"- Analysis results saved in: {analyzer.analysis_dir}")
        if enable_shap:
            print(f"- SHAP analysis saved in: {analyzer.shap_dir}")
        
        return analyzer
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise e

def main():
    """Main function - run LASSO+RFE analysis for all models"""
    models = ['RandomForest', 'GradientBoosting', 'SVR']
    
    enable_shap = False  # Set to True to enable SHAP analysis
    
    print(f"SHAP analysis status: {'Enabled' if enable_shap else 'Disabled'}")
    
    for model_type in models:
        print("\n" + "="*80)
        print(f"Running {model_type} regression analysis - LASSO+RFE feature selection")
        print("="*80)
        
        try:
            analyzer = run_model_analysis(
                model_type=model_type, 
                enable_shap=enable_shap
            )
            print(f"\n✓ {model_type} analysis completed")
        except Exception as e:
            print(f"\n✗ {model_type} analysis failed: {str(e)}")

if __name__ == "__main__":
    main()
