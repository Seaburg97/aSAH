import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SVRAnalyzer:
    def __init__(self):
        """Initialize SVR analyzer"""
        self.model = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.clinical_scaler = StandardScaler()
        self.pca = PCA(n_components=1)
        self.feature_names = None
        self.excluded_columns = [0, 1, 2, 3, 7, 8, 12, 13, 14, 15, 21, 22]
        self.prediction_results = []
        self.mrs_threshold = 2.5
        self.collinear_features = ['WFNS_score', 'Hunt-Hess_score', 'GCS_score']
        self.pca_explained_variance_ = None
        self.pca_weights = None  # Store PCA weights
        
    def process_data(self, data, is_training=True):
        """Process dataset including encoding, standardization and PCA transformation"""
        self.patient_ids = data.iloc[:, 1].values
        data_copy = data.copy()
        collinear_data = data_copy[self.collinear_features]
        
        columns_to_drop = data_copy.columns[self.excluded_columns].tolist()
        X = data_copy.drop(columns_to_drop + ['mRS'] + self.collinear_features, axis=1)
        y = data_copy['mRS']
        
        if is_training:
            self.feature_names = list(X.columns)
            
            for column in X.columns:
                if X[column].dtype == 'object':
                    self.encoders[column] = LabelEncoder()
                    X[column] = self.encoders[column].fit_transform(X[column].astype(str))
            
            collinear_data_scaled = self.clinical_scaler.fit_transform(collinear_data)
            Consciousness_score = self.pca.fit_transform(collinear_data_scaled)
            
            self.pca_weights = self.pca.components_[0]
            self.pca_explained_variance_ = self.pca.explained_variance_ratio_
            
            X = self.scaler.fit_transform(X)
            
        else:
            for column in X.columns:
                if column in self.encoders:
                    try:
                        X[column] = self.encoders[column].transform(X[column].astype(str))
                    except ValueError as e:
                        print(f"Warning: New categories found in column {column}")
                        X[column] = X[column].map(lambda x: -1 if x not in self.encoders[column].classes_ else self.encoders[column].transform([x])[0])
            
            collinear_data_scaled = self.clinical_scaler.transform(collinear_data)
            Consciousness_score = self.pca.transform(collinear_data_scaled)
            
            X = self.scaler.transform(X)
        
        X = np.hstack((X, Consciousness_score))
        
        return X, y

    def analyze_pca_components(self, data):
        """Analyze PCA component contributions and loadings"""
        collinear_data = data[self.collinear_features]
        collinear_data_scaled = self.clinical_scaler.fit_transform(collinear_data)
        self.pca.fit(collinear_data_scaled)
        
        pca_analysis = {
            'explained_variance_ratio': self.pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(self.pca.explained_variance_ratio_),
            'component_loadings': pd.DataFrame(
                self.pca.components_,
                columns=self.collinear_features,
                index=[f'PC{i+1}' for i in range(len(self.pca.components_))]
            )
        }
        
        return pca_analysis

    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        acc = accuracy_score(y_true.round(), y_pred.round())
        
        y_true_binary = (y_true > self.mrs_threshold).astype(int)
        y_pred_binary = (y_pred > self.mrs_threshold).astype(int)
        
        auc = roc_auc_score(y_true_binary, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        
        return {
            'mae': mae,
            'acc': acc,
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity
        }

    def store_predictions(self, dataset_name, patient_ids, y_true, y_pred):
        """Store prediction results"""
        for pid, true_val, pred_val in zip(patient_ids, y_true, y_pred):
            self.prediction_results.append({
                'Dataset': dataset_name,
                'Patient_ID': pid,
                'True_Value': true_val,
                'Predicted_Value': pred_val,
                'Binary_True': 1 if true_val > self.mrs_threshold else 0,
                'Binary_Pred': 1 if pred_val > self.mrs_threshold else 0
            })

    def save_predictions_to_csv(self, filename='prediction_results.csv'):
        """Save all predictions to CSV file"""
        df = pd.DataFrame(self.prediction_results)
        df.to_csv(filename, index=False)
        print(f"\nPrediction results saved to: {filename}")

    def train_model(self, train_data, params=None):
        """Train SVR model and perform cross-validation"""
        X_train, y_train = self.process_data(train_data, is_training=True)
        
        if params is None:
            params = {'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1, 'gamma': 'scale'}
        
        self.model = SVR(**params)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_train)
        self.store_predictions('Train', self.patient_ids, y_train, y_pred)
        
        train_metrics = self.calculate_metrics(y_train, y_pred)
        
        return train_metrics

    def calculate_feature_importance(self, X, y, n_repeats=30):
        """Calculate feature importance including PCA weight information"""
        baseline_mae = mean_absolute_error(y, self.model.predict(X))
        importance_scores = []
        importance_stds = []
        
        for feature_idx in range(X.shape[1]):
            scores = []
            for _ in range(n_repeats):
                X_permuted = X.copy()
                X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])
                permuted_score = mean_absolute_error(y, self.model.predict(X_permuted))
                scores.append(permuted_score - baseline_mae)
            
            importance_scores.append(np.mean(scores))
            importance_stds.append(np.std(scores))
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names + ['Consciousness_score'],
            'Importance': importance_scores,
            'Std': importance_stds
        })
        
        pca_weights_str = ' + '.join([f'{w:.2f}×{f}' for w, f in zip(self.pca_weights, self.collinear_features)])
        importance_df.loc[importance_df['Feature'] == 'Consciousness_score', 'PCA_composition'] = f'({pca_weights_str})'
        
        return importance_df.sort_values('Importance', ascending=False)

    def plot_integrated_analysis(self, importance_df, pca_analysis, save_path='integrated_analysis.png', dpi=300):
        """Create feature importance analysis plot with optimized fonts and label positions"""
        plt.style.use('default')
        sns.set_style("whitegrid")
        
        plt.figure(figsize=(18, 12))
        
        importance_df = importance_df.sort_values('Importance', ascending=True)
        y_pos = np.arange(len(importance_df))
        
        ax = plt.gca()
        
        bars = ax.barh(y_pos, importance_df['Importance'],
                    color=np.where(importance_df['Importance'] > 0, 'green', 'darkblue'),
                    alpha=0.6)
        
        ax.errorbar(importance_df['Importance'], y_pos,
                    xerr=importance_df['Std'],
                    fmt='none', color='gray',
                    capsize=5, alpha=0.8,
                    capthick=2, elinewidth=2)
        
        y_labels = importance_df['Feature'].tolist()
        
        x_min, x_max = ax.get_xlim()
        x_range = x_max - x_min
        
        consciousness_value_end = 0
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            value = importance_df['Importance'].iloc[i]
            std = importance_df['Std'].iloc[i]
            
            if value >= 0:
                x_pos = width + std + x_range * 0.02
            else:
                x_pos = width - std - x_range * 0.02
            
            text = ax.text(x_pos,
                        bar.get_y() + bar.get_height()/2,
                        f'{value:.3f}',
                        va='center',
                        ha='left' if value >= 0 else 'right',
                        color='black',
                        fontsize=20)
            
            if y_labels[i] == 'Consciousness_score':
                bbox = text.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
                bbox_data = bbox.transformed(ax.transData.inverted())
                consciousness_value_end = bbox_data.x1
        
        try:
            consciousness_idx = y_labels.index('Consciousness_score')
            
            pca_explained_var = pca_analysis['explained_variance_ratio'][0]
            loadings = pca_analysis['component_loadings'].iloc[0]
            features = pca_analysis['component_loadings'].columns
            
            weights_str = ' + '.join([f'{loadings[i]:.2f}×{feat}'
                                for i, feat in enumerate(features)])
            
            x_pos = consciousness_value_end + x_range * 0.05
            
            ax.text(x_pos, consciousness_idx,
                    f'({weights_str}) [var: {pca_explained_var:.1%}]',
                    va='center', ha='left',
                    color='black',
                    fontsize=18,
                    bbox=dict(facecolor='lightblue', alpha=0.6, pad=5))
            
        except Exception as e:
            print(f"Warning: Could not add PCA information due to: {str(e)}")
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=30)
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        ax.set_title('Feature Importance Analysis', pad=20, size=40)
        ax.set_xlabel('Impact on Model Error', size=0)
        
        ax.tick_params(axis='x', labelsize=20)
        
        plt.margins(x=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"\nIntegrated analysis plot saved to: {save_path}")

    def external_validate(self, external_data, dataset_name):
        """Perform validation on external dataset"""
        X_test, y_test = self.process_data(external_data, is_training=False)
        y_pred = self.model.predict(X_test)
        
        self.store_predictions(dataset_name, self.patient_ids, y_test, y_pred)
        metrics = self.calculate_metrics(y_test, y_pred)
        
        return metrics, y_pred

def main():
    """Main function"""
    try:
        # Load datasets
        data_A = pd.read_csv('yjs.csv')
        data_B = pd.read_csv('ay.csv')
        data_C = pd.read_csv('tl.csv')
        data_D = pd.read_csv('fy.csv')
        data_E = pd.read_csv('testall.csv')
        
        # Create output directory
        import os
        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize analyzer
        analyzer = SVRAnalyzer()
        
        print("\nAnalyzing PCA components...")
        pca_analysis = analyzer.analyze_pca_components(data_A)
        
        print("\nTraining on dataset A...")
        train_metrics = analyzer.train_model(data_A)
        print("\nTraining metrics:")
        for metric, value in train_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        X_train, y_train = analyzer.process_data(data_A, is_training=True)
        
        importance_df = analyzer.calculate_feature_importance(X_train, y_train)
        print("\nFeature importance:")
        print(importance_df)
        
        analyzer.plot_integrated_analysis(
            importance_df,
            pca_analysis,
            save_path=os.path.join(output_dir, 'integrated_analysis.png'),
            dpi=300
        )
        
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
            print("Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        
        # Save all prediction results to CSV file
        analyzer.save_predictions_to_csv(os.path.join(output_dir, 'svr_predictions.csv'))