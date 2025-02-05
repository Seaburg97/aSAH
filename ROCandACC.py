import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, mean_absolute_error
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from statsmodels.stats.contingency_tables import mcnemar

def read_and_merge_data(file_paths):
    """Merge multiple CSV files into a single DataFrame"""
    dfs = [pd.read_csv(path) for path in file_paths]
    return pd.concat(dfs, axis=0).reset_index(drop=True)

def calculate_binary_metrics(y_true, y_pred, threshold=2.5):
    """Calculate binary classification metrics: AUC, sensitivity, specificity"""
    y_true_binary = (y_true > threshold).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)
    
    auc = roc_auc_score(y_true_binary, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    return auc, sensitivity, specificity

def calculate_auc_ci(y_true, y_pred, threshold=2.5, confidence=0.95):
    """Calculate AUC and its confidence interval"""
    y_true_binary = (y_true > threshold).astype(int)
    auc = roc_auc_score(y_true_binary, y_pred)
    
    n_pos = sum(y_true_binary == 1)
    n_neg = sum(y_true_binary == 0)
    
    q0 = auc * (1 - auc)
    q1 = auc / (2 - auc) - auc**2
    q2 = 2 * auc**2 / (1 + auc) - auc**2
    
    se = np.sqrt((q0 + (n_pos - 1) * q1 + (n_neg - 1) * q2) / (n_pos * n_neg))
    z = norm.ppf(1 - (1 - confidence) / 2)
    
    ci_lower = auc - z * se
    ci_upper = auc + z * se
    
    return auc, (ci_lower, ci_upper)

def calculate_performance_metrics(y_true, y_pred):
    """Calculate MAE and accuracy"""
    mae = mean_absolute_error(y_true, y_pred)
    accuracy = np.mean(np.round(y_pred) == y_true)
    return mae, accuracy

def delong_test(y_true, pred1, pred2, threshold=2.5):
    """Perform DeLong test to compare two AUCs"""
    y_true_binary = (y_true > threshold).astype(int)
    
    n_bootstrap = 1000
    n_samples = len(y_true)
    auc_diffs = []
    
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n_samples, n_samples)
        auc1 = roc_auc_score(y_true_binary[indices], pred1[indices])
        auc2 = roc_auc_score(y_true_binary[indices], pred2[indices])
        auc_diffs.append(auc1 - auc2)
    
    p_value = 2 * min(
        np.mean(np.array(auc_diffs) <= 0),
        np.mean(np.array(auc_diffs) >= 0)
    )
    
    return p_value

def compare_mae_t_test(pred1, pred2, y_true):
    """Compare MAEs using paired t-test"""
    mae1 = np.abs(y_true - pred1)
    mae2 = np.abs(y_true - pred2)
    return stats.ttest_rel(mae1, mae2)

def analyze_predictions(df):
    """Main analysis function for model predictions"""
    predictions = ['Prediction_pre', 'Prediction_post', 'stacking_pred']
    results = {}
    
    for pred in predictions:
        auc, ci = calculate_auc_ci(df['True_Label'], df[pred])
        _, sens, spec = calculate_binary_metrics(df['True_Label'], df[pred])
        mae, acc = calculate_performance_metrics(df['True_Label'], df[pred])
        
        results[pred] = {
            'AUC': auc,
            'AUC_CI': ci,
            'Sensitivity': sens,
            'Specificity': spec,
            'MAE': mae,
            'ACC': acc
        }
    
    comparisons = []
    for i, pred1 in enumerate(predictions):
        for pred2 in predictions[i+1:]:
            delong_p = delong_test(
                df['True_Label'].values,
                df[pred1].values,
                df[pred2].values
            )
            
            t_stat, t_p = compare_mae_t_test(
                df[pred1],
                df[pred2],
                df['True_Label']
            )
            
            comparisons.append({
                'Models': f'{pred1} vs {pred2}',
                'DeLong_p': delong_p,
                'T_test_p': t_p,
                'T_statistic': t_stat
            })
    
    return results, comparisons

def plot_roc_curves(df):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    predictions = ['Prediction_pre', 'Prediction_post', 'stacking_pred']
    colors = ['blue', 'red', 'green']
    
    y_true_binary = (df['True_Label'] > 2.5).astype(int)
    
    for pred, color in zip(predictions, colors):
        fpr, tpr, _ = roc_curve(y_true_binary, df[pred])
        auc = roc_auc_score(y_true_binary, df[pred])
        plt.plot(fpr, tpr, color=color, label=f'{pred} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.grid(True)
    return plt

def generate_summary_table(results, comparisons):
    """Generate summary tables for metrics and comparisons"""
    metrics_data = []
    for model, metrics in results.items():
        metrics_data.append({
            'Model': model,
            'AUC': f"{metrics['AUC']:.3f} ({metrics['AUC_CI'][0]:.3f}-{metrics['AUC_CI'][1]:.3f})",
            'Sensitivity': f"{metrics['Sensitivity']:.3f}",
            'Specificity': f"{metrics['Specificity']:.3f}",
            'MAE': f"{metrics['MAE']:.3f}",
            'ACC': f"{metrics['ACC']:.3f}"
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    comparison_data = []
    for comp in comparisons:
        comparison_data.append({
            'Comparison': comp['Models'],
            'DeLong Test p-value': f"{comp['DeLong_p']:.4f}",
            'T-test p-value': f"{comp['T_test_p']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    return metrics_df, comparison_df

def main():
    # Input data paths
    file_paths = ['results.csv', 
                  '/results.csv', 
                  '/fyresults.csv']
    df = read_and_merge_data(file_paths)
    
    # Perform analysis
    results, comparisons = analyze_predictions(df)
    
    # Print results
    print("\n=== Model Performance Metrics ===")
    for model, metrics in results.items():
        print(f"\n{model}:")
        print(f"AUC: {metrics['AUC']:.3f} (95% CI: {metrics['AUC_CI'][0]:.3f}-{metrics['AUC_CI'][1]:.3f})")
        print(f"Sensitivity: {metrics['Sensitivity']:.3f}")
        print(f"Specificity: {metrics['Specificity']:.3f}")
        print(f"MAE: {metrics['MAE']:.3f}")
        print(f"ACC: {metrics['ACC']:.3f}")
    
    print("\n=== Model Comparisons ===")
    for comp in comparisons:
        print(f"\n{comp['Models']}:")
        print(f"DeLong test p-value: {comp['DeLong_p']:.4f}")
        print(f"MAE t-test p-value: {comp['T_test_p']:.4f}")
    
    # Generate and save plots
    plot_roc_curves(df)
    plt.savefig('roc_curves.png')
    plt.close()
    
    # Generate and save summary tables
    metrics_df, comparison_df = generate_summary_table(results, comparisons)
    metrics_df.to_csv('performance_metrics.csv', index=False)
    comparison_df.to_csv('model_comparisons.csv', index=False)
    
    print("\n=== Performance Metrics Summary ===")
    print(metrics_df.to_string(index=False))
    print("\n=== Model Comparisons Summary ===")
    print(comparison_df.to_string(index=False))

if __name__ == "__main__":
    main()