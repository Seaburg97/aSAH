import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations
import os
from scipy.stats import norm

def calculate_metrics(y_true, y_pred, threshold=2.5):
    if len(y_true) == 0 or len(y_pred) == 0:
        return None, None, None, None, None
    
    try:
        mae = mean_absolute_error(y_true, y_pred)

        y_true_rounded = np.round(y_true)
        y_pred_rounded = np.round(y_pred)
        accuracy = np.mean(y_true_rounded == y_pred_rounded)
        
        y_true_binary = y_true >= threshold
        y_pred_binary = y_pred >= threshold
        
        sensitivity = np.sum((y_true_binary == True) & (y_pred_binary == True)) / (np.sum(y_true_binary == True) + 1e-10)
        
        specificity = np.sum((y_true_binary == False) & (y_pred_binary == False)) / (np.sum(y_true_binary == False) + 1e-10)
        
        if len(np.unique(y_true_binary)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(y_true_binary, y_pred)
        
        return mae, accuracy, auc, sensitivity, specificity
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None, None, None, None, None

def bootstrap_metrics(y_true, y_pred, n_iterations=1000, threshold=2.5):
    """Calculate metrics and their confidence intervals using bootstrap"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.array([np.nan] * 5), np.array([np.nan] * 5), np.array([np.nan] * 5)
    
    n_samples = len(y_true)
    metrics_boot = []
    
    for _ in range(n_iterations):
        indices = np.random.randint(0, n_samples, n_samples)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        metrics = calculate_metrics(y_true_boot, y_pred_boot, threshold)
        if metrics[0] is not None:
            metrics_boot.append(metrics)
    
    if not metrics_boot:
        return np.array([np.nan] * 5), np.array([np.nan] * 5), np.array([np.nan] * 5)
    
    metrics_boot = np.array(metrics_boot)
    
    ci_lower = np.nanpercentile(metrics_boot, 2.5, axis=0)
    ci_upper = np.nanpercentile(metrics_boot, 97.5, axis=0)
    metrics_mean = np.nanmean(metrics_boot, axis=0)
    
    return metrics_mean, ci_lower, ci_upper

def compute_delong_variance(y_true, y_pred1, y_pred2):
    """Compute variance for DeLong test with modifications to match pROC package"""
    y_true = np.array(y_true)
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)
    
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    n1, n0 = len(pos_idx), len(neg_idx)
    
    placement_1 = np.zeros((n1, n0))
    placement_2 = np.zeros((n1, n0))
    
    for i, pos in enumerate(pos_idx):
        placement_1[i, :] = (y_pred1[pos] > y_pred1[neg_idx]).astype(float) + 0.5 * (y_pred1[pos] == y_pred1[neg_idx]).astype(float)
        placement_2[i, :] = (y_pred2[pos] > y_pred2[neg_idx]).astype(float) + 0.5 * (y_pred2[pos] == y_pred2[neg_idx]).astype(float)
    
    theta_1 = placement_1.mean(axis=1)
    theta_2 = placement_2.mean(axis=1)
    
    v10_1 = placement_1.mean(axis=0) - placement_1.mean()
    v10_2 = placement_2.mean(axis=0) - placement_2.mean()
    
    v01_1 = theta_1 - placement_1.mean()
    v01_2 = theta_2 - placement_2.mean()
    
    var1 = (np.var(v10_1) / n0 + np.var(v01_1) / n1)
    var2 = (np.var(v10_2) / n0 + np.var(v01_2) / n1)
    cov = (np.cov(v10_1, v10_2)[0,1] / n0 + np.cov(v01_1, v01_2)[0,1] / n1)
    
    return var1, var2, cov

def delong_test(y_true, y_pred1, y_pred2):
    """Perform DeLong's test for comparing two AUCs"""
    y_true = np.array(y_true)
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)
    
    auc1 = roc_auc_score(y_true, y_pred1)
    auc2 = roc_auc_score(y_true, y_pred2)
    
    var1, var2, cov = compute_delong_variance(y_true, y_pred1, y_pred2)
    
    z = (auc1 - auc2) / np.sqrt(var1 + var2 - 2*cov)
    
    p_value = 2 * (1 - norm.cdf(abs(z)))
    
    return p_value

def calculate_metric_differences(df, model_names, metric_type):
    """Calculate metric differences between models. Positive values indicate improvement"""
    n_models = len(model_names)
    diff_matrix = np.zeros((n_models, n_models))
    
    valid_mask = ~pd.isna(df['True_Value'])
    for model in model_names:
        valid_mask = valid_mask & ~pd.isna(df[model])
    
    y_true = df.loc[valid_mask, 'True_Value'].values
    y_true_binary = y_true >= 2.5
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            y_pred1 = df.loc[valid_mask, model1].values
            y_pred2 = df.loc[valid_mask, model2].values
            
            if metric_type == 'accuracy':
                acc1 = np.mean(np.round(y_true) == np.round(y_pred1))
                acc2 = np.mean(np.round(y_true) == np.round(y_pred2))
                diff_matrix[i, j] = (acc2 - acc1) * 100
            elif metric_type == 'mae':
                mae1 = mean_absolute_error(y_true, y_pred1)
                mae2 = mean_absolute_error(y_true, y_pred2)
                diff_matrix[i, j] = -(mae2 - mae1)
            elif metric_type == 'auc':
                auc1 = roc_auc_score(y_true_binary, y_pred1)
                auc2 = roc_auc_score(y_true_binary, y_pred2)
                diff_matrix[i, j] = auc2 - auc1
    
    return diff_matrix

def perform_paired_tests(df, model_names):
    """Perform paired t-tests and DeLong tests"""
    valid_mask = ~pd.isna(df['True_Value'])
    for model in model_names:
        valid_mask = valid_mask & ~pd.isna(df[model])
    
    y_true = df.loc[valid_mask, 'True_Value'].values
    y_true_binary = y_true >= 2.5
    
    n_models = len(model_names)
    results = {
        'accuracy': {'p_values': np.zeros((n_models, n_models))},
        'mae': {'p_values': np.zeros((n_models, n_models))},
        'auc': {'p_values': np.zeros((n_models, n_models))}
    }
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i != j:
                y_pred1 = df.loc[valid_mask, model1].values
                y_pred2 = df.loc[valid_mask, model2].values
                
                acc1 = (np.round(y_true) == np.round(y_pred1)).astype(int)
                acc2 = (np.round(y_true) == np.round(y_pred2)).astype(int)
                _, p_val = stats.ttest_rel(acc1, acc2)
                results['accuracy']['p_values'][i, j] = p_val
                
                mae1 = np.abs(y_true - y_pred1)
                mae2 = np.abs(y_true - y_pred2)
                _, p_val = stats.ttest_rel(mae1, mae2)
                results['mae']['p_values'][i, j] = p_val
                
                try:
                    p_val = delong_test(y_true_binary, y_pred1, y_pred2)
                    results['auc']['p_values'][i, j] = p_val
                except Exception as e:
                    print(f"Error in DeLong test for {model1} vs {model2}: {e}")
                    results['auc']['p_values'][i, j] = np.nan
    
    return results

def create_pvalue_heatmap(p_values, model_names, metric_name, save_path, df):
    diff_matrix = calculate_metric_differences(df, model_names, metric_name.lower())
    
    display_names = [name.replace(' ', '\n') for name in model_names]
    
    annot_matrix = np.empty_like(p_values, dtype=object)
    annot_matrix.fill('')
    
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            if i < j:
                p_val = p_values[i, j]
                diff_val = diff_matrix[i, j]
                
                if metric_name.lower() == 'accuracy':
                    if abs(diff_val) < 0.01:
                        diff_str = '<0.01'
                    else:
                        sign = '+' if diff_val > 0 else ''
                        diff_str = f'{sign}{diff_val:.2f}'
                else:
                    if abs(diff_val) < 0.001:
                        diff_str = '<0.001'
                    else:
                        sign = '+' if diff_val > 0 else ''
                        diff_str = f'{sign}{diff_val:.3f}'
                
                if p_val < 0.001:
                    p_str = 'P<0.001'
                else:
                    p_str = f'P={p_val:.3f}'
                
                annot_matrix[i, j] = f'{diff_str}\n{p_str}'
                
            elif i > j:
                diff_val = diff_matrix[i, j]
                
                if metric_name.lower() == 'accuracy':
                    if abs(diff_val) < 0.01:
                        diff_str = '<0.01'
                    else:
                        sign = '+' if diff_val > 0 else ''
                        diff_str = f'{sign}{diff_val:.2f}'
                else:
                    if abs(diff_val) < 0.001:
                        diff_str = '<0.001'
                    else:
                        sign = '+' if diff_val > 0 else ''
                        diff_str = f'{sign}{diff_val:.3f}'
                
                annot_matrix[i, j] = diff_str
    
    plt.figure(figsize=(16, 14))
    plt.rcParams.update({'font.size': 30})
    
    mask = np.zeros_like(p_values, dtype=bool)
    mask[np.tril_indices_from(mask)] = True
    
    heatmap_matrix = -np.log10(p_values + 1e-10)
    heatmap_matrix[mask] = np.nan
    
    fig, ax = plt.subplots(figsize=(16, 14))
    
    p_cmap = sns.color_palette("RdYlBu_r", as_cmap=True)
    
    ax.grid(True, color='gray', linestyle='--', alpha=0.3)
    
    sns.heatmap(heatmap_matrix,
                annot=annot_matrix,
                fmt='',
                cmap=p_cmap,
                square=True,
                xticklabels=display_names,
                yticklabels=display_names,
                annot_kws={'size': 20, 'va': 'center', 'ha': 'center'},
                cbar_kws={'label': 'Significance Level'},
                ax=ax,
                mask=np.isnan(heatmap_matrix))
    
    metric_type_map = {
        'MAE': 'Mean Absolute Error',
        'ACCURACY': 'Accuracy',
        'AUC': 'Area Under Curve'
    }
    full_metric_name = metric_type_map.get(metric_name, metric_name)
    plt.title(f'{full_metric_name} Comparison', fontsize=30, pad=20)
    
    plt.xticks(rotation=0, fontsize=20)
    plt.yticks(rotation=0, fontsize=20)
    
    ax.set_xticklabels(ax.get_xticklabels(), ha='center', va='center')
    ax.set_yticklabels(ax.get_yticklabels(), ha='center', va='center')
    
    plt.grid(True, which='major', color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_plot(bootstrap_results, model_names, center_name, save_path):
    """Create visualization of performance metrics"""
    display_names = [name.replace(' ', '\n') for name in model_names]
    
    plt.figure(figsize=(20, 12))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    metric_names = ['MAE', 'Accuracy', 'AUC', 'Sensitivity', 'Specificity']
    
    n_models = len(model_names)
    model_width = 0.8
    metric_width = model_width / 5
    
    for i in range(n_models-1):
        plt.axvline(x=i + 0.5, color='gray', linestyle='--', alpha=0.5)
    
    for model_idx, (model_name, model_metrics) in enumerate(zip(model_names, bootstrap_results)):
        if not model_metrics:
            continue
        
        model_metrics = np.array(model_metrics)
        
        for metric_idx in range(5):
            metric_data = model_metrics[:, metric_idx]
            positions = [model_idx + (metric_idx - 2) * metric_width]
            
            bp = plt.boxplot(metric_data, positions=positions,
                           widths=metric_width * 0.8,
                           patch_artist=True,
                           showfliers=False)
            
            plt.setp(bp['boxes'], facecolor=colors[metric_idx], alpha=0.6)
            plt.setp(bp['medians'], color='black', linewidth=2)
            plt.setp(bp['whiskers'], linewidth=2)
            plt.setp(bp['caps'], linewidth=2)
            
            plt.plot(positions + np.random.normal(0, metric_width*0.1, len(metric_data)),
                    metric_data, 'o', color=colors[metric_idx], alpha=0.3, markersize=6)
    
    plt.title(f'Performance Metrics - {center_name}', fontsize=16, pad=20)
    plt.xlabel('Models', fontsize=14, labelpad=10)
    plt.ylabel('Value', fontsize=14, labelpad=10)
    
    plt.xticks(range(len(display_names)), display_names, rotation=0, ha='center')
    
    legend_elements = [plt.Line2D([0], [0], marker='s', color='w',
                                markerfacecolor=color, markersize=10,
                                label=name)
                      for color, name in zip(colors, metric_names)]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        save_dir = 'analysis_results'
        os.makedirs(save_dir, exist_ok=True)
        
        df = pd.read_csv('allPro.csv')
        print(f"Total rows in dataset: {len(df)}")
        
        model_names = ['Pre-operative model', 'Post-operative model',
                      'Stacking imaging model', 'Clinical model', 'Fusion model']
        
        print("\nPerforming paired tests for Test-combined group")
        print("=" * 50)
        
        test_combined_df = df[df['Center'] == 'Test-combined']
        results = perform_paired_tests(test_combined_df, model_names)
        
        for metric in ['accuracy', 'mae', 'auc']:
            create_pvalue_heatmap(
                results[metric]['p_values'],
                model_names,
                metric.upper(),
                os.path.join(save_dir, f'{metric}_pvalue_heatmap.png'),
                test_combined_df
            )
        
        unique_centers = sorted(df['Center'].unique())
        print(f"\nFound centers: {unique_centers}")
        
        for center in unique_centers:
            print(f"\nProcessing center {center}")
            print("=" * 50)
            
            center_df = df[df['Center'] == center]
            center_bootstrap_results = []
            
            for model in model_names:
                y_true = center_df['True_Value'].values
                y_pred = center_df[model].values
                
                valid_mask = ~pd.isna(y_true) & ~pd.isna(y_pred)
                if not np.all(valid_mask):
                    print(f"Warning: Found NaN values in center {center}, model {model}")
                y_true = y_true[valid_mask]
                y_pred = y_pred[valid_mask]
                
                all_bootstrap_metrics = []
                n_iterations = 1000
                for _ in range(n_iterations):
                    indices = np.random.randint(0, len(y_true), len(y_true))
                    metrics = calculate_metrics(y_true[indices], y_pred[indices])
                    if metrics[0] is not None:
                        all_bootstrap_metrics.append(metrics)
                
                center_bootstrap_results.append(all_bootstrap_metrics)
                
                metrics_mean, ci_lower, ci_upper = bootstrap_metrics(y_true, y_pred)
                metric_names = ['MAE', 'Accuracy', 'AUC', 'Sensitivity', 'Specificity']
                
                print(f"\n{model}:")
                for i, metric in enumerate(metric_names):
                    print(f"{metric}: {metrics_mean[i]:.3f} ({ci_lower[i]:.3f} - {ci_upper[i]:.3f})")
            
            create_performance_plot(
                center_bootstrap_results,
                model_names,
                center,
                os.path.join(save_dir, f'metrics_{center}.png')
            )
        
        print("\nAnalysis completed! Results saved in", save_dir)
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()