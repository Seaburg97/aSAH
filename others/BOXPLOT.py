import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations
import os
from scipy.stats import norm

def calculate_metrics(y_true, y_pred, threshold=2.5):
    """Calculate evaluation metrics"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return None, None, None, None, None
    
    try:
        # Calculate MAE
        mae = mean_absolute_error(y_true, y_pred)
        
        # Binary conversion for other metrics
        y_true_binary = y_true >= threshold
        y_pred_binary = y_pred >= threshold
        
        # Calculate confusion matrix elements
        tp = np.sum((y_true_binary == True) & (y_pred_binary == True))
        tn = np.sum((y_true_binary == False) & (y_pred_binary == False))
        fp = np.sum((y_true_binary == False) & (y_pred_binary == True))
        fn = np.sum((y_true_binary == True) & (y_pred_binary == False))
        
        # Calculate metrics
        sensitivity = tp / (tp + fn + 1e-10)
        specificity = tn / (tn + fp + 1e-10)
        ppv = tp / (tp + fp + 1e-10)
        
        # Calculate AUC
        if len(np.unique(y_true_binary)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(y_true_binary, y_pred)
        
        return mae, auc, sensitivity, specificity, ppv
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None, None, None, None, None

def bootstrap_metrics(y_true, y_pred, n_iterations=1000, threshold=2.5):
    """Calculate metrics and confidence intervals using bootstrap"""
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
    """Compute variance for DeLong test (matching pROC package)"""
    y_true = np.array(y_true)
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)
    
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    n1, n0 = len(pos_idx), len(neg_idx)
    
    # Calculate placement matrices
    placement_1 = np.zeros((n1, n0))
    placement_2 = np.zeros((n1, n0))
    
    for i, pos in enumerate(pos_idx):
        placement_1[i, :] = (y_pred1[pos] > y_pred1[neg_idx]).astype(float) + 0.5 * (y_pred1[pos] == y_pred1[neg_idx]).astype(float)
        placement_2[i, :] = (y_pred2[pos] > y_pred2[neg_idx]).astype(float) + 0.5 * (y_pred2[pos] == y_pred2[neg_idx]).astype(float)
    
    # Calculate theta and variance components
    theta_1 = placement_1.mean(axis=1)
    theta_2 = placement_2.mean(axis=1)
    
    v10_1 = placement_1.mean(axis=0) - placement_1.mean()
    v10_2 = placement_2.mean(axis=0) - placement_2.mean()
    
    v01_1 = theta_1 - placement_1.mean()
    v01_2 = theta_2 - placement_2.mean()
    
    # Calculate variances and covariance
    var1 = (np.var(v10_1) / n0 + np.var(v01_1) / n1)
    var2 = (np.var(v10_2) / n0 + np.var(v01_2) / n1)
    cov = (np.cov(v10_1, v10_2)[0,1] / n0 + np.cov(v01_1, v01_2)[0,1] / n1)
    
    return var1, var2, cov

def delong_test(y_true, y_pred1, y_pred2):
    """Perform DeLong's test for comparing two AUCs"""
    y_true = np.array(y_true)
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)
    
    # Calculate AUCs
    auc1 = roc_auc_score(y_true, y_pred1)
    auc2 = roc_auc_score(y_true, y_pred2)
    
    # Calculate variance components
    var1, var2, cov = compute_delong_variance(y_true, y_pred1, y_pred2)
    
    # Calculate z-score
    z = (auc1 - auc2) / np.sqrt(var1 + var2 - 2*cov)
    
    # Calculate two-sided p-value
    p_value = 2 * (1 - norm.cdf(abs(z)))
    
    return p_value

def calculate_metric_differences(df, model_names, metric_type):
    """Calculate metric differences between models
    diff_matrix[i,j] represents improvement of column j over row i
    Positive values mean column model is better than row model
    """
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
            
            if metric_type == 'mae':
                # MAE difference (lower is better)
                mae1 = mean_absolute_error(y_true, y_pred1)
                mae2 = mean_absolute_error(y_true, y_pred2)
                diff_matrix[i, j] = -(mae2 - mae1)
            elif metric_type == 'auc':
                # AUC difference (higher is better)
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
        'mae': {
            'p_values': np.zeros((n_models, n_models))
        },
        'auc': {
            'p_values': np.zeros((n_models, n_models))
        }
    }
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i != j:
                y_pred1 = df.loc[valid_mask, model1].values
                y_pred2 = df.loc[valid_mask, model2].values
                
                # Paired t-test for MAE
                mae1 = np.abs(y_true - y_pred1)
                mae2 = np.abs(y_true - y_pred2)
                _, p_val = stats.ttest_rel(mae1, mae2)
                results['mae']['p_values'][i, j] = p_val
                
                # DeLong test for AUC
                try:
                    p_val = delong_test(y_true_binary, y_pred1, y_pred2)
                    results['auc']['p_values'][i, j] = p_val
                except Exception as e:
                    print(f"Error in DeLong test for {model1} vs {model2}: {e}")
                    results['auc']['p_values'][i, j] = np.nan
    
    return results

def create_pvalue_heatmap(p_values, model_names, metric_name, save_path, df=None, bootstrap_diff=None, bootstrap_ci_lower=None, bootstrap_ci_upper=None):
    """Create model comparison heatmap with upper triangular display"""
    if bootstrap_diff is None and df is not None:
        diff_matrix = calculate_metric_differences(df, model_names, metric_name.lower())
    else:
        diff_matrix = bootstrap_diff
    
    plt.figure(figsize=(15, 13))
    plt.rcParams.update({'font.size': 20})
    
    n_models = len(model_names)
    compare_matrix = np.zeros((n_models, n_models))
    
    # Create lower triangle mask (to display upper triangle)
    mask = np.zeros_like(p_values, dtype=bool)
    np.fill_diagonal(mask, True)
    mask[np.tril_indices_from(mask, k=-1)] = True
    
    annot_matrix = np.empty_like(p_values, dtype=object)
    
    # Add model names on diagonal
    for i in range(n_models):
        annot_matrix[i, i] = model_names[i]
    
    # Create annotations for upper triangular
    for i in range(n_models):
        for j in range(n_models):
            if i < j:
                p_val = p_values[i, j]
                diff_val = diff_matrix[i, j]
                
                # Significance marker
                if p_val < 0.001:
                    sig_mark = "***"
                elif p_val < 0.01:
                    sig_mark = "**"
                elif p_val < 0.05:
                    sig_mark = "*"
                else:
                    sig_mark = "ns"
                
                compare_matrix[i, j] = diff_val
                
                # Direction arrow
                direction = "↑" if diff_val > 0 else "↓"
                
                # Format difference value
                if abs(diff_val) < 0.001:
                    diff_str = '<0.001'
                else:
                    diff_str = f'{abs(diff_val):.3f}'
                
                # Build annotation
                if abs(diff_val) < 0.001:
                    annot_matrix[i, j] = f'≈\n{sig_mark}'
                else:
                    annot_matrix[i, j] = f'{diff_str} {direction}\n{sig_mark}'
    
    # Color map
    if metric_name.lower() == 'mae':
        cmap = sns.diverging_palette(10, 240, as_cmap=True)
    else:
        cmap = sns.diverging_palette(10, 240, as_cmap=True)
    
    # Normalization
    max_abs_val = np.max(np.abs(compare_matrix[~np.isnan(compare_matrix) & ~mask]))
    norm = plt.Normalize(-max_abs_val, max_abs_val)
    
    # Draw heatmap
    ax = sns.heatmap(
        compare_matrix,
        annot=annot_matrix,
        fmt='',
        cmap=cmap,
        norm=norm,
        mask=mask,
        square=True,
        linewidths=1,
        linecolor='black',
        cbar_kws={'label': f'Performance Difference ({metric_name.upper()})'},
        annot_kws={'size': 22, 'va': 'center', 'ha': 'center'},
        xticklabels=model_names,
        yticklabels=model_names
    )

    # Frame styling
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.spines["top"].set_linewidth(1)
    ax.spines["top"].set_color("black")
    ax.spines["right"].set_linewidth(1)
    ax.spines["right"].set_color("black")

    # Remove lower triangle grid lines
    for i in range(n_models):
        for j in range(n_models):
            if i > j or i == j:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, facecolor='white' if i > j else 'lightgray', 
                                         linewidth=0 if i > j else 1, edgecolor=None if i > j else 'black'))
    
    plt.xlabel('', fontsize=24)
    plt.ylabel('', fontsize=24)
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(fontsize=20)
    
    # Legend
    if metric_name.lower() == 'auc':
        legend_lines = [
            "Blue: Column model better",
            "Red: Row model better",
            "↑: Column model better",
            "↓: Row model better",
            "≈: Similar performance",
            "*: p<0.05  **: p<0.01",
            "***: p<0.001  ns: not significant"
        ]
    else:
        legend_lines = [
            "Blue: Column model better (lower MAE)",
            "Red: Row model better (lower MAE)",
            "↑: Column model better (lower MAE)",
            "↓: Row model better (lower MAE)",
            "≈: Similar performance",
            "*: p<0.05  **: p<0.01",
            "***: p<0.001  ns: not significant"
        ]
    
    # Add legend
    legend_x = 0.05
    legend_y = 0.16
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', pad=1.0)
    plt.text(legend_x, legend_y, '\n'.join(legend_lines), transform=ax.transAxes, fontsize=16,
             verticalalignment='center', horizontalalignment='left',
             bbox=props)
    
    # Diagonal cells
    for i in range(n_models):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, facecolor='lightgray', 
                                  linewidth=1, edgecolor='black'))
    
    # Title
    metric_type_map = {
        'MAE': 'Mean Absolute Error',
        'AUC': 'Area Under Curve'
    }
    full_metric_name = metric_type_map.get(metric_name.upper(), metric_name)
    plt.title(f'Model Comparison: {full_metric_name}', fontsize=35, pad=30)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_plot(bootstrap_results, model_names, center_name, save_path):
    """Create performance metrics visualization with metrics on x-axis"""
    metric_names = ['MAE', 'AUC', 'Sensitivity', 'Specificity', 'PPV']
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = colors[:len(model_names)]
    
    plt.figure(figsize=(16, 10))
    
    n_metrics = len(metric_names)
    n_models = len(model_names)
    metric_width = 0.8
    model_width = metric_width / n_models
    
    # Add vertical separator lines
    for i in range(n_metrics-1):
        if i == 0:  # Between MAE and AUC
            plt.axvline(x=i + 0.5, color='black', linestyle='-',linewidth=3, alpha=1)
        else:
            plt.axvline(x=i + 0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Plot data by metric
    for metric_idx, metric_name in enumerate(metric_names):
        for model_idx, (model_name, model_metrics) in enumerate(zip(model_names, bootstrap_results)):
            if not model_metrics:
                continue
            
            model_metrics = np.array(model_metrics)
            metric_data = model_metrics[:, metric_idx]
            
            # Box plot position
            positions = [metric_idx + (model_idx - n_models/2 + 0.5) * model_width]
            
            bp = plt.boxplot(metric_data, positions=positions,
                           widths=model_width * 0.8,
                           patch_artist=True,
                           showfliers=False)
            
            # Styling
            plt.setp(bp['boxes'], facecolor=colors[model_idx], alpha=0.6)
            plt.setp(bp['medians'], color='black', linewidth=2)
            plt.setp(bp['whiskers'], linewidth=2)
            plt.setp(bp['caps'], linewidth=2)
            
            # Add scatter points
            plt.plot(positions + np.random.normal(0, model_width*0.1, len(metric_data)),
                    metric_data, 'o', color=colors[model_idx], alpha=0.3, markersize=6)
    
    plt.title(f'Performance Metrics - {center_name}', fontsize=35, pad=20)
    plt.xlabel('', fontsize=45, labelpad=10)
    plt.ylabel('Value', fontsize=35, labelpad=10)
    
    plt.xticks(range(len(metric_names)), metric_names, fontsize=30, rotation=0, ha='center')
    
    # Legend with model names
    legend_elements = [plt.Line2D([0], [0], marker='s', color='w',
                                markerfacecolor=color, markersize=20,
                                label=name)
                      for color, name in zip(colors, model_names)]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=20, frameon=True, edgecolor='black', framealpha=0.8)
    
    plt.grid(True, axis='y',linestyle='--', linewidth=0.7, alpha=0.6)
    plt.ylim(0, 1.75)
    
    # Frame styling
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color('black')
        spine.set_linestyle('-')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_confusion_matrix(df, model_names, center_name, save_path, bootstrap_results=None, threshold=2.5):
    """Create binary confusion matrix visualization for each model"""
    fig, axes = plt.subplots(1, len(model_names), figsize=(4*len(model_names), 4))
    
    if len(model_names) == 1:
        axes = [axes]
    
    cmap = plt.cm.Blues
    valid_mask = ~pd.isna(df['True_Value'])
    
    for i, (model, ax) in enumerate(zip(model_names, axes)):
        model_valid_mask = valid_mask & ~pd.isna(df[model])
        if not any(model_valid_mask):
            continue
            
        y_true = df.loc[model_valid_mask, 'True_Value'].values
        y_pred = df.loc[model_valid_mask, model].values
        
        # Binary classification
        y_true_binary = (y_true >= threshold).astype(int)
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Confusion matrix
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(y_true_binary, y_pred_binary):
            cm[t, p] += 1
        
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        
        # Add counts only
        for i_row in range(cm.shape[0]):
            for j_col in range(cm.shape[1]):
                ax.text(j_col, i_row, f"{cm[i_row, j_col]}",
                       ha="center", va="center", fontsize=14,
                       color="white" if cm[i_row, j_col] > cm.max()/2 else "black")
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["< 2.5", "≥ 2.5"])
        ax.set_yticklabels(["< 2.5", "≥ 2.5"])
        
        ax.set_xlabel("Predicted", fontsize=12)
        if i == 0:
            ax.set_ylabel("True", fontsize=12)
            
        ax.set_title(model, fontsize=14)
        
        for edge, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color('black')
    
    plt.suptitle(f"Confusion Matrices - {center_name}", fontsize=18, y=0.98)
    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_full_confusion_matrix(df, model_names, center_name, save_path, bootstrap_results=None):
    """Create full score (0-6) confusion matrix visualization"""
    fig, axes = plt.subplots(1, len(model_names), figsize=(4*len(model_names), 5))
    
    if len(model_names) == 1:
        axes = [axes]
    
    cmap = plt.cm.Blues
    valid_mask = ~pd.isna(df['True_Value'])
    
    for i, (model, ax) in enumerate(zip(model_names, axes)):
        model_valid_mask = valid_mask & ~pd.isna(df[model])
        if not any(model_valid_mask):
            continue
            
        y_true = df.loc[model_valid_mask, 'True_Value'].values
        y_pred = df.loc[model_valid_mask, model].values
        
        # Round to integer scores
        y_true_rounded = np.round(y_true).astype(int)
        y_pred_rounded = np.round(y_pred).astype(int)
        
        # Clip to 0-6 range
        y_true_rounded = np.clip(y_true_rounded, 0, 6)
        y_pred_rounded = np.clip(y_pred_rounded, 0, 6)
        
        # Create 7x7 confusion matrix
        cm = np.zeros((7, 7), dtype=int)
        for t, p in zip(y_true_rounded, y_pred_rounded):
            cm[t, p] += 1
        
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        
        # Add counts
        for i_row in range(cm.shape[0]):
            for j_col in range(cm.shape[1]):
                if cm[i_row, j_col] > 0:
                    ax.text(j_col, i_row, f"{cm[i_row, j_col]}",
                           ha="center", va="center", fontsize=8,
                           color="white" if cm[i_row, j_col] > cm.max()/2 else "black")
        
        ax.set_xticks(range(7))
        ax.set_yticks(range(7))
        ax.set_xticklabels([str(i) for i in range(7)])
        ax.set_yticklabels([str(i) for i in range(7)])
        
        ax.set_xlabel("Predicted Score", fontsize=12)
        if i == 0:
            ax.set_ylabel("True Score", fontsize=12)
            
        ax.set_title(model, fontsize=14)
        
        for edge, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color('black')
    
    plt.suptitle(f"Score Confusion Matrices (0-6) - {center_name}", fontsize=18, y=0.98)
    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def bootstrap_metric_differences(df, model_names, metric_type, n_iterations=1000):
    """Calculate metric differences with bootstrap confidence intervals"""
    n_models = len(model_names)
    diff_matrix = np.zeros((n_models, n_models))
    diff_ci_lower = np.zeros((n_models, n_models))
    diff_ci_upper = np.zeros((n_models, n_models))
    
    valid_mask = ~pd.isna(df['True_Value'])
    for model in model_names:
        valid_mask = valid_mask & ~pd.isna(df[model])
    
    y_true = df.loc[valid_mask, 'True_Value'].values
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i == j:
                continue
                
            y_pred1 = df.loc[valid_mask, model1].values
            y_pred2 = df.loc[valid_mask, model2].values
            
            diff_values = []
            
            for _ in range(n_iterations):
                # Bootstrap sample
                indices = np.random.randint(0, len(y_true), len(y_true))
                y_true_boot = y_true[indices]
                y_pred1_boot = y_pred1[indices]
                y_pred2_boot = y_pred2[indices]
                
                if metric_type == 'mae':
                    mae1 = mean_absolute_error(y_true_boot, y_pred1_boot)
                    mae2 = mean_absolute_error(y_true_boot, y_pred2_boot)
                    diff_values.append(-(mae2 - mae1))
                    
                elif metric_type == 'auc':
                    y_true_binary = y_true_boot >= 2.5
                    if len(np.unique(y_true_binary)) < 2:
                        continue
                    auc1 = roc_auc_score(y_true_binary, y_pred1_boot)
                    auc2 = roc_auc_score(y_true_binary, y_pred2_boot)
                    diff_values.append(auc2 - auc1)
            
            if not diff_values:
                diff_matrix[i, j] = np.nan
                diff_ci_lower[i, j] = np.nan
                diff_ci_upper[i, j] = np.nan
                continue
                
            diff_matrix[i, j] = np.mean(diff_values)
            diff_ci_lower[i, j] = np.percentile(diff_values, 2.5)
            diff_ci_upper[i, j] = np.percentile(diff_values, 97.5)
    
    return diff_matrix, diff_ci_lower, diff_ci_upper

def main():
    try:
        # Create save directories
        save_dir = 'analysis_results'
        os.makedirs(save_dir, exist_ok=True)
        
        confusion_matrix_dir = os.path.join(save_dir, 'confusion_matrices')
        os.makedirs(confusion_matrix_dir, exist_ok=True)
        
        full_confusion_matrix_dir = os.path.join(save_dir, 'full_confusion_matrices')
        os.makedirs(full_confusion_matrix_dir, exist_ok=True)
        
        # Read data
        df = pd.read_csv('/home/yinpengzhan/DL_project/C.csv')
        print(f"Total rows in dataset: {len(df)}")
        
        # Define model names
        model_names = ['CL method11', 'FS method11', 'Clinical model4', 'Fusion model4']
        
        # Perform paired tests for Test-combined group
        print("\nPerforming paired tests for Test-combined group")
        print("=" * 50)
        
        test_combined_df = df[df['Center'] == 'Test-combined']
        
        # Run paired tests
        results = perform_paired_tests(test_combined_df, model_names)
        
        # Bootstrap analysis for Test-combined
        test_combined_bootstrap_results = []
        
        for model in model_names:
            y_true = test_combined_df['True_Value'].values
            y_pred = test_combined_df[model].values
            
            # Handle missing values
            valid_mask = ~pd.isna(y_true) & ~pd.isna(y_pred)
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
            
            # Bootstrap analysis
            all_bootstrap_metrics = []
            n_iterations = 1000
            for _ in range(n_iterations):
                indices = np.random.randint(0, len(y_true), len(y_true))
                metrics = calculate_metrics(y_true[indices], y_pred[indices])
                if metrics[0] is not None:
                    all_bootstrap_metrics.append(metrics)
            
            test_combined_bootstrap_results.append(all_bootstrap_metrics)
            
            # Calculate and print summary metrics
            metrics_mean, ci_lower, ci_upper = bootstrap_metrics(y_true, y_pred)
            metric_names_display = ['MAE', 'AUC', 'Sensitivity', 'Specificity', 'PPV']
            
            print(f"\n{model}:")
            for i, metric in enumerate(metric_names_display):
                print(f"{metric}: {metrics_mean[i]:.3f} ({ci_lower[i]:.3f} - {ci_upper[i]:.3f})")

        # Bootstrap calculation for model differences
        bootstrap_diff_matrices = {}
        for metric in ['mae', 'auc']:
            diff, _, _ = bootstrap_metric_differences(test_combined_df, model_names, metric)
            bootstrap_diff_matrices[metric] = diff

        # Print p-values
        print("\nP-values for all model comparisons:")
        print("=" * 50)
        for metric in ['mae', 'auc']:
            print(f"\n{metric.upper()} p-values:")
            print("-" * 30)
            p_values = results[metric]['p_values']
            
            # Print header
            header = "Model"
            for model in model_names:
                header += f" | {model}"
            print(header)
            print("-" * len(header))
            
            # Print p-values for each row
            for i, row_model in enumerate(model_names):
                row = f"{row_model}"
                for j, col_model in enumerate(model_names):
                    if i == j:
                        row += f" | -"
                    else:
                        row += f" | {p_values[i, j]:.4f}"
                print(row)
                
        # Create p-value heatmaps for Test-combined
        for metric in ['mae', 'auc']:
            create_pvalue_heatmap(
                results[metric]['p_values'],
                model_names,
                metric.upper(),
                os.path.join(save_dir, f'{metric}_pvalue_heatmap.png'),
                bootstrap_diff=bootstrap_diff_matrices[metric]
            )
        
        # Create performance plot for Test-combined
        create_performance_plot(
            test_combined_bootstrap_results,
            model_names,
            'Test-combined',
            os.path.join(save_dir, 'metrics_Test-combined.png')
        )
        
        # Create confusion matrices for Test-combined
        create_confusion_matrix(
            test_combined_df,
            model_names,
            'Test-combined',
            os.path.join(confusion_matrix_dir, 'confusion_matrix_Test-combined.png'),
            bootstrap_results=test_combined_bootstrap_results
        )
        
        create_full_confusion_matrix(
            test_combined_df,
            model_names,
            'Test-combined',
            os.path.join(full_confusion_matrix_dir, 'full_confusion_matrix_Test-combined.png'),
            bootstrap_results=test_combined_bootstrap_results
        )
        
        # Process all centers
        unique_centers = sorted(df['Center'].unique())
        print(f"\nFound centers: {unique_centers}")
        
        for center in unique_centers:
            if center == 'Test-combined':
                continue
                
            print(f"\nProcessing center {center}")
            print("=" * 50)
            
            center_df = df[df['Center'] == center]
            center_bootstrap_results = []
            
            # Calculate metrics for each model
            for model in model_names:
                y_true = center_df['True_Value'].values
                y_pred = center_df[model].values
                
                # Handle missing values
                valid_mask = ~pd.isna(y_true) & ~pd.isna(y_pred)
                if not np.all(valid_mask):
                    print(f"Warning: Found NaN values in center {center}, model {model}")
                y_true = y_true[valid_mask]
                y_pred = y_pred[valid_mask]
                
                # Bootstrap analysis
                all_bootstrap_metrics = []
                n_iterations = 1000
                for _ in range(n_iterations):
                    indices = np.random.randint(0, len(y_true), len(y_true))
                    metrics = calculate_metrics(y_true[indices], y_pred[indices])
                    if metrics[0] is not None:
                        all_bootstrap_metrics.append(metrics)
                
                center_bootstrap_results.append(all_bootstrap_metrics)
                
                # Calculate and print summary metrics
                metrics_mean, ci_lower, ci_upper = bootstrap_metrics(y_true, y_pred)
                metric_names_display = ['MAE', 'AUC', 'Sensitivity', 'Specificity', 'PPV']
                
                print(f"\n{model}:")
                for i, metric in enumerate(metric_names_display):
                    print(f"{metric}: {metrics_mean[i]:.3f} ({ci_lower[i]:.3f} - {ci_upper[i]:.3f})")
            
            # Create visualizations
            create_performance_plot(
                center_bootstrap_results,
                model_names,
                center,
                os.path.join(save_dir, f'metrics_{center}.png')
            )
            
            create_confusion_matrix(
                center_df,
                model_names,
                center,
                os.path.join(confusion_matrix_dir, f'confusion_matrix_{center}.png'),
                bootstrap_results=center_bootstrap_results
            )
            
            create_full_confusion_matrix(
                center_df,
                model_names,
                center,
                os.path.join(full_confusion_matrix_dir, f'full_confusion_matrix_{center}.png'),
                bootstrap_results=center_bootstrap_results
            )
        
        print("\nAnalysis completed! Results saved in", save_dir)
        print(f"Binary confusion matrices saved in {confusion_matrix_dir}")
        print(f"Full score confusion matrices saved in {full_confusion_matrix_dir}")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()
