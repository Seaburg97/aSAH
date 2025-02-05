import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import norm, ttest_ind
from sklearn.metrics import mean_absolute_error
import seaborn as sns

def compute_delong_variance(y_true_group1, y_pred_group1, y_true_group2, y_pred_group2):
    """
    Calculate variance components for DeLong test
    """
    y_true_group1 = np.array(y_true_group1)
    y_pred_group1 = np.array(y_pred_group1)
    y_true_group2 = np.array(y_true_group2)
    y_pred_group2 = np.array(y_pred_group2)
    
    pos_idx1 = np.where(y_true_group1 == 1)[0]
    neg_idx1 = np.where(y_true_group1 == 0)[0]
    pos_idx2 = np.where(y_true_group2 == 1)[0]
    neg_idx2 = np.where(y_true_group2 == 0)[0]
    
    n11, n10 = len(pos_idx1), len(neg_idx1)
    n21, n20 = len(pos_idx2), len(neg_idx2)
    
    # Calculate placement matrices
    placement_1 = np.zeros((n11, n10))
    for i, pos in enumerate(pos_idx1):
        placement_1[i, :] = (y_pred_group1[pos] > y_pred_group1[neg_idx1]).astype(float) + \
                           0.5 * (y_pred_group1[pos] == y_pred_group1[neg_idx1]).astype(float)
    
    placement_2 = np.zeros((n21, n20))
    for i, pos in enumerate(pos_idx2):
        placement_2[i, :] = (y_pred_group2[pos] > y_pred_group2[neg_idx2]).astype(float) + \
                           0.5 * (y_pred_group2[pos] == y_pred_group2[neg_idx2]).astype(float)
    
    # Compute variance components
    v10_1 = placement_1.mean(axis=0) - placement_1.mean()
    v10_2 = placement_2.mean(axis=0) - placement_2.mean()
    
    v01_1 = placement_1.mean(axis=1) - placement_1.mean()
    v01_2 = placement_2.mean(axis=1) - placement_2.mean()
    
    var1 = np.var(v10_1) / n10 + np.var(v01_1) / n11
    var2 = np.var(v10_2) / n20 + np.var(v01_2) / n21
    
    return var1, var2

def delong_test(y_true_group1, y_pred_group1, y_true_group2, y_pred_group2):
    """
    Perform modified DeLong test to compare AUCs between two independent groups
    """
    auc1 = roc_auc_score(y_true_group1, y_pred_group1)
    auc2 = roc_auc_score(y_true_group2, y_pred_group2)
    
    var1, var2 = compute_delong_variance(y_true_group1, y_pred_group1, y_true_group2, y_pred_group2)
    
    z = (auc1 - auc2) / np.sqrt(var1 + var2)
    p_value = 2 * (1 - norm.cdf(abs(z)))
    
    return z, p_value, auc1, auc2

def bootstrap_mae(y_true, y_pred, n_iterations=1000):
    """Calculate bootstrap confidence intervals for MAE"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    maes = []
    n_samples = len(y_true)
    original_mae = mean_absolute_error(y_true, y_pred)
    
    for _ in range(n_iterations):
        indices = np.random.randint(0, n_samples, n_samples)
        sample_true = y_true[indices]
        sample_pred = y_pred[indices]
        mae = mean_absolute_error(sample_true, sample_pred)
        maes.append(mae)
    
    ci_lower = np.percentile(maes, 2.5)
    ci_upper = np.percentile(maes, 97.5)
    
    return original_mae, ci_lower, ci_upper

def bootstrap_auc(y_true, y_score, n_iterations=1000):
    """Calculate bootstrap confidence intervals for AUC"""
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    aucs = []
    n_samples = len(y_true)
    original_auc = roc_auc_score(y_true, y_score)
    
    for _ in range(n_iterations):
        indices = np.random.randint(0, n_samples, n_samples)
        sample_true = y_true[indices]
        sample_score = y_score[indices]
        auc_value = roc_auc_score(sample_true, sample_score)
        aucs.append(auc_value)
    
    ci_lower = np.percentile(aucs, 2.5)
    ci_upper = np.percentile(aucs, 97.5)
    
    return original_auc, ci_lower, ci_upper

def analyze_subgroup_differences(data, group_column, models):
    """Analyze differences between subgroups for multiple models"""
    results = {}
    
    for model in models:
        model_results = {}
        group_0_data = data[data[group_column] == 0]
        group_1_data = data[data[group_column] == 1]
        
        y_true_0 = (group_0_data['mRS'] >= 2.5).astype(int)
        y_true_1 = (group_1_data['mRS'] >= 2.5).astype(int)
        
        y_pred_0 = group_0_data[model].values
        y_pred_1 = group_1_data[model].values
        
        mae_0, mae_ci_lower_0, mae_ci_upper_0 = bootstrap_mae(
            group_0_data['mRS'].values,
            group_0_data[model].values
        )
        mae_1, mae_ci_lower_1, mae_ci_upper_1 = bootstrap_mae(
            group_1_data['mRS'].values,
            group_1_data[model].values
        )
        
        t_stat, p_value_t = ttest_ind(
            np.abs(group_0_data['mRS'].values - group_0_data[model].values),
            np.abs(group_1_data['mRS'].values - group_1_data[model].values)
        )
        
        auc_0, auc_ci_lower_0, auc_ci_upper_0 = bootstrap_auc(y_true_0, y_pred_0)
        auc_1, auc_ci_lower_1, auc_ci_upper_1 = bootstrap_auc(y_true_1, y_pred_1)
        
        _, p_value_delong, _, _ = delong_test(y_true_0, y_pred_0, y_true_1, y_pred_1)
        
        model_results = {
            'Group_0': {
                'mae': mae_0,
                'mae_ci': (mae_ci_lower_0, mae_ci_upper_0),
                'auc': auc_0,
                'auc_ci': (auc_ci_lower_0, auc_ci_upper_0),
                'n_samples': len(y_true_0)
            },
            'Group_1': {
                'mae': mae_1,
                'mae_ci': (mae_ci_lower_1, mae_ci_upper_1),
                'auc': auc_1,
                'auc_ci': (auc_ci_lower_1, auc_ci_upper_1),
                'n_samples': len(y_true_1)
            },
            'mae_p': p_value_t,
            'auc_p': p_value_delong
        }
        
        results[model] = model_results
    
    return results

def format_p_value(p_value):
    """Format p-value with special handling for p < 0.001"""
    if p_value < 0.001:
        return 'P<0.001'
    else:
        return f'P={p_value:.3f}'

def plot_subgroup_differences(results, group_column, models):
    """Plot comparison of subgroup differences"""
    plt.style.use('default')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    x = np.arange(len(models))
    width = 0.35
    
    color_0 = '#2196F3'
    color_1 = '#F44336'
    
    plt.rcParams.update({'font.size': 30})
    
    def get_label(i):
        if group_column == 'DCI':
            return 'Without DCI (n=473)' if i == 0 else 'DCI (n=199)'
        elif group_column == 'CH':
            return 'Without CH (n=513)' if i == 0 else 'CH (n=159)'
        return f'{group_column}={i}'
    
    # Plot MAE comparison
    mae_max = 0
    for i, (group, color) in enumerate([('Group_0', color_0), ('Group_1', color_1)]):
        maes = [results[model][group]['mae'] for model in models]
        mae_errors = [(results[model][group]['mae_ci'][1] - 
                      results[model][group]['mae_ci'][0])/2 for model in models]
        
        bars = ax1.bar(x + i*width, maes, width, 
                      label=get_label(i),
                      color=color,
                      alpha=0.7,
                      yerr=mae_errors,
                      capsize=5)
        
        for j, v in enumerate(maes):
            mae_max = max(mae_max, v + mae_errors[j] + 0.1)
            ax1.text(x[j] + i*width, v + mae_errors[j] + 0.02,
                    f'{v:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=30)
            
            if i == 1:
                p_value = results[models[j]]['mae_p']
                mid_x = x[j] + width/2
                max_y = max(maes[j] + mae_errors[j],
                          results[models[j]]['Group_0']['mae'] + 
                          (results[models[j]]['Group_0']['mae_ci'][1] - 
                           results[models[j]]['Group_0']['mae_ci'][0])/2)
                ax1.text(mid_x, max_y + 0.2,
                        format_p_value(p_value),
                        ha='center',
                        va='bottom',
                        fontsize=30)
    
    ax1.legend(fontsize=30, loc='center', bbox_to_anchor=(0.5, 1.35))
    ax1.set_ylabel('MAE', fontsize=30)
    ax1.set_title(f'{group_column}', fontsize=0, pad=20)
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels([m.replace('_', ' ') for m in models], fontsize=30)
    ax1.tick_params(axis='both', which='major', labelsize=30)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, mae_max + 0.1)
    
    # Plot AUC comparison
    auc_max = 0
    for i, (group, color) in enumerate([('Group_0', color_0), ('Group_1', color_1)]):
        aucs = [results[model][group]['auc'] for model in models]
        auc_errors = [(results[model][group]['auc_ci'][1] - 
                      results[model][group]['auc_ci'][0])/2 for model in models]
        
        bars = ax2.bar(x + i*width, aucs, width,
                      color=color,
                      alpha=0.7,
                      yerr=auc_errors,
                      capsize=5)
        
        for j, v in enumerate(aucs):
            auc_max = max(auc_max, v + auc_errors[j] + 0.1)
            ax2.text(x[j] + i*width, v + auc_errors[j] + 0.02,
                    f'{v:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=30)
            
            if i == 1:
                p_value = results[models[j]]['auc_p']
                mid_x = x[j] + width/2
                max_y = max(aucs[j] + auc_errors[j],
                          results[models[j]]['Group_0']['auc'] + 
                          (results[models[j]]['Group_0']['auc_ci'][1] - 
                           results[models[j]]['Group_0']['auc_ci'][0])/2)
                ax2.text(mid_x, max_y + 0.2,
                        format_p_value(p_value),
                        ha='center',
                        va='bottom',
                        fontsize=30)
    
    ax2.set_ylabel('AUC', fontsize=30)
    ax2.set_xticks(x + width/2)
    ax2.set_xticklabels([m.replace('_', ' ') for m in models], fontsize=30)
    ax2.tick_params(axis='both', which='major', labelsize=30)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(0, auc_max + 0.2)
    
    plt.tight_layout()
    plt.savefig(f'{group_column}_subgroup_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function for analyzing subgroup differences"""
    # Read data
    data = pd.read_csv('/testall.csv')
    
    # Define models to analyze
    models = [
        'Stacking_imaging_model',
        'Fusion_model'
    ]
    
    # DCI subgroup analysis
    print("\nAnalyzing DCI subgroup differences...")
    dci_results = analyze_subgroup_differences(data, 'DCI', models)
    plot_subgroup_differences(dci_results, 'DCI', models)
    
    # CH subgroup analysis
    print("\nAnalyzing CH subgroup differences...")
    ch_results = analyze_subgroup_differences(data, 'CH', models)
    plot_subgroup_differences(ch_results, 'CH', models)

if __name__ == "__main__":
    main()