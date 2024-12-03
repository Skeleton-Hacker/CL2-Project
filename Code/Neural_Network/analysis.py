import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_metrics_over_folds(metrics, folds):
    """Plot individual metrics across folds."""
    plt.figure(figsize=(10, 6))
    for metric, values in metrics.items():
        if metric != 'Loss':
            plt.plot(folds, values, 'o-', label=metric, linewidth=2, markersize=8)
    plt.title('Performance Metrics Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    plt.xticks(folds)
    plt.tight_layout()
    plt.savefig('Results/Graphs/metrics_over_folds.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_curve(loss_values, folds):
    """Plot loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(folds, loss_values, 'ro-', linewidth=2, markersize=8)
    plt.title('Loss Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.xticks(folds)
    plt.tight_layout()
    plt.savefig('Results/Graphs/loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metric_distributions_box(metrics, metric_names):
    """Plot box plots for metrics distribution."""
    plt.figure(figsize=(10, 6))
    box_data = [metrics[m] for m in metric_names]
    plt.boxplot(box_data, labels=metric_names)
    plt.title('Distribution of Metrics (Box Plot)')
    plt.ylabel('Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Results/Graphs/metric_distributions_box.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metric_distributions_violin(metrics, metric_names):
    """Plot violin plots for metrics distribution."""
    plt.figure(figsize=(10, 6))
    violin_data = [metrics[m] for m in metric_names]
    plt.violinplot(violin_data)
    plt.title('Metric Distributions (Violin Plot)')
    plt.xticks(range(1, len(metric_names) + 1), metric_names)
    plt.ylabel('Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Results/Graphs/metric_distributions_violin.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_final_metrics_bar(final_metrics, metric_names):
    """Plot bar chart of final metrics with error bars."""
    plt.figure(figsize=(10, 6))
    means = [final_metrics[m.lower()]['mean'] for m in metric_names]
    stds = [final_metrics[m.lower()]['std'] for m in metric_names]
    plt.bar(metric_names, means, yerr=stds, capsize=5)
    plt.title('Final Metrics with Standard Deviation')
    plt.ylabel('Score')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('Results/Graphs/final_metrics_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_correlation(metrics, metric_names):
    """Plot correlation heatmap of metrics."""
    plt.figure(figsize=(10, 8))
    metric_values = np.array([metrics[m] for m in metric_names])
    correlation_matrix = np.corrcoef(metric_values)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                xticklabels=metric_names, yticklabels=metric_names)
    plt.title('Metrics Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('Results/Graphs/metrics_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(metrics, total_samples=4000):
    """Plot average confusion matrix."""
    plt.figure(figsize=(10, 8))
    fold_accuracies = metrics['Accuracy']
    fold_precisions = metrics['Precision']
    fold_recalls = metrics['Recall']
    
    avg_tn = avg_fp = avg_fn = avg_tp = 0
    for i in range(len(fold_accuracies)):
        precision = fold_precisions[i]
        recall = fold_recalls[i]
        
        tp = int(recall * (total_samples/2))
        fn = int((total_samples/2) - tp)
        fp = int(tp * (1-precision) / precision)
        tn = total_samples - (tp + fn + fp)
        
        avg_tp += tp
        avg_fn += fn
        avg_fp += fp
        avg_tn += tn
    
    avg_cm = np.array([[avg_tn/5, avg_fp/5], 
                      [avg_fn/5, avg_tp/5]]).astype(int)
    
    sns.heatmap(avg_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Average Confusion Matrix Across Folds')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('Results/Graphs/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_all_plots(results_data):
    """Create all visualization plots."""
    folds = range(1, 6)
    metrics = {
        'Accuracy': [fold['accuracy'] for fold in results_data['fold_results']],
        'Precision': [fold['precision'] for fold in results_data['fold_results']],
        'Recall': [fold['recall'] for fold in results_data['fold_results']],
        'F1': [fold['f1'] for fold in results_data['fold_results']],
        'Loss': [fold['loss'] for fold in results_data['fold_results']]
    }
    
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    
    # Generate individual plots
    plot_metrics_over_folds(metrics, folds)
    plot_loss_curve(metrics['Loss'], folds)
    plot_metric_distributions_box(metrics, metric_names)
    plot_metric_distributions_violin(metrics, metric_names)
    plot_final_metrics_bar(results_data['final_metrics'], metric_names)
    plot_metrics_correlation(metrics, metric_names)
    plot_confusion_matrix(metrics)

def print_summary_statistics(results_data):
    """Print detailed summary statistics."""
    print("\n=== Model Configuration ===")
    for key, value in results_data['config'].items():
        print(f"{key}: {value}")

    print("\n=== Performance Across Folds ===")
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for fold_idx, fold_result in enumerate(results_data['fold_results'], 1):
        print(f"\nFold {fold_idx}:")
        print(f"Loss: {fold_result['loss']:.4f}")
        for metric in metrics:
            print(f"{metric.capitalize()}: {fold_result[metric]:.4f}")

    print("\n=== Final Metrics ===")
    for metric, values in results_data['final_metrics'].items():
        print(f"\n{metric.capitalize()}:")
        print(f"  Mean: {values['mean']:.4f}")
        print(f"  Std:  {values['std']:.4f}")

# Load and process the results
with open('Results/results.json', 'r') as f:
    results_data = json.load(f)

create_all_plots(results_data)
print_summary_statistics(results_data)
