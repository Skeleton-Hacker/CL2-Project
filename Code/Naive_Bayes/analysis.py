import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import pandas as pd
from model import main as train_model
import os
import json
import pickle

def setup_output_directory(base_dir='Results/'):
    """Create output directory with timestamp"""
    output_dir = f"{base_dir}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_model_and_results(model_dir="Results/"):
    """Load model and results from files"""
    # Load model using pickle
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    
    # Load results using JSON
    with open(os.path.join(model_dir, "results.json"), "r") as f:
        results = json.load(f)
    
    return model, results

def save_plot(plt, output_dir, name):
    """Save plot to output directory"""
    plt.savefig(os.path.join(output_dir, f"{name}.png"))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plot confusion matrix with percentages and absolute numbers"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum() * 100
    
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    
    plt.title('Confusion Matrix\n(percentages)')
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    
    # Add absolute numbers
    for i in range(2):
        for j in range(2):
            plt.text(j + 0.2, i + 0.4, f'n: {cm[i,j]}',
                    color='black', fontweight='bold')
    
    save_plot(plt, output_dir, 'confusion_matrix')

def plot_metrics(y_true, y_pred, output_dir):
    """Plot various performance metrics"""
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics = {
        'Precision (Pos)': report['1']['precision'],
        'Recall (Pos)': report['1']['recall'],
        'F1-Score (Pos)': report['1']['f1-score'],
        'Precision (Neg)': report['0']['precision'],
        'Recall (Neg)': report['0']['recall'],
        'F1-Score (Neg)': report['0']['f1-score'],
        'Accuracy': report['accuracy']
    }
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(metrics.keys(), metrics.values())
    plt.title('Model Performance Metrics')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    save_plot(plt, output_dir, 'metrics')
    
    # Save metrics to CSV
    pd.DataFrame([metrics]).to_csv(
        os.path.join(output_dir, 'metrics.csv'),
        index=False
    )
    
    return metrics

def plot_roc_curve(y_true, y_prob, output_dir):
    """Plot ROC curve with probability validation"""
    # Ensure probabilities are finite and in [0,1]
    y_prob = np.array(y_prob)
    y_prob = np.clip(y_prob, 0, 1)
    
    # Replace any remaining NaN values with 0.5
    y_prob = np.nan_to_num(y_prob, nan=0.5)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    save_plot(plt, output_dir, 'roc_curve')
    return roc_auc

def plot_word_importance(model, output_dir, top_n=20):
    """Analyze and visualize most distinctive words"""
    # Calculate log ratio of probabilities
    word_scores = {
        word: np.log(model['pos_probability'][word] / model['neg_probability'][word])
        for word in model['vocabulary']
    }
    
    # Get top words for each class
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1])
    top_neg = sorted_words[:top_n]
    top_pos = sorted_words[-top_n:][::-1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Plot positive words
    ax1.barh([word[0] for word in top_pos],
             [score[1] for score in top_pos])
    ax1.set_title('Most Positive Words')
    ax1.set_xlabel('Log Probability Ratio')
    
    # Plot negative words
    ax2.barh([word[0] for word in top_neg],
             [score[1] for score in top_neg])
    ax2.set_title('Most Negative Words')
    ax2.set_xlabel('Log Probability Ratio')
    
    plt.tight_layout()
    save_plot(plt, output_dir, 'word_importance')

def generate_report(metrics, output_dir):
    """Generate a text report of the analysis"""
    report = f"""
Sentiment Analysis Report

Performance Metrics:
------------------
"""
    for metric, value in metrics.items():
        report += f"{metric}: {value:.3f}\n"
    
    with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
        f.write(report)

def analyze_results(model, results, output_dir):
    """Generate all analysis and visualizations"""
    # Plot confusion matrix
    plot_confusion_matrix(
        results['true_labels'],
        results['predictions'],
        output_dir
    )
    
    # Plot metrics
    metrics = plot_metrics(
        results['true_labels'],
        results['predictions'],
        output_dir
    )
    
    # Plot ROC curve
    plot_roc_curve(
        results['true_labels'],
        results['probabilities'],
        output_dir
    )
    
    # Plot word importance
    plot_word_importance(model, output_dir)
    
    # Generate report
    generate_report(metrics, output_dir)

def main():
    # Create output directory
    output_dir = setup_output_directory()
    print(f"Analysis results will be saved to: {output_dir}")
    
    # Load model and results
    print("Loading model and results...")
    model, results = load_model_and_results()
    
    # Generate analysis
    print("Generating visualizations and analysis...")
    analyze_results(model, results, output_dir)
    
    print(f"Analysis complete! Results saved in '{output_dir}'")

if __name__ == "__main__":
    main()