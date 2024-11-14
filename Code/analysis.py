import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import os

def load_results(results_path):
    """Load and return results from the specified JSON file"""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results

def plot_metrics_distribution(fold_metrics):
    """Plot distribution of metrics across folds"""
    metrics_df = pd.DataFrame(fold_metrics)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=metrics_df)
    plt.title('Distribution of Metrics Across Folds')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('Result/metrics_distribution.png')
    plt.close()
    
    # Print summary statistics
    print("\nMetrics Summary Statistics:")
    print(metrics_df.describe())
    
    return metrics_df.describe()

def analyze_confusion_matrix(true_labels, predictions):
    """Generate and visualize confusion matrix"""
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('Result/confusion_matrix.png')
    plt.close()
    
    # Print and return classification report
    class_report = classification_report(true_labels, predictions)
    print("\nClassification Report:")
    print(class_report)
    
    return cm, class_report

def analyze_error_cases(texts, predictions, true_labels):
    """Analyze cases where the model made mistakes"""
    error_cases = []
    for text, pred, true in zip(texts, predictions, true_labels):
        if pred != true:
            error_cases.append({
                'text': text,
                'predicted': pred,
                'true': true
            })
    
    # Create error analysis DataFrame
    error_df = pd.DataFrame(error_cases)
    error_df.to_csv('Result/error_analysis.csv', index=False)
    
    print("\nError Analysis:")
    print(f"Total error cases: {len(error_cases)}")
    print("\nSample error cases:")
    print(error_df.head())
    
    return error_df

def analyze_feature_importance():
    """Analyze feature importance using TF-IDF weights"""
    results_dir = Path('Result')
    feature_files = list(results_dir.glob('fold_*_features.csv'))
    
    if not feature_files:
        print("No feature files found for analysis")
        return None
    
    # Combine features from all folds
    all_features = []
    for file in feature_files:
        fold_features = pd.read_csv(file)
        all_features.append(fold_features)
    
    combined_features = pd.concat(all_features)
    
    # Calculate mean feature importance
    mean_importance = combined_features.mean()
    top_features = mean_importance.nlargest(20)
    
    # Plot top features
    plt.figure(figsize=(12, 6))
    top_features.plot(kind='bar')
    plt.title('Top 20 Most Important Features')
    plt.xlabel('Features')
    plt.ylabel('Mean TF-IDF Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('Result/feature_importance.png')
    plt.close()
    
    return top_features

def analyze_fold_performance(fold_metrics):
    """Analyze performance variation across folds"""
    metrics_df = pd.DataFrame(fold_metrics)
    
    # Calculate statistics
    stats = {
        'mean': metrics_df.mean(),
        'std': metrics_df.std(),
        'min': metrics_df.min(),
        'max': metrics_df.max()
    }
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    metrics_df.boxplot()
    plt.title('Performance Metrics Across Folds')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('Result/fold_performance.png')
    plt.close()
    
    return stats

def generate_text_length_analysis(texts, predictions, true_labels):
    """Analyze relationship between text length and prediction accuracy"""
    text_lengths = [len(text.split()) for text in texts]
    correct_predictions = [pred == true for pred, true in zip(predictions, true_labels)]
    
    length_analysis = pd.DataFrame({
        'text_length': text_lengths,
        'correct': correct_predictions
    })
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='correct', y='text_length', data=length_analysis)
    plt.title('Text Length vs Prediction Accuracy')
    plt.xlabel('Correct Prediction')
    plt.ylabel('Text Length (words)')
    plt.tight_layout()
    plt.savefig('Result/text_length_analysis.png')
    plt.close()
    
    return length_analysis.groupby('correct')['text_length'].describe()

def generate_full_report(results_path):
    """Generate a comprehensive analysis report"""
    print("Loading results and generating comprehensive analysis report...")
    
    # Create results directory if it doesn't exist
    os.makedirs('Result', exist_ok=True)
    
    # Load results
    results = load_results(results_path)
    
    # Extract needed data
    predictions = results['predictions']
    true_labels = results['true_labels']
    fold_metrics = results['fold_metrics']
    final_metrics = results['final_metrics']
    detailed_results = results['detailed_results']
    texts = detailed_results['texts']
    
    # Run all analyses
    metrics_summary = plot_metrics_distribution(fold_metrics)
    cm, class_report = analyze_confusion_matrix(true_labels, predictions)
    error_df = analyze_error_cases(texts, predictions, true_labels)
    top_features = analyze_feature_importance()
    fold_stats = analyze_fold_performance(fold_metrics)
    length_analysis = generate_text_length_analysis(texts, predictions, true_labels)
    
    # Generate summary report
    report = {
        'final_metrics': final_metrics,
        'fold_metrics_summary': fold_stats,
        'error_rate': 1 - final_metrics['accuracy'],
        'total_samples': len(predictions),
        'error_cases_count': len(error_df),
        'text_length_analysis': length_analysis.to_dict()
    }
    
    # Save summary report
    with open('Result/analysis_summary.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    print("\nAnalysis complete. Results saved in 'Result' directory.")
    return report

def main():
    # Find the most recent results file
    results_dir = Path('Result')
    if not results_dir.exists():
        print("No results directory found. Please run the model first.")
        return
    
    results_files = list(results_dir.glob('results_*.json'))
    if not results_files:
        print("No results files found. Please run the model first.")
        return
    
    latest_results = max(results_files, key=lambda x: x.stat().st_mtime)
    
    # Generate comprehensive report
    report = generate_full_report(latest_results)
    
    # Print key findings
    print("\nKey Findings:")
    print(f"Overall Accuracy: {report['final_metrics']['accuracy']:.4f}")
    print(f"Total Samples Analyzed: {report['total_samples']}")
    print(f"Error Rate: {report['error_rate']:.4f}")
    print(f"Number of Error Cases: {report['error_cases_count']}")

if __name__ == "__main__":
    main()