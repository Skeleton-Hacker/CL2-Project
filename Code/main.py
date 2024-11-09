import os
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import re
from collections import defaultdict
import random
import contractions
import warnings
import nltk
# from nltk.corpus import stopwords

# Download required NLTK data
# nltk.download('stopwords', quiet=True)
warnings.filterwarnings("ignore")

def create_residual_block(dim):
    """Create an improved residual block with layer normalization"""
    return {
        'layer1_weight': torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(dim, dim)) * 0.075),
        'layer1_bias': torch.nn.Parameter(torch.zeros(dim)),
        'ln1_weight': torch.nn.Parameter(torch.ones(dim)),
        'ln1_bias': torch.nn.Parameter(torch.zeros(dim)),
        'layer2_weight': torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(dim, dim)) * 0.075),
        'layer2_bias': torch.nn.Parameter(torch.zeros(dim)),
        'ln2_weight': torch.nn.Parameter(torch.ones(dim)),
        'ln2_bias': torch.nn.Parameter(torch.zeros(dim))
    }

def forward_residual_block(x, block_params, training=True):
    """Forward pass through improved residual block with layer normalization"""
    identity = x
    
    # First layer with normalization
    out = torch.mm(x, block_params['layer1_weight'].t()) + block_params['layer1_bias']
    
    # Layer normalization
    mean = out.mean(-1, keepdim=True)
    var = out.var(-1, keepdim=True, unbiased=False)
    out = (out - mean) / torch.sqrt(var + 1e-5)
    out = out * block_params['ln1_weight'] + block_params['ln1_bias']
    
    out = torch.relu(out)
    
    if training:
        out = torch.dropout(out, p=0.2, train=True)
    
    # Second layer with normalization
    out = torch.mm(out, block_params['layer2_weight'].t()) + block_params['layer2_bias']
    
    mean = out.mean(-1, keepdim=True)
    var = out.var(-1, keepdim=True, unbiased=False)
    out = (out - mean) / torch.sqrt(var + 1e-5)
    out = out * block_params['ln2_weight'] + block_params['ln2_bias']
    
    return identity + out * 0.075

def create_model_params(input_dim):
    """Create improved model parameters with deeper architecture"""
    return {
        'embedding': {
            'weight': torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(768, input_dim)) * 0.075),
            'bias': torch.nn.Parameter(torch.zeros(768))
        },
        'ln1': {
            'weight': torch.nn.Parameter(torch.ones(768)),
            'bias': torch.nn.Parameter(torch.zeros(768))
        },
        'residual1': create_residual_block(768),
        'residual2': create_residual_block(768),
        'residual3': create_residual_block(768),
        'ln_out': {
            'weight': torch.nn.Parameter(torch.ones(768)),
            'bias': torch.nn.Parameter(torch.zeros(768))
        },
        'output': {
            'weight': torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(1, 768)) * 0.075),
            'bias': torch.nn.Parameter(torch.zeros(1))
        }
    }

def get_all_params(params):
    """Helper function to get all parameters for optimization"""
    all_params = []
    for key, value in params.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                if isinstance(inner_value, torch.nn.Parameter):
                    all_params.append(inner_value)
        elif isinstance(value, torch.nn.Parameter):
            all_params.append(value)
    return all_params

def forward(x, params, training=True):
    """Improved forward pass with consistent layer normalization"""
    # Initial embedding
    x = torch.mm(x, params['embedding']['weight'].t()) + params['embedding']['bias']
    
    # Layer normalization
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    x = (x - mean) / torch.sqrt(var + 1e-5)
    x = x * params['ln1']['weight'] + params['ln1']['bias']
    
    if training:
        x = torch.dropout(x, p=0.2, train=True)
    
    # Residual blocks
    x = forward_residual_block(x, params['residual1'], training)
    x = forward_residual_block(x, params['residual2'], training)
    x = forward_residual_block(x, params['residual3'], training)
    
    # Final layer normalization
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    x = (x - mean) / torch.sqrt(var + 1e-5)
    x = x * params['ln_out']['weight'] + params['ln_out']['bias']
    
    # Output projection
    x = torch.mm(x, params['output']['weight'].t()) + params['output']['bias']
    
    return x

def load_data(pos_dir, neg_dir):
    """Load text files from positive and negative directories"""
    texts = []
    labels = []
    
    # Load positive samples
    for filename in os.listdir(pos_dir):
        with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as f:
            texts.append(f.read().strip())
            labels.append(1)
    
    # Load negative samples
    for filename in os.listdir(neg_dir):
        with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as f:
            texts.append(f.read().strip())
            labels.append(0)
    
    return texts, labels

def preprocess_text(text):
    """Enhanced text preprocessing with stop words removal"""
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'(?<!\w)\d+(?!\w)', '', text)
    text = re.sub(r'[^\w\s\']', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = ' '.join(text.split())
    words = text.split()
    # words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def train_fold(params, train_loader, val_loader, num_epochs=20, learning_rate=0.001, patience=5):
    """Improved training function with better optimization"""
    # Get all parameters for optimization
    all_params = get_all_params(params)
    
    optimizer = torch.optim.AdamW(all_params, lr=learning_rate, weight_decay=0.01)
    
    # Rest of the training function remains the same
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    best_val_accuracy = 0
    best_params = None
    best_metrics = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = forward(batch_X.float(), params, training=True)
            
            # Label smoothing
            smooth_y = batch_y.float() * 0.9 + 0.05
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                outputs.squeeze(), smooth_y
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=0.5)
            optimizer.step()
            total_loss += loss.item()
        
        # Validation phase remains the same...
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = forward(batch_X.float(), params, training=False)
                predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
                val_predictions.extend(predictions.tolist())
                val_labels.extend(batch_y.tolist())
        
        val_accuracy = accuracy_score(val_labels, val_predictions)
        val_precision = precision_score(val_labels, val_predictions)
        val_recall = recall_score(val_labels, val_predictions)
        val_f1 = f1_score(val_labels, val_predictions)
        
        scheduler.step(val_accuracy)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_params = {k: v.clone() if isinstance(v, torch.nn.Parameter) else 
                          {k2: v2.clone() for k2, v2 in v.items()}
                          for k, v in params.items()}
            best_metrics = {
                'accuracy': val_accuracy,
                'precision': val_precision,
                'recall': val_recall,
                'f1': val_f1
            }
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, '
              f'Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}')
        
        # if patience_counter >= patience:
        #     print(f'Early stopping triggered after epoch {epoch+1}')
        #     break
    
    return best_params, best_metrics

def k_fold_training(texts, labels, k=5, num_epochs=20, batch_size=32, learning_rate=0.001):
    """Complete k-fold cross validation training pipeline with stratification"""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics = []
    all_predictions = []
    all_true_labels = []
    
    # stop_words = set(stopwords.words('english'))
    vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 3),
        stop_words='english'
    )
    
    processed_texts = [preprocess_text(text) for text in texts]
    
    print(f"Starting {k}-fold cross validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(processed_texts, labels)):
        print(f"\nTraining Fold {fold + 1}/{k}")
        
        X_train = [processed_texts[i] for i in train_idx]
        y_train = [labels[i] for i in train_idx]
        X_val = [processed_texts[i] for i in val_idx]
        y_val = [labels[i] for i in val_idx]
        
        X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
        X_val_tfidf = vectorizer.transform(X_val).toarray()
        
        X_train_tensor = torch.FloatTensor(X_train_tfidf)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val_tfidf)
        y_val_tensor = torch.LongTensor(y_val)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        params = create_model_params(X_train_tfidf.shape[1])
        params, fold_metric = train_fold(
            params,
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            patience=3
        )
        
        fold_metrics.append(fold_metric)
        
        with torch.no_grad():
            val_predictions = []
            for batch_X, _ in val_loader:
                outputs = forward(batch_X.float(), params, training=False)
                predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
                val_predictions.extend(predictions.tolist())
        
        all_predictions.extend(val_predictions)
        all_true_labels.extend(y_val)
        
        print(f"\nFold {fold + 1} Results:")
        print(f"Accuracy: {fold_metric['accuracy']:.4f}")
        print(f"Precision: {fold_metric['precision']:.4f}")
        print(f"Recall: {fold_metric['recall']:.4f}")
        print(f"F1-score: {fold_metric['f1']:.4f}")
    
    print("\nOverall K-Fold Metrics:")
    metrics = defaultdict(list)
    for fold_metric in fold_metrics:
        for metric_name, value in fold_metric.items():
            metrics[metric_name].append(value)
    
    for metric_name, values in metrics.items():
        mean_value = np.mean(values)
        std_value = np.std(values)
        print(f"{metric_name.capitalize()}:")
        print(f"  Mean: {mean_value:.4f}")
        print(f"  Std: {std_value:.4f}")
    
    final_metrics = {
        'accuracy': accuracy_score(all_true_labels, all_predictions),
        'precision': precision_score(all_true_labels, all_predictions),
        'recall': recall_score(all_true_labels, all_predictions),
        'f1': f1_score(all_true_labels, all_predictions)
    }
    
    print("\nFinal Metrics on All Folds:")
    for metric_name, value in final_metrics.items():
        print(f"{metric_name.capitalize()}: {value:.4f}")
    
    return final_metrics, fold_metrics

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Load your data
    directory = "../Datasets/"
    pos_dir = directory + "pos"
    neg_dir = directory + "neg"
    
    # Load texts and labels
    texts, labels = load_data(pos_dir, neg_dir)
    
    # Set number of folds
    k = min(5, len(texts) // 2)
    
    # Run k-fold training
    final_metrics, fold_metrics = k_fold_training(
        texts=texts,
        labels=labels,
        k=k,
        num_epochs=20,
        batch_size=min(32, len(texts) // k),
        learning_rate=0.001
    )

if __name__ == "__main__":
    main()