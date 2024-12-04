import os
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import re
import random
import contractions
import warnings
import json
import pandas as pd
import nlpaug.augmenter.word as naw
import torch.nn.functional as F
from collections import defaultdict
import logging
import math
import time

# Disable warnings
warnings.filterwarnings("ignore")

# Implement text augmentation techniques
def augment_text(text):
    augmenter = naw.SynonymAug(aug_src='wordnet')
    aug_text = augmenter.augment(text)[0]
    return aug_text

def load_data(pos_dir, neg_dir):
    """Load text files with efficient processing"""
    texts = []
    labels = []
    
    def process_directory(directory, label):
        processed = []
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r') as f:
                    processed.append((f.read().strip(), label))
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
        return processed
    
    # Process positive and negative samples
    pos_samples = process_directory(pos_dir, 1)
    neg_samples = process_directory(neg_dir, 0)
    
    # Combine and shuffle samples
    all_samples = pos_samples + neg_samples
    random.shuffle(all_samples)
    
    # Split into texts and labels
    texts, labels = zip(*all_samples)
    
    return list(texts), list(labels)

def preprocess_text(text):
    """Enhanced text preprocessing with stop words removal"""
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'(?<!\w)\d+(?!\w)', '', text)
    text = re.sub(r'[^\w\s\']', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = ' '.join(text.split())
    return text

def prepare_data_loaders(texts, labels, vectorizer=None, test_size=0.2, batch_size=32):
    """Comprehensive data preparation with advanced vectorization"""
    try:
        # Text preprocessing
        processed_texts = [preprocess_text(text) for text in texts]
        
        # TF-IDF Vectorization
        if vectorizer is None:
            vectorizer = TfidfVectorizer(
                max_features=15000,
                min_df=3,
                max_df=0.95,
                ngram_range=(1, 3),
                stop_words='english'
            )
        
        # Fit and transform
        X_tfidf = vectorizer.fit_transform(processed_texts).toarray()
        y = np.array(labels)
        
        # Ensure test_size is valid (between 0 and 1)
        test_size = max(0.1, min(0.5, test_size))  # Constrain between 0.1 and 0.5
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, 
            test_size=test_size,  # Now using the corrected test_size
            stratify=y, 
            random_state=42
        )
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
        
        # Create DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        return train_loader, test_loader, vectorizer
    
    except Exception as e:
        print(f"Data preparation failed: {e}")
        return None, None, None


def create_model_params(input_dim, num_attention_heads=8, use_positional_encoding=True, num_residual_blocks=3, base_dim=512):
    """Create enhanced model parameters with attention mechanism"""
    params = {
        'embedding': {
            'weight': torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(base_dim, input_dim))),
            'bias': torch.nn.Parameter(torch.zeros(base_dim))
        },
        'attention': {
            'query': torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(num_attention_heads, base_dim, base_dim // num_attention_heads))),
            'key': torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(num_attention_heads, base_dim, base_dim // num_attention_heads))),
            'value': torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(num_attention_heads, base_dim, base_dim // num_attention_heads))),
            'output': torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(base_dim, base_dim)))
        },
        'ln1': {
            'weight': torch.nn.Parameter(torch.ones(base_dim)),
            'bias': torch.nn.Parameter(torch.zeros(base_dim))
        }
    }
    
    # Add positional encoding parameters if enabled
    if use_positional_encoding:
        params['positional_encoding'] = {
            'weight': torch.nn.Parameter(torch.zeros(1000, base_dim))  # Support sequences up to length 1000
        }
    
    # Create residual blocks
    for i in range(num_residual_blocks):
        params[f'residual{i+1}'] = create_residual_block(base_dim)
    
    # Output layer normalization
    params['ln_out'] = {
        'weight': torch.nn.Parameter(torch.ones(base_dim)),
        'bias': torch.nn.Parameter(torch.zeros(base_dim))
    }
    
    # Output projection
    params['output'] = {
        'weight': torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(1, base_dim))),
        'bias': torch.nn.Parameter(torch.zeros(1))
    }
    
    return params


def create_residual_block(dim, dropout_rate=0.1):
    """Create an enhanced residual block with improved regularization"""
    return {
        'layer1_weight': torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(dim, dim))),
        'layer1_bias': torch.nn.Parameter(torch.zeros(dim)),
        'ln1_weight': torch.nn.Parameter(torch.ones(dim)),
        'ln1_bias': torch.nn.Parameter(torch.zeros(dim)),
        'layer2_weight': torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(dim, dim))),
        'layer2_bias': torch.nn.Parameter(torch.zeros(dim)),
        'ln2_weight': torch.nn.Parameter(torch.ones(dim)),
        'ln2_bias': torch.nn.Parameter(torch.zeros(dim)),
        'dropout_rate': dropout_rate
    }

def forward(x, params, training=True):
    """Enhanced forward pass with attention mechanism"""
    # Initial embedding
    x = torch.mm(x, params['embedding']['weight'].t()) + params['embedding']['bias']
    
    # Add positional encoding if enabled
    if 'positional_encoding' in params:
        seq_len = x.size(0)
        x = x + params['positional_encoding']['weight'][:seq_len]
    
    # Initial layer normalization
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    x = (x - mean) / torch.sqrt(var + 1e-5)
    x = x * params['ln1']['weight'] + params['ln1']['bias']
    
    # Multi-head attention
    batch_size = x.size(0)
    num_heads = params['attention']['query'].size(0)
    head_dim = params['attention']['query'].size(2)
    
    # Reshape for attention
    queries = torch.stack([torch.mm(x, params['attention']['query'][h]) for h in range(num_heads)])
    keys = torch.stack([torch.mm(x, params['attention']['key'][h]) for h in range(num_heads)])
    values = torch.stack([torch.mm(x, params['attention']['value'][h]) for h in range(num_heads)])
    
    # Scaled dot-product attention
    scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(head_dim)
    attention_weights = F.softmax(scores, dim=-1)
    
    if training:
        attention_weights = F.dropout(attention_weights, p=0.1, training=True)
    
    attention_output = torch.matmul(attention_weights, values)
    
    # Concatenate heads and project
    attention_output = attention_output.transpose(0, 1).contiguous().view(batch_size, -1)
    x = torch.mm(attention_output, params['attention']['output'])
    
    # Apply dropout if in training mode
    if training:
        x = F.dropout(x, p=0.1, training=True)
    
    # Process through residual blocks
    num_blocks = len([k for k in params.keys() if k.startswith('residual')])
    for i in range(num_blocks):
        x = forward_residual_block(x, params[f'residual{i+1}'], training)
    
    # Final layer normalization
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    x = (x - mean) / torch.sqrt(var + 1e-5)
    x = x * params['ln_out']['weight'] + params['ln_out']['bias']
    
    # Output projection
    x = torch.mm(x, params['output']['weight'].t()) + params['output']['bias']
    
    return x


def forward_residual_block(x, block_params, training=True):
    """Enhanced residual block forward pass with improved normalization"""
    identity = x
    
    # First layer
    out = torch.mm(x, block_params['layer1_weight'].t()) + block_params['layer1_bias']
    
    # First layer normalization
    mean = out.mean(-1, keepdim=True)
    var = out.var(-1, keepdim=True, unbiased=False)
    out = (out - mean) / torch.sqrt(var + 1e-5)
    out = out * block_params['ln1_weight'] + block_params['ln1_bias']
    
    # Activation function (GELU)
    out = F.gelu(out)
    
    # Dropout
    if training:
        out = F.dropout(out, p=block_params['dropout_rate'], training=True)
    
    # Second layer
    out = torch.mm(out, block_params['layer2_weight'].t()) + block_params['layer2_bias']
    
    # Second layer normalization
    mean = out.mean(-1, keepdim=True)
    var = out.var(-1, keepdim=True, unbiased=False)
    out = (out - mean) / torch.sqrt(var + 1e-5)
    out = out * block_params['ln2_weight'] + block_params['ln2_bias']
    
    # Residual connection with scaling
    return identity + 0.1 * out

def get_all_params(params):
    """Helper function to collect all trainable parameters"""
    all_params = []
    for key, value in params.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                if isinstance(inner_value, torch.nn.Parameter):
                    all_params.append(inner_value)
        elif isinstance(value, torch.nn.Parameter):
            all_params.append(value)
    return all_params

def mixup_data(x, y, alpha=0.2):
    """Performs mixup on the input data and returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_model(model_params, train_loader, test_loader, num_epochs=50, learning_rate=1e-3, 
                warmup_steps=None, label_smoothing=0.1, mixup_alpha=0.2):
    """Enhanced training with warmup, label smoothing, and mixup"""
    try:
        # Initialize optimizer with CPU-specific parameters
        all_params = get_all_params(model_params)
        optimizer = torch.optim.AdamW(
            all_params,
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Calculate total steps for warmup
        total_steps = num_epochs * len(train_loader)
        if warmup_steps is None:
            warmup_steps = total_steps // 10  # Default to 10% of total steps
        
        # Initialize learning rate scheduler with warmup
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / 
                      float(max(1, total_steps - warmup_steps)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        best_val_loss = float('inf')
        patience_counter = 0
        accumulation_steps = 4  # Reduced for CPU
        
        for epoch in range(num_epochs):
            model_params['training'] = True
            total_train_loss = 0
            optimizer.zero_grad()
            
            # Training loop with gradient accumulation
            for i, (batch_X, batch_y) in enumerate(train_loader):
                # Apply mixup augmentation
                if mixup_alpha > 0:
                    mixed_x, y_a, y_b, lam = mixup_data(batch_X, batch_y, mixup_alpha)
                    outputs = forward(mixed_x, model_params)
                    loss_a = F.binary_cross_entropy_with_logits(outputs, y_a)
                    loss_b = F.binary_cross_entropy_with_logits(outputs, y_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    outputs = forward(batch_X, model_params)
                    # Apply label smoothing
                    if label_smoothing > 0:
                        smooth_targets = (1 - label_smoothing) * batch_y + label_smoothing * 0.5
                        loss = F.binary_cross_entropy_with_logits(outputs, smooth_targets)
                    else:
                        loss = F.binary_cross_entropy_with_logits(outputs, batch_y)
                
                loss = loss / accumulation_steps
                loss.backward()
                
                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                    optimizer.step()
                    scheduler.step()  # Update learning rate
                    optimizer.zero_grad()
                
                total_train_loss += loss.item() * accumulation_steps
            
            # Validation phase
            val_metrics = evaluate_model(model_params, test_loader)
            
            # Model checkpointing and early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_params = {k: v.clone() if isinstance(v, torch.nn.Parameter) 
                                   else v for k, v in model_params.items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f'Epoch {epoch+1}/{num_epochs}:', end=' ')
            print(f'Train Loss: {total_train_loss/len(train_loader):.4f}', end=' ')
            print(f'Val Loss: {val_metrics["loss"]:.4f}', end=' ')
            print(f'Val Accuracy: {val_metrics["accuracy"]:.4f}')
            
            if patience_counter >= 10:
                print("Early stopping triggered")
                break
        
        return best_model_params
    
    except Exception as e:
        print(f"Model training failed: {e}")
        return None



def evaluate_model(model_params, test_loader):
    """Comprehensive model evaluation"""
    model_params['training'] = False
    total_loss = 0
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = forward(batch_X, model_params)
            
            # Calculate loss
            loss = F.binary_cross_entropy_with_logits(
                outputs,
                batch_y,
                reduction='mean'
            )
            total_loss += loss.item()
            
            # Get predictions
            predictions = (torch.sigmoid(outputs) > 0.5).long()
            all_predictions.extend(predictions.numpy().flatten())
            all_true_labels.extend(batch_y.numpy().flatten())
    
    # Calculate metrics
    metrics = {
        'loss': total_loss / len(test_loader),
        'accuracy': accuracy_score(all_true_labels, all_predictions),
        'precision': precision_score(all_true_labels, all_predictions),
        'recall': recall_score(all_true_labels, all_predictions),
        'f1': f1_score(all_true_labels, all_predictions)
    }
    
    return metrics

def k_fold_cross_validation(texts, labels, k=5, num_epochs=20):
    """K-fold cross-validation with CPU optimization"""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"\nFold {fold + 1}/{k}")
        
        # Split data
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        # Prepare data loaders
        train_loader, val_loader, vectorizer = prepare_data_loaders(
            train_texts,
            train_labels,
            test_size=0.0,  # No additional split needed
            batch_size=32  # Reduced for CPU
        )
        
        # Create and train model
        input_dim = train_loader.dataset.tensors[0].shape[1]
        model_params = create_model_params(input_dim)
        best_model_params = train_model(
            model_params,
            train_loader,
            val_loader,
            num_epochs=num_epochs
        )
        
        # Evaluate fold
        fold_metrics.append(evaluate_model(best_model_params, val_loader))
    
    # Calculate average metrics
    avg_metrics = {
        metric: np.mean([fold[metric] for fold in fold_metrics])
        for metric in fold_metrics[0].keys()
    }
    
    return avg_metrics, fold_metrics

# Monitoring and Error Handling (Changes #6 and #7)
class MetricsLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.history = defaultdict(list)
    
    def log_metrics(self, metrics, step):
        for key, value in metrics.items():
            self.history[key].append(value)
        with open(os.path.join(self.log_dir, f'metrics_step_{step}.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

class CustomLogger:
    def __init__(self, log_file):
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def log_error(self, error, stack_trace=None):
        logging.error(f"Error: {error}")
        if stack_trace:
            logging.error(f"Stack trace: {stack_trace}")
    
    def log_progress(self, message):
        logging.info(message)
        print(message)

class ErrorRecovery:
    def __init__(self, max_retries=3):
        self.max_retries = max_retries
    
    def execute_with_retry(self, func, *args, **kwargs):
        retries = 0
        while retries < self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                if retries == self.max_retries:
                    raise e
                time.sleep(1)

def main():
    """Enhanced main execution function with comprehensive logging and error handling"""
    try:
        # Initialize custom logger (Change #7)
        error_recovery = ErrorRecovery(max_retries=3)
        metrics_logger = MetricsLogger('Results/metrics_logs')
        logger = CustomLogger('Results/training.log')

        with open('Results/training.log', 'w'):
            pass
        
        logger.log_progress("Starting sentiment analysis model training...")
        
        # Set deterministic behavior for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # Set number of threads for CPU optimization
        torch.set_num_threads(8)
        
        # Load and validate data
        pos_dir = "../../Datasets/pos"
        neg_dir = "../../Datasets/neg"
        
        if not (os.path.exists(pos_dir) and os.path.exists(neg_dir)):
            logger.log_error("Dataset directories not found")
            raise FileNotFoundError("Dataset directories not found")
        
        logger.log_progress("Loading data...")
        texts, labels = load_data(pos_dir, neg_dir)
        # logger.log_progress(f"Loaded {len(texts)} samples ({sum(labels)} positive, {len(labels)-sum(labels)} negative)")
        
        # Configure training parameters with new enhancements (Changes #2, #3, #4)
        config = {
            'batch_size': 32,
            'num_epochs': 20,
            'learning_rate': 1e-3,
            'early_stopping_patience': 10,
            'num_folds': 5,
            'warmup_steps': 100,
            'label_smoothing': 0.1,
            'mixup_alpha': 0.2,
            'attention_heads': 8,
            'positional_encoding': True,
            'augmentation_prob': 0.3
        }
        
        logger.log_progress("\nStarting k-fold cross-validation...")
        fold_results = []
        
        # Perform k-fold cross-validation with enhanced monitoring (Change #6)
        skf = StratifiedKFold(n_splits=config['num_folds'], shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
            try:
                logger.log_progress(f"\nTraining Fold {fold + 1}/{config['num_folds']}")
                
                # Split data for current fold with augmentation (Change #4)
                train_texts = [texts[i] for i in train_idx]
                train_labels = [labels[i] for i in train_idx]
                if config['augmentation_prob'] > 0:
                    augmented_texts, augmented_labels = [], []
                    for text, label in zip(train_texts, train_labels):
                        if random.random() < config['augmentation_prob']:
                            aug_text = augment_text(text)
                            augmented_texts.append(aug_text)
                            augmented_labels.append(label)
                    train_texts.extend(augmented_texts)
                    train_labels.extend(augmented_labels)
                
                # Prepare data loaders with dynamic batching
                train_loader, val_loader, vectorizer = error_recovery.execute_with_retry(
                    prepare_data_loaders,
                    train_texts,
                    train_labels,
                    test_size=0.0,
                    batch_size=config['batch_size']
                )
                
                if train_loader is None or val_loader is None:
                    logger.log_error(f"Skipping fold {fold + 1} due to data preparation failure")
                    continue
                
                # Create enhanced model with attention (Change #2)
                input_dim = train_loader.dataset.tensors[0].shape[1]
                model_params = create_model_params(
                    input_dim,
                    num_attention_heads=config['attention_heads'],
                    use_positional_encoding=config['positional_encoding']
                )
                
                # Train model with enhanced features (Changes #3, #5)
# In the main function, update the model training call:
                best_model_params = train_model(
                    model_params,
                    train_loader,
                    val_loader,
                    num_epochs=config['num_epochs'],
                    learning_rate=config['learning_rate'],
                    warmup_steps=100,  # Add warmup steps
                    label_smoothing=0.1,  # Add label smoothing
                    mixup_alpha=0.2  # Add mixup augmentation
                )

                
                if best_model_params is None:
                    logger.log_error(f"Skipping fold {fold + 1} due to training failure")
                    continue
                
                # Evaluate fold with enhanced metrics (Change #6)
                fold_metrics = evaluate_model(best_model_params, val_loader)
                fold_results.append(fold_metrics)
                metrics_logger.log_metrics(fold_metrics, fold)
                
                logger.log_progress(f"\nFold {fold + 1} Results:")
                for metric, value in fold_metrics.items():
                    logger.log_progress(f"{metric.capitalize()}: {value:.4f}")
                
            except Exception as e:
                logger.log_error(f"Error in fold {fold + 1}: {e}")
                continue
        
        # Calculate and display final results with enhanced reporting
        if fold_results:
            logger.log_progress("\nFinal Cross-Validation Results:")
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            
            results = {
                'config': config,
                'fold_results': fold_results,
                'final_metrics': {}
            }
            
            for metric in metrics:
                values = [result[metric] for result in fold_results if metric in result]
                if values:
                    mean_value = np.mean(values)
                    std_value = np.std(values)
                    results['final_metrics'][metric] = {
                        'mean': float(mean_value),
                        'std': float(std_value)
                    }
                    logger.log_progress(f"{metric.capitalize()}: {mean_value:.4f} (±{std_value:.4f})")
            
            # Save results with timestamp
            # timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            with open(f'Results/results.json', 'w') as f:
                json.dump(results, f, indent=4)
            
            logger.log_progress(f"\nResults saved to results.json")
            return results
        
        else:
            logger.log_error("No valid results obtained from any fold")
            return None
            
    except Exception as e:
        logger.log_error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    try:
        results = main()
        if results is not None:
            print("\nModel training completed successfully!")
    except Exception as e:
        print(f"Program failed with error: {e}")
