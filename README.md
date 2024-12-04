# Text Classification using Distributed Semantic Models

This repository is part of the **Computational Linguistics - 2** project, focusing on text classification using distributed semantic models and alternative approaches.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Usage](#usage)
- [Models](#models)
  - [Neural Network Model](#neural-network-model)
  - [Naive Bayes Model](#naive-bayes-model)
- [Directory Structure](#directory-structure)

---

## Overview

This repository implements sentiment analysis using two approaches:
- An advanced Neural Network with attention mechanisms, positional encoding, and various optimization techniques
- A Naive Bayes classifier as a baseline model

Both models are designed to classify text into positive and negative sentiments.

---

## Features
- **Advanced Neural Architecture**: Implements attention mechanisms, positional encoding, and residual connections
- **Enhanced Training**: Includes mixup augmentation, label smoothing, and learning rate warmup
- **Robust Evaluation**: K-fold cross-validation with comprehensive metrics
- **Data Augmentation**: Text augmentation using synonym replacement
- **Error Handling**: Comprehensive logging and error recovery mechanisms
- **Performance Monitoring**: Detailed metrics tracking and visualization

---

## Installation

### Prerequisites
- Python 3.8 or later
- PyTorch
- NLTK
- scikit-learn
- Other dependencies listed in requirements.txt

### Setup
1. Clone the repository
2. Install the dependencies via
```bash
pip install -r requirements.txt
```

---

## Usage

1. Prepare the dataset in the following structure
```bash
├── Datasets/
│   ├── neg/          
│   ├── pos/
```
2. Run the model after entering the respective directories
```bash
cd Code/Neural_Network
python3 model.py
python3 analysis.py
```
	 
```bash
cd Code/Naie_Bayes
python3 model.py
python3 analysis.py
```
The results will be saved in the `Results` directory in the respective model directory. 

---

## Models

### Neural Network Model

1. Multi-head attention mechanism
2. Positional encoding
3. Residual connections
4. Label smoothing
5. Mixup augmentation
6. Learning rate warmup
7. Gradient accumulation
8. Early stopping


### Naive Bayes Model

1. Text preprocessing
2. Stop word removal
3. Laplace smoothing
4. Cross-validation
5. Performance metrics calculation

---

## Directory Structure
``` bash
.
├── Code/
│   ├── Neural_Network/
│   │   ├── model.py
│   │   ├── analysis.py
│   ├── Naive_Bayes/
│       ├── model.py
│       ├── analysis.py
├── Datasets/
│   ├── neg/          
│   ├── pos/    
├── Results/
│   ├── DSM/
│   │   ├── CSV_files/
│   │   ├── Fold_analysis/
│   │   ├── Graphs/
│   │   ├── JSON_files/
│   ├── Naive_Bayes/
├── requirements.txt  
└── README.md         
```