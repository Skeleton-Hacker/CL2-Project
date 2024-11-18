# Text Classification using Distributed Semantic Models

This repository is part of the **Computational Linguistics - 2** project, focusing on text classification using distributed semantic models and alternative approaches.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
  - [Automated Setup](#automated-setup)
  - [Manual Setup](#manual-setup)
- [Usage](#usage)
- [Models](#models)
  - [Distributed Semantic Models](#distributed-semantic-models)
  - [Naive Bayes Model](#naive-bayes-model)
- [Directory Structure](#directory-structure)

---

## Overview

This repository implements text classification using **Distributed Semantic Models** alongside a baseline **Naive Bayes classifier**. It enables users to experiment with these models and analyze their performance on text classification tasks.

---

## Features
- **Automated Setup**: Use a shell script to configure the environment and run the models with minimal effort.
- **Model Comparisons**: Test and compare performance between semantic models and traditional approaches.
- **Output Analysis**: Includes scripts for comprehensive performance evaluation and visualization.

---

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8 or later
- pip (Python package manager)
- Bash shell (for running scripts)

### Automated Setup
1. Make the script executable:
   ```bash
   chmod +x script.sh
   ```
2. Run the script:
   ```bash
   ./script.sh
   ```
This script will:
- Ask you if you want to create a virtual environment
- Set up the virtual environment and install the necessary dependencies from `requirements.txt`
- Prompt you to choose which model(s) to run
- Make the necessary directories accordingly

### Manual Setup
1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure the required directories are created manually. Refer to [Directory Structure](#directory-structure) for more details

---

## Usage

To run the models, follow the interactive prompts in the `script.sh` file. 
1. Run the script:
   ```bash
   ./script.sh
   ```
2. Answer the prompts to 
    - Decide whether to create a virtual environment.
    - Choose which model(s) to execute:
        - DSM for the Distributed Semantic Model.
        - NB for the Naive Bayes model.
        - both to run both models.
        - none to skip execution.
3. Results will be saved in the appropriate directories under the Results folder.

---

## Script Functionality

The script automates the following tasks:

1. **Virtual Environment Setup**:
   - Creates a virtual environment (`Project`) and installs dependencies from `requirements.txt`.
   - Skips this step if you choose not to set up the virtual environment.

2. **Model Execution**:
   - **Distributed Semantic Model (DSM)**:
     - Cleans up old results under `Results/DSM/` (if they exist).
     - Creates necessary subdirectories for outputs:
       - `CSV_files`, `Fold_analysis`, `Graphs`, `JSON_files`.
     - Runs `Code/Dist_Semantic_Model/model.py` and `Code/Dist_Semantic_Model/analysis.py`.
     - Saves the analysis output to `Results/DSM/Analysis.txt`.
   - **Naive Bayes Model (NB)**:
     - Cleans up old results under `Results/Naive_Bayes/` (if they exist).
     - Creates the output directory if it doesn't exist.
     - Runs `Code/Naive_Bayes/model.py` and `Code/Naive_Bayes/analysis.py`.

3. **Custom Execution**:
   - Users can choose to run one model, both models, or none.

---

## Models

### Distributed Semantic Models

This model utilizes advanced distributed representations for words and sentences, leveraging vector embeddings and fine-tuned features for text classification tasks.

### Naive Bayes Model

A traditional probabilistic classifier based on the Bayes theorem, used as a baseline for comparison.

---

## Directory Structure
``` bash
.
├── Code/
│   ├── Dist_Semantic_Model/
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
├── script.sh         
└── README.md         
```