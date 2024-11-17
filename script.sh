#!/bin/bash
read -p "Do you want to create a virtual environment? (y/n): " response
response=${response,,}
if [ "$response" == "y" ]; then
    python3 -m venv Project
    echo "Virtual environment created in 'Project'"
    source Project/bin/activate
    if [ -f "requirements.txt" ]; then
        echo "Installing requirements..."
        pip install -r requirements.txt
    else
        echo "No requirements.txt file found"
    fi
else
    echo "Skipping virtual environment creation"
    echo "Please ensure the required packages are installed before proceeding"
fi

read -p "Which model do you want to run? (DSM/NB/both/none): " model_choice
echo ""
model_choice=${model_choice,,}
case $model_choice in
    dsm)
        if [ -d "Project" ]; then
            source Project/bin/activate
        fi
        if [ -d "Results/DSM" ]; then
        rm -f Results/DSM/CSV_files/*
        rm -f Results/DSM/Fold_analysis/*
        rm -f Results/DSM/Graphs/*
        rm -f Results/DSM/JSON_files/*
        else
            mkdir Results
            mkdir Results/DSM
            mkdir Results/DSM/CSV_files
            mkdir Results/DSM/Fold_analysis
            mkdir Results/DSM/Graphs
            mkdir Results/DSM/JSON_files
        fi
        python3 Code/Dist_Semantic_Model/model.py && python3 Code/Dist_Semantic_Model/analysis.py > Results/DSM/Analysis.txt
        ;;
    nb)
        if [ -d "Project" ]; then
            source Project/bin/activate
        fi
        if [ -d "Results/Naive_Bayes"]; then
            rm -f Results/Naive_Bayes/*
        else
            mkdir Results
            mkdir Results/Naive_Bayes
        fi
        python3 Code/Naive_Bayes/model.py && python3 Code/Naive_Bayes/analysis.py
        ;;
    both)
        if [ -d "Project" ]; then
            source Project/bin/activate
        fi
        echo "Running DSM model..."
        if [ -d "Results/DSM" ]; then
        rm -f Results/DSM/CSV_files/*
        rm -f Results/DSM/Fold_analysis/*
        rm -f Results/DSM/Graphs/*
        rm -f Results/DSM/JSON_files/*
        else
            mkdir Results
            mkdir Results/DSM
            mkdir Results/DSM/CSV_files
            mkdir Results/DSM/Fold_analysis
            mkdir Results/DSM/Graphs
            mkdir Results/DSM/JSON_files
        fi
        python3 Code/Dist_Semantic_Model/model.py && python3 Code/Dist_Semantic_Model/analysis.py > Results/DSM/Analysis.txt
        echo ""
        echo "Running Naive Bayes model..."
        if [ -d "Results/Naive_Bayes" ]; then
            rm -f Results/Naive_Bayes/*
        else
            mkdir -p Results/Naive_Bayes
        fi
        python3 Code/Naive_Bayes/model.py && python3 Code/Naive_Bayes/analysis.py
        ;;
    none)
        echo "No model selected. Exiting the script..."
        ;;
    *)
        echo "Invalid choice. Exiting the script..."
        ;;
esac
