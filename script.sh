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

read -p "Do you want to run the model? (y/n): " response
response=${response,,}
if [ "$response" == "y" ]; then
    if [ -d "Project" ]; then
        source Project/bin/activate
    fi
    if [ -d "Results" ]; then
        rm -f Results/CSV_files/*
        rm -f Results/Fold_analysis/*
        rm -f Results/Graphs/*
        rm -f Results/JSON_files/*
    else
        mkdir Results
        mkdir Results/CSV_files
        mkdir Results/Fold_analysis
        mkdir Results/Graphs
        mkdir Results/JSON_files
    fi
    python3 Code/neural.py && python3 Code/analysis.py > Results/Analysis.txt
else
    echo "Exiting script"
fi