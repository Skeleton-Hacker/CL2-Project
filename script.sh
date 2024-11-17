#!/bin/bash
read -p "Do you want to create a virtual environment? (y/n): " response
response=${response,,}
if [ "$response" == "y" ]; then
    python3 -m venv venv
    echo "Virtual environment created in 'venv'"
    source venv/bin/activate
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
    rm -f Result/CSV_file/*
    rm -f Result/Fold_analysis/*
    rm -f Result/Graphs/*
    rm -f Result/JSON_files/*
    python3 Code/neural.py && python3 Code/analysis.py
else
    echo "Exiting script"
fi