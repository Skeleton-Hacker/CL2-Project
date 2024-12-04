from math import log
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
from random import shuffle, seed
from tabulate import tabulate
import string
import numpy as np
import json
import pickle
import warnings

warnings.filterwarnings("ignore")

def load_data(directory):
    seed(42)
    trainingPos = []
    trainingNeg = []
    testingPos = []
    testingNeg = []

    posFiles = os.listdir(os.path.join(directory, "pos"))
    shuffle(posFiles)
    for i, file in enumerate(posFiles):
        with open(os.path.join(directory, "pos", file), "r") as fileDesc:
            if i < 900:
                trainingPos.append(preprocessor(fileDesc.read()))
            else:
                testingPos.append(preprocessor(fileDesc.read()))
    
    negFiles = os.listdir(os.path.join(directory, "neg"))
    shuffle(negFiles)
    for i, file in enumerate(negFiles):
        with open(os.path.join(directory, "neg", file), "r") as fileDesc:
            if i < 900:
                trainingNeg.append(preprocessor(fileDesc.read()))
            else:
                testingNeg.append(preprocessor(fileDesc.read()))
    
    return trainingPos, trainingNeg, testingPos, testingNeg

def preprocessor(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text)
    wordFrequency = {word: 1 for word in words}
    return wordFrequency

def trainer(trainingPos, trainingNeg):
    vocabulary = set()
    training = trainingPos + trainingNeg
    for review in training:
        vocabulary = vocabulary.union(set(review.keys()))
    
    posFrequency = {word: 0 for word in vocabulary}
    negFrequency = {word: 0 for word in vocabulary}

    for review in trainingPos:
        for word in review:
            posFrequency[word] += review[word]
    for review in trainingNeg:
        for word in review:
            negFrequency[word] += review[word]

    totalPos = sum(posFrequency.values())
    totalNeg = sum(negFrequency.values())

    posProbability = {word:
                        (posFrequency[word]+1)/(totalPos+len(vocabulary))
                        for word in vocabulary
                    }
    negProbability = {word:
                        (negFrequency[word]+1)/(totalNeg+len(vocabulary))
                        for word in vocabulary
                    }
    
    return posProbability, negProbability, vocabulary

def get_probability_scores(review, posProbability, negProbability):
    """Calculate probability scores for a single review"""
    posScore = log(0.5)
    negScore = log(0.5)
    
    for word in review:
        if word in posProbability:
            posScore += log(posProbability[word])
        if word in negProbability:
            negScore += log(negProbability[word])
            
    return posScore, negScore

def predict_single(review, posProbability, negProbability):
    """Make prediction for a single review"""
    posScore, negScore = get_probability_scores(review, posProbability, negProbability)
    return 1 if posScore > negScore else 0, np.exp(posScore) / (np.exp(posScore) + np.exp(negScore))

def test_model(testingPos, testingNeg, posProbability, negProbability):
    """Test model and return predictions, true labels, and probabilities"""
    predictions = []
    probabilities = []
    true_labels = []
    
    # Process positive reviews
    for review in testingPos:
        pred, prob = predict_single(review, posProbability, negProbability)
        predictions.append(pred)
        probabilities.append(float(prob))  # Convert numpy float to Python float for JSON serialization
        true_labels.append(1)
    
    # Process negative reviews
    for review in testingNeg:
        pred, prob = predict_single(review, posProbability, negProbability)
        predictions.append(pred)
        probabilities.append(float(prob))  # Convert numpy float to Python float for JSON serialization
        true_labels.append(0)
    
    return predictions, true_labels, probabilities

def save_model_and_results(model_dict, results_dict, output_dir="model_output"):
    """Save model and results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model using pickle (since it contains sets which aren't JSON serializable)
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model_dict, f)
    
    # Save results using JSON
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results_dict, f)

def analyser(model_name, tp, tn, fp, fn):
    """Analyse model performance"""
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    print(f"\n{model_name} Analysis:")
    print(tabulate(
        [
            ["Accuracy", accuracy],
            ["Precision", precision],
            ["Recall", recall],
            ["F1 Score", f1]
        ],
        headers=["Metric", "Value"]
    ))

def main():
    directory = "../../Datasets/"
    output_dir = "Results/"
    
    # Load and prepare data
    print("Loading and preparing data...")
    trainingPos, trainingNeg, testingPos, testingNeg = load_data(directory)
    
    # Train model
    print("Training model...")
    posProbability, negProbability, vocabulary = trainer(trainingPos, trainingNeg)
    
    # Test model
    print("Testing model...")
    predictions, true_labels, probabilities = test_model(
        testingPos, testingNeg, posProbability, negProbability
    )
    
    # Prepare dictionaries
    model_dict = {
        'pos_probability': posProbability,
        'neg_probability': negProbability,
        'vocabulary': vocabulary
    }
    
    results_dict = {
        'predictions': predictions,
        'true_labels': true_labels,
        'probabilities': probabilities
    }
    
    # Save to files
    print(f"Saving model and results to {output_dir}/...")
    save_model_and_results(model_dict, results_dict, output_dir)
    
    # Print original analysis
    tp = sum([1 for p, t in zip(predictions, true_labels) if p == 1 and t == 1])
    tn = sum([1 for p, t in zip(predictions, true_labels) if p == 0 and t == 0])
    fp = sum([1 for p, t in zip(predictions, true_labels) if p == 1 and t == 0])
    fn = sum([1 for p, t in zip(predictions, true_labels) if p == 0 and t == 1])
    analyser("Naive Bayes", tp, tn, fp, fn)
    
    print("\nModel and results have been saved successfully!")

if __name__ == "__main__":
    main()