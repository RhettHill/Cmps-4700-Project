# Version: v0.1
# Date Last Updated: 02-26-2025

#%% MODULE BEGINS
module_name = 'spam_detection_model_training'
'''
Version: v0.1
Description:
This module loads the preprocessed email data, splits it into training and testing sets,
trains a Support Vector Machine (SVM) classifier, and evaluates its performance.
Authors:
<Your Name>
Date Created : 02-26-2025
Date Last Updated: 02-26-2025
Doc:
Loads the CSV with TF-IDF features, converts the TARGET column into binary labels,
and runs model training and evaluation.
Notes:
Uses scikit-learn for ML tasks.
'''

#%% IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    import os

import pandas as pd
import numpy as np
from copy import deepcopy as dpcpy

# Sklearn imports for model building and evaluation
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#%% CONSTANTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DATA_CSV = "processed_emails.csv"  # Path to the processed CSV file

#%% FUNCTION DEFINITIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def load_and_prepare_data():
    """
    Loads the processed email data from CSV,
    converts TARGET to binary (0 for ham, 1 for spam),
    and splits into features (X) and labels (y).
    """
    # Load CSV file
    df = pd.read_csv(DATA_CSV)
    
    # Extract features and target; drop 'SAMPLE ID'
    X = df.drop(["SAMPLE ID", "TARGET"], axis=1)
    
    # Convert TARGET to binary: ham -> 0, spam -> 1
    y = df["TARGET"].map({"ham": 0, "spam": 1})
    
    return X, y

def train_and_evaluate_model(X, y):
    """
    Splits the data into training and test sets, trains an SVM classifier,
    and prints out the accuracy, classification report, and confusion matrix.
    """
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the SVM classifier (using a linear kernel)
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = svm_model.predict(X_test)
    
    # Evaluate the model
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("Model Accuracy:", acc)
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)

#%% MAIN CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    """
    Main function to load data, train the SVM model, and evaluate its performance.
    """
    print("Loading and preparing data...")
    X, y = load_and_prepare_data()
    
    print("Training and evaluating the SVM model...")
    train_and_evaluate_model(X, y)

#%% SELF-RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    print(f"\"{module_name}\" module begins.")
    main()
