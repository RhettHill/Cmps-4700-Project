# Version: v0.2
# Date Last Updated: 04-03-2025

#%% MODULE BEGINS
module_name = 'dt_model'
'''
Version: v0.2
Description:
This module trains a Decision Tree classifier on processed TF-IDF email data for spam detection.
Authors:
<Your Name>
Date Created : 02-27-2025
Date Last Updated: 04-03-2025
Doc:
Loads preprocessed email data, trains a Decision Tree classifier, and evaluates performance.
Includes hyperparameter tuning, cross-validation, and visualization.
'''

#%% IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sklearn imports for model training and evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#%% CONSTANTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DATA_CSV = "processed_emails.csv"  # Path to processed CSV file

#%% FUNCTION DEFINITIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def load_and_prepare_data():
    """
    Loads the processed email data from CSV,
    converts TARGET to binary (0 for ham, 1 for spam),
    and splits into features (X) and labels (y).
    """
    df = pd.read_csv(DATA_CSV)
    X = df.drop(["SAMPLE ID", "TARGET"], axis=1)
    y = df["TARGET"].map({"ham": 0, "spam": 1})
    return X, y

def train_and_evaluate_model(X, y):
    """
    Splits the data into training and test sets, trains a Decision Tree classifier,
    evaluates its performance, and visualizes the tree.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Decision Tree with tuned parameters
    dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
    dt_model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(dt_model, X_train, y_train, cv=5)
    print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.4f}")

    # Predictions
    y_pred = dt_model.predict(X_test)
    
    # Evaluation Metrics
    print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Decision Tree Visualization
    plt.figure(figsize=(12, 8))
    plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=["Ham", "Spam"])
    plt.show()

#%% MAIN CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    """ Main function to load data, train the Decision Tree model, and evaluate it. """
    print("Loading data...")
    X, y = load_and_prepare_data()
    print("Training Decision Tree model...")
    train_and_evaluate_model(X, y)

#%% SELF-RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    print(f"\"{module_name}\" module begins.")
    main()
