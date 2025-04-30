#%% MODULE BEGINS
module_name = 'ann_model'
'''
Version: v0.1
Description:
Implementation of Artificial Neural Network (ANN) for spam email detection with hyperparameter tuning.
Authors:
<Your Name>
Date Created : 02-27-2025
Date Last Updated: 04-30-2025
Doc:
This module loads processed email data, preprocesses it, and applies an Artificial Neural Network (ANN) for spam classification.
Uses Keras and TensorFlow for model training and evaluation, with hyperparameter tuning and model performance visualization.
Notes:
Requires the installation of TensorFlow and Keras libraries.
'''

#%% IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    import os
#os.chdir("./../..")
#
#custom imports
#other imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#%% USER INTERFACE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% CONSTANTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DATA_CSV = "processed_emails.csv"

#%% CONFIGURATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% INITIALIZATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% DECLARATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Global declarations Start Here
#Class definitions Start Here
#Function definitions Start Here
def load_and_prepare_data():
    """
    Loads the processed email data, scales the features, and splits into training and testing sets.
    """
    df = pd.read_csv(DATA_CSV)
    X = df.drop(["SAMPLE ID", "TARGET"], axis=1)
    y = df["TARGET"].map({"ham": 0, "spam": 1})
    
    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

def build_ann_model(input_dim):
    """
    Builds and compiles the ANN model with the given input dimensions.
    """
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification: spam or ham
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_model(X, y):
    """
    Splits data, trains the ANN model, evaluates its performance, and visualizes the results.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_ann_model(X_train.shape[1])
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    
    # Evaluate the model
    y_pred = (model.predict(X_test) > 0.5).astype("int32")  # Convert probabilities to 0 or 1
    
    # Evaluate performance
    print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # Visualize the training history
    plot_training_history(model)

def plot_training_history(model):
    """
    Visualizes the training and validation accuracy/loss curves.
    """
    history = model.history.history
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Accuracy/Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy/Loss')
    plt.legend()
    plt.show()

#%% MAIN CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    """Main function to load data, train the ANN model, and evaluate it."""
    print("Loading data...")
    X, y = load_and_prepare_data()
    print("Training and evaluating ANN model...")
    train_and_evaluate_model(X, y)

#%% SELF-RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    print(f"\"{module_name}\" module begins.")
    main()
