#%% MODULE BEGINS
module_name = 'knn_model'
'''
Version: 1.1
Description:
Implementation of K-Nearest Neighbors (KNN) for spam email detection with hyperparameter tuning.
Authors:
<Your Name>
Date Created : <Date>
Date Last Updated: <Date>
Doc:
This module loads processed email data, preprocesses it, and applies KNN with GridSearchCV to find the best k.
Notes:
Uses scikit-learn for model training and evaluation, logging for progress tracking.
'''

#%% IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    import os
#os.chdir("./../..")
#
#custom imports
#other imports
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

#%% USER INTERFACE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% CONSTANTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DATA_CSV = "processed_emails.csv"

#%% CONFIGURATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#%% INITIALIZATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% DECLARATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Global declarations Start Here
#Class definitions Start Here
#Function definitions Start Here
def load_and_prepare_data():
    logging.info("Starting to load and prepare data...")
    try:
        df = pd.read_csv(DATA_CSV)
        X = df.drop(["SAMPLE ID", "TARGET"], axis=1)
        y = df["TARGET"].map({"ham": 0, "spam": 1})
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        logging.info("Data loaded and preprocessed successfully.")
        return X, y
    except FileNotFoundError:
        logging.error(f"Error: The file {DATA_CSV} was not found.")
        exit(1)
    except Exception as e:
        logging.error(f"Unexpected error loading data: {e}")
        exit(1)


def train_and_evaluate_model(X, y):
    """
    Splits the data, finds the best k using GridSearchCV,
    trains the best model, and evaluates its performance.
    """
    # Split data into training (60%), validation (20%), and test (20%) sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Hyperparameter tuning for best k
    param_grid = {'n_neighbors': np.arange(1, 31, 2)}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_k = grid_search.best_params_['n_neighbors']
    logging.info(f"Best k found: {best_k} with accuracy: {grid_search.best_score_:.4f}")
    
    # Train final model with best k
    knn_model = KNeighborsClassifier(n_neighbors=best_k)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    
    # Evaluate final model
    logging.info(f"Final Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logging.info("Classification Report:\n" + classification_report(y_test, y_pred))
    logging.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))
    visualize_knn_decision_boundary(knn_model, X_train, X_test, y_test)
    
def visualize_knn_decision_boundary(model, X_train, X_test, y_test):
    """
    Visualizes the KNN decision boundary for the first two principal components.
    """
    # Use PCA to reduce the data to 2 dimensions
    pca = PCA(n_components=2)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)

    # Create a meshgrid for plotting the decision boundaries
    x_min, x_max = X_train_reduced[:, 0].min() - 1, X_train_reduced[:, 0].max() + 1
    y_min, y_max = X_train_reduced[:, 1].min() - 1, X_train_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                         np.linspace(y_min, y_max, 100))

    # Transform the meshgrid points into the same space as the training data
    meshgrid_points = np.c_[xx.ravel(), yy.ravel()]

    # Inverse transform the meshgrid points using PCA
    meshgrid_points_pca = pca.inverse_transform(meshgrid_points)
    
    # Scale the meshgrid points using the same scaler used for training
    meshgrid_points_scaled = StandardScaler().fit(X_train).transform(meshgrid_points_pca)
    
    # Predict using the model (trained on the original feature space)
    Z = model.predict(meshgrid_points_scaled)
    Z = Z.reshape(xx.shape)

    # Plot the decision boundaries
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)


    # Plot the test points (in the reduced 2D space)
    scatter =plt.scatter(X_test_reduced[:, 0], X_test_reduced[:, 1], c=y_test, marker='o', edgecolor='k', cmap=plt.cm.coolwarm, label='', alpha=0.8)
    
    plt.legend(handles=scatter.legend_elements()[0], labels=['Ham', 'Spam'], title="Classes")

    plt.title(f"KNN Decision Boundary with k={model.n_neighbors}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()





def main():
    """Main function to load data, train the K-NN model, and evaluate it."""
    logging.info("Loading data...")
    X, y = load_and_prepare_data()
    logging.info("Training K-NN model...")
    train_and_evaluate_model(X, y)
    

#%% MAIN CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here
#%% SELF-RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    logging.info(f'"{module_name}" module begins.')
    main()
