# Version: v0.1
# Date Last Updated: 02-26-2025

#%% MODULE BEGINS
module_name = 'spam_detection'
'''
Version: v0.1
Description:
Authors:
Rhett Hill, Zachary Gros
Date Created : 02-26-2025
Date Last Updated: 04-03-2025
Doc:
This module loads email data, preprocesses it, and extracts features.
'''

#%% IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    import os

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.metrics import classification_report, confusion_matrix, r2_score, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from keras.utils import to_categorical


#%% CONSTANTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DATA_PATH = "Data/enron1" #Change enron 1-6 to process different dataset

#%% INITIALIZATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
email_data = []  # To store extracted emails and labels

#%% FUNCTION DEFINITIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def load_emails():
    '''
    Reads emails from the Enron dataset and formats them into SAMPLE ID, TARGET, and raw text.
    '''
    global email_data
    sample_id = 1

    for label in ["ham", "spam"]:
        label_path = os.path.join(DATA_PATH, label)

        if os.path.exists(label_path):
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                try:
                    with open(file_path, "r", encoding="latin-1") as f:
                        email_content = f.read()
                    email_data.append([sample_id, label, email_content])
                    sample_id += 1
                #
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                #
            #
        #
    #
#

def preprocess_text(text):
    '''
    Cleans email text by removing special characters, numbers, and stopwords.
    '''
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)   # Remove numbers
    return text.strip()
#

def plot_pca(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y,  alpha=0.7)
    plt.title("PCA Projection of Emails")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Class")
    plt.show()
#

def plot_pred(X, y_p, y_t, type):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    df = pd.DataFrame({'x': X_pca[:, 0], 'y': X_pca[:, 1], 'LABEL': y_p, 'TARGET': y_t})
    
    plt.figure(figsize=(8, 6))
    plt.scatter(df[(df['LABEL'] == 'ham') & (df['TARGET'] == 'ham')]['x'], df[(df['LABEL'] == 'ham') & (df['TARGET'] == 'ham')]['y'], color='blue', marker='o', label='True Ham', alpha=0.6)
    plt.scatter(df[(df['LABEL'] == 'ham') & (df['TARGET'] == 'spam')]['x'], df[(df['LABEL'] == 'ham') & (df['TARGET'] == 'spam')]['y'], color='blue', marker='^', label='False Ham', alpha=0.6)

    plt.scatter(df[(df['LABEL'] == 'spam') & (df['TARGET'] == 'ham')]['x'], df[(df['LABEL'] == 'spam') & (df['TARGET'] == 'ham')]['y'], color='orange', marker='^', label='False Spam', alpha=0.6)
    plt.scatter(df[(df['LABEL'] == 'spam') & (df['TARGET'] == 'spam')]['x'], df[(df['LABEL'] == 'spam') & (df['TARGET'] == 'spam')]['y'], color='orange', marker='o', label='True Spam', alpha=0.6)
    plt.legend(title="Class")
    plt.title(f"Predicted Classes ({type})")
    plt.show()
#

def calculatePerformanceScores(cm):
    TN, FP, FN, TP = cm.ravel()
    
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    F1 = (2 * (precision * sensitivity)) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0
    
    return sensitivity, specificity, accuracy, F1
#

def plotPerformanceScores(Y_Test, Y_pred, type):
    cm = confusion_matrix(Y_Test, Y_pred, labels=['ham', 'spam'])
    sensitivity, specificity, accuracy, f1 = calculatePerformanceScores(cm)
    
    scores = {'Accuracy': round(accuracy, 4), 'Recall (Sensitivity)': round(sensitivity, 4), 'Specificity': round(specificity, 4), 'F1-Score': round(f1, 4)}
    
    def addLabels(x, y):
        for i in range(len(x)):
            plt.text(i, y[i] + .01, y[i])
        #
    #
    
    plt.bar(scores.keys(), scores.values())
    plt.title(f"Performance Scores for {type}")
    addLabels(scores.keys(), list(scores.values()))
    plt.show()
    
    m = ConfusionMatrixDisplay(cm, display_labels=['ham', 'spam'])
    m.plot()
    plt.show()
#

def plotROC(Y_Test, prob, type):
    map = {'ham': 1, 'spam': 0}
    Y = Y_Test.map(lambda s: map.get(s) if s in map else s)
    
    fpr, tpr, thresholds = roc_curve(Y, prob)
    roc_auc = roc_auc_score(Y, prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"Receiver Operating Characteristic (ROC) Curve for {type}")
    plt.legend(loc='lower right')
    plt.show()
#

#%% MAIN CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    df = pd.read_csv("data_features.csv")
    target = df["TARGET"]
   
    df_features = df.drop(columns=["SAMPLE ID", "TARGET"])
    col_labels = df.columns.tolist()
    
    #Split into train/testing
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(df_features, target, test_size=0.3, random_state=1)
    
    #Convert data to work with model
    X_Train = tf.convert_to_tensor(X_Train)
    
    map = {'ham': 1, 'spam': 0}
    Y_Train = Y_Train.map(lambda s: map.get(s) if s in map else s)
    
    Y_Train = tf.convert_to_tensor(Y_Train)
    
    X_Test = tf.convert_to_tensor(X_Test)

    #Create ANN Model
    model = Sequential()
    
    #Add Layers
    model.add(Dense(units=12, input_dim=1000))
    model.add(Activation('sigmoid'))
    
    model.add(Dense(units=32))
    model.add(Activation('sigmoid'))
    
    model.add(Dense(units=64))
    model.add(Activation('sigmoid'))
    
    model.add(Dense(units=1))
    model.add(Activation('sigmoid'))
    
    #Fit Model
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.fit(X_Train, Y_Train, epochs=2, batch_size=10)
    
    prob = model.predict(X_Test)
    pred = (prob > 0.5).astype(str)
    
    for i in range(len(pred)):
        if pred[i][0] == 'True':
            pred[i][0] = 'ham'
        #
        else:
            pred[i][0] = 'spam'
        #
    #
    pred = pred.flatten()

    plot_pred(X_Test, pred, Y_Test, 'ANN')
    plotPerformanceScores(Y_Test, pred, 'ANN')
    plotROC(Y_Test, prob, 'ANN')
#

#%% SELF-RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    print(f"\"{module_name}\" module begins.")
    
    main()
    
    
#EPOC vs ERROR curve
#Interpretation of plots

#Interpretation of performance scores