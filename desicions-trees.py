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
import re
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score

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
    
    #Begin Desicion Tree Classifier
    dtree = tree.DecisionTreeClassifier(random_state=1)
    dtree = dtree.fit(X_Train, Y_Train)
    
    y_pred = dtree.predict(X_Test)
    print("Accuracy: ", metrics.accuracy_score(Y_Test, y_pred))
        
    plot_pred(X_Test, y_pred, Y_Test, 'Decision Tree')
    #Plot Tree
    '''plt.figure(figsize=(25, 20))
    tree.plot_tree(dtree, feature_names=col_labels, filled=True)
    plt.savefig("tree.png", dpi=1000)'''
    
    plotPerformanceScores(Y_Test, y_pred, 'Decision Tree')
    plotROC(Y_Test, dtree.predict_proba(X_Test)[:, 1], 'Decision Tree')
#

#%% SELF-RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    print(f"\"{module_name}\" module begins.")
    
    main()