#%% MODULE BEGINS
module_name = 'spam_detection'

'''
Version: v1.0

Description:
    Uses various machine learning algorithms to detect spam emails.

Authors:
    Rhett Hill, Zachary Gros

Date Created     :  02-26-2025
Date Last Updated:  05-04-2025
'''

#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
   import os
   #os.chdir("./../..")
#

import logging
import pandas as pd
import numpy as np
import re
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import minmax_scale, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.svm import SVC

#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DATA_PATH = "INPUT"

#%% CONFIGURATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#%% INITIALIZATIONS             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    
    df = pd.DataFrame(email_data, columns=["SAMPLE ID", "TARGET", "TEXT"])
    df.to_csv("INPUT/Data.csv", index=False)
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

def prepare_data():
    #Loads and splits data into Training, Validation, and Testing
    logging.info("Starting to load and prepare data...")
    DATA_CSV = "OUTPUT/data_features.csv"
    try:
        df = pd.read_csv(DATA_CSV)
        X = df.drop(["SAMPLE ID", "TARGET"], axis=1)
        y = df["TARGET"].map({"ham": 0, "spam": 1})
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split data into training (60%), validation (20%), and test (20%) sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        logging.info("Data loaded and preprocessed successfully.")
        return X_train, y_train, X_val, y_val, X_test, y_test
    #
    except FileNotFoundError:
        logging.error(f"Error: The file {DATA_CSV} was not found.")
        exit(1)
    #
    except Exception as e:
        logging.error(f"Unexpected error loading data: {e}")
        exit(1)
    #
#

def plot_text(X, t):
    x = []
    y = []
    
    for i in X["TEXT"]:
        x.append(len(i))
        y.append(len(i.split()))
    #
    
    if t == "PROCESSED_MINMAX":
        x = minmax_scale(x)
        y = minmax_scale(y)
    #
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x, y=y, hue=X["TARGET"], alpha=0.7)
    plt.title("Comparison of String Length to Word Count ("+t+")")
    plt.xlabel("String Length")
    plt.ylabel("Word Count")
    plt.legend(title="Class")
    plt.savefig(f"OUTPUT/{t}.png")
    plt.close()
#

def plot_pca(X, y, s=None):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y,  alpha=0.7)
    plt.title("PCA Projection of Emails")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Class")
    if s == None:
        plt.savefig(f"OUTPUT/processed_pca.png")
    #
    else:
        plt.savefig(f"OUTPUT/processed_pca{s}.png")
    #
    plt.close()
#

def plot_pred(X, y_p, y_t, type):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    df = pd.DataFrame({'x': X_pca[:, 0], 'y': X_pca[:, 1], 'LABEL': y_p, 'TARGET': y_t})
    
    plt.figure(figsize=(8, 6))
    plt.scatter(df[(df['LABEL'] == 0) & (df['TARGET'] == 0)]['x'], df[(df['LABEL'] == 0) & (df['TARGET'] == 0)]['y'], color='blue', marker='o', label='True Ham', alpha=0.6)
    plt.scatter(df[(df['LABEL'] == 0) & (df['TARGET'] == 1)]['x'], df[(df['LABEL'] == 0) & (df['TARGET'] == 1)]['y'], color='blue', marker='^', label='False Ham', alpha=0.6)

    plt.scatter(df[(df['LABEL'] == 1) & (df['TARGET'] == 0)]['x'], df[(df['LABEL'] == 1) & (df['TARGET'] == 0)]['y'], color='orange', marker='^', label='False Spam', alpha=0.6)
    plt.scatter(df[(df['LABEL'] == 1) & (df['TARGET'] == 1)]['x'], df[(df['LABEL'] == 1) & (df['TARGET'] == 1)]['y'], color='orange', marker='o', label='True Spam', alpha=0.6)
    plt.legend(title="Class")
    plt.title(f"Predicted Classes ({type})")
    plt.savefig(f"OUTPUT/plot_{type}.png")
    plt.close()
#

def splitData(features, target, ratio=(0.6, 0.2, 0.2)):
    trRatio, vRatio, tsRatio = ratio
    
    X_Train, X_Temp, Y_Train, Y_Temp = train_test_split(features, target, test_size=(1 - trRatio), random_state=1)
    X_Val, X_Test, Y_Val, Y_Test = train_test_split(X_Temp, Y_Temp, test_size=tsRatio, random_state=1)

    return X_Train, Y_Train, X_Val, Y_Val, X_Test, Y_Test
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
    cm = confusion_matrix(Y_Test, Y_pred, labels=[1, 0])
    sensitivity, specificity, accuracy, f1 = calculatePerformanceScores(cm)
    
    scores = {'Accuracy': round(accuracy, 4), 'Recall (Sensitivity)': round(sensitivity, 4), 'Specificity': round(specificity, 4), 'F1-Score': round(f1, 4)}
    
    def addLabels(x, y):
        for i in range(len(x)):
            plt.text(i, y[i] + .01, y[i])
        #
    #
    
    plt.figure(figsize=(12, 8))
    plt.bar(scores.keys(), scores.values())
    plt.title(f"Performance Scores for {type}")
    addLabels(scores.keys(), list(scores.values()))
    plt.savefig(f"OUTPUT/performance_scores_{type}.png")
    plt.close()
    
    m = ConfusionMatrixDisplay(cm, display_labels=['ham', 'spam'])
    m.plot()
    plt.title(f"Confusion Matrix for {type}")
    plt.savefig(f"OUTPUT/confusion_matrix_{type}.png")
    plt.close()
#

def plotROC(y_test, y_prob, type):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='orange', label=f'ROC curve (AUC = {roc_auc})')
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"Receiver Operating Characteristic (ROC) Curve for {type}")
    plt.legend()
    plt.savefig(f"OUTPUT/ROC_AUC_{type}.png")
    plt.close()
#

def error_curve(X_train, y_train, X_val, y_val, type):
    ac_scores = []
    cv_scores = []
    if type == 'KNN':
        for k in range (1, 21, 1):
            knn = KNeighborsClassifier(k)
            knn.fit(X_train, y_train)
            p = knn.predict(X_val)
            acc = accuracy_score(y_val, p)
            ac_scores.append(acc)
            score = cross_val_score(knn, X_val, y_val, cv=5)
            cv_scores.append(np.mean(score))
        #
    #
    elif type == 'DT':
        for i in range (1, 21, 1):
            dt = tree.DecisionTreeClassifier(max_depth=i)
            dt.fit(X_train, y_train)
            p = dt.predict(X_val)
            acc = accuracy_score(y_val, p)
            ac_scores.append(acc)
            score = cross_val_score(dt, X_val, y_val, cv=5)
            cv_scores.append(np.mean(score))
        #
    #
    elif type == 'SVM':
        ac_scores = cross_val_score(SVC(kernel='linear').fit(X_train, y_train), X_train, y_train, cv = 20, scoring='accuracy')
        cv_scores = cross_val_score(SVC(kernel='linear').fit(X_train, y_train), X_val, y_val, cv = 20)
    #
    plt.figure(figsize=(12, 8))
    plt.ylim(0, 1)
    plt.plot(ac_scores, label='Training Accuracy')
    plt.plot(cv_scores, label='Validation Accuracy')
    plt.plot(1 - np.array(ac_scores), label='Training Loss')
    plt.plot(1 - np.array(cv_scores), label='Validation Loss')
    plt.title(f"Training and Validation Accuracy/Loss for {type}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy/Loss")
    plt.xticks(range(1, 21, 2))
    plt.legend()
    plt.savefig(f"OUTPUT/training_history_{type}.png")
    plt.close()
#

def plot_training_history_ANN(model):
    """
    Visualizes the training and validation accuracy/loss curves.
    """
    history = model.history.history
    plt.figure(figsize=(12, 8))
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Accuracy/Loss for ANN')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy/Loss')
    plt.legend()
    plt.savefig("OUTPUT/training_history_ANN.png")
    plt.close()
#

def KNN(X_train, y_train, X_val, y_val, X_test, y_test):
    # Hyperparameter tuning for best k
    param_grid = {'n_neighbors': np.arange(1, 31, 2)}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_k = grid_search.best_params_['n_neighbors']
    logging.info(f"Best k found: {best_k} with accuracy: {grid_search.best_score_:.4f}")

    model = KNeighborsClassifier(best_k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    plot_pred(X_test, y_pred, y_test, 'KNN')
    plotPerformanceScores(y_test, y_pred, 'KNN')
    plotROC(y_test, model.predict_proba(X_test)[:, 1], 'KNN')
    error_curve(X_train, y_train, X_val, y_val, 'KNN')
#

def DT(X_train, y_train, X_val, y_val, X_test, y_test, cols):
    dtree = tree.DecisionTreeClassifier(random_state=1)
    dtree = dtree.fit(X_train, y_train)
    
    y_pred = dtree.predict(X_test)
        
    # Decision Tree Visualization
    plt.figure(figsize=(25, 20))
    tree.plot_tree(dtree, filled=True, feature_names=cols, class_names=["ham", "spam"])
    plt.savefig("OUTPUT/tree.png", dpi=350)
    plt.close()
    
    plot_pred(X_test, y_pred, y_test, 'DT')
    plotPerformanceScores(y_test, y_pred, 'DT')
    plotROC(y_test, dtree.predict_proba(X_test)[:, 1], 'DT')
    error_curve(X_train, y_train, X_val, y_val, 'DT')
#

def ANN(X_train, y_train, X_val, y_val, X_test, y_test):
    #Combine the training and validation datasets since the split is built into the model
    X_train = np.concat((X_train, X_val), axis=0)
    y_train = np.concat((y_train, y_val), axis=0)
    
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification: spam or ham
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    
    # Evaluate the model
    y_prob = model.predict(X_test)
    y_pred = (y_prob > 0.5).astype(int)  # Convert probabilities to 0 or 1
    y_pred = y_pred.flatten()
    
    plot_pred(X_test, y_pred, y_test, 'ANN')
    plotPerformanceScores(y_test, y_pred, 'ANN')
    plotROC(y_test, y_prob, 'ANN')
    plot_training_history_ANN(model)
#

def SVM(X_train, y_train, X_val, y_val, X_test, y_test):
    #We use the default C value of 1.0 since it performed great for our data
    model = SVC(kernel='linear', random_state=42, probability=True)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    plot_pred(X_test, y_pred, y_test, 'SVM')
    plotPerformanceScores(y_test, y_pred, 'SVM')
    plotROC(y_test, model.predict_proba(X_test)[:, 1], 'SVM')
    error_curve(X_train, y_train, X_val, y_val, 'SVM')
#

#Random Trials
def KNN_random_trials(X_train, y_train, X_val, y_val, X_test, y_test):
    #Choose arbitrary k values of 1, 5, 15, 25, and 50
    k_values = [1, 5, 15, 25, 50]
    n = 0
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    for i in k_values:
        model = KNeighborsClassifier(i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
        sensitivity, specificity, accuracy, f1 = calculatePerformanceScores(cm)
        scores = {'Accuracy': round(accuracy, 4), 'Recall (Sensitivity)': round(sensitivity, 4), 'Specificity': round(specificity, 4), 'F1-Score': round(f1, 4)}
        
        def addLabels(x, y):
            for j in range(len(x)):
                ax.text(j, y[j], y[j])
            #
        #
        
        ax = axes[n // 2, n % 2]
        ax.bar(scores.keys(), scores.values())
        ax.set_title(f"Performace Scores for KNN with K={i}")
        addLabels(scores.keys(), list(scores.values()))
        ax.set_ylim(0, 1)
        n = n + 1
    #
    plt.tight_layout()
    plt.savefig(f"OUTPUT/performance_scores_random_k.png")
    plt.close()
#

def DT_random_trials(X_train, y_train, X_val, y_val, X_test, y_test):
    #Choose arbitrary depth values of 25, 50, 100, 150, and 250
    depth_values = [25, 50, 100, 150, 250]
    n = 0
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    for i in depth_values:
        model = tree.DecisionTreeClassifier(max_depth=i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
        sensitivity, specificity, accuracy, f1 = calculatePerformanceScores(cm)
        scores = {'Accuracy': round(accuracy, 4), 'Recall (Sensitivity)': round(sensitivity, 4), 'Specificity': round(specificity, 4), 'F1-Score': round(f1, 4)}
        
        def addLabels(x, y):
            for j in range(len(x)):
                ax.text(j, y[j], y[j])
            #
        #
        
        ax = axes[n // 2, n % 2]
        ax.bar(scores.keys(), scores.values())
        ax.set_title(f"Performace Scores for DT with max_depth={i}")
        addLabels(scores.keys(), list(scores.values()))
        ax.set_ylim(0, 1)
        n = n + 1
    #
    plt.tight_layout()
    plt.savefig(f"OUTPUT/performance_scores_random_dt.png")
    plt.close()
#

def ANN_random_trials(X_train, y_train, X_val, y_val, X_test, y_test):
    #Choose arbitrary input units of 2, 4, 8, 16, 32 and the hidden layer input units of double the initial layer
    input_units = [2, 4, 8, 16, 32]
    n = 0
    #Combine the training and validation datasets since the split is built into the model
    X_train = np.concat((X_train, X_val), axis=0)
    y_train = np.concat((y_train, y_val), axis=0)
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    for i in input_units:
        model = Sequential()
        model.add(Dense(i, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(2*i, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))  # Binary classification: spam or ham
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
        
        # Evaluate the model
        y_prob = model.predict(X_test)
        y_pred = (y_prob > 0.5).astype(int)  # Convert probabilities to 0 or 1
        y_pred = y_pred.flatten()
        
        cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
        sensitivity, specificity, accuracy, f1 = calculatePerformanceScores(cm)
        scores = {'Accuracy': round(accuracy, 4), 'Recall (Sensitivity)': round(sensitivity, 4), 'Specificity': round(specificity, 4), 'F1-Score': round(f1, 4)}
        
        def addLabels(x, y):
            for j in range(len(x)):
                ax.text(j, y[j], y[j])
            #
        #
        
        ax = axes[n // 2, n % 2]
        ax.bar(scores.keys(), scores.values())
        ax.set_title(f"Performace Scores for ANN with input units={i}")
        addLabels(scores.keys(), list(scores.values()))
        ax.set_ylim(0, 1)
        n = n + 1
    #
    plt.tight_layout()
    plt.savefig(f"OUTPUT/performance_scores_random_ann.png")
    plt.close()
#

def SVM_random_trials(X_train, y_train, X_val, y_val, X_test, y_test):
    #Choose arbitrary C values of 0.01, 0.1, 1, 10, and 15
    c_values = [0.01, 0.1, 1, 10, 15]
    n = 0
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    for i in c_values:
        model = SVC(kernel='linear', random_state=42, C=i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
        sensitivity, specificity, accuracy, f1 = calculatePerformanceScores(cm)
        scores = {'Accuracy': round(accuracy, 4), 'Recall (Sensitivity)': round(sensitivity, 4), 'Specificity': round(specificity, 4), 'F1-Score': round(f1, 4)}
        
        def addLabels(x, y):
            for j in range(len(x)):
                ax.text(j, y[j], y[j])
            #
        #
        
        ax = axes[n // 2, n % 2]
        ax.bar(scores.keys(), scores.values())
        ax.set_title(f"Performace Scores for SVM with C={i}")
        addLabels(scores.keys(), list(scores.values()))
        ax.set_ylim(0, 1)
        n = n + 1
    #
    plt.tight_layout()
    plt.savefig(f"OUTPUT/performance_scores_random_svm.png")
    plt.close()
#

def tr_va_ts_split_visualization(X_train, X_val, X_test):
    train = pd.DataFrame(data=X_train)
    train['split'] = 'Training'
    
    val = pd.DataFrame(data=X_val)
    val['split'] = 'Validation'
    
    test = pd.DataFrame(data=X_test)
    test['split'] = 'Test'
    
    df = pd.concat([train, val, test])
    
    y = df['split']
    df = df.drop(columns='split')
    
    plot_pca(df, y, '_tr_val_ts')
    
    def addLabels(x, y):
        for i in range(len(x)):
            plt.text(i, y[i] + .01, y[i])
        #
    #
    
    dict = {'tr': round(len(X_train)/(len(X_train) + len(X_val) + len(X_test)), 4), 'val': round(len(X_val)/(len(X_train) + len(X_val) + len(X_test)), 4), 'test': round(len(X_test)/(len(X_train) + len(X_val) + len(X_test)), 4)}
    
    plt.figure(figsize=(12, 8))
    plt.bar(['Training', 'Validation', 'Test'], dict.values())
    plt.title(f"Training/Validation/Test Data Split Ratio")
    addLabels(['Training', 'Validation', 'Test'], list(dict.values()))
    plt.ylim(0, 1)
    plt.savefig(f"OUTPUT/tr_val_ts_ratio_visualization.png")
    plt.close()
#

#%% MAIN CODE                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here
def main():
    '''
    Main function to load emails and preprocess them into structured data.
    '''
    print("Loading emails...")
    if os.path.isfile("INPUT/Data.csv") == False:
        load_emails()
    #

    df = pd.read_csv("INPUT/Data.csv")

    plot_text(df, "RAW")
    plt.close()

    # Preprocess text
    vectorizer = CountVectorizer(max_features=1000)
    df["TEXT"] = df["TEXT"].apply(preprocess_text)
    df.to_csv("OUTPUT/data_preprocessed.csv", index=False)
    print("Preprocessing complete. Data saved as 'data_preprocessed.csv'.")

    plot_text(df, "PREPROCESSED")
    plt.close()
    plot_text(df, "PREPROCESSED_MINMAX")
    plt.close()
    
    # Convert text into numerical features (TF-IDF)
    print("Extracting features...")
    vectorizer = TfidfVectorizer(max_features=1000)  # Limit to 1000 most important 
    features = vectorizer.fit_transform(df["TEXT"]).toarray()
    
    feature_names = vectorizer.get_feature_names_out()
    df_features = pd.DataFrame(features, columns=feature_names)
    
    final_df = pd.concat([df[["SAMPLE ID", "TARGET"]], df_features], axis=1)
    
    final_df.to_csv("OUTPUT/data_features.csv", index=False)
    print("Preprocessing complete. Data saved as 'data_features.csv'.")
    
    #plots the data in 2d visualization
    plot_pca(features, df["TARGET"])
    plt.close()
    
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
    
    tr_va_ts_split_visualization(X_train, X_val, X_test)
    
    KNN(X_train, y_train, X_val, y_val, X_test, y_test)
    DT(X_train, y_train, X_val, y_val, X_test, y_test, list(feature_names))
    ANN(X_train, y_train, X_val, y_val, X_test, y_test)
    SVM(X_train, y_train, X_val, y_val, X_test, y_test)
    
    KNN_random_trials(X_train, y_train, X_val, y_val, X_test, y_test)
    DT_random_trials(X_train, y_train, X_val, y_val, X_test, y_test)
    ANN_random_trials(X_train, y_train, X_val, y_val, X_test, y_test)
    SVM_random_trials(X_train, y_train, X_val, y_val, X_test, y_test)
#

#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    print(f"\"{module_name}\" module begins.")
    
    #TEST Code
    main()