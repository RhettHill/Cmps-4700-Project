# Version: v0.1
# Date Last Updated: 02-26-2025

#%% MODULE BEGINS
module_name = 'spam_detection'
'''
Version: v0.1
Description:
A machine learning project to classify spam emails using various ML algorithms.
Authors:
<Your Name>
Date Created : 02-26-2025
Date Last Updated: 02-26-2025
Doc:
This module loads email data, preprocesses it, and extracts features.
Notes:
Uses Enron Spam Dataset stored in "Data/enron1/" folder.
'''

#%% IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    import os

# Standard imports
import pandas as pd
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn imports for text processing and ML
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

#%% CONSTANTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DATA_PATH = "Data/enron2"  # Use only enron1 dataset

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
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

def preprocess_text(text):
    '''
    Cleans email text by removing special characters, numbers, and stopwords.
    '''
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)   # Remove numbers
    return text.strip()

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

#%% MAIN CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    '''
    Main function to load emails and preprocess them into structured data.
    '''
    print("Loading emails...")
    load_emails()

    # Convert to Pandas DataFrame
    df = pd.DataFrame(email_data, columns=["SAMPLE ID", "TARGET", "TEXT"])
    
    # Preprocess text
    vectorizer = CountVectorizer(max_features=1000)
    df["TEXT"] = df["TEXT"].apply(preprocess_text)
    
    # Convert text into numerical features (TF-IDF)
    print("Extracting features...")
    vectorizer = TfidfVectorizer(max_features=1000)  # Limit to 1000 most important 
    features = vectorizer.fit_transform(df["TEXT"]).toarray()
    
    # Create final dataset
    feature_names = vectorizer.get_feature_names_out()
    df_features = pd.DataFrame(features, columns=feature_names)
    
    # Merge with SAMPLE ID and TARGET
    final_df = pd.concat([df[["SAMPLE ID", "TARGET"]], df_features], axis=1)
    
    # Save to Excel
    final_df.to_excel("processed_emails.xlsx", index=False)
    print("Preprocessing complete. Data saved as 'processed_emails.xlsx'.")
    
    # Plot PCA projection of processed data
    plot_pca(features, df["TARGET"])

#%% SELF-RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    print(f"\"{module_name}\" module begins.")
    
    main()