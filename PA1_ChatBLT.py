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
Date Last Updated: 03-27-2025
Doc:
This module loads email data, preprocesses it, and extracts features.
'''

#%% IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    import os

import pandas as pd
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import minmax_scale

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
    plt.show()   
#

#%% MAIN CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    '''
    Main function to load emails and preprocess them into structured data.
    '''
    print("Loading emails...")
    load_emails()

    df = pd.DataFrame(email_data, columns=["SAMPLE ID", "TARGET", "TEXT"])
    df.to_csv("data_raw.csv", index=False)
    
    plot_text(df, "RAW")
    
    # Preprocess text
    vectorizer = CountVectorizer(max_features=1000)
    df["TEXT"] = df["TEXT"].apply(preprocess_text)
    df.to_csv("data_preprocessed.csv", index=False)
    print("Preprocessing complete. Data saved as 'data_preprocessed.csv'.")
    
    plot_text(df, "PREPROCESSED")
    plot_text(df, "PREPROCESSED_MINMAX")
#

#%% SELF-RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    print(f"\"{module_name}\" module begins.")
    
    main()