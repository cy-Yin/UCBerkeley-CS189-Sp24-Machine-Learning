from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import os

def load_data(folder_ham, folder_spam):
    texts = []
    labels = []
    
    # read the ham folder
    for filename in os.listdir(folder_ham):
        filepath = os.path.join(folder_ham, filename)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            texts.append(f.read())
            labels.append(0)  # ham label: 0
    
    # read the spam folder
    for filename in os.listdir(folder_spam):
        filepath = os.path.join(folder_spam, filename)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            texts.append(f.read())
            labels.append(1)  # spam label: 1
    
    return texts, labels

# Load data
train_texts, train_labels = load_data('data/ham', 'data/spam')
print("file get.")

# Construct Bag of Words features using CountVectorizer
vectorizer = CountVectorizer()  # simple bag of words without tf-idf
train_data = vectorizer.fit_transform(train_texts) # returns a sparse matrix
print("train_data get.")

# Here I use Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(train_data, train_labels)
print("fit done")

# Load test texts and filenames
def load_test_data(test_folder):
    test_texts = []
    # Sort files by their numeric filename (assuming filenames are like 1.txt, 2.txt ...)
    for filename in sorted(os.listdir(test_folder), key=lambda x: int(os.path.splitext(x)[0])):
        filepath = os.path.join(test_folder, filename)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            test_texts.append(f.read())
    return test_texts

# Load test data
test_texts = load_test_data('data/test')

# Transform test texts with the previously trained vectorizer
test_data = vectorizer.transform(test_texts)  # Use transform, not fit_transform

# Predict the test labels
pred_test_labels = clf.predict(test_data)

# Usage: results_to_csv(model.predict(X_test))
# Copied from hw3/scripts/save_csv.py
def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1 # Ensures that the index starts at 1
    df.to_csv('spam_bagofwords_submission.csv', index_label='Id')

results_to_csv(pred_test_labels)

# In Kaggle competition
# https://www.kaggle.com/competitions/cs189-hw3-spam-spring-2024/
# Private score: 0.972
# Public score:  0.980