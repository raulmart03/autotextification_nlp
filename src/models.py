import pandas as pd
from datasets import Dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from typing import List
from sklearn.svm import SVC
import joblib
import os

# Initialization of tools
model_LR = LogisticRegression()
model_SVC = SVC(kernel='linear')
model_MNB = MultinomialNB()
vectorizer = TfidfVectorizer()


"""Trains the model with the training dataset and saves the model and the vectorizer."""
def fit(train_dataset: Dataset) -> None:
    global model_LR, model_SVC, vectorizer

    X_train = train_dataset["processed_text"]
    y_train = train_dataset["label"]

    # Vectorize the text data
    X_train_vec = vectorizer.fit_transform(X_train)

    # Train the classification model
    print("Fitting Support Vector Machine model...")
    model_SVC.fit(X_train_vec, y_train)
    print("Fitting Logistic Regression model...")
    model_LR.fit(X_train_vec, y_train)
    print("Fitting MultinomialNB model...")
    model_MNB.fit(X_train_vec, y_train)

    # Save the model and the vectorizer
    os.makedirs('./models', exist_ok=True)
    joblib.dump(model_LR, './models/logistic_regression_model.joblib')
    joblib.dump(model_SVC, './models/support_vector_machine_model.joblib')
    joblib.dump(model_MNB, './models/multinomialNB_model.joblib')
    joblib.dump(vectorizer, './models/tfidf_vectorizer.joblib')
    print("Models and vectorizer saved in './models/'")



def main():
    # Load the dataset from the Parquet file
    df_train_loaded = pd.read_parquet('./data/preprocessed_train_dataset.parquet')

    # Convert to Hugging Face Dataset if necessary
    train_dataset = Dataset.from_pandas(df_train_loaded)
    print("Training dataset loaded:")
    print(train_dataset)

    # Train the model
    fit(train_dataset)

    print("All the models were trained")

if __name__ == "__main__":
    main()
