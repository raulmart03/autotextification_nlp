import pandas as pd
import re
from nltk import RegexpTokenizer
from sklearn.model_selection import train_test_split
from typing import List
import os
from transformers import BertTokenizer

# Initialization of tools
tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')

"""Preprocess the text by removing numbers, punctuation, and extra whitespace."""
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Expected a string for text preprocessing")
    text = text.lower()
    text = re.sub(r'\b\d+\b', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

"""Tokenize the text into a list of words."""
def tokenize_text(text: str) -> List[str]:
    if not isinstance(text, str):
        raise ValueError("Expected a string for text tokenization")
    return tokenizer.tokenize(text)

"""Preprocess and tokenize the text column of a DataFrame."""
def preprocess_and_tokenize(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    df['processed_text'] = df[text_column].apply(preprocess_text)
    df['processed_text'] = df['processed_text'].apply(tokenize_text)
    return df

"""Save the DataFrame to a Parquet file."""
def save_dataframe_as_parquet(df: pd.DataFrame, file_path: str):
    # Ensure the destination folder exists
    output_dir = os.path.dirname(file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_parquet(file_path, index=False)
    print(f"File saved to '{file_path}'")

def main():

    # Define the path to the .jsonl file
    file_path = './data/old_train.jsonl'

    # Read the .jsonl file using pandas
    df = pd.read_json(file_path, lines=True)

    # Verify that the 'text' and 'label' columns exist
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("The DataFrame does not contain the necessary 'text' and 'label' columns.")

    # Apply the preprocessing and tokenization function to the text column
    df = preprocess_and_tokenize(df, 'text')

    print("DataFrame after preprocessing and tokenization:")
    print(df.head())

    X = df['processed_text'].values
    y = df['label'].values

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Combine X_train and y_train into a list of dictionaries
    train_data = [{'processed_text': ' '.join(text), 'label': label} for text, label in zip(X_train, y_train)]
    test_data = [{'processed_text': ' '.join(text), 'label': label} for text, label in zip(X_test, y_test)]

    # Convert the list of dictionaries to a DataFrame
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Save the DataFrame to a Parquet file
    save_dataframe_as_parquet(train_df, './data/preprocessed_train_dataset.parquet')
    save_dataframe_as_parquet(test_df, './data/preprocessed_test_dataset.parquet')

if __name__ == "__main__":
    main()
