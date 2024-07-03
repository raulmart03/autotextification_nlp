# Machine Learning Model Evaluation Project

## Description

This project aims to train and evaluate several text classification models using preprocessed data. The trained models include:

- Logistic Regression
- Support Vector Machine (SVM)
- Multinomial Naive Bayes

Evaluation results include performance metrics such as accuracy, recall, precision, and F1-score. It also includes confusion matrices and comparative charts.

## Requirements

- Python 3.10
- Pandas
- scikit-learn
- datasets (Hugging Face)
- joblib
- matplotlib
- seaborn

## Usage

# Preprocess 
To preprocess, rum the preprocess.py script located in the src directory. The data preprocessing performs several key tasks to preprocess the dataset stored in a .jsonl file, tokenize the text data, and split it into training and test sets. Finally, it saves the processed data into Parquet files. 

python src/preprocess.py 

# Training Models

To train the models, run the train.py script located in the src directory. This script loads the training data, vectorizes the text, and trains the models. The models and vectorizer will be saved in the models directory.

python src/train.py 

# Evaluating Models

To evaluate the trained models, run the evaluation.py script located in the src directory. This script loads the models and the vectorizer, evaluates the model performance on the test data, and generates evaluation reports and charts, which are saved in the results directory.

python src/evaluation.py

# Donwload files
There are files missing in the folders:
The dataset (included in folder./data): https://drive.google.com/file/d/1tBUEAdSW5sPkd4d-cUE910xGpq26QUsu/view?usp=sharing
The preprocesses train dataset (included in folder ./data): https://drive.google.com/file/d/12YFSQ2wouaXWDRSJFz7fBE7gUooXGsKQ/view?usp=sharing
The tfidf vectorizer (included in ./models): https://drive.google.com/file/d/10DEO_urIJZh2lMHAq6vaF3d-ijjMg2uO/view?usp=sharing
The support vector machine model (included in ./models): https://drive.google.com/file/d/1B2ghQ2xuP9GyB1-av3zkMx2gn7AWxdPf/view?usp=sharing
