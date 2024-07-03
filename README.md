# Machine Learning Model Evaluation Project

## Description

This project aims to train and evaluate several text classification models using preprocessed data. The trained models include:

- Logistic Regression
- Support Vector Machine (SVM)
- Multinomial Naive Bayes

Evaluation results include performance metrics such as accuracy, recall, precision, and F1-score. It also includes confusion matrices and comparative charts.

## Project Structure
├── data
│ ├── preprocessed_train_dataset.parquet
│ ├── preprocessed_test_dataset.parquet
| └── old_train.jsonl
├── models
│ ├── logistic_regression_model.joblib
│ ├── support_vector_machine_model.joblib
│ ├── multinomialNB_model.joblib
│ └── tfidf_vectorizer.joblib
├── results
│ ├── lg_model.txt
│ ├── svm_model.txt
│ ├── mnb_model.txt
│ ├── lg_model_conf_matrix.png
│ ├── svm_model_conf_matrix.png
│ ├── mnb_model_conf_matrix.png
│ └── metrics_comparison.png
├── src
│ ├── train.py
| ├── preprocess.py
│ └── evaluation.py
├── README.md
└── requirements.tx

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


# autotextification_nlp
# autotextification_nlp