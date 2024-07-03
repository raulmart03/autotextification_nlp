import pandas as pd
from datasets import Dataset
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_model_and_vectorizer(model_path: str, vectorizer_path: str):
    """Loads the model and the vectorizer from the specified files."""
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("The model or the vectorizer were not found in the specified paths.")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def plot_confusion_matrix(conf_matrix, labels, title, output_path):
    """Plots the confusion matrix and saves it as an image."""
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.savefig(output_path)
    plt.close()

def evaluate_model(test_dataset: Dataset, model, vectorizer, report_path: str, plot_path: str):
    """Evaluates the model on the test dataset."""
    X_test = test_dataset["processed_text"]
    y_test = test_dataset["label"]

    # Vectorize the text data
    X_test_vec = vectorizer.transform(X_test)

    # Make predictions using the trained model
    y_pred = model.predict(X_test_vec)

    # Calculate and display evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Ensure the directory for the report path exists
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # Save the report and confusion matrix to files
    with open(report_path, 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix))

    # Plot and save the confusion matrix
    plot_confusion_matrix(conf_matrix, labels=model.classes_, title="Confusion Matrix", output_path=plot_path)

    return accuracy, precision, recall, f1

def plot_metrics_comparison(metrics, output_path):
    """Plots a bar chart comparing the accuracy, precision, recall, and f1-score of different models."""
    df_metrics = pd.DataFrame(metrics, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    df_metrics.set_index('Model', inplace=True)
    df_metrics.plot(kind='bar', figsize=(12, 8))
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.savefig(output_path)
    plt.close()

def main():
    # Define the paths to the model and the vectorizer
    model1_path = './models/logistic_regression_model.joblib'
    model2_path = './models/support_vector_machine_model.joblib'
    model3_path = './models/multinomialNB_model.joblib'
    vectorizer_path = './models/tfidf_vectorizer.joblib'

    # Load the test dataset from the Parquet file
    df_test_loaded = pd.read_parquet('./data/preprocessed_test_dataset.parquet')

    # Convert to Hugging Face Dataset if necessary
    test_dataset = Dataset.from_pandas(df_test_loaded)
    print("Test dataset loaded.")
    print(test_dataset)

    metrics = []

    # Load model 1 and the vectorizer
    model1, vectorizer = load_model_and_vectorizer(model1_path, vectorizer_path)
    print("Model 1 (Logistic regression) and vectorizer loaded successfully.")
    # Evaluate the model
    metrics.append(['Logistic Regression'] + list(evaluate_model(test_dataset, model1, vectorizer, './results/lg_model.txt', './results/lg_model_conf_matrix.png')))

    # Load model 2 and the vectorizer
    model2, vectorizer = load_model_and_vectorizer(model2_path, vectorizer_path)
    print("Model 2 (Support Vector Machines) and vectorizer loaded successfully.")
    # Evaluate the model
    metrics.append(['Support Vector Machine'] + list(evaluate_model(test_dataset, model2, vectorizer, './results/svm_model.txt', './results/svm_model_conf_matrix.png')))

    # Load model 3 and the vectorizer
    model3, vectorizer = load_model_and_vectorizer(model3_path, vectorizer_path)
    print("Model 3 (Multinomial NB) and vectorizer loaded successfully.")
    # Evaluate the model
    metrics.append(['Multinomial NB'] + list(evaluate_model(test_dataset, model3, vectorizer, './results/mnb_model.txt', './results/mnb_model_conf_matrix.png')))

    # Ensure the directory for the metrics comparison exists
    os.makedirs('./results', exist_ok=True)

    # Plot and save the comparison of metrics
    plot_metrics_comparison(metrics, './results/metrics_comparison.png')
    print("Metrics comparison chart saved in './results/metrics_comparison.png'")

if __name__ == "__main__":
    main()
