# =============================================================================
# model_evaluation.py
# =============================================================================
# Évaluation du modèle LightGBM sur les commentaires YouTube.
# Inclut :
# - Chargement des données et du modèle
# - Transformation TF-IDF
# - Évaluation et logging (classification report + matrice de confusion)
# - Sauvegarde des artefacts et informations du run MLflow
# =============================================================================

import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature

# -----------------------------
# Configuration logging
# -----------------------------
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# -----------------------------
# Fonctions utilitaires
# -----------------------------
def safe_load_csv(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        logger.error(f"CSV file not found: {file_path}")
        raise FileNotFoundError(file_path)
    df = pd.read_csv(file_path)
    df.fillna('', inplace=True)
    logger.debug(f"Data loaded from {file_path}, shape: {df.shape}")
    return df

def safe_load_pickle(file_path: str):
    if not os.path.exists(file_path):
        logger.error(f"Pickle file not found: {file_path}")
        raise FileNotFoundError(file_path)
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    logger.debug(f"Pickle loaded from {file_path}")
    return obj

def safe_load_yaml(file_path: str) -> dict:
    if not os.path.exists(file_path):
        logger.error(f"YAML file not found: {file_path}")
        raise FileNotFoundError(file_path)
    with open(file_path, 'r') as f:
        params = yaml.safe_load(f)
    logger.debug(f"Parameters loaded from {file_path}")
    return params

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    logger.info("Model evaluation done")
    return report, cm

def log_confusion_matrix(cm, dataset_name):
    os.makedirs('reports', exist_ok=True)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_path = os.path.join('reports', f'confusion_matrix_{dataset_name}.png')
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)
    logger.info(f"Confusion matrix saved at {cm_path}")

def save_model_info(run_id: str, model_path: str, file_path: str):
    model_info = {"run_id": run_id, "model_path": model_path}
    with open(file_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    logger.debug(f"Model info saved to {file_path}")

# -----------------------------
# Fonction principale
# -----------------------------
def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment('dvc-pipeline-runs')

    with mlflow.start_run() as run:
        try:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

            # Charger paramètres
            params = safe_load_yaml(os.path.join(root_dir, 'params.yaml'))
            for key, val in params.items():
                mlflow.log_param(key, val)

            # Charger modèle et TF-IDF
            model = safe_load_pickle(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = safe_load_pickle(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Charger données test
            test_data = safe_load_csv(os.path.join(root_dir, 'data/interim/test_processed.csv'))
            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            # Signature et log modèle MLflow
            input_example = pd.DataFrame(X_test_tfidf.toarray()[:5], columns=vectorizer.get_feature_names_out())
            signature = infer_signature(input_example, model.predict(X_test_tfidf[:5]))
            mlflow.sklearn.log_model(model, "lgbm_model", signature=signature, input_example=input_example)
            mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Sauvegarder info run
            save_model_info(run.info.run_id, "lgbm_model", 'experiment_info.json')

            # Évaluation modèle
            report, cm = evaluate_model(model, X_test_tfidf, y_test)
            log_confusion_matrix(cm, "Test_Data")

            # Log métriques
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })

            # Tags MLflow
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

            logger.info("Model evaluation completed successfully!")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
