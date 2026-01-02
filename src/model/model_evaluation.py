# =============================================================================
# model_evaluation.py
# =============================================================================
# Ce script permet d'évaluer le modèle LightGBM entraîné sur les commentaires
# YouTube. Il inclut :
# - le chargement des données et du modèle,
# - la transformation TF-IDF des données de test,
# - l'évaluation du modèle (classification report et matrice de confusion),
# - la journalisation (logging) et l'enregistrement des artefacts avec MLflow.
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
# Configuration du logging
# -----------------------------
# Crée un logger dédié pour ce script
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')  # Niveau DEBUG pour tout tracer

# Logging console (infos)
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# Logging fichier (erreurs)
file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

# Format des messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Ajout des handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# -----------------------------
# Fonctions utilitaires
# -----------------------------

def load_data(file_path: str) -> pd.DataFrame:
    """
    Charge un fichier CSV et remplace les NaN par des chaînes vides.
    """
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise


def load_model(model_path: str):
    """
    Charge le modèle entraîné à partir d'un fichier pickle.
    """
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """
    Charge le vectorizer TF-IDF sauvegardé.
    """
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise


def load_params(params_path: str) -> dict:
    """
    Charge les paramètres du fichier YAML.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """
    Évalue le modèle : classification report et matrice de confusion.
    """
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        logger.debug('Model evaluation completed')
        return report, cm
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def log_confusion_matrix(cm, dataset_name):
    """
    Crée et enregistre la matrice de confusion, puis log l'image dans MLflow.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    cm_file_path = f'confusion_matrix_{dataset_name}.png'
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """
    Sauvegarde les informations du modèle (run_id et chemin) dans un fichier JSON.
    """
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


# -----------------------------
# Fonction principale
# -----------------------------
def main():
    # Configuration MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment('dvc-pipeline-runs')
    
    with mlflow.start_run() as run:
        try:
            # Déterminer le répertoire racine
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

            # Charger les paramètres
            params = load_params(os.path.join(root_dir, 'params.yaml'))

            # Log des paramètres
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # Charger modèle et vectorizer
            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Charger les données de test
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))

            # Transformation TF-IDF
            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            # Création d'un exemple pour signature MLflow
            input_example = pd.DataFrame(X_test_tfidf.toarray()[:5], columns=vectorizer.get_feature_names_out())
            signature = infer_signature(input_example, model.predict(X_test_tfidf[:5]))

            # Log modèle avec signature et exemple
            mlflow.sklearn.log_model(model, "lgbm_model", signature=signature, input_example=input_example)

            # Sauvegarder infos run
            model_path = "lgbm_model"
            save_model_info(run.info.run_id, model_path, 'experiment_info.json')

            # Log du vectorizer
            mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Évaluer le modèle
            report, cm = evaluate_model(model, X_test_tfidf, y_test)

            # Log des métriques
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })

            # Log matrice de confusion
            log_confusion_matrix(cm, "Test Data")

            # Tags MLflow
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")


# -----------------------------
# Lancer la fonction principale
# -----------------------------
if __name__ == '__main__':
    main()
