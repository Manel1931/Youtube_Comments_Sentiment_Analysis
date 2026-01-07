# =============================================================================
# register_model.py
# =============================================================================
# Script pour enregistrer un modèle dans MLflow Model Registry.
# Étapes principales :
# - Charger les informations du modèle depuis un JSON
# - Enregistrer le modèle dans MLflow
# - Passer le modèle à l'étape spécifiée (Staging ou Production)
# - Sauvegarder un log local des enregistrements
# =============================================================================

import json
import mlflow
import logging
import os
from mlflow.exceptions import MlflowException

# -----------------------------
# Configuration MLflow
# -----------------------------
# Définition de l'URI du serveur MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# -----------------------------
# Configuration du logging
# -----------------------------
logger = logging.getLogger('model_registration')
logger.setLevel(logging.DEBUG)

# Affichage des logs sur la console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Enregistrement des erreurs critiques dans un fichier
file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel(logging.ERROR)

# Format standard des messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Ajout des handlers au logger principal
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# -----------------------------
# Fonctions utilitaires
# -----------------------------
def load_model_info(file_path: str) -> dict:
    """
    Chargement sécurisé des informations du modèle depuis un fichier JSON.

    Args:
        file_path (str): Chemin vers le fichier JSON contenant 'run_id' et 'model_path'.

    Returns:
        dict: Dictionnaire avec les informations du modèle.
    """
    if not os.path.exists(file_path):
        logger.error(f'File not found: {file_path}')
        raise FileNotFoundError(f"{file_path} does not exist.")
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.info(f'Model info loaded successfully from {file_path}')
        return model_info
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading model info: {e}")
        raise


def register_model(model_name: str, model_info: dict, stage: str = "Staging"):
    """
    Enregistre un modèle dans MLflow Model Registry et le passe à l'étape spécifiée.

    Args:
        model_name (str): Nom du modèle dans MLflow.
        model_info (dict): Dictionnaire contenant 'run_id' et 'model_path'.
        stage (str): Étape MLflow souhaitée ("Staging" ou "Production").
    """
    try:
        # Construction de l'URI du modèle pour MLflow
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        model_version = mlflow.register_model(model_uri, model_name)

        # Passage du modèle à l'étape spécifiée
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage=stage
        )
        logger.info(f'Model "{model_name}" version {model_version.version} registered and moved to {stage}.')

        # Enregistrement local des informations d'enregistrement pour traçabilité
        log_path = 'registered_models_log.json'
        log_entry = {
            "model_name": model_name,
            "version": model_version.version,
            "stage": stage,
            "run_id": model_info['run_id']
        }
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(log_entry)
        with open(log_path, 'w') as f:
            json.dump(logs, f, indent=4)
        logger.debug(f"Registration info saved locally to {log_path}")

    except MlflowException as e:
        logger.error(f"MLflow error during registration: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during model registration: {e}")
        raise


# -----------------------------
# Fonction principale
# -----------------------------
def main():
    """
    Orchestration principale pour l'enregistrement d'un modèle :
    - Charger le JSON avec les infos du modèle
    - Enregistrer le modèle dans MLflow et le passer en Staging
    """
    try:
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "yt_chrome_plugin_model"
        register_model(model_name, model_info, stage="Staging")
    except Exception as e:
        logger.error(f"Failed to complete the model registration process: {e}")
        print(f"Error: {e}")


# -----------------------------
# Exécution principale
# -----------------------------
if __name__ == '__main__':
    main()
