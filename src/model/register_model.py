# =============================================================================
# register_model.py
# =============================================================================
# Ce script permet d'enregistrer un modèle MLflow dans le Model Registry.
# Il récupère les informations du modèle depuis un fichier JSON généré après
# l'entraînement et l'évaluation, puis enregistre le modèle dans MLflow et
# le passe à l'étape "Staging".
# =============================================================================

import json
import mlflow
import logging
import os

# -----------------------------
# Configuration MLflow
# -----------------------------
mlflow.set_tracking_uri("http://localhost:5000")  # URI du serveur MLflow

# -----------------------------
# Configuration du logging
# -----------------------------
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

# Console handler pour logs infos
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# Fichier handler pour logs erreurs
file_handler = logging.FileHandler('model_registration_errors.log')
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

def load_model_info(file_path: str) -> dict:
    """
    Charge les informations du modèle depuis un fichier JSON.
    Le JSON doit contenir :
      - run_id : ID de l'exécution MLflow
      - model_path : chemin relatif du modèle dans les artefacts MLflow
    """
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise


def register_model(model_name: str, model_info: dict):
    """
    Enregistre le modèle dans le MLflow Model Registry et le passe en "Staging".
    
    Args:
        model_name (str): Nom sous lequel le modèle sera enregistré.
        model_info (dict): Dictionnaire contenant 'run_id' et 'model_path'.
    """
    try:
        # Construire l'URI du modèle à partir du run_id et du chemin
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        # Enregistrer le modèle dans MLflow
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Passer le modèle à l'étape "Staging"
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        
        logger.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise


# -----------------------------
# Fonction principale
# -----------------------------
def main():
    try:
        # Charger le fichier JSON créé par model_evaluation.py
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        # Nom sous lequel enregistrer le modèle
        model_name = "yt_chrome_plugin_model"
        # model_name = "my_model"  # Exemple si tu veux changer
        
        # Enregistrer le modèle
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")


# -----------------------------
# Exécution principale
# -----------------------------
if __name__ == '__main__':
    main()
