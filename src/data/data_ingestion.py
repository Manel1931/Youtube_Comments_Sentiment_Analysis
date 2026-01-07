import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

# =========================
# Logging configuration
# =========================
# Logger pour suivre les événements et erreurs du script
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

# Handler console pour afficher les logs de debug et info
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Handler fichier pour enregistrer uniquement les erreurs critiques
file_handler = logging.FileHandler('data_ingestion_errors.log')
file_handler.setLevel(logging.ERROR)

# Format standard des messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Ajout des handlers au logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# =========================
# Functions
# =========================

def load_params(params_path: str) -> dict:
    """
    Chargement des paramètres depuis un fichier YAML.

    Args:
        params_path (str): Chemin vers le fichier YAML

    Returns:
        dict: Dictionnaire contenant les paramètres
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f'Parameters retrieved from {params_path}')
        return params
    except FileNotFoundError:
        logger.error(f'File not found: {params_path}')
        raise
    except yaml.YAMLError as e:
        logger.error(f'YAML error: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """
    Chargement des données depuis une URL ou un chemin local.

    Args:
        data_url (str): URL ou chemin local vers le CSV

    Returns:
        pd.DataFrame: DataFrame chargé
    """
    try:
        df = pd.read_csv(data_url)
        logger.debug(f'Data loaded from {data_url} with shape {df.shape}')
        print(f"[INFO] Dataset loaded. Number of rows: {df.shape[0]}, columns: {df.shape[1]}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse the CSV file: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error occurred while loading the data: {e}')
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prétraitement des données :
    - Suppression des valeurs manquantes
    - Suppression des doublons
    - Suppression des lignes vides dans 'clean_comment'
    - Reset de l'index

    Args:
        df (pd.DataFrame): DataFrame brut

    Returns:
        pd.DataFrame: DataFrame prétraité
    """
    try:
        initial_shape = df.shape
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df = df[df['clean_comment'].str.strip() != '']
        df.reset_index(drop=True, inplace=True)

        logger.debug(f'Data preprocessing completed. Initial shape: {initial_shape}, Final shape: {df.shape}')
        print(f"[INFO] Preprocessing done. Removed {initial_shape[0] - df.shape[0]} rows.")
        return df
    except KeyError as e:
        logger.error(f'Missing column in the dataframe: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error during preprocessing: {e}')
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """
    Sauvegarde des datasets train et test dans le dossier 'data/raw'.

    Args:
        train_data (pd.DataFrame): Jeu d'entraînement
        test_data (pd.DataFrame): Jeu de test
        data_path (str): Chemin racine pour enregistrer les données
    """
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        train_path = os.path.join(raw_data_path, "train.csv")
        test_path = os.path.join(raw_data_path, "test.csv")

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        logger.debug(f'Train and test data saved to {raw_data_path}')
        print(f"[INFO] Train data saved to {train_path}")
        print(f"[INFO] Test data saved to {test_path}")
    except Exception as e:
        logger.error(f'Unexpected error occurred while saving the data: {e}')
        raise


# =========================
# Main execution
# =========================
def main():
    """
    Orchestration complète de l'ingestion des données :
    - Chargement des paramètres
    - Chargement des données brutes
    - Prétraitement des données
    - Split train/test
    - Sauvegarde des datasets
    """
    try:
        # Détermination du répertoire racine
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

        # Charger les paramètres YAML
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        test_size = params['data_ingestion']['test_size']

        # Charger le dataset brut
        df = load_data(
            data_url='https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv'
        )

        # Prétraiter le dataset
        df_clean = preprocess_data(df)

        # Split en train et test
        train_data, test_data = train_test_split(df_clean, test_size=test_size, random_state=42)
        logger.debug(f'Train/Test split done. Train shape: {train_data.shape}, Test shape: {test_data.shape}')
        print(f"[INFO] Train/Test split completed. Train rows: {train_data.shape[0]}, Test rows: {test_data.shape[0]}")

        # Sauvegarder les datasets
        save_data(train_data, test_data, os.path.join(root_dir, 'data'))

        logger.info('Data ingestion process completed successfully.')
        print("[SUCCESS] Data ingestion completed!")

    except Exception as e:
        logger.error(f'Failed to complete the data ingestion process: {e}')
        print(f"[ERROR] {e}")


if __name__ == '__main__':
    main()
