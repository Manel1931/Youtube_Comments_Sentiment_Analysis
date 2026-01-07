import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------
# Logging configuration
# -------------------------------
# Configuration d'un logger pour suivre les événements du script (debug, erreurs, infos)
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)  # Niveau minimal pour capturer tous les messages debug

# Handler pour afficher les logs sur la console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Handler pour enregistrer uniquement les erreurs critiques dans un fichier
file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel(logging.ERROR)

# Format standard des messages de log
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Ajout des handlers au logger principal
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# -------------------------------
# Utility functions
# -------------------------------
def get_root_directory() -> str:
    """
    Retourne le répertoire racine du projet (2 niveaux au-dessus de ce script).
    Utile pour construire des chemins de fichiers relatifs à la racine.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def load_params(params_path: str) -> dict:
    """
    Charge les hyperparamètres depuis un fichier YAML.
    Paramètres:
        - params_path: chemin vers le fichier YAML
    Retour:
        - dictionnaire contenant les paramètres
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f'Parameters successfully loaded from {params_path}')
        return params
    except FileNotFoundError:
        logger.error(f'YAML file not found: {params_path}')
        raise
    except yaml.YAMLError as e:
        logger.error(f'YAML parsing error: {e}')
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """
    Charge un fichier CSV prétraité et remplace les valeurs manquantes par une chaîne vide.
    Paramètres:
        - file_path: chemin vers le fichier CSV
    Retour:
        - DataFrame Pandas prêt pour l'entraînement
    """
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Remplacement des NaN pour éviter les erreurs lors du TF-IDF
        logger.debug(f'Data loaded from {file_path} (NaNs filled, shape: {df.shape})')
        return df
    except Exception as e:
        logger.error(f'Error loading data from {file_path}: {e}')
        raise


# -------------------------------
# Feature engineering
# -------------------------------
def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    """
    Transforme les textes en vecteurs TF-IDF.
    Sauvegarde le vectorizer pour l'utilisation future lors de l'inférence.

    Paramètres:
        - train_data: DataFrame contenant les colonnes 'clean_comment' et 'category'
        - max_features: nombre maximum de features TF-IDF
        - ngram_range: tuple (min_n, max_n) pour n-grammes
    Retour:
        - X_train_tfidf: matrice TF-IDF des textes
        - y_train: labels correspondants
    """
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values

        # Fit du TF-IDF sur les données d'entraînement et transformation
        X_train_tfidf = vectorizer.fit_transform(X_train)
        logger.debug(f'TF-IDF transformation complete. Shape: {X_train_tfidf.shape}')

        # Sauvegarde du vectorizer pour l'utiliser lors de la prédiction
        vectorizer_path = os.path.join(get_root_directory(), 'tfidf_vectorizer.pkl')
        os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        logger.debug(f'TF-IDF vectorizer saved at {vectorizer_path}')

        return X_train_tfidf, y_train
    except KeyError as e:
        logger.error(f'Missing column in training data: {e}')
        raise
    except Exception as e:
        logger.error(f'Error during TF-IDF transformation: {e}')
        raise


# -------------------------------
# Model training
# -------------------------------
def train_lgbm(X_train: np.ndarray, y_train: np.ndarray, learning_rate: float, max_depth: int,
               n_estimators: int) -> lgb.LGBMClassifier:
    """
    Entraîne un modèle LightGBM pour la classification multi-classes.
    Paramètres:
        - X_train: matrice des features TF-IDF
        - y_train: labels correspondants
        - learning_rate, max_depth, n_estimators: hyperparamètres LightGBM
    Retour:
        - Modèle LightGBM entraîné
    """
    try:
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=len(np.unique(y_train)),
            metric='multi_logloss',
            is_unbalance=True,
            class_weight='balanced',
            reg_alpha=0.1,
            reg_lambda=0.1,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
        )
        model.fit(X_train, y_train)
        logger.debug('LightGBM model training completed')
        return model
    except Exception as e:
        logger.error(f'Error during LightGBM training: {e}')
        raise


# -------------------------------
# Save model
# -------------------------------
def save_model(model, file_path: str) -> None:
    """
    Persiste le modèle entraîné sur le disque.
    Paramètres:
        - model: instance du modèle LightGBM
        - file_path: chemin complet pour sauvegarder le modèle
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug(f'Model saved successfully at {file_path}')
    except Exception as e:
        logger.error(f'Error saving model at {file_path}: {e}')
        raise


# -------------------------------
# Main execution
# -------------------------------
def main():
    """
    Fonction principale orchestrant le processus complet:
        - Chargement des paramètres
        - Chargement des données
        - Feature engineering TF-IDF
        - Entraînement du modèle LightGBM
        - Sauvegarde du modèle
    """
    try:
        root_dir = get_root_directory()
        params = load_params(os.path.join(root_dir, 'params.yaml'))

        # Extraction des hyperparamètres depuis le fichier YAML
        max_features = params['model_building']['max_features']
        ngram_range = tuple(params['model_building']['ngram_range'])
        learning_rate = params['model_building']['learning_rate']
        max_depth = params['model_building']['max_depth']
        n_estimators = params['model_building']['n_estimators']

        # Chargement des données prétraitées
        train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))

        # Transformation TF-IDF
        X_train_tfidf, y_train = apply_tfidf(train_data, max_features, ngram_range)

        # Entraînement du modèle LightGBM
        model = train_lgbm(X_train_tfidf, y_train, learning_rate, max_depth, n_estimators)

        # Sauvegarde du modèle entraîné
        save_model(model, os.path.join(root_dir, 'lgbm_model.pkl'))

        logger.info('Model building process completed successfully!')

    except Exception as e:
        logger.error(f'Model building failed: {e}')
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
