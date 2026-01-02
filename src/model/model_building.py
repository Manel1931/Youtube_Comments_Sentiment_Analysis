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
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# -------------------------------
# Utility functions
# -------------------------------
def get_root_directory() -> str:
    """Return the project root directory (two levels up from this script)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))

def load_params(params_path: str) -> dict:
    """Load hyperparameters from a YAML file."""
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
    """Load preprocessed CSV data and fill missing values."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug(f'Data loaded from {file_path} (NaNs filled)')
        return df
    except Exception as e:
        logger.error(f'Error loading data from {file_path}: {e}')
        raise

# -------------------------------
# Feature engineering
# -------------------------------
def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    """
    Transform text data into TF-IDF features.
    Saves the vectorizer for future use.
    """
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values

        # Fit and transform the training data
        X_train_tfidf = vectorizer.fit_transform(X_train)
        logger.debug(f'TF-IDF transformation complete. Shape: {X_train_tfidf.shape}')

        # Save the vectorizer to the root directory
        vectorizer_path = os.path.join(get_root_directory(), 'tfidf_vectorizer.pkl')
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        logger.debug(f'TF-IDF vectorizer saved at {vectorizer_path}')

        return X_train_tfidf, y_train
    except Exception as e:
        logger.error(f'Error during TF-IDF transformation: {e}')
        raise

# -------------------------------
# Model training
# -------------------------------
def train_lgbm(X_train: np.ndarray, y_train: np.ndarray, learning_rate: float, max_depth: int, n_estimators: int) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier with given hyperparameters."""
    try:
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
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
    """Persist the trained model to disk."""
    try:
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
    try:
        root_dir = get_root_directory()
        params = load_params(os.path.join(root_dir, 'params.yaml'))

        # Extract model hyperparameters
        max_features = params['model_building']['max_features']
        ngram_range = tuple(params['model_building']['ngram_range'])
        learning_rate = params['model_building']['learning_rate']
        max_depth = params['model_building']['max_depth']
        n_estimators = params['model_building']['n_estimators']

        # Load preprocessed training data
        train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))

        # TF-IDF feature engineering
        X_train_tfidf, y_train = apply_tfidf(train_data, max_features, ngram_range)

        # Train LightGBM model
        model = train_lgbm(X_train_tfidf, y_train, learning_rate, max_depth, n_estimators)

        # Save trained model
        save_model(model, os.path.join(root_dir, 'lgbm_model.pkl'))

        logger.info('Model building process completed successfully!')

    except Exception as e:
        logger.error(f'Model building failed: {e}')
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
