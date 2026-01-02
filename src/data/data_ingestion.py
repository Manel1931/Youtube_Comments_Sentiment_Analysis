import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

# =========================
# Logging configuration
# =========================
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('data_ingestion_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# =========================
# Functions
# =========================

def load_params(params_path: str) -> dict:
    """
    Load parameters from a YAML file.
    Returns a dictionary of parameters.
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
    Load data from a CSV URL or local path.
    """
    try:
        df = pd.read_csv(data_url)
        logger.debug(f'Data loaded from {data_url} with shape {df.shape}')
        return df
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse the CSV file: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error occurred while loading the data: {e}')
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataframe:
    - Remove missing values
    - Remove duplicates
    - Remove rows with empty 'clean_comment'
    """
    try:
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df = df[df['clean_comment'].str.strip() != '']
        
        logger.debug(f'Data preprocessing completed. Final shape: {df.shape}')
        return df
    except KeyError as e:
        logger.error(f'Missing column in the dataframe: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error during preprocessing: {e}')
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """
    Save train and test data into 'data/raw' folder.
    Creates the folder if it doesn't exist.
    """
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        
        logger.debug(f'Train and test data saved to {raw_data_path}')
    except Exception as e:
        logger.error(f'Unexpected error occurred while saving the data: {e}')
        raise

# =========================
# Main execution
# =========================

def main():
    try:
        # Load parameters
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        test_size = params['data_ingestion']['test_size']
        
        # Load raw data
        df = load_data(
            data_url='https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv'
        )
        
        # Preprocess
        df_clean = preprocess_data(df)
        
        # Split into train and test sets
        train_data, test_data = train_test_split(df_clean, test_size=test_size, random_state=42)
        
        # Save the datasets
        save_data(train_data, test_data, os.path.join(root_dir, 'data'))
        
        logger.info('Data ingestion process completed successfully.')
    except Exception as e:
        logger.error(f'Failed to complete the data ingestion process: {e}')
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
