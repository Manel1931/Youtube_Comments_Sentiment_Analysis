import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# =========================
# Logging configuration
# =========================
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# =========================
# NLTK resources
# =========================
nltk.download('wordnet')
nltk.download('stopwords')

# =========================
# Functions
# =========================

def preprocess_comment(comment):
    """
    Apply preprocessing transformations to a comment:
    - Lowercasing
    - Strip whitespaces
    - Remove newlines
    - Keep only alphanumeric and basic punctuation
    - Remove stopwords (except key sentiment words)
    - Lemmatization
    """
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Keep important sentiment words
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return comment  # Return original comment if preprocessing fails

def normalize_text(df):
    """
    Apply preprocessing to the 'clean_comment' column of a dataframe.
    """
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """
    Save processed train and test datasets into data/interim.
    """
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        os.makedirs(interim_data_path, exist_ok=True)
        logger.debug(f"Directory {interim_data_path} ready")

        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)

        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

# =========================
# Main execution
# =========================

def main():
    try:
        logger.debug("Starting data preprocessing...")

        # Load raw datasets
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Raw data loaded successfully')

        # Apply preprocessing
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Save processed datasets
        save_data(train_processed_data, test_processed_data, data_path='./data')

        logger.info("Data preprocessing completed successfully.")

    except Exception as e:
        logger.error(f'Failed to complete the data preprocessing process: {e}')
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
