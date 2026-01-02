import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import json

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

def preprocess_comment(comment: str) -> str:
    """Preprocess a single comment for modeling."""
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return comment

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to the 'clean_comment' column."""
    try:
        if 'clean_comment' not in df.columns:
            raise KeyError("Column 'clean_comment' missing from dataframe")
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        df['comment_length'] = df['clean_comment'].apply(len)  # New feature
        logger.debug(f'Text normalization completed. Sample lengths: {df["comment_length"].head()}')
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save processed datasets and preprocessing stats."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        os.makedirs(interim_data_path, exist_ok=True)
        train_path = os.path.join(interim_data_path, "train_processed.csv")
        test_path = os.path.join(interim_data_path, "test_processed.csv")

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logger.debug(f"Processed data saved: {train_path} & {test_path}")

        # Save preprocessing statistics
        stats = {
            "train_size": len(train_data),
            "test_size": len(test_data),
            "avg_train_comment_length": train_data['comment_length'].mean(),
            "avg_test_comment_length": test_data['comment_length'].mean()
        }
        stats_path = os.path.join(interim_data_path, "preprocessing_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        logger.debug(f"Preprocessing stats saved at {stats_path}")

    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

# =========================
# Main execution
# =========================

def main():
    try:
        logger.debug("Starting data preprocessing...")

        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Raw data loaded successfully')

        train_processed = normalize_text(train_data)
        test_processed = normalize_text(test_data)

        save_data(train_processed, test_processed, data_path='./data')

        logger.info("Data preprocessing completed successfully.")
    except Exception as e:
        logger.error(f'Failed to complete the data preprocessing process: {e}')
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
