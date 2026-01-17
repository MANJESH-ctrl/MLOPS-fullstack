import pandas as pd
import logging
import yaml
from src.logger import logging
from sklearn.model_selection import train_test_split


# data sampling utility
def simple_sample_data(source_path, target_path, sample_frac):
    """
    Simple function to sample data and save it.
    
    source_path: where your original data.csv is
    target_path: where to save sampled data (e.g., 'locals3/sampled.csv')
    sample_frac: what percent to sample (0.1 = 10%)
    """
    # Load data
    df = pd.read_csv(source_path)
    
    # Sample it (stratified by target column if exists)
    if 'Response' in df.columns:
        X = df.drop('Response', axis=1)
        y = df['Response']
        X_sample, _, y_sample, _ = train_test_split(
            X, y, 
            train_size=sample_frac, 
            random_state=42, 
            stratify=y
        )
        df_sample = pd.concat([X_sample, y_sample], axis=1)
    else:
        df_sample = df.sample(frac=sample_frac, random_state=42)
    
    # Save it
    df_sample.to_csv(target_path, index=False)
    
    print(f"Original: {len(df):,} rows")
    print(f"Sampled: {len(df_sample):,} rows ({sample_frac*100}%)")
    print(f"Saved to: {target_path}")
    
    return df_sample



df_sample = simple_sample_data('references/data.csv', 'locals3/sampled.csv', 0.01)


# parameters loading utility
def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise


# data loading utility
def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logging.info('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise