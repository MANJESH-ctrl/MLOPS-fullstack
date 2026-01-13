#model_building.py
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from src.logger import logging


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train the Random Forest model."""
    try:
        # Using Random Forest with parameters for your imbalanced data
        clf = RandomForestClassifier(
            n_estimators=100,           # Number of trees
            max_depth=10,               # Control tree depth
            min_samples_split=5,        # Minimum samples to split
            min_samples_leaf=2,         # Minimum samples in leaf
            random_state=42,            # For reproducibility
            class_weight='balanced',    # Important for your 88%/12% imbalance
            n_jobs=-1                   # Use all CPU cores
        )
        clf.fit(X_train, y_train)
        logging.info('Random Forest model training completed')
        logging.info(f'Trained with {clf.n_estimators} trees')
        return clf
    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        # Load processed data
        train_data = load_data('./data/processed/train_final.csv')
        
        # Split into features and target
        # Method 1: If target column is named 'Response'
        X_train = train_data.drop('Response', axis=1).values
        y_train = train_data['Response'].values
        
        # Method 2: If you're not sure about column names (use as fallback)
        # X_train = train_data.iloc[:, :-1].values  # All except last column
        # y_train = train_data.iloc[:, -1].values   # Last column
        
        logging.info(f'Training data shape: X={X_train.shape}, y={y_train.shape}')
        logging.info(f'Class distribution: 0={sum(y_train==0)}, 1={sum(y_train==1)}')
        
        # Train model
        clf = train_model(X_train, y_train)
        
        # Save model
        save_model(clf, 'models/model.pkl')
        
        # Quick evaluation on training data
        train_accuracy = clf.score(X_train, y_train)
        logging.info(f'Training accuracy: {train_accuracy:.4f}')
        
        # Show feature importance (top 10)
        if hasattr(clf, 'feature_importances_'):
            feature_names = train_data.drop('Response', axis=1).columns.tolist()
            importances = clf.feature_importances_
            top_features = sorted(zip(feature_names, importances), 
                                  key=lambda x: x[1], reverse=True)[:10]
            
            logging.info('Top 10 feature importances:')
            for feature, importance in top_features:
                logging.info(f'  {feature}: {importance:.4f}')
            
            # Save feature names for reference
            with open('models/feature_names.pkl', 'wb') as f:
                pickle.dump(feature_names, f)
            logging.info(f'Saved {len(feature_names)} feature names')
        
    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    # Ensure models directory exists
    import os
    os.makedirs('models', exist_ok=True)
    
    main()