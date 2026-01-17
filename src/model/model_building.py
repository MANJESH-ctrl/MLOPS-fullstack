#model_building.py
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from src.logger import logging
from src.utility.utils import load_data, load_params
 
params = load_params(params_path='params.yaml')
n_estimators = params['model_building']['n_estimators']
max_depth = params['model_building']['max_depth']
min_samples_split = params['model_building']['min_samples_split']
min_samples_leaf = params['model_building']['min_samples_leaf']
random_state = params['model_building']['random_state']
class_weight = params['model_building']['class_weight']
n_jobs = params['model_building']['n_jobs'] 

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train the Random Forest model."""
    try:
        # Using Random Forest with parameters for your imbalanced data
        clf = RandomForestClassifier(
            n_estimators=n_estimators,           
            max_depth= max_depth,               
            min_samples_split=min_samples_split,        
            min_samples_leaf=min_samples_leaf,         
            random_state=random_state,            
            class_weight=class_weight,    
            n_jobs=n_jobs                   
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
        train_data = load_data('./data/processed/train_final.csv')
        
        X_train = train_data.drop('Response', axis=1).values
        y_train = train_data['Response'].values
        
        
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
    main()