import numpy as np
import pandas as pd
import os
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import pickle
import yaml
from src.logger import logging
from src.utility.utils import load_params, load_data


def preprocess_dataframe(train_data: pd.DataFrame, test_data: pd.DataFrame, fit_encoders: bool = True, encoders_dict: dict = None):
    """
    Preprocess the insurance dataframe (structured data).
    
    Args:
        train_data (pd.DataFrame): Training DataFrame to preprocess
        test_data (pd.DataFrame): Test DataFrame to preprocess
        fit_encoders (bool): Whether to fit new encoders or use existing ones
        encoders_dict (dict): Dictionary containing pre-fitted encoders
        
    Returns:
        tuple: (train_processed, test_processed, encoders_dict)
    """
    try:
        logging.info("Starting data preprocessing...")
        
        # Create copies to avoid modifying originals
        train_processed = train_data.copy()
        test_processed = test_data.copy()
        
        # 1. Handle missing values if any
        if train_processed.isnull().sum().any() or test_processed.isnull().sum().any():
            logging.info("Handling missing values...")
            # Fill numeric columns with median (more robust than mean)
            numeric_cols = train_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in train_processed.columns:
                    median_val = train_processed[col].median()
                    train_processed[col].fillna(median_val, inplace=True)
                    test_processed[col].fillna(median_val, inplace=True)
        
        # 2. Encode categorical variables
        logging.info("Encoding categorical variables...")
        
        # Initialize or use existing encoders
        if fit_encoders:
            label_encoders = {}
            
            # Gender encoding (simple mapping)
            gender_map = {'Female': 0, 'Male': 1}
            train_processed['Gender'] = train_processed['Gender'].map(gender_map)
            test_processed['Gender'] = test_processed['Gender'].map(gender_map)
            
            # Vehicle_Damage encoding
            damage_map = {'No': 0, 'Yes': 1}
            train_processed['Vehicle_Damage'] = train_processed['Vehicle_Damage'].map(damage_map)
            test_processed['Vehicle_Damage'] = test_processed['Vehicle_Damage'].map(damage_map)
            
            # Vehicle_Age - create dummy variables
            # Get dummies for train
            vehicle_age_dummies_train = pd.get_dummies(train_processed['Vehicle_Age'], prefix='Vehicle_Age', drop_first=True)
            train_processed = pd.concat([train_processed, vehicle_age_dummies_train], axis=1)
            
            # Get dummies for test (align with train)
            vehicle_age_dummies_test = pd.get_dummies(test_processed['Vehicle_Age'], prefix='Vehicle_Age', drop_first=True)
            # Ensure test has same columns as train
            for col in vehicle_age_dummies_train.columns:
                if col not in vehicle_age_dummies_test.columns:
                    vehicle_age_dummies_test[col] = 0
            
            test_processed = pd.concat([test_processed, vehicle_age_dummies_test], axis=1)
            
            # Drop original Vehicle_Age column
            train_processed = train_processed.drop('Vehicle_Age', axis=1)
            test_processed = test_processed.drop('Vehicle_Age', axis=1)
            
            # Save mappings
            encoders_dict = {
                'gender_map': gender_map,
                'damage_map': damage_map,
                'vehicle_age_columns': list(vehicle_age_dummies_train.columns)
            }
        else:
            # Use provided encoders
            if 'gender_map' in encoders_dict:
                train_processed['Gender'] = train_processed['Gender'].map(encoders_dict['gender_map'])
                test_processed['Gender'] = test_processed['Gender'].map(encoders_dict['gender_map'])
            
            if 'damage_map' in encoders_dict:
                train_processed['Vehicle_Damage'] = train_processed['Vehicle_Damage'].map(encoders_dict['damage_map'])
                test_processed['Vehicle_Damage'] = test_processed['Vehicle_Damage'].map(encoders_dict['damage_map'])
            
            # Handle Vehicle_Age dummies
            if 'vehicle_age_columns' in encoders_dict:
                # Get dummies and align
                vehicle_age_dummies_train = pd.get_dummies(train_processed['Vehicle_Age'], prefix='Vehicle_Age', drop_first=True)
                vehicle_age_dummies_test = pd.get_dummies(test_processed['Vehicle_Age'], prefix='Vehicle_Age', drop_first=True)
                
                # Ensure all expected columns exist
                for col in encoders_dict['vehicle_age_columns']:
                    if col not in vehicle_age_dummies_train.columns:
                        vehicle_age_dummies_train[col] = 0
                    if col not in vehicle_age_dummies_test.columns:
                        vehicle_age_dummies_test[col] = 0
                
                train_processed = pd.concat([train_processed, vehicle_age_dummies_train], axis=1)
                test_processed = pd.concat([test_processed, vehicle_age_dummies_test], axis=1)
                
                # Drop original
                train_processed = train_processed.drop('Vehicle_Age', axis=1)
                test_processed = test_processed.drop('Vehicle_Age', axis=1)
        
        # 3. Drop unnecessary columns
        columns_to_drop = ['id']  # ID is not useful for modeling
        for col in columns_to_drop:
            if col in train_processed.columns:
                train_processed = train_processed.drop(col, axis=1)
            if col in test_processed.columns:
                test_processed = test_processed.drop(col, axis=1)
        
        logging.info(f"Preprocessing complete. Train shape: {train_processed.shape}, Test shape: {test_processed.shape}")
        
        if fit_encoders:
            return train_processed, test_processed, encoders_dict
        else:
            return train_processed, test_processed
        
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        os.makedirs(interim_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)
        logging.info('Preprocessed data saved to %s', interim_data_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        # Load raw data
        train_data = load_data('./data/raw/train.csv')
        test_data = load_data('./data/raw/test.csv')
        logging.info(f'Train data loaded: {train_data.shape}, Test data loaded: {test_data.shape}')
        
        # Process data
        train_processed, test_processed, encoders_dict = preprocess_dataframe(train_data, test_data, fit_encoders=True)
        
        # Save encoders for future use
        models_dir = './models'
        os.makedirs(models_dir, exist_ok=True)
        with open(os.path.join(models_dir, 'preprocessing_encoders.pkl'), 'wb') as f:
            pickle.dump(encoders_dict, f)
        logging.info('Preprocessing encoders saved to models/preprocessing_encoders.pkl')
        
        # Save processed data
        save_data(train_processed, test_processed, data_path='./data')
        
    except Exception as e:
        logging.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()