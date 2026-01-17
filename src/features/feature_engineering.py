# feature engineering
import numpy as np
import pandas as pd
import os
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
import yaml
import pickle
import src
from src.logger import logging
from src.utility.utils import load_params, load_data    

def apply_feature_scaling(train_data: pd.DataFrame, test_data: pd.DataFrame, fit_scalers: bool = True, scalers_dict: dict = None):
    """
    Apply feature scaling to the data.
    
    Args:
        train_data (pd.DataFrame): Training data
        test_data (pd.DataFrame): Test data
        fit_scalers (bool): Whether to fit new scalers or use existing ones
        scalers_dict (dict): Dictionary containing pre-fitted scalers
        
    Returns:
        tuple: (train_scaled, test_scaled, scalers_dict)
    """
    try:
        logging.info("Applying feature scaling...")
        
        # Separate features and target
        X_train = train_data.drop('Response', axis=1) if 'Response' in train_data.columns else train_data
        y_train = train_data['Response'] if 'Response' in train_data.columns else None
        X_test = test_data.drop('Response', axis=1) if 'Response' in test_data.columns else test_data
        y_test = test_data['Response'] if 'Response' in test_data.columns else None
        
        # Define which features to scale and how
        # StandardScaler for normally distributed features
        std_features = ['Age', 'Vintage'] if all(f in X_train.columns for f in ['Age', 'Vintage']) else []
        
        # MinMaxScaler for features with bounds or large ranges
        minmax_features = ['Annual_Premium'] if 'Annual_Premium' in X_train.columns else []
        
        # Other features that don't need scaling (binary, already encoded)
        other_features = [col for col in X_train.columns 
                         if col not in std_features + minmax_features]
        
        if fit_scalers:
            scalers_dict = {}
            
            # Apply StandardScaler
            if std_features:
                std_scaler = StandardScaler()
                X_train_std = std_scaler.fit_transform(X_train[std_features])
                X_test_std = std_scaler.transform(X_test[std_features])
                
                # Convert back to DataFrame
                X_train_std_df = pd.DataFrame(X_train_std, columns=std_features, index=X_train.index)
                X_test_std_df = pd.DataFrame(X_test_std, columns=std_features, index=X_test.index)
                
                scalers_dict['std_scaler'] = std_scaler
            
            # Apply MinMaxScaler
            if minmax_features:
                minmax_scaler = MinMaxScaler()
                X_train_mm = minmax_scaler.fit_transform(X_train[minmax_features])
                X_test_mm = minmax_scaler.transform(X_test[minmax_features])
                
                # Convert back to DataFrame
                X_train_mm_df = pd.DataFrame(X_train_mm, columns=minmax_features, index=X_train.index)
                X_test_mm_df = pd.DataFrame(X_test_mm, columns=minmax_features, index=X_test.index)
                
                scalers_dict['minmax_scaler'] = minmax_scaler
            
            # Combine scaled features with unscaled ones
            X_train_scaled = pd.concat([
                X_train_std_df if std_features else pd.DataFrame(),
                X_train_mm_df if minmax_features else pd.DataFrame(),
                X_train[other_features]
            ], axis=1)
            
            X_test_scaled = pd.concat([
                X_test_std_df if std_features else pd.DataFrame(),
                X_test_mm_df if minmax_features else pd.DataFrame(),
                X_test[other_features]
            ], axis=1)
            
        else:
            # Use existing scalers
            if scalers_dict and 'std_scaler' in scalers_dict and std_features:
                X_train_std = scalers_dict['std_scaler'].transform(X_train[std_features])
                X_test_std = scalers_dict['std_scaler'].transform(X_test[std_features])
                X_train_std_df = pd.DataFrame(X_train_std, columns=std_features, index=X_train.index)
                X_test_std_df = pd.DataFrame(X_test_std, columns=std_features, index=X_test.index)
            
            if scalers_dict and 'minmax_scaler' in scalers_dict and minmax_features:
                X_train_mm = scalers_dict['minmax_scaler'].transform(X_train[minmax_features])
                X_test_mm = scalers_dict['minmax_scaler'].transform(X_test[minmax_features])
                X_train_mm_df = pd.DataFrame(X_train_mm, columns=minmax_features, index=X_train.index)
                X_test_mm_df = pd.DataFrame(X_test_mm, columns=minmax_features, index=X_test.index)
            
            # Combine
            X_train_scaled = pd.concat([
                X_train_std_df if std_features and 'std_scaler' in scalers_dict else pd.DataFrame(),
                X_train_mm_df if minmax_features and 'minmax_scaler' in scalers_dict else pd.DataFrame(),
                X_train[other_features]
            ], axis=1)
            
            X_test_scaled = pd.concat([
                X_test_std_df if std_features and 'std_scaler' in scalers_dict else pd.DataFrame(),
                X_test_mm_df if minmax_features and 'minmax_scaler' in scalers_dict else pd.DataFrame(),
                X_test[other_features]
            ], axis=1)
        
        # Add target back if it exists
        if y_train is not None:
            train_scaled = pd.concat([X_train_scaled, y_train], axis=1)
            test_scaled = pd.concat([X_test_scaled, y_test], axis=1)
        else:
            train_scaled = X_train_scaled
            test_scaled = X_test_scaled
        
        logging.info('Feature scaling applied')
        
        if fit_scalers:
            return train_scaled, test_scaled, scalers_dict
        else:
            return train_scaled, test_scaled
        
    except Exception as e:
        logging.error('Error during feature scaling: %s', e)
        raise

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features for the insurance data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with interaction features
    """
    try:
        logging.info("Creating interaction features...")
        
        df_with_interactions = df.copy()
        
        # 1. Age group feature
        if 'Age' in df.columns:
            df_with_interactions['Age_Group'] = pd.cut(df['Age'], 
                                                      bins=[0, 30, 50, 100], 
                                                      labels=['Young', 'Middle', 'Senior'])
            
            # Convert to one-hot encoding
            age_group_dummies = pd.get_dummies(df_with_interactions['Age_Group'], prefix='Age_Group', drop_first=True)
            df_with_interactions = pd.concat([df_with_interactions, age_group_dummies], axis=1)
            df_with_interactions = df_with_interactions.drop('Age_Group', axis=1)
        
        # 2. Interaction: Previously_Insured * Vehicle_Damage
        if all(col in df.columns for col in ['Previously_Insured', 'Vehicle_Damage']):
            df_with_interactions['Insured_x_Damage'] = df['Previously_Insured'] * df['Vehicle_Damage']
        
        # 3. Interaction: Age * Annual_Premium (normalized)
        if all(col in df.columns for col in ['Age', 'Annual_Premium']):
            df_with_interactions['Age_Premium_Ratio'] = df['Annual_Premium'] / (df['Age'] + 1)  # +1 to avoid division by zero
        
        # 4. Vintage group
        if 'Vintage' in df.columns:
            df_with_interactions['Vintage_Group'] = pd.qcut(df['Vintage'], q=4, labels=['New', 'Medium', 'Long', 'Very_Long'])
            vintage_dummies = pd.get_dummies(df_with_interactions['Vintage_Group'], prefix='Vintage', drop_first=True)
            df_with_interactions = pd.concat([df_with_interactions, vintage_dummies], axis=1)
            df_with_interactions = df_with_interactions.drop('Vintage_Group', axis=1)
        
        logging.info(f'Created interaction features. New shape: {df_with_interactions.shape}')
        return df_with_interactions
        
    except Exception as e:
        logging.error('Error creating interaction features: %s', e)
        # Return original df if error
        return df

def select_features(train_data: pd.DataFrame, test_data: pd.DataFrame, k: int = 20):
    """
    Select top k features using ANOVA F-value.
    
    Args:
        train_data (pd.DataFrame): Training data
        test_data (pd.DataFrame): Test data
        k (int): Number of top features to select
        
    Returns:
        tuple: (train_selected, test_selected, selector)
    """
    try:
        logging.info(f"Selecting top {k} features...")
        
        # Separate features and target
        X_train = train_data.drop('Response', axis=1)
        y_train = train_data['Response']
        X_test = test_data.drop('Response', axis=1)
        
        # Ensure k is not greater than number of features
        k = min(k, X_train.shape[1])
        
        # Apply feature selection
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature names
        selected_features = X_train.columns[selector.get_support()].tolist()
        
        # Convert back to DataFrame
        train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
        train_selected['Response'] = y_train.values
        
        test_selected = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
        if 'Response' in test_data.columns:
            test_selected['Response'] = test_data['Response'].values
        
        logging.info(f'Selected {k} features: {selected_features}')
        
        return train_selected, test_selected, selector
        
    except Exception as e:
        logging.error('Error during feature selection: %s', e)
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info('Data saved to %s', file_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        # Load parameters
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']
        create_interaction = params['feature_engineering']['create_interaction']
        

        # Load processed data
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')
        logging.info(f'Train data loaded: {train_data.shape}, Test data loaded: {test_data.shape}')
        
        # Step 1: Create interaction features (optional)
        if create_interaction:
            train_data = create_interaction_features(train_data)
            test_data = create_interaction_features(test_data)
        
        # Step 2: Apply feature scaling
        train_scaled, test_scaled, scalers_dict = apply_feature_scaling(train_data, test_data, fit_scalers=True)
        
        # Step 3: Feature selection (optional)
        if max_features > 0:
            train_final, test_final, selector = select_features(train_scaled, test_scaled, k=max_features)
        else:
            train_final, test_final = train_scaled, test_scaled
            selector = None
        
        # Save scalers and selector
        models_dir = './models'
        os.makedirs(models_dir, exist_ok=True)
        
        with open(os.path.join(models_dir, 'feature_scalers.pkl'), 'wb') as f:
            pickle.dump(scalers_dict, f)
        logging.info('Feature scalers saved to models/feature_scalers.pkl')
        
        if selector:
            with open(os.path.join(models_dir, 'feature_selector.pkl'), 'wb') as f:
                pickle.dump(selector, f)
            logging.info('Feature selector saved to models/feature_selector.pkl')
        
        # Save processed data
        save_data(train_final, os.path.join("./data", "processed", "train_final.csv"))
        save_data(test_final, os.path.join("./data", "processed", "test_final.csv"))
        
        logging.info(f'Feature engineering complete. Final train shape: {train_final.shape}, Test shape: {test_final.shape}')
        
    except Exception as e:
        logging.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()