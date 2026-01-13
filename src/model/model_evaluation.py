import numpy as np
import pandas as pd
import pickle
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import dagshub
from src.logger import logging


def setup_mlflow_tracking(mode='local'):
    """
    Setup MLflow tracking URI based on mode.
    """
    try:
        if mode == 'production':
            # PRODUCTION: Use environment variables
            dagshub_token = os.getenv("DAGSHUB_TOKEN")
            if not dagshub_token:
                raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")
            
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
            
            tracking_uri = 'https://dagshub.com/MANJESH-ctrl/MLOPS-fullstack.mlflow'
            mlflow.set_tracking_uri(tracking_uri)
            
        else:
            # LOCAL: Direct URL
            mlflow.set_tracking_uri('https://dagshub.com/MANJESH-ctrl/MLOPS-fullstack.mlflow')
            dagshub.init(repo_owner='MANJESH-ctrl', repo_name='MLOPS-fullstack', mlflow=True)
            
    except Exception as e:
        logging.error(f"Error setting up MLflow tracking: {e}")
        raise

def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model: %s', e)
        raise

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

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logging.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise

def main(mode='local'):
    """
    Main function to run model evaluation.
    
    Args:
        mode (str): 'local' or 'production'
    """
    try:
        # Setup MLflow tracking
        setup_mlflow_tracking(mode)
        
        # Set experiment
        mlflow.set_experiment("insurance-prediction")
        
        # Start MLflow run
        with mlflow.start_run() as run:
            logging.info(f"Started MLflow run: {run.info.run_id}")
            
            # Load model and test data
            clf = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_final.csv')
            
            # Split features and target
            X_test = test_data.drop('Response', axis=1).values
            y_test = test_data['Response'].values

            # Evaluate model
            metrics = evaluate_model(clf, X_test, y_test)
            
            # Save metrics
            save_metrics(metrics, 'reports/metrics.json')
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model parameters
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(clf, "model")
            
            # Log the metrics file
            mlflow.log_artifact('reports/metrics.json')
            
            print(f"\n✅ Evaluation complete!")
            print(f"📊 Accuracy: {metrics['accuracy']:.4f}")
            print(f"📊 Precision: {metrics['precision']:.4f}")
            print(f"📊 Recall: {metrics['recall']:.4f}")
            print(f"📊 AUC: {metrics['auc']:.4f}")
            print(f"\n🔗 View run at: https://dagshub.com/MANJESH-ctrl/MLOPS-fullstack.mlflow")
            print(f"\n🔗 View run at: https://dagshub.com/MANJESH-ctrl/MLOPS-fullstack.mlflow#/experiments/1/runs/{run.info.run_id}")

            # model registration info
            model_info = {
                 'run_id': run.info.run_id,
                 'model_path': "model"
            }           
            info_path = 'reports/experiment_info.json'
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=4)
            mlflow.log_artifact(info_path)


    except Exception as e:
        logging.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    # Create reports directory
    os.makedirs('reports', exist_ok=True)
    
    # Run evaluation (change 'local' to 'production' if needed)
    main(mode='local')