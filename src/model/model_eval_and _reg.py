import json
import logging
import os
import pickle
from matplotlib.pyplot import clf
import mlflow
import mlflow.sklearn
import dagshub
from src.logger import logging
from src.utility.utils import load_data 
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


dagshub.init(repo_owner='MANJESH-ctrl', repo_name='MLOPS', mlflow=True)

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


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise


def main():
    mlflow.set_experiment("pipeline")
    with mlflow.start_run() as run:
        try:
            clf = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_final.csv')

            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            # Evaluate predictions
            try:
                y_pred = clf.predict(X_test)

                # Some classifiers don't implement predict_proba
                if hasattr(clf, "predict_proba"):
                    y_pred_proba = clf.predict_proba(X_test)[:, 1]
                else:
                    y_pred_proba = None

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                auc = None
                if y_pred_proba is not None:
                    auc = roc_auc_score(y_test, y_pred_proba)

                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall
                }
                if auc is not None:
                    metrics['auc'] = auc

                logging.info('Model evaluation metrics calculated')

            except Exception as e:
                logging.error('Error during model evaluation: %s', e)
                raise

            # --- Now save and log metrics (outside inner try) ---
            save_metrics(metrics, 'reports/metrics.json')  # make sure reports/ exists
            mlflow.log_artifact('reports/metrics.json')

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            try:
                mlflow.sklearn.log_model(
                    clf,
                    artifact_path="model",
                    registered_model_name="my_model"
                )

                logging.info("Model logged successfully")
            except Exception as e:
                logging.error(f"Model logging failed: {e}")
                raise

            # Log model parameters to MLflow
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            
        except Exception as e:
            logging.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")
            raise

    
if __name__ == '__main__':
    main()