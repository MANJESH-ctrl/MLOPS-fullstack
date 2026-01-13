import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub
from mlflow.exceptions import MlflowException

import warnings
warnings.filterwarnings("ignore")

def setup_mlflow_tracking(mode='local'):
    """
    Setup MLflow tracking URI based on mode.
    """
    try:
        if mode == 'production':
            dagshub_token = os.getenv("DAGSHUB_TOKEN")
            if not dagshub_token:
                raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")
            
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
            
            tracking_uri = 'https://dagshub.com/MANJESH-ctrl/MLOPS-fullstack.mlflow'
            
        else:
            tracking_uri = 'https://dagshub.com/MANJESH-ctrl/MLOPS-fullstack.mlflow'
        
        # Set tracking URI for BOTH modes
        mlflow.set_tracking_uri(tracking_uri)
        print(f"✅ MLflow tracking URI set to: {tracking_uri}")
        
        # Initialize DagsHub for BOTH modes
        dagshub.init(repo_owner='MANJESH-ctrl', repo_name='MLOPS-fullstack', mlflow=True)
            
    except Exception as e:
        logging.error(f"Error setting up MLflow tracking: {e}")
        raise

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.info('Model info loaded from %s', file_path)
        
        print(f"\n📋 Loaded model info:")
        print(f"   Run ID: {model_info.get('run_id')}")
        print(f"   Model path: {model_info.get('model_path')}")
        
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def check_run_exists(run_id: str) -> bool:
    """Check if the run exists in MLflow."""
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        print(f"✅ Run found: {run_id}")
        print(f"   Experiment: {run.info.experiment_id}")
        print(f"   Status: {run.info.status}")
        
        # Check artifacts
        artifacts = client.list_artifacts(run_id)
        print(f"\n📁 Artifacts in run:")
        for artifact in artifacts:
            print(f"   - {artifact.path}")
        
        return True
    except MlflowException as e:
        print(f"❌ Run not found: {run_id}")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking run: {e}")
        return False

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        run_id = model_info['run_id']
        model_path = model_info['model_path']
        
        # First, check if run exists
        if not check_run_exists(run_id):
            print("\n❌ Cannot register model - run not found!")
            print("   Try running model_evaluation.py first")
            return
        
        model_uri = f"runs:/{run_id}/{model_path}"
        
        print(f"\n🔗 Model URI: {model_uri}")
        print(f"📝 Registering model: {model_name}...")
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        
        print(f"\n✅ Model registered successfully!")
        print(f"   Model Name: {model_name}")
        print(f"   Version: {model_version.version}")
        print(f"   Stage: Staging")
        print(f"   Run ID: {run_id}")
        print(f"\n🔗 View model at: https://dagshub.com/MANJESH-ctrl/MLOPS-fullstack.mlflow#/models")
        
    except MlflowException as e:
        logging.error(f'MLflow error during registration: {e}')
        print(f"\n❌ MLflow Error: {e}")
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        print(f"\n❌ Error: {e}")

def main(mode='local'):
    try:
        # Setup MLflow tracking
        setup_mlflow_tracking(mode)
        
        # Check current tracking URI
        print(f"Current tracking URI: {mlflow.get_tracking_uri()}")
        
        # Load model info
        model_info = load_model_info('reports/experiment_info.json')
        
        # Check if run_id exists
        if 'run_id' not in model_info:
            print("❌ No 'run_id' found in experiment_info.json!")
            print("   Make sure model_evaluation.py ran successfully")
            return
            
        # Register model
        model_name = "insurance-prediction-rf"
        register_model(model_name, model_info)
        
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main(mode='local')