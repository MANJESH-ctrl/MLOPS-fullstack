import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub
import warnings
warnings.filterwarnings("ignore")

# Set up MLflow tracking
print("🔧 Setting up MLflow tracking...")
mlflow.set_tracking_uri('https://dagshub.com/MANJESH-ctrl/MLOPS-fullstack.mlflow')

# Try to initialize with DagsHub (optional for read operations)
try:
    dagshub.init(repo_owner='MANJESH-ctrl', repo_name='MLOPS-fullstack', mlflow=True)
except:
    print("⚠️  DagsHub init optional, continuing...")

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        print(f"✅ Model info loaded from {file_path}")
        print(f"   Run ID: {model_info.get('run_id')}")
        print(f"   Model Path: {model_info.get('model_path')}")
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        run_id = model_info['run_id']
        model_path = model_info['model_path']
        
        # Construct model URI
        model_uri = f"runs:/{run_id}/{model_path}"
        print(f"📦 Model URI: {model_uri}")
        
        # Get client
        client = mlflow.tracking.MlflowClient()
        
        # Check if model already exists
        try:
            existing_versions = client.search_model_versions(f"name='{model_name}'")
            print(f"🔍 Found {len(existing_versions)} existing versions of {model_name}")
        except:
            pass
        
        # Register the model
        print(f"📝 Registering model '{model_name}' from run {run_id}...")
        model_version = mlflow.register_model(model_uri, model_name)
        
        print(f"✅ Model registered successfully!")
        print(f"   Model Name: {model_name}")
        print(f"   Version: {model_version.version}")
        print(f"   Stage: {model_version.current_stage}")
        
        # Transition the model to "Staging" stage
        print(f"🔄 Transitioning model to 'Staging' stage...")
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        
        # Add description
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description="Insurance prediction model trained on customer data"
        )
        
        print(f"🎉 Model {model_name} version {model_version.version} registered and transitioned to Staging.")
        
        return model_version
        
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        print(f"❌ Registration failed: {e}")
        raise

def main():
    try:
        # Load model info
        model_info_path = 'reports/experiment_info.json'
        if not os.path.exists(model_info_path):
            print(f"❌ File not found: {model_info_path}")
            print("   Make sure to run model_evaluation.py first!")
            return
        
        model_info = load_model_info(model_info_path)
        
        # Register model
        model_name = "insurance-prediction-model"
        model_version = register_model(model_name, model_info)
        
        # Print success info
        print("\n" + "="*50)
        print("🎯 REGISTRATION SUCCESSFUL!")
        print("="*50)
        print(f"Model: {model_name}")
        print(f"Version: {model_version.version}")
        print(f"Stage: Staging")
        print(f"Run ID: {model_info['run_id']}")
        print("\n🔗 View in MLflow UI:")
        print(f"https://dagshub.com/MANJESH-ctrl/MLOPS-fullstack.mlflow#/models")
        
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("🚀 Starting model registration...")
    main()