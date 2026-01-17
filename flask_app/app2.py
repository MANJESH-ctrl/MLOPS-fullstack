from flask import Flask, render_template, request, jsonify
import mlflow
import pickle
import os
import pandas as pd
import numpy as np
import json
import dagshub
from datetime import datetime

# For monitoring (optional)
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time

import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# MLflow Setup - Choose LOCAL or PRODUCTION
# ============================================================================

# Choose mode: 'local' or 'production'
MODE = 'local'

def setup_mlflow():
    """Setup MLflow tracking based on mode."""
    try:
        if MODE == 'production':
            # PRODUCTION: Use environment variables
            dagshub_token = os.getenv("DAGSHUB_TOKEN")
            if not dagshub_token:
                raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")
            
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

            dagshub_url = "https://dagshub.com"
            repo_owner = "MANJESH-ctrl"
            repo_name = "MLOPS"

            tracking_uri = f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow'
            mlflow.set_tracking_uri(tracking_uri)
            
        else:
            # LOCAL: Direct URL
            tracking_uri = 'https://dagshub.com/MANJESH-ctrl/MLOPS-fullstack.mlflow'
            mlflow.set_tracking_uri(tracking_uri)
            
        print(f"✅ MLflow tracking URI: {tracking_uri}")
        return True
        
    except Exception as e:
        print(f"❌ MLflow setup error: {e}")
        return False

# ============================================================================
# Initialize Flask App
# ============================================================================

app = Flask(__name__)

# Optional: Prometheus metrics
registry = CollectorRegistry()
REQUEST_COUNT = Counter(
    "insurance_app_request_count", 
    "Total requests to insurance app", 
    ["method", "endpoint"], 
    registry=registry
)
REQUEST_LATENCY = Histogram(
    "insurance_app_request_latency_seconds", 
    "Request latency", 
    ["endpoint"], 
    registry=registry
)
PREDICTION_COUNT = Counter(
    "insurance_model_prediction_count", 
    "Prediction counts", 
    ["prediction"], 
    registry=registry
)

# ============================================================================
# Load Model and Preprocessing Objects
# ============================================================================

print("🔄 Loading model and preprocessing objects...")

# Setup MLflow first
setup_mlflow()
dagshub.init(repo_owner='MANJESH-ctrl', repo_name='MLOPS', mlflow=True)
# Load model from MLflow Model Registry
model_name = "my_model"

# model_stage = "Staging"  # or "Production" if you've promoted it

def load_production_model():
    """Load the latest model from MLflow Model Registry."""
    try:
        # Try to load from model registry
        model_uri = f"models:/{model_name}@latest"
        print(f"🔍 Loading model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"✅ Model loaded from MLflow Registry")
        return model
    except Exception as e:
        print(f"⚠️ Could not load from MLflow: {e}")
        print("🔄 Falling back to local model file...")
        try:
            # Fallback to local pickle file
            with open('models/model.pkl', 'rb') as f:
                model = pickle.load(f)
            print("✅ Model loaded from local file")
            return model
        except Exception as e2:
            print(f"❌ Could not load model: {e2}")
            return None

# Load the model
model = load_production_model()

# Load preprocessing objects
try:
    with open('models/preprocessing_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    print("✅ Preprocessing encoders loaded")
except:
    encoders = None
    print("⚠️ Could not load encoders")

try:
    with open('models/feature_scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    print("✅ Feature scalers loaded")
except:
    scalers = None
    print("⚠️ Could not load scalers")

try:
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    print(f"✅ Feature names loaded: {len(feature_names)} features")
except:
    feature_names = None
    print("⚠️ Could not load feature names")

# ============================================================================
# Preprocessing Functions
# ============================================================================

def preprocess_input(data):
    """
    Preprocess input data for prediction.
    Takes a dictionary of form inputs and returns processed features.
    """
    try:
        # Create DataFrame from input
        df = pd.DataFrame([data])
        
        # 1. Encode categorical variables
        if 'Gender' in df.columns:
            # Use mapping from encoders or default
            gender_map = encoders.get('gender_map', {'Female': 0, 'Male': 1}) if encoders else {'Female': 0, 'Male': 1}
            df['Gender'] = df['Gender'].map(gender_map)
        
        if 'Vehicle_Damage' in df.columns:
            damage_map = encoders.get('damage_map', {'No': 0, 'Yes': 1}) if encoders else {'No': 0, 'Yes': 1}
            df['Vehicle_Damage'] = df['Vehicle_Damage'].map(damage_map)
        
        # 2. Handle Vehicle_Age - create dummy variables
        if 'Vehicle_Age' in df.columns:
            # Create dummy variables
            if encoders and 'vehicle_age_columns' in encoders:
                expected_columns = encoders['vehicle_age_columns']
                for col in expected_columns:
                    df[col] = 0
                
                # Set the appropriate column to 1
                if df['Vehicle_Age'].iloc[0] == '< 1 Year':
                    # This is the dropped reference category
                    pass
                elif df['Vehicle_Age'].iloc[0] == '1-2 Year':
                    df['Vehicle_Age_1-2 Year'] = 1
                elif df['Vehicle_Age'].iloc[0] == '> 2 Years':
                    df['Vehicle_Age_> 2 Years'] = 1
            
            df = df.drop('Vehicle_Age', axis=1)
        
        # 3. Apply scaling
        if scalers:
            # StandardScaler for Age and Vintage
            if 'std_scaler' in scalers and all(col in df.columns for col in ['Age', 'Vintage']):
                df[['Age', 'Vintage']] = scalers['std_scaler'].transform(df[['Age', 'Vintage']])
            
            # MinMaxScaler for Annual_Premium
            if 'minmax_scaler' in scalers and 'Annual_Premium' in df.columns:
                df[['Annual_Premium']] = scalers['minmax_scaler'].transform(df[['Annual_Premium']])
        
        # 4. Ensure all expected features are present
        if feature_names:
            # Add missing columns with 0
            for col in feature_names:
                if col not in df.columns and col != 'Response':
                    df[col] = 0
            
            # Reorder columns to match training
            df = df[feature_names]
        
        return df
        
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        raise

# ============================================================================
# Flask Routes
# ============================================================================

@app.route("/")
def home():
    """Home page with input form."""
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    
    # Render the form
    response = render_template("index.html", result=None)
    
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests."""
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()
    
    try:
        # Get form data
        form_data = {
            'Gender': request.form.get('gender'),
            'Age': float(request.form.get('age')),
            'Driving_License': int(request.form.get('driving_license')),
            'Region_Code': int(request.form.get('region_code')),
            'Previously_Insured': int(request.form.get('previously_insured')),
            'Vehicle_Age': request.form.get('vehicle_age'),
            'Vehicle_Damage': request.form.get('vehicle_damage'),
            'Annual_Premium': float(request.form.get('annual_premium')),
            'Policy_Sales_Channel': int(request.form.get('policy_sales_channel')),
            'Vintage': int(request.form.get('vintage'))
        }
        
        # Log the input
        print(f"📥 Received input: {form_data}")
        
        # Preprocess the input
        processed_data = preprocess_input(form_data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = None
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(processed_data)[0][1]
        
        # Determine result message
        result_text = "INTERESTED in buying insurance" if prediction == 1 else "NOT INTERESTED in buying insurance"
        
        # Update prediction metrics
        PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
        
        # Calculate latency
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
        
        # Log the prediction
        print(f"✅ Prediction: {prediction} ({result_text})")
        if probability:
            print(f"   Probability: {probability:.2%}")
        
        # Return result
        return render_template("index.html", 
                             result=result_text,
                             prediction=prediction,
                             probability=f"{probability:.2%}" if probability else None,
                             input_data=form_data)
                             
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
        return render_template("index.html", 
                             error=f"Prediction failed: {str(e)}",
                             result=None)

@app.route("/predict_api", methods=["POST"])
def predict_api():
    """API endpoint for programmatic access."""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Preprocess and predict
        processed_data = preprocess_input(data)
        prediction = model.predict(processed_data)[0]
        
        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            probability = float(model.predict_proba(processed_data)[0][1])
        
        # Return JSON response
        response = {
            "prediction": int(prediction),
            "prediction_text": "interested" if prediction == 1 else "not_interested",
            "probability": probability,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }
    return jsonify(status), 200

@app.route("/metrics", methods=["GET"])
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}



# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    

    # Check if model loaded
    if model is None:
        print("❌ WARNING: Model not loaded. App may not work properly.")
    
    # Run the app
    print("🚀 Starting Flask app...")
    print("🌐 Web Interface: http://localhost:5000")
    print("📊 Health Check: http://localhost:5000/health")
    print("📈 Metrics: http://localhost:5000/metrics")
    
    # For development
    app.run(debug=True, host="0.0.0.0", port=5000)
    
    # For production, use:
    # app.run(host="0.0.0.0", port=5000)