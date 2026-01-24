import os
import json
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from preprocessing_pipeline import create_model_features 
from flask_cors import CORS 

# --- 1. Model and Artifact Loading ---

ARTIFACTS_DIR = 'artifacts'
MODEL_PIPELINE = None
FEATURE_NAMES = []
CATEGORICAL_MAPPINGS = {}

try:
    # Load the full Pipeline object (Preprocessor + Classification Model)
    MODEL_PIPELINE = joblib.load(os.path.join(ARTIFACTS_DIR, 'churn_prediction_model.joblib'))
    FEATURE_NAMES = joblib.load(os.path.join(ARTIFACTS_DIR, 'feature_names.joblib'))
    CATEGORICAL_MAPPINGS = joblib.load(os.path.join(ARTIFACTS_DIR, 'categorical_mappings.joblib'))
    
    EXPECTED_FEATURE_COUNT = 10
    
    print("API: All model artifacts loaded successfully. Ready for churn prediction.")
except Exception as e:
    print(f"ERROR: Could not load artifacts. Ensure 'artifacts/' folder exists and files are present. Error: {e}")
    # Setting model to None ensures API calls will fail gracefully
    MODEL_PIPELINE = None 

# --- 2. Initialize Flask App ---

app = Flask(__name__)
CORS(app) 

def predict_churn_internal(cleaned_input_data: dict) -> dict:
    """
    Internal function to process input and predict customer churn 
    using the loaded SKLearn Pipeline.
    """
    if MODEL_PIPELINE is None:
        raise Exception("Model pipeline is not loaded. Cannot predict.")

    # 1. Use the preprocessing helper to create the input DataFrame (1 row, 10 columns)
    processed_df = create_model_features(cleaned_input_data)
    
    # 2. Reindex to ensure EXACT column order from training (CRITICAL!)
    final_features_df = processed_df.reindex(columns=FEATURE_NAMES, fill_value=0)
    
    # 3. Get the churn prediction using the full pipeline
    # The pipeline handles encoding internally before prediction.
    churn_prediction = MODEL_PIPELINE.predict(final_features_df)[0]
    
    # 4. Get the probability of churn
    churn_probability = MODEL_PIPELINE.predict_proba(final_features_df)[0]
    
    # 5. Determine risk level based on probability
    risk_level = "Low"
    if churn_probability[1] > 0.7:
        risk_level = "High"
    elif churn_probability[1] > 0.4:
        risk_level = "Medium"
    
    return {
        'churn_prediction': int(churn_prediction),
        'churn_probability': float(churn_probability[1]),
        'risk_level': risk_level
    }

# --- 3. Define the API Endpoints ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint that receives the customer features and returns the churn prediction.
    """
    try:
        data = request.get_json(force=True)
        
        # Validation: Check for the exact number of features
        if len(data) != EXPECTED_FEATURE_COUNT:
             return jsonify({
                'status': 'error', 
                'message': f'Input validation failed: Expected {EXPECTED_FEATURE_COUNT} features, received {len(data)}. Please check all required inputs.'
            }), 400
            
        prediction_result = predict_churn_internal(data)
        
        return jsonify({
            'status': 'success',
            'churn_prediction': prediction_result['churn_prediction'],
            'churn_probability': round(prediction_result['churn_probability'], 4),
            'risk_level': prediction_result['risk_level']
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'status': 'error', 'message': f'Prediction failed due to internal model error: {str(e)}'}), 400

@app.route('/categories', methods=['GET'])
def get_categories():
    """Endpoint to provide categorical mappings for the frontend dropdowns."""
    try:
        return jsonify({
            'categorical_mappings': CATEGORICAL_MAPPINGS
        })
    except Exception as e:
        return jsonify({'error': 'Could not retrieve category data. Artifacts may be missing.'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)