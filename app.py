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
LOCATION_MAP = {}
PROPERTY_TYPES = []

try:
    # Load the full Pipeline object (Preprocessor + Random Forest/Other Model)
    MODEL_PIPELINE = joblib.load(os.path.join(ARTIFACTS_DIR, 'house_rent_prediction_model.joblib'))
    FEATURE_NAMES = joblib.load(os.path.join(ARTIFACTS_DIR, 'feature_names.joblib'))
    LOCATION_MAP = joblib.load(os.path.join(ARTIFACTS_DIR, 'location_filter_map.joblib'))
    PROPERTY_TYPES = joblib.load(os.path.join(ARTIFACTS_DIR, 'property_types.joblib'))
    
    EXPECTED_FEATURE_COUNT = 10
    
    print("API: All model artifacts loaded successfully. Ready for Random Forest prediction.")
except Exception as e:
    print(f"ERROR: Could not load artifacts. Ensure 'artifacts/' folder exists and files are present. Error: {e}")
    # Setting model to None ensures API calls will fail gracefully
    MODEL_PIPELINE = None 

# --- 2. Initialize Flask App ---

app = Flask(__name__)
CORS(app) 

def predict_price_internal(cleaned_input_data: dict) -> float:
    """
    Internal function to process input, predict log price, and inverse transform 
    using the loaded SKLearn Pipeline.
    """
    if MODEL_PIPELINE is None:
        raise Exception("Model pipeline is not loaded. Cannot predict.")

    # 1. Use the preprocessing helper to create the input DataFrame (1 row, 10 columns)
    processed_df = create_model_features(cleaned_input_data)
    
    # 2. Reindex to ensure EXACT column order from training (CRITICAL!)
    final_features_df = processed_df.reindex(columns=FEATURE_NAMES, fill_value=0)
    
    # 3. Get the Log Price Prediction using the full pipeline
    # The pipeline handles OHE internally before prediction.
    log_price_pred = MODEL_PIPELINE.predict(final_features_df)[0]
    
    # 4. Inverse Transform (e^(log_price) - 1)
    predicted_price = np.expm1(log_price_pred)
    
    return predicted_price

# --- 3. Define the API Endpoints ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint that receives the 10 CLEANED features and returns the price prediction.
    """
    try:
        data = request.get_json(force=True)
        
        # Validation: Check for the exact 10 features
        if len(data) != EXPECTED_FEATURE_COUNT:
             return jsonify({
                'status': 'error', 
                'message': f'Input validation failed: Expected {EXPECTED_FEATURE_COUNT} features, received {len(data)}. Please check all required inputs.'
            }), 400
            
        prediction_result = predict_price_internal(data)
        
        return jsonify({
            'status': 'success',
            'predicted_price': round(prediction_result, 2),
            'currency': 'NGN' 
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'status': 'error', 'message': f'Prediction failed due to internal model error: {str(e)}'}), 400

@app.route('/locations', methods=['GET'])
def get_locations():
    """Endpoint to provide location map and property type data for the frontend dropdowns."""
    try:
        return jsonify({
            'location_map': LOCATION_MAP,
            'property_types': PROPERTY_TYPES
        })
    except Exception as e:
        return jsonify({'error': 'Could not retrieve map data. Artifacts may be missing.'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)