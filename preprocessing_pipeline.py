import pandas as pd
from typing import Dict, Any

def create_model_features(input_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Creates a standardized DataFrame (X) from the raw JSON input dictionary for churn prediction.

    Args:
        input_data: Dictionary containing the customer features submitted by the frontend.

    Returns:
        A pandas DataFrame with the features, ready for reindexing and prediction.
    """
    # 1. Convert the single input dictionary into a DataFrame with one row
    # This ensures consistency for the scikit-learn pipeline
    df = pd.DataFrame([input_data])
    
    # 2. Ensure all numerical columns are of appropriate type
    numerical_cols = ['tenure', 'monthly_charges', 'total_charges']
    for col in numerical_cols:
        if col in df.columns:
            if col == 'tenure':
                df[col] = df[col].astype(int)
            else:
                df[col] = df[col].astype(float)

    # 3. Ensure categorical columns are object type (required for encoding)
    categorical_cols = ['contract_type', 'payment_method', 'internet_service', 
                        'online_security', 'tech_support', 'streaming_tv', 'streaming_movies']
    for col in categorical_cols:
         if col in df.columns:
            df[col] = df[col].astype(object)

    return df