import pandas as pd
from typing import Dict, Any

def create_model_features(input_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Creates a standardized DataFrame (X) from the raw JSON input dictionary.

    Args:
        input_data: Dictionary containing the 10 raw features submitted by the frontend.

    Returns:
        A pandas DataFrame with the 10 features, ready for reindexing and prediction.
    """
    # 1. Convert the single input dictionary into a DataFrame with one row
    # This ensures consistency for the scikit-learn pipeline
    df = pd.DataFrame([input_data])
    
    # 2. Ensure all numerical columns are of integer type
    numerical_cols = ['No_of_Bedrooms', 'No_of_Bathrooms']
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # 3. Ensure boolean features are correctly typed (Python bools will work)
    # The scikit-learn pipeline will treat these as numerical (0 or 1)
    
    # 4. Ensure categorical columns are object type (required for OHE)
    categorical_cols = ['Locality', 'Area', 'Property_Type']
    for col in categorical_cols:
         if col in df.columns:
            df[col] = df[col].astype(object)

    return df