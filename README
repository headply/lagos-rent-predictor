# Lagos Rent Price Prediction Service

This project provides an end-to-end service for predicting house and apartment rental prices in Lagos, Nigeria, based on property features. The underlying data used for training the model was scraped from the PropertyPro website. It is designed to power a frontend application for real-time price predictions.

The core prediction engine uses a trained Random Forest machine learning model, which is exposed via a lightweight Flask web service.

---

## Key Technologies

- **Flask**: Lightweight web server framework for the backend.  
- **gunicorn**: Production-ready WSGI server to run the Flask app.  
- **scikit-learn**: Used for the Random Forest model, preprocessing, and ML utilities.  
- **pandas**: Handles data manipulation and feature preparation.  
- **flask-cors**: Manages Cross-Origin Resource Sharing for frontend communication.  
- **RandomForestRegressor**: The ensemble learning model used for prediction (loaded via joblib).

---

## Model Artifacts

The service relies on pre-trained model files stored in an `artifacts/` folder:

- `random_forest_house_price_model.joblib`: Trained Random Forest model object.  
- `feature_names.joblib`: List of feature names used during training to ensure correct input order.  
- `location_filter_map.joblib`: Dictionary for mapping locations, used by the frontend.

⚠️ **Note**: Model artifacts must be created using the same versions of `scikit-learn`, `pandas`, and `numpy` as the deployment environment to avoid compatibility issues.

---

## Feature Inputs

The prediction service expects a JSON input with the following 13 features:

| Feature Name           | Type          | Description |
|------------------------|---------------|-------------|
| `Locality`             | String        | Major area or Local Government Area (e.g., "Ikeja") |
| `Area`                 | String        | Specific neighborhood within the Locality (e.g., "Gbagada") |
| `Property_Type`        | String        | Type of rental (e.g., "Apartment", "Terrace") |
| `No_of_Bedrooms`       | Integer       | Number of bedrooms |
| `No_of_Bathrooms`      | Integer       | Number of bathrooms |
| `Is_New`               | Binary (0/1) | 1 if the property is brand new |
| `amen_none_specified`  | Binary (0/1) | 1 if no specific amenities were listed |
| `amen_furnished`       | Binary (0/1) | 1 if the property is furnished |
| `amen_security`        | Binary (0/1) | 1 if the property has visible security features |
| `amen_big_compound`    | Binary (0/1) | 1 if the property has a notably large compound |

---

## Service Output

The prediction endpoint returns a JSON object containing the predicted rental price in Nigerian Naira.  

**Example Response:**

```json
{
    "predicted_price_naira": 7500000.00
}
