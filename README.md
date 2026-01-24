# Customer Churn Prediction Service

This project provides an end-to-end service for predicting customer churn in the telecommunications industry using open telecom data. The service is designed to help telecom companies identify customers who are likely to discontinue their services, enabling proactive retention strategies.

The core prediction engine uses a trained machine learning model (Random Forest Classifier or similar), which is exposed via a lightweight Flask web service.

---

## Key Technologies

- **Flask**: Lightweight web server framework for the backend.  
- **gunicorn**: Production-ready WSGI server to run the Flask app.  
- **scikit-learn**: Used for classification models, preprocessing, and ML utilities.  
- **pandas**: Handles data manipulation and feature preparation.  
- **flask-cors**: Manages Cross-Origin Resource Sharing for frontend communication.  
- **RandomForestClassifier**: The ensemble learning model used for churn prediction (loaded via joblib).

---

## Model Artifacts

The service relies on pre-trained model files stored in an `artifacts/` folder:

- `churn_prediction_model.joblib`: Trained classification model object.  
- `feature_names.joblib`: List of feature names used during training to ensure correct input order.  
- `categorical_mappings.joblib`: Dictionary for mapping categorical variables.

⚠️ **Note**: Model artifacts must be created using the same versions of `scikit-learn`, `pandas`, and `numpy` as the deployment environment to avoid compatibility issues.

---

## Feature Inputs

The prediction service expects a JSON input with customer features. Typical telecom churn features include:

| Feature Name           | Type          | Description |
|------------------------|---------------|-------------|
| `tenure`               | Integer       | Number of months the customer has been with the company |
| `monthly_charges`      | Float         | Monthly service charges in USD |
| `total_charges`        | Float         | Total charges accumulated over tenure |
| `contract_type`        | String        | Contract type: "Month-to-month", "One year", "Two year" |
| `payment_method`       | String        | Payment method: "Electronic check", "Mailed check", "Bank transfer", "Credit card" |
| `internet_service`     | String        | Internet service type: "DSL", "Fiber optic", "No" |
| `online_security`      | String        | Has online security: "Yes", "No", "No internet service" |
| `tech_support`         | String        | Has tech support: "Yes", "No", "No internet service" |
| `streaming_tv`         | String        | Has streaming TV: "Yes", "No", "No internet service" |
| `streaming_movies`     | String        | Has streaming movies: "Yes", "No", "No internet service" |

---

## Service Output

The prediction endpoint returns a JSON object containing the churn prediction and probability.

**Example Response:**

```json
{
    "churn_prediction": 1,
    "churn_probability": 0.78,
    "risk_level": "High"
}
