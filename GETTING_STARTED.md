# Getting Started with Customer Churn Prediction

This guide will help you set up and run the customer churn prediction service.

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Git (for cloning the repository)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/headply/lagos-rent-predictor.git
cd lagos-rent-predictor
```

### 2. Create a Virtual Environment (Recommended)

```bash
# On Linux/Mac
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Preparing the Model

Before you can run the prediction service, you need to train a model and generate the required artifacts.

### Option 1: Use Pre-trained Model (If Available)

If you have pre-trained model artifacts, place them in the `artifacts/` directory:

```
artifacts/
  ├── churn_prediction_model.joblib
  ├── feature_names.joblib
  └── categorical_mappings.joblib
```

### Option 2: Train Your Own Model

1. Download a telecom churn dataset (see `DATA_SOURCES.md` for recommendations)
2. Create a training script using the example in `DATA_SOURCES.md`
3. Train the model and save artifacts to the `artifacts/` directory

### Quick Training Example

```python
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load your dataset
df = pd.read_csv('telco_churn_data.csv')

# Define features
numerical_features = ['tenure', 'monthly_charges', 'total_charges']
categorical_features = ['contract_type', 'payment_method', 'internet_service', 
                        'online_security', 'tech_support', 'streaming_tv', 'streaming_movies']

# Prepare data
X = df[numerical_features + categorical_features]
y = df['Churn']

# Create pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# Save artifacts
import os
os.makedirs('artifacts', exist_ok=True)
joblib.dump(model_pipeline, 'artifacts/churn_prediction_model.joblib')
joblib.dump(X.columns.tolist(), 'artifacts/feature_names.joblib')

categorical_mappings = {
    'contract_type': ['Month-to-month', 'One year', 'Two year'],
    'payment_method': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
    'internet_service': ['DSL', 'Fiber optic', 'No'],
    'online_security': ['Yes', 'No', 'No internet service'],
    'tech_support': ['Yes', 'No', 'No internet service'],
    'streaming_tv': ['Yes', 'No', 'No internet service'],
    'streaming_movies': ['Yes', 'No', 'No internet service']
}
joblib.dump(categorical_mappings, 'artifacts/categorical_mappings.joblib')

print(f"Model accuracy: {model_pipeline.score(X_test, y_test):.4f}")
```

## Running the Service

### Development Mode

```bash
python app.py
```

The service will start on `http://localhost:5000`

### Production Mode (Using Gunicorn)

```bash
gunicorn app:app --bind 0.0.0.0:5000 --workers 4
```

## Testing the API

### Health Check

```bash
curl http://localhost:5000/categories
```

### Make a Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "monthly_charges": 70.00,
    "total_charges": 840.00,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes"
  }'
```

### Expected Response

```json
{
  "status": "success",
  "churn_prediction": 1,
  "churn_probability": 0.7842,
  "risk_level": "High"
}
```

## API Endpoints

### POST /predict
Predicts customer churn based on input features.

**Request Body**: JSON with 10 customer features (see `SAMPLE_DATA.md`)

**Response**:
- `status`: "success" or "error"
- `churn_prediction`: 0 (no churn) or 1 (churn)
- `churn_probability`: Probability of churn (0.0 to 1.0)
- `risk_level`: "Low", "Medium", or "High"

### GET /categories
Returns available options for categorical features.

**Response**:
- `categorical_mappings`: Dictionary of categorical feature options

## Deployment

### Deploying to Heroku

1. Install the Heroku CLI
2. Create a new Heroku app:
   ```bash
   heroku create your-app-name
   ```

3. Push to Heroku:
   ```bash
   git push heroku main
   ```

4. Ensure `Procfile` and `runtime.txt` are configured correctly

### Deploying to Other Platforms

The service can be deployed to:
- AWS (Elastic Beanstalk, ECS, Lambda)
- Google Cloud Platform (App Engine, Cloud Run)
- Azure (App Service)
- DigitalOcean App Platform
- Render
- Railway

Ensure you have the `artifacts/` directory with model files included in your deployment.

## Troubleshooting

### Model Not Loading

**Error**: `Could not load artifacts`

**Solution**: 
- Ensure the `artifacts/` directory exists
- Verify all three required files are present
- Check that scikit-learn versions match between training and deployment

### Prediction Fails

**Error**: `Input validation failed`

**Solution**:
- Verify you're sending exactly 10 features
- Check feature names match exactly (case-sensitive)
- Ensure data types are correct (integers for tenure, floats for charges)

### CORS Issues

If you're accessing the API from a web browser and encounter CORS errors, the `flask-cors` package should handle this. Ensure it's installed:

```bash
pip install flask-cors
```

## Next Steps

1. Review `DATA_SOURCES.md` for information on open telecom datasets
2. Check `SAMPLE_DATA.md` for more examples of API usage
3. Train your model using your preferred dataset
4. Customize the feature set based on your data
5. Deploy the service to your production environment

## Support

For issues or questions:
- Check the documentation files in this repository
- Review the code comments in `app.py` and `preprocessing_pipeline.py`
- Open an issue on GitHub

## License

See the LICENSE file in the repository.
