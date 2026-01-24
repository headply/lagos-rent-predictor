# Data Sources and Model Training

## Open Telecom Datasets

This project uses publicly available telecom customer churn datasets. Here are recommended open data sources:

### 1. IBM Telecom Customer Churn Dataset
- **Source**: IBM Sample Data Sets / Kaggle
- **Link**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Description**: Contains customer account information, demographics, and service details for 7,043 customers
- **Features**:
  - Customer demographics (gender, senior citizen status, partner, dependents)
  - Account information (tenure, contract, payment method, paperless billing)
  - Services (phone, internet, online security, backup, device protection, tech support, streaming)
  - Charges (monthly charges, total charges)
  - Churn status (target variable)

### 2. Orange Telecom Dataset
- **Source**: Kaggle / Orange S.A.
- **Link**: https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets
- **Description**: Real anonymized customer data from Orange Telecom
- **Features**: Includes call patterns, service usage, and customer churn labels

### 3. Cell2Cell Dataset
- **Source**: Duke University / Teradata Center
- **Description**: Wireless telecommunications customer data
- **Features**: Customer demographics, usage patterns, billing information

## Feature Engineering

Common features used in telecom churn prediction:

### Demographic Features
- Age, gender, senior citizen status
- Family composition (partner, dependents)

### Service Features
- Contract type (month-to-month, one year, two year)
- Internet service type (DSL, Fiber optic, None)
- Additional services (online security, backup, tech support, streaming)

### Billing Features
- Monthly charges
- Total charges
- Payment method
- Paperless billing status

### Usage Features
- Tenure (months as customer)
- Call patterns (if available)
- Data usage (if available)

## Model Training Guide

### Prerequisites
```bash
pip install scikit-learn pandas numpy joblib
```

### Sample Training Script

```python
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv('telco_churn_data.csv')

# Define features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Define numerical and categorical features
numerical_features = ['tenure', 'monthly_charges', 'total_charges']
categorical_features = ['contract_type', 'payment_method', 'internet_service', 
                        'online_security', 'tech_support', 'streaming_tv', 'streaming_movies']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create full pipeline with model
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model_pipeline.fit(X_train, y_train)

# Evaluate
accuracy = model_pipeline.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")

# Save artifacts
joblib.dump(model_pipeline, 'artifacts/churn_prediction_model.joblib')
joblib.dump(X.columns.tolist(), 'artifacts/feature_names.joblib')

# Save categorical mappings for frontend
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
```

## Model Evaluation Metrics

For churn prediction, consider these metrics:

- **Accuracy**: Overall correctness
- **Precision**: Of predicted churners, how many actually churned
- **Recall**: Of actual churners, how many were caught
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

## Deployment Notes

1. Ensure scikit-learn versions match between training and deployment
2. Test the model with edge cases before deployment
3. Monitor model performance over time
4. Consider retraining periodically with new data
5. Implement proper error handling for missing or invalid features
