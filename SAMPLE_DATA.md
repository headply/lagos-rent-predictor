# Sample Customer Data for Churn Prediction

This file demonstrates the expected input format for the customer churn prediction API.

## Sample API Request

### Endpoint
```
POST /predict
Content-Type: application/json
```

### Request Body Example 1: High Risk Customer
```json
{
  "tenure": 2,
  "monthly_charges": 85.50,
  "total_charges": 171.00,
  "contract_type": "Month-to-month",
  "payment_method": "Electronic check",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "tech_support": "No",
  "streaming_tv": "Yes",
  "streaming_movies": "Yes"
}
```

### Expected Response
```json
{
  "status": "success",
  "churn_prediction": 1,
  "churn_probability": 0.78,
  "risk_level": "High"
}
```

### Request Body Example 2: Low Risk Customer
```json
{
  "tenure": 48,
  "monthly_charges": 45.20,
  "total_charges": 2170.00,
  "contract_type": "Two year",
  "payment_method": "Credit card",
  "internet_service": "DSL",
  "online_security": "Yes",
  "tech_support": "Yes",
  "streaming_tv": "No",
  "streaming_movies": "No"
}
```

### Expected Response
```json
{
  "status": "success",
  "churn_prediction": 0,
  "churn_probability": 0.15,
  "risk_level": "Low"
}
```

## Feature Descriptions

### Numerical Features

1. **tenure** (integer)
   - Range: 0-72 (months)
   - Description: Number of months the customer has been with the company
   - Example: 12

2. **monthly_charges** (float)
   - Range: 18.00-120.00 (USD)
   - Description: Monthly service charges
   - Example: 65.50

3. **total_charges** (float)
   - Range: 0.00-8000.00 (USD)
   - Description: Total charges accumulated over tenure
   - Example: 786.00

### Categorical Features

4. **contract_type** (string)
   - Options: "Month-to-month", "One year", "Two year"
   - Description: Type of service contract
   - Example: "Month-to-month"

5. **payment_method** (string)
   - Options: "Electronic check", "Mailed check", "Bank transfer", "Credit card"
   - Description: How the customer pays for services
   - Example: "Credit card"

6. **internet_service** (string)
   - Options: "DSL", "Fiber optic", "No"
   - Description: Type of internet service
   - Example: "Fiber optic"

7. **online_security** (string)
   - Options: "Yes", "No", "No internet service"
   - Description: Whether customer has online security add-on
   - Example: "Yes"

8. **tech_support** (string)
   - Options: "Yes", "No", "No internet service"
   - Description: Whether customer has tech support add-on
   - Example: "No"

9. **streaming_tv** (string)
   - Options: "Yes", "No", "No internet service"
   - Description: Whether customer has streaming TV service
   - Example: "Yes"

10. **streaming_movies** (string)
    - Options: "Yes", "No", "No internet service"
    - Description: Whether customer has streaming movies service
    - Example: "Yes"

## Testing the API

### Using curl
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

### Using Python requests
```python
import requests
import json

url = "http://localhost:5000/predict"
data = {
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
}

response = requests.post(url, json=data)
print(json.dumps(response.json(), indent=2))
```

## Getting Categorical Options

To retrieve available options for categorical fields:

```bash
curl http://localhost:5000/categories
```

Response:
```json
{
  "categorical_mappings": {
    "contract_type": ["Month-to-month", "One year", "Two year"],
    "payment_method": ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    "internet_service": ["DSL", "Fiber optic", "No"],
    "online_security": ["Yes", "No", "No internet service"],
    "tech_support": ["Yes", "No", "No internet service"],
    "streaming_tv": ["Yes", "No", "No internet service"],
    "streaming_movies": ["Yes", "No", "No internet service"]
  }
}
```

## Churn Prediction Output

### Prediction Fields

- **churn_prediction** (integer): 0 = No churn, 1 = Churn
- **churn_probability** (float): Probability of churn (0.0 to 1.0)
- **risk_level** (string): "Low", "Medium", or "High"
  - Low: probability < 0.4
  - Medium: 0.4 ≤ probability ≤ 0.7
  - High: probability > 0.7
