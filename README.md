# 🏘️ Lagos Rent Predictor

[![Python](https://img.shields.io/badge/Python-3.11.9-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-Latest-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A machine learning-powered web service for predicting house and apartment rental prices in Lagos, Nigeria. Built with Flask and scikit-learn, this service provides accurate rent predictions based on property features using a trained Random Forest model.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project provides an end-to-end service for predicting rental prices in Lagos, Nigeria. The data used for training the model was sourced from PropertyPro, a leading Nigerian real estate platform. The service exposes a REST API that can be integrated with frontend applications for real-time price predictions.

**Key Highlights:**
- 🤖 Random Forest Regressor model for accurate predictions
- 🚀 Production-ready Flask API with CORS support
- 📊 Trained on real Lagos property data
- 🎨 Includes a beautiful web interface (index.html)
- ☁️ Ready for deployment on Heroku, Render, or similar platforms

## ✨ Features

- **Real-time Predictions**: Get instant rental price estimates based on property features
- **Location-aware**: Considers locality and specific area within Lagos
- **Multiple Property Types**: Supports apartments, terraces, and other property types
- **Amenity Analysis**: Factors in property amenities (furnished, security, compound size)
- **Easy Integration**: RESTful API for seamless frontend integration
- **Interactive UI**: Includes a ready-to-use web interface

## 🛠️ Technology Stack

| Technology | Purpose |
|------------|---------|
| **Flask** | Lightweight web server framework |
| **Gunicorn** | Production WSGI HTTP server |
| **scikit-learn 1.6.1** | Machine learning model and preprocessing |
| **pandas 2.2.3** | Data manipulation and feature engineering |
| **flask-cors 6.0.1** | Cross-Origin Resource Sharing support |
| **joblib** | Model serialization and loading |
| **NumPy** | Numerical computations |

## 📦 Installation

### Prerequisites

- Python 3.11.9 or compatible version
- pip (Python package manager)
- Git

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/headply/lagos-rent-predictor.git
cd lagos-rent-predictor
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify model artifacts**

Ensure the `artifacts/` directory contains:
- `house_rent_prediction_model.joblib`
- `feature_names.joblib`
- `location_filter_map.joblib`
- `property_types.joblib`

> ⚠️ **Note**: Model artifacts must be compatible with scikit-learn 1.6.1, pandas 2.2.3, and the same Python version.

## 🚀 Usage

### Starting the Development Server

```bash
python app.py
```

The API will be available at `http://localhost:5000`

### Using the Web Interface

1. Open `index.html` in your browser
2. Fill in the property details
3. Click "Predict Rent" to get an estimate

### Making API Calls

**Using cURL:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Locality": "Ikeja",
    "Area": "Gbagada",
    "Property_Type": "Apartment",
    "No_of_Bedrooms": 3,
    "No_of_Bathrooms": 2,
    "Is_New": false,
    "amen_none_specified": false,
    "amen_furnished": true,
    "amen_security": true,
    "amen_big_compound": false
  }'
```

**Using Python:**
```python
import requests

url = "http://localhost:5000/predict"
data = {
    "Locality": "Ikeja",
    "Area": "Gbagada",
    "Property_Type": "Apartment",
    "No_of_Bedrooms": 3,
    "No_of_Bathrooms": 2,
    "Is_New": False,
    "amen_none_specified": False,
    "amen_furnished": True,
    "amen_security": True,
    "amen_big_compound": False
}

response = requests.post(url, json=data)
print(response.json())
```

## 📚 API Documentation

### Endpoints

#### 1. Predict Rent Price

**Endpoint:** `POST /predict`

**Description:** Predicts rental price based on property features.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `Locality` | string | Yes | Major area or LGA (e.g., "Ikeja", "Lekki") |
| `Area` | string | Yes | Specific neighborhood within locality |
| `Property_Type` | string | Yes | Type of property (e.g., "Apartment", "Terrace", "Flat") |
| `No_of_Bedrooms` | integer | Yes | Number of bedrooms (1-6) |
| `No_of_Bathrooms` | integer | Yes | Number of bathrooms (1-6) |
| `Is_New` | boolean | Yes | Whether property is brand new |
| `amen_none_specified` | boolean | Yes | No specific amenities listed |
| `amen_furnished` | boolean | Yes | Property is furnished |
| `amen_security` | boolean | Yes | Has security features |
| `amen_big_compound` | boolean | Yes | Has large compound |

**Example Request:**
```json
{
  "Locality": "Ikeja",
  "Area": "Gbagada",
  "Property_Type": "Apartment",
  "No_of_Bedrooms": 3,
  "No_of_Bathrooms": 2,
  "Is_New": false,
  "amen_none_specified": false,
  "amen_furnished": true,
  "amen_security": true,
  "amen_big_compound": false
}
```

**Success Response (200 OK):**
```json
{
  "status": "success",
  "predicted_price": 7500000.00,
  "currency": "NGN"
}
```

**Error Response (400 Bad Request):**
```json
{
  "status": "error",
  "message": "Input validation failed: Expected 10 features, received 8. Please check all required inputs."
}
```

#### 2. Get Locations and Property Types

**Endpoint:** `GET /locations`

**Description:** Returns available locations and property types for dropdown menus.

**Success Response (200 OK):**
```json
{
  "location_map": {
    "Ikeja": ["Gbagada", "Oshodi", "..."],
    "Lekki": ["Phase 1", "Phase 2", "..."]
  },
  "property_types": ["Apartment", "Terrace", "Flat", "Duplex", "..."]
}
```

## 📁 Project Structure

```
lagos-rent-predictor/
│
├── app.py                          # Main Flask application
├── preprocessing_pipeline.py        # Feature engineering utilities
├── index.html                       # Frontend web interface
├── requirements.txt                 # Python dependencies
├── runtime.txt                      # Python version specification
├── Procfile                         # Heroku deployment configuration
│
├── artifacts/                       # Model artifacts directory
│   ├── house_rent_prediction_model.joblib
│   ├── feature_names.joblib
│   ├── location_filter_map.joblib
│   └── property_types.joblib
│
└── README.md                        # This file
```

## 🤖 Model Details

### Model Architecture

- **Algorithm**: Random Forest Regressor
- **Target Variable**: Log-transformed rental price (NGN)
- **Training Data**: PropertyPro scraped data
- **Features**: 10 input features (3 categorical, 2 numerical, 5 binary)
- **Preprocessing**: One-Hot Encoding for categorical features

### Feature Engineering

The `preprocessing_pipeline.py` module handles:
- Conversion of input JSON to pandas DataFrame
- Type casting for numerical and categorical features
- Ensuring correct column order for model input

### Model Performance

The model uses log transformation for the target variable to handle the wide range of rental prices in Lagos. Predictions are inverse-transformed back to Nigerian Naira.

> 💡 **Note**: Model artifacts must match the exact versions of scikit-learn (1.6.1), pandas (2.2.3), and numpy used in production to avoid deserialization issues.

## ☁️ Deployment

### Deploying to Heroku

1. **Install Heroku CLI**
```bash
# Follow instructions at https://devcenter.heroku.com/articles/heroku-cli
```

2. **Login to Heroku**
```bash
heroku login
```

3. **Create a new app**
```bash
heroku create your-app-name
```

4. **Deploy**
```bash
git push heroku main
```

5. **Open your app**
```bash
heroku open
```

### Deploying to Render

1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn app:app`
5. Deploy

### Environment Variables

No environment variables are required for basic operation. All configuration is handled through the artifacts directory.

## 🔧 Troubleshooting

### Model Loading Errors

**Issue**: `Could not load artifacts` error on startup

**Solutions**:
- Verify all `.joblib` files exist in the `artifacts/` directory
- Ensure scikit-learn version matches: `pip install scikit-learn==1.6.1`
- Rebuild artifacts using the same Python and library versions

### Prediction Errors

**Issue**: `Input validation failed` error

**Solutions**:
- Ensure all 10 required features are included in the request
- Check data types match the API specification
- Verify boolean fields are `true`/`false` (not `1`/`0` as strings)

### CORS Issues

**Issue**: Frontend cannot access API from different domain

**Solutions**:
- flask-cors is already configured
- Check browser console for specific CORS errors
- Ensure the API URL in frontend code is correct

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines for Python code
- Add tests for new features
- Update documentation for API changes
- Ensure backward compatibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Data sourced from PropertyPro Nigeria
- Built with Flask and scikit-learn
- Inspired by the need for transparent rental pricing in Lagos

## 📞 Contact

For questions, issues, or suggestions:

- **GitHub Issues**: [Create an issue](https://github.com/headply/lagos-rent-predictor/issues)
- **Repository**: [headply/lagos-rent-predictor](https://github.com/headply/lagos-rent-predictor)

---

**Made with ❤️ for the Lagos real estate community**
