"""
Training Script for Customer Churn Prediction Model

This script trains a Random Forest classifier for predicting customer churn
using telecom customer data. It preprocesses the data, trains the model,
evaluates performance, and saves the required artifacts.

Usage:
    python train_model.py --data telco_churn_data.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def load_and_prepare_data(filepath):
    """Load data and prepare features."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df


def create_pipeline(numerical_features, categorical_features):
    """Create the preprocessing and model pipeline."""
    # Preprocessing for numerical data
    numerical_transformer = StandardScaler()
    
    # Preprocessing for categorical data
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Bundle preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Create the full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    return pipeline


def train_model(X_train, y_train, pipeline):
    """Train the model."""
    print("\nTraining the model...")
    pipeline.fit(X_train, y_train)
    print("Training completed!")
    
    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    """Evaluate model performance."""
    print("\nEvaluating model performance...")
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("="*50)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }


def save_artifacts(pipeline, feature_names, categorical_mappings, output_dir='artifacts'):
    """Save model and related artifacts."""
    print(f"\nSaving model artifacts to '{output_dir}/' directory...")
    
    # Create artifacts directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the full pipeline
    joblib.dump(pipeline, os.path.join(output_dir, 'churn_prediction_model.joblib'))
    print(f"✓ Saved: churn_prediction_model.joblib")
    
    # Save feature names
    joblib.dump(feature_names, os.path.join(output_dir, 'feature_names.joblib'))
    print(f"✓ Saved: feature_names.joblib")
    
    # Save categorical mappings
    joblib.dump(categorical_mappings, os.path.join(output_dir, 'categorical_mappings.joblib'))
    print(f"✓ Saved: categorical_mappings.joblib")
    
    print("\nAll artifacts saved successfully!")


def main(data_path):
    """Main training pipeline."""
    print("="*60)
    print("CUSTOMER CHURN PREDICTION MODEL TRAINING")
    print("="*60)
    
    # 1. Load data
    df = load_and_prepare_data(data_path)
    
    # 2. Define features
    # These should match your actual dataset columns
    numerical_features = ['tenure', 'monthly_charges', 'total_charges']
    categorical_features = [
        'contract_type', 
        'payment_method', 
        'internet_service',
        'online_security', 
        'tech_support', 
        'streaming_tv', 
        'streaming_movies'
    ]
    
    all_features = numerical_features + categorical_features
    target_column = 'Churn'  # Adjust this to match your target column name
    
    # 3. Prepare features and target
    print(f"\nPreparing features and target...")
    X = df[all_features]
    y = df[target_column]
    
    # If target is string (e.g., 'Yes'/'No'), encode it
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"Target classes: {le.classes_}")
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # 4. Split data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 5. Create pipeline
    pipeline = create_pipeline(numerical_features, categorical_features)
    
    # 6. Train model
    trained_pipeline = train_model(X_train, y_train, pipeline)
    
    # 7. Evaluate model
    metrics = evaluate_model(trained_pipeline, X_test, y_test)
    
    # 8. Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
    print(f"Cross-validation ROC-AUC scores: {cv_scores}")
    print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 9. Save artifacts
    categorical_mappings = {
        'contract_type': ['Month-to-month', 'One year', 'Two year'],
        'payment_method': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
        'internet_service': ['DSL', 'Fiber optic', 'No'],
        'online_security': ['Yes', 'No', 'No internet service'],
        'tech_support': ['Yes', 'No', 'No internet service'],
        'streaming_tv': ['Yes', 'No', 'No internet service'],
        'streaming_movies': ['Yes', 'No', 'No internet service']
    }
    
    save_artifacts(trained_pipeline, all_features, categorical_mappings)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nYou can now run the Flask API with: python app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a customer churn prediction model'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to the CSV file containing customer data'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"Error: Data file '{args.data}' not found!")
        exit(1)
    
    main(args.data)
