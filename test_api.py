"""
API Test Script

This script tests the basic structure and endpoints of the churn prediction API.
Note: This will fail if model artifacts are not present, but it verifies the API structure.
"""

import requests
import json


def test_predict_endpoint():
    """Test the /predict endpoint with sample data."""
    print("Testing /predict endpoint...")
    
    url = "http://localhost:5000/predict"
    
    # Sample customer data
    sample_data = {
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
    
    try:
        response = requests.post(url, json=sample_data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            result = response.json()
            if 'churn_prediction' in result and 'churn_probability' in result:
                print("✓ Prediction endpoint working correctly!")
                return True
        else:
            print("✗ Prediction endpoint returned error (expected if model not trained)")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to API. Is the server running?")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_categories_endpoint():
    """Test the /categories endpoint."""
    print("\nTesting /categories endpoint...")
    
    url = "http://localhost:5000/categories"
    
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            result = response.json()
            if 'categorical_mappings' in result:
                print("✓ Categories endpoint working correctly!")
                return True
        else:
            print("✗ Categories endpoint returned error")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to API. Is the server running?")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_invalid_input():
    """Test the API with invalid input."""
    print("\nTesting with invalid input (should return 400)...")
    
    url = "http://localhost:5000/predict"
    
    # Invalid data - only 5 features instead of 10
    invalid_data = {
        "tenure": 12,
        "monthly_charges": 70.00,
        "total_charges": 840.00,
        "contract_type": "Month-to-month",
        "payment_method": "Electronic check"
    }
    
    try:
        response = requests.post(url, json=invalid_data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 400:
            print("✓ Validation working correctly!")
            return True
        else:
            print("✗ Expected 400 error for invalid input")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to API. Is the server running?")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("CHURN PREDICTION API TESTS")
    print("=" * 60)
    print("\nNote: Make sure the API is running (python app.py)")
    print("Some tests may fail if model artifacts are not present.\n")
    
    results = []
    
    # Run tests
    results.append(("Predict Endpoint", test_predict_endpoint()))
    results.append(("Categories Endpoint", test_categories_endpoint()))
    results.append(("Invalid Input Validation", test_invalid_input()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
