import requests
import json

def test_health_check():
    """Test the health check endpoint."""
    response = requests.get('http://localhost:5000/health')
    print("Health Check Response:", response.json())
    assert response.status_code == 200

def test_ecommerce_prediction():
    """Test the e-commerce fraud prediction endpoint."""
    # Sample e-commerce transaction data
    sample_data = {
        'user_id': 247547,
        'purchase_value':0.5496066136744162 ,
        'device_id': 46780,
        'age': -0.363124425973929,
        'ip_address':-1.72872419284537,
        'ip_int': -1.72872419245195,
        'hour_of_day':-1.23112401227313,
        'day_of_week':1.487910955460511,
        'is_weekend':1.566179094584172,
        'time_to_purchase':-0.413799987928256,
        'transaction_count':0,
        'user_activity_period_days':0,
        'transaction_velocity':0, 
        'avg_purchase_value':0.5496066136744162,
        'age_group':1,
        'source_Direct':False,
        'source_SEO':True,
        'browser_FireFox':False,
        'browser_IE':False,
        'browser_Opera':False,
        'browser_Safari':True,
        'sex_M':False,
        'signup_hour':22,
        'signup_day':1,
        'signup_year':2015,
        'signup_month':2,
        'purchase_hour':2,
        'purchase_day':5,
        'purchase_year':2015,
        'purchase_month':4,
    }


    response = requests.post(
        'http://localhost:5000/predict/ecommerce',
        json=sample_data
    )
    print("E-commerce Prediction Response:", response.json())
    assert response.status_code == 200

def test_credit_prediction():
    """Test the credit card fraud prediction endpoint."""
    # Sample credit card transaction data
    sample_data = {
        'Time':-1.996822,
        'V1': -0.701082,
        'V2': -0.041687,
        'V3': 1.680101,
        'V4': 0.976623,
        'V5': -0.247020,
        'V6': 0.348012,
        'V7': 0.193700,
        'V8': 0.084434,
        'V9': 0.333534,
        'V10': 0.085687,
        'V11': -0.541662,
        'V12': -0.620391,
        'V13': -0.996549,
        'V14': -0.327050,
        'V15': 1.603614,
        'V16': -0.539733,
        'V17': 0.246646,
        'V18': 0.028989,
        'V19': 0.497010,
        'V20': 0.326273,
        'V21': -0.024776,
        'V22': 0.383483,
        'V23': -0.177444,
        'V24': 0.110156,
        'V25': 0.247058,
        'V26': -0.392622,
        'V27': 0.333032,
        'V28': -0.065849,
        'Amount': 0.244199
    }
    
    response = requests.post(
        'http://localhost:5000/predict/credit',
        json=sample_data
    )
    print("Credit Card Prediction Response:", response.json())
    assert response.status_code == 200

if __name__ == '__main__':
    print("Testing API endpoints...")
    test_health_check()
    test_ecommerce_prediction()
    test_credit_prediction()
    print("All tests passed!")