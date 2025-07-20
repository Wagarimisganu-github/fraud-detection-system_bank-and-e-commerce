from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    filename='./logs/fraud_detection.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load the models and scalers
try:
    ecommerce_model = joblib.load('./models/model_features/best_ecommerce_fraud_model.joblib')
    credit_model = joblib.load('./models/model_features/best_credit_fraud_model.joblib')
    ecommerce_scaler = joblib.load('./models/model_features/ecommerce_scaler.joblib')
    credit_scaler = joblib.load('./models/model_features/credit_scaler.joblib')
    logger.info("Models and scalers loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

def preprocess_ecommerce_data(data):
    """Preprocess e-commerce transaction data."""
    try:
        # Convert data to DataFrame if it's not already
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame([data])
        
        # Scale the numerical features
        scaled_data = ecommerce_scaler.transform(data)
        
        return scaled_data
    except Exception as e:
        logger.error(f"Error preprocessing e-commerce data: {str(e)}")
        raise

def preprocess_credit_data(data):
    """Preprocess credit card transaction data."""
    try:
        # Convert data to DataFrame if it's not already
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame([data])
        
        # Scale the numerical features
        scaled_data = credit_scaler.transform(data)
        
        return scaled_data
    except Exception as e:
        logger.error(f"Error preprocessing credit card data: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/predict/ecommerce', methods=['POST'])
def predict_ecommerce():
    """Endpoint for e-commerce fraud prediction."""
    try:
        # Get data from request
        data = request.get_json()
        
        # Log the incoming request
        logger.info(f"Received e-commerce prediction request: {data}")
        
        # Preprocess the data
        processed_data = preprocess_ecommerce_data(data)
        
        # Make prediction
        prediction = ecommerce_model.predict(processed_data)
        prediction_proba = ecommerce_model.predict_proba(processed_data)
        
        # Prepare response
        response = {
            'prediction': int(prediction[0]),
            'probability': float(prediction_proba[0][1]),
            'timestamp': datetime.now().isoformat()
        }
        
        # Log the prediction
        logger.info(f"E-commerce prediction made: {response}")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing e-commerce prediction request: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/credit', methods=['POST'])
def predict_credit():
    """Endpoint for credit card fraud prediction."""
    try:
        # Get data from request
        data = request.get_json()
        
        # Log the incoming request
        logger.info(f"Received credit card prediction request: {data}")
        
        # Preprocess the data
        processed_data = preprocess_credit_data(data)
        
        # Make prediction
        prediction = credit_model.predict(processed_data)
        prediction_proba = credit_model.predict_proba(processed_data)
        
        # Prepare response
        response = {
            'prediction': int(prediction[0]),
            'probability': float(prediction_proba[0][1]),
            'timestamp': datetime.now().isoformat()
        }
        
        # Log the prediction
        logger.info(f"Credit card prediction made: {response}")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing credit card prediction request: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)