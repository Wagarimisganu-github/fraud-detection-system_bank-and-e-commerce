# Fraud Detection System

A fraud detection system for e-commerce and banking transactions using advanced machine learning techniques and real-time monitoring. This project implements multiple ML models to detect fraudulent activities in both e-commerce and credit card transactions, featuring geolocation analysis, transaction pattern recognition, and an interactive dashboard for fraud insights.

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Dashboard](#dashboard)
- [Model Performance](#model-performance)

## ✨ Features

- **Fraud Detection**
  - E-commerce transaction fraud detection
  - Credit card fraud detection

- **Advanced ML Models**
  - Traditional ML models (Logistic Regression,Decision Tree, Random Forest, Gradient Boosting, MLP)
  - Deep Learning models (CNN, RNN, LSTM)
  - Model explainability using SHAP and LIME

- **Real-time Processing**
  - Flask REST API for real-time predictions
  - Dockerized deployment
  - Scalable architecture

- **Interactive Dashboard**
  - Real-time fraud monitoring
  - Geographical fraud distribution
  - Transaction pattern visualization
  - Device and browser analysis


## 🏗️ Project Structure

```
fraud-detection-system/
│
├── scripts/
│   ├── preprocess.py
|   ├── model_utils.py
│   └── README.md
├── notebooks/
│   ├── data_analysis.ipynb
│   ├── model_training_explainability.ipynb
│   ├── model_training.ipynb
│   └── README.md
├── fraud_detection_dashboard/    # directory for flask dashboard
│   └── app.py
├── fraud-detection-api/     # directory for flaskapi call 
│   ├── Dockerfile
│   ├── serve_model.py
│   └── test_api.py
│   └── logs/    # log files 
│   └── models/
|   |   └── model_features/
|   |   └── pkl/
|   └── models/
|   └── requirements.txt
|
├── models/   # model files 
├── tests/
├── requirements.txt    # all dependencies 
└── README.md
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/OL-YAD/fraud-detection-system.git
cd fraud-detection-system
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Training Models

```bash
python scripts/model_utils.py 
```

### Running and Testing the API

```bash
python fraud-detection-api/serve_model.py
python fraud-detection-api/test_api.py
```

### Running with Docker

```bash
docker build -t fraud-detection-api .
docker run -p 5000:5000 fraud-detection-api
```

### Accessing the Dashboard

Navigate to `http://127.0.0.1:8050/` after starting the application.

## 📚 API Documentation

### Endpoints

#### Fraud Detection Endpoint

**Request Body:**
```json
{
    "user_id": 247547,
    "purchase_value": 0.5496066136744162,
    "device_id": 46780,
    "age": -0.363124425973929,
    "ip_address": -1.72872419284537,
    "ip_int": -1.72872419245195,
    "hour_of_day": -1.23112401227313,
    "day_of_week": 1.487910955460511,
    "is_weekend": 1.566179094584172,
    "time_to_purchase": -0.413799987928256,
    "transaction_count": 0,
    "user_activity_period_days": 0,
    "transaction_velocity": 0,
    "avg_purchase_value": 0.5496066136744162,
    "age_group": 1,
    "source_Direct": false,
    "source_SEO": true,
    "browser_FireFox": false,
    "browser_IE": false,
    "browser_Opera": false,
    "browser_Safari": true,
    "sex_M": false,
    "signup_hour": 22,
    "signup_day": 1,
    "signup_year": 2015,
    "signup_month": 2,
    "purchase_hour": 2,
    "purchase_day": 5,
    "purchase_year": 2015,
    "purchase_month": 4
}
```

**Response:**
```json
{
    "prediction": 0,
    "probability": 0.11403758098435984,
    "timestamp": '2024-10-31T10:15:01.099631',
}
```


#### Credit Card Fraud Detection Endpoint

**Request Body:**
```json
{
    "Time": -1.996822,
    "V1": -0.701082,
    "V2": -0.041687,
    "V3": 1.680101,
    ...
    "V28": -0.065849,
    "Amount": 0.244199
}
```

**Response:**
```json
{
    "prediction": 0,
    "probability": 0.01,
    "timestamp": '2024-10-31T10:26:54.569307',
}
```

## 📊 Dashboard

The dashboard provides real-time monitoring and visualization of:
- Transaction volumes and fraud rates
- Geographical distribution of fraud
- Device and browser-based fraud patterns
- Time-based fraud trends

## 📈 Model Performance

## 🏆 Model Performance Comparison for E-commerce Fraud Detection

| Model               | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.662    | 0.880     | 0.662  | 0.732    | 0.764   |
| Decision Tree       | 0.893    | 0.907     | 0.893  | 0.899    | 0.755   |
| Random Forest       | 0.955    | 0.956     | 0.955  | 0.950    | 0.769   |
| Gradient Boosting   | 0.956    | 0.957     | 0.956  | 0.950    | 0.771   |
| MLP                 | 0.836    | 0.889     | 0.836  | 0.857    | 0.755   |
| CNN                 | 0.899    | 0.907     | 0.899  | 0.903    | 0.761   |
| RNN                 | 0.914    | 0.915     | 0.914  | 0.915    | 0.767   |
| LSTM                | 0.895    | 0.907     | 0.895  | 0.900    | 0.770   |


## 🏆 Model Performance Comparison for Credit Card Fraud Detection

| Model               | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.974    | 0.998     | 0.974  | 0.985    | 0.963   |
| Decision Tree       | 0.997    | 0.998     | 0.997  | 0.998    | 0.820   |
| Random Forest       | 0.999    | 0.999     | 0.999  | 0.999    | 0.969   |
| Gradient Boosting   | 0.989    | 0.998     | 0.989  | 0.993    | 0.977   |
| MLP                 | 0.999    | 0.999     | 0.999  | 0.999    | 0.946   |
| CNN                 | 0.999    | 0.999     | 0.999  | 0.999    | 0.933   |
| RNN                 | 0.988    | 0.998     | 0.988  | 0.993    | 0.952   |
| LSTM                | 0.998    | 0.999     | 0.998  | 0.998    | 0.924   |



## 🙏 Acknowledgments
- 10 Academy for the training and support

## 📧 Contact

For questions and feedback:
- Email: olyadtemesgen@gmail.com
- LinkedIn: [https://www.linkedin.com/in/olyad-temesgen/]
