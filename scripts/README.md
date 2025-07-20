# Fraud Detection System

## Overview
A fraud detection system for e-commerce and banking transactions using advanced machine learning techniques and real-time monitoring. This project implements multiple ML models to detect fraudulent activities in both e-commerce and credit card transactions, featuring geolocation analysis, transaction pattern recognition, and an interactive dashboard for fraud insights.

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


## Folder Structure

- `notebooks/` : Jupyter notebooks for all the analysis.
- `scripts/`   : Python scripts files.


## scripts Folder/ 

```
    `preprocess.py`           : a script file for data cleaning and EDA analysis  
    `model_utils`             : a script file for model training pipeline  
```

## Setup Instructions
1. Clone the repository.
2. Set up the virtual environment.
3. Install dependencies using `pip install -r requirements.txt`
