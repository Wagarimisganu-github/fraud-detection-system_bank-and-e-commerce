import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import mlflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM
import tensorflow as tf

class FraudDetectionModeling:
    def __init__(self, experiment_name="fraud_detection"):
        """Initialize the modeling class with experiment tracking."""
        mlflow.set_experiment(experiment_name)
        self.models = {}
        self.results = {}
        
    def prepare_data(self, X, y, test_size=0.2, random_state=42, sampling_strategy='balanced'):
        """
        Prepare and split the data for modeling with optional sampling for imbalanced datasets.
        """
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        if sampling_strategy == 'balanced':
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state)
            self.X_train_scaled, self.y_train = smote.fit_resample(self.X_train_scaled, self.y_train)
            
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def initialize_traditional_models(self, random_state=42):
        """Initialize traditional machine learning models."""
        self.models.update({
            'logistic_regression': LogisticRegression(
                class_weight='balanced',
                random_state=random_state
            ),
            'decision_tree': DecisionTreeClassifier(
                class_weight='balanced',
                random_state=random_state
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=random_state
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=random_state
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=100,
                random_state=random_state
            )
        })
        
    def initialize_deep_models(self, input_shape):
        """Initialize deep learning models."""
        # CNN Model
        cnn_model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(100, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        # RNN Model
        rnn_model = Sequential([
            tf.keras.layers.SimpleRNN(64, input_shape=input_shape, return_sequences=True),
            tf.keras.layers.SimpleRNN(32),
            Dense(1, activation='sigmoid')
        ])
        
        # LSTM Model
        lstm_model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            LSTM(32),
            Dense(1, activation='sigmoid')
        ])
        
        self.models.update({
            'cnn': cnn_model,
            'rnn': rnn_model,
            'lstm': lstm_model
        })
    
    def train_traditional_model(self, model_name):
        """Train and evaluate a traditional machine learning model."""
        model = self.models[model_name]
        
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_params(model.get_params())
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
            
            # Log metrics
            mlflow.log_metrics({
                "accuracy": metrics['classification_report']['accuracy'],
                "precision": metrics['classification_report']['weighted avg']['precision'],
                "recall": metrics['classification_report']['weighted avg']['recall'],
                "f1": metrics['classification_report']['weighted avg']['f1-score'],
                "auc_roc": metrics['roc_curve'][2]
            })
            
            # Log model
            mlflow.sklearn.log_model(model, model_name)
            
            self.results[model_name] = metrics
            
    def train_deep_model(self, model_name, epochs=10, batch_size=32):
        """Train and evaluate a deep learning model."""
        model = self.models[model_name]
        
        # Reshape input data for deep learning models
        X_train_reshaped = self.X_train_scaled.reshape(
            self.X_train_scaled.shape[0], 
            self.X_train_scaled.shape[1], 
            1
        )
        X_test_reshaped = self.X_test_scaled.reshape(
            self.X_test_scaled.shape[0], 
            self.X_test_scaled.shape[1], 
            1
        )
        
        with mlflow.start_run(run_name=f"deep_{model_name}"):
            # Compile model
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            history = model.fit(
                X_train_reshaped, self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test_reshaped, self.y_test),
                verbose=1
            )
            
            # Make predictions
            y_pred_proba = model.predict(X_test_reshaped)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
            metrics['history'] = history.history
            
            # Log metrics
            mlflow.log_metrics({
                "test_accuracy": metrics['classification_report']['accuracy'],
                "test_precision": metrics['classification_report']['weighted avg']['precision'],
                "test_recall": metrics['classification_report']['weighted avg']['recall'],
                "test_f1": metrics['classification_report']['weighted avg']['f1-score'],
                "test_auc_roc": metrics['roc_curve'][2]
            })
            
            # Log model
            mlflow.keras.log_model(model, model_name)
            
            self.results[model_name] = metrics
            
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate and return all evaluation metrics."""
        metrics = {
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        metrics['roc_curve'] = (fpr, tpr, auc(fpr, tpr))
        
        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        metrics['pr_curve'] = (precision, recall, auc(recall, precision))
        
        return metrics
    
    def plot_results(self):
        """Plot evaluation metrics for all models."""
        for model_name, metrics in self.results.items():
            # Plot confusion matrix
            plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            sns.heatmap(
                metrics['confusion_matrix'], 
                annot=True, 
                fmt='d',
                cmap='Blues'
            )
            plt.title(f'{model_name} - Confusion Matrix')
            
            # Plot ROC curve
            plt.subplot(132)
            fpr, tpr, roc_auc = metrics['roc_curve']
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} - ROC Curve')
            plt.legend()
            
            # Plot Precision-Recall curve
            plt.subplot(133)
            precision, recall, pr_auc = metrics['pr_curve']
            plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'{model_name} - Precision-Recall Curve')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
            # For deep learning models, plot training history
            if 'history' in metrics:
                plt.figure(figsize=(12, 4))
                
                plt.subplot(121)
                plt.plot(metrics['history']['accuracy'], label='Training')
                plt.plot(metrics['history']['val_accuracy'], label='Validation')
                plt.title(f'{model_name} - Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                
                plt.subplot(122)
                plt.plot(metrics['history']['loss'], label='Training')
                plt.plot(metrics['history']['val_loss'], label='Validation')
                plt.title(f'{model_name} - Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                
                plt.tight_layout()
                plt.show()

def main():
    # Example usage
    # Initialize the modeling class
    fraud_detector = FraudDetectionModeling(experiment_name="fraud_detection")
    
    
    # Initialize models
    fraud_detector.initialize_traditional_models()
    fraud_detector.initialize_deep_models(input_shape=(X_train_scaled.shape[1], 1))
    
    # Train traditional models
    for model_name in ['logistic_regression', 'decision_tree', 'random_forest', 
                      'gradient_boosting', 'mlp']:
        fraud_detector.train_traditional_model(model_name)
    
    # Train deep learning models
    for model_name in ['cnn', 'rnn', 'lstm']:
        fraud_detector.train_deep_model(model_name)
    
    # Plot results
    fraud_detector.plot_results()

if __name__ == "__main__":
    main()
