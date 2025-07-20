import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import ipaddress
from typing import Tuple, Dict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import socket
import struct

def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all required datasets
    
    Returns:
        Tuple containing fraud_data, ip_country, and credit_card DataFrames
    """
    fraud_data = pd.read_csv('../data/Fraud_Data.csv')
    ip_country = pd.read_csv('../data/IpAddress_to_Country.csv')
    credit_card = pd.read_csv('../data/creditcard.csv')
    return fraud_data, ip_country, credit_card

def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Check for missing values in the dataframe
    
    Args:
        df: Input DataFrame
    
    Returns:
        Series containing count of missing values for each column
    """
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_info = pd.concat([missing_values, missing_percentage], axis=1)
    missing_info.columns = ['Missing Count', 'Missing Percentage']
    return missing_info[missing_info['Missing Count'] > 0]

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataframe
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with handled missing values
    """
    df_copy = df.copy()
    
    # For numeric columns, impute with median
    numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df_copy[col].isnull().sum() > 0:
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    
    # For categorical columns, impute with mode
    categorical_columns = df_copy.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df_copy[col].isnull().sum() > 0:
            df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
    
    return df_copy

def check_duplicates(df: pd.DataFrame) -> Dict:
    """
    Check for duplicate entries in the dataframe
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dictionary containing duplicate statistics
    """
    total_duplicates = df.duplicated().sum()
    duplicate_percentage = (total_duplicates / len(df)) * 100
    return {
        'total_rows': len(df),
        'duplicate_rows': total_duplicates,
        'duplicate_percentage': duplicate_percentage
    }

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data by removing duplicates and correcting data types
    
    Args:
        df: Input DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    df_copy = df.copy()
    
    # Remove duplicates
    df_copy = df_copy.drop_duplicates()
    
    # Convert timestamps to datetime
    if 'signup_time' in df_copy.columns:
        df_copy['signup_time'] = pd.to_datetime(df_copy['signup_time'])
    if 'purchase_time' in df_copy.columns:
        df_copy['purchase_time'] = pd.to_datetime(df_copy['purchase_time'])
    
    return df_copy

def get_data_info(df: pd.DataFrame) -> None:
    """
    Print basic information about the dataset on separate lines
    
    Args:
        df: Input DataFrame
    """
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Data types:\n{df.dtypes.value_counts().to_dict()}")



def ip_to_int(ip):
    """Convert an IP address to its integer representation."""
    try:
        return struct.unpack("!I", socket.inet_aton(ip))[0]
    except socket.error:
        return None  # Handle invalid IPs gracefully



def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features for fraud detection
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with additional time-based features
    """
    df_copy = df.copy()
    
    if 'purchase_time' in df_copy.columns:
        df_copy['hour_of_day'] = df_copy['purchase_time'].dt.hour
        df_copy['day_of_week'] = df_copy['purchase_time'].dt.dayofweek
        df_copy['month'] = df_copy['purchase_time'].dt.month
        df_copy['is_weekend'] = df_copy['day_of_week'].isin([5, 6]).astype(int)
        
        # Calculate time difference between signup and purchase
        if 'signup_time' in df_copy.columns:
            df_copy['time_to_purchase'] = (df_copy['purchase_time'] - df_copy['signup_time']).dt.total_seconds()
    
    return df_copy

def calculate_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate transaction frequency and velocity features
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with additional transaction-based features
    """
    df_copy = df.copy()
    
    if 'user_id' in df_copy.columns:
        # Transaction frequency per user
        user_freq = df_copy.groupby('user_id').size().reset_index(name='transaction_count')
        df_copy = df_copy.merge(user_freq, on='user_id', how='left')
        
        # Transaction velocity (transactions per day)
        if 'purchase_time' in df_copy.columns:
            user_velocity = df_copy.groupby('user_id').agg({
                'purchase_time': lambda x: (x.max() - x.min()).total_seconds() / 86400
            }).reset_index()
            user_velocity.columns = ['user_id', 'user_activity_period_days']
            df_copy = df_copy.merge(user_velocity, on='user_id', how='left')
            df_copy['transaction_velocity'] = df_copy['transaction_count'] / df_copy['user_activity_period_days'].clip(lower=1)
            
        # Average purchase value per user
        user_avg_purchase = df_copy.groupby('user_id')['purchase_value'].mean().reset_index()
        user_avg_purchase.columns = ['user_id', 'avg_purchase_value']
        df_copy = df_copy.merge(user_avg_purchase, on='user_id', how='left')
    
    return df_copy

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features using one-hot encoding
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with encoded categorical features
    """
    df_copy = df.copy()
    
    categorical_columns = df_copy.select_dtypes(include=['object']).columns
    exclude_columns = ['ip_address', 'user_id', 'device_id']
    
    categorical_columns = [col for col in categorical_columns if col not in exclude_columns]
    
    # Apply one-hot encoding
    df_encoded = pd.get_dummies(df_copy, columns=categorical_columns, drop_first=True)
    
    return df_encoded

def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize numerical features using StandardScaler
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with normalized features
    """
    df_copy = df.copy()
    scaler = StandardScaler()
    
    numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
    exclude_columns = ['class', 'Class', 'user_id', 'device_id']

    numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
    
    df_copy[numeric_columns] = scaler.fit_transform(df_copy[numeric_columns])
    
    return df_copy

def plot_distribution(df: pd.DataFrame, column: str, title: str = None):
    """
    Plot distribution of a column using plotly
    
    Args:
        df: Input DataFrame
        column: Column name to plot
        title: Plot title
    """
    if df[column].dtype in ['int64', 'float64']:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[column], name=column))
        fig.update_layout(
            title=title or f'Distribution of {column}',
            xaxis_title=column,
            yaxis_title='Count'
        )
    else:
        value_counts = df[column].value_counts()
        fig = go.Figure(data=[go.Bar(x=value_counts.index, y=value_counts.values)])
        fig.update_layout(
            title=title or f'Distribution of {column}',
            xaxis_title=column,
            yaxis_title='Count'
        )
    fig.show()

def plot_correlation_matrix(df: pd.DataFrame, title: str = None):
    """
    Plot correlation matrix using plotly
    
    Args:
        df: Input DataFrame
        title: Plot title
    """
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu'
    ))
    
    fig.update_layout(
        title=title or 'Correlation Matrix',
        width=800,
        height=800
    )
    fig.show()

def analyze_fraud_patterns(df: pd.DataFrame):
    """
    Analyze and plot fraud patterns
    
    Args:
        df: Input DataFrame with 'class' column (0 for non-fraud, 1 for fraud)
    """
    if 'class' not in df.columns and 'Class' not in df.columns:
        raise ValueError("DataFrame must contain 'class' or 'Class' column")
    
    target_col = 'class' if 'class' in df.columns else 'Class'
    
    # Create subplots
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Fraud Distribution',
                                      'Fraud by Hour of Day',
                                      'Fraud by Day of Week',
                                      'Average Purchase Value by Class'))
    
    # Fraud Distribution
    fraud_dist = df[target_col].value_counts(normalize=True) * 100
    fig.add_trace(
        go.Bar(x=['Non-Fraud', 'Fraud'], y=fraud_dist.values),
        row=1, col=1
    )
    
    # Fraud by Hour of Day
    if 'hour_of_day' in df.columns:
        hourly_fraud = df[df[target_col] == 1]['hour_of_day'].value_counts().sort_index()
        fig.add_trace(
            go.Scatter(x=hourly_fraud.index, y=hourly_fraud.values, mode='lines+markers'),
            row=1, col=2
        )
    
    # Fraud by Day of Week
    if 'day_of_week' in df.columns:
        daily_fraud = df[df[target_col] == 1]['day_of_week'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], y=daily_fraud.values),
            row=2, col=1
        )
    
    # Average Purchase Value by Class
    if 'purchase_value' in df.columns:
        avg_purchase = df.groupby(target_col)['purchase_value'].mean()
        fig.add_trace(
            go.Bar(x=['Non-Fraud', 'Fraud'], y=avg_purchase.values),
            row=2, col=2
        )
    
    fig.update_layout(height=800, width=1000, showlegend=False)
    fig.show()