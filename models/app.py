from dash import Dash, html, dcc
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask
import pandas as pd
from datetime import datetime
import numpy as np
import socket
import struct

# Initialize Flask app
server = Flask(__name__)

# Initialize Dash app
app = Dash(__name__, server=server)

def ip_to_int(ip):
    """Convert an IP address to its integer representation."""
    try:
        return struct.unpack("!I", socket.inet_aton(ip))[0]
    except socket.error:
        return None  # Handle invalid IPs gracefully

# Load datasets
def load_data():
    # Load e-commerce fraud data
    fraud_data = pd.read_csv('../data/Fraud_Data.csv')
    # Load credit card fraud data
    credit_data = pd.read_csv('../data/creditcard.csv')
    # Load IP to country mapping
    ip_country = pd.read_csv('../data/IpAddress_to_Country.csv')
    
    return fraud_data, credit_data, ip_country

# Data processing functions
def process_ecommerce_data(fraud_data, ip_country):
    # Create a copy to avoid modifying original data
    fraud_data_cleaned = fraud_data.copy()
    ip_country_cleaned = ip_country.copy()
    
    # Convert timestamps
    fraud_data_cleaned['signup_time'] = pd.to_datetime(fraud_data_cleaned['signup_time'])
    fraud_data_cleaned['purchase_time'] = pd.to_datetime(fraud_data_cleaned['purchase_time'])
    
    # Add day and hour columns
    fraud_data_cleaned['purchase_day'] = fraud_data_cleaned['purchase_time'].dt.day_name()
    fraud_data_cleaned['purchase_hour'] = fraud_data_cleaned['purchase_time'].dt.hour
    
    # Convert IP addresses to integers
    fraud_data_cleaned['ip_int'] = fraud_data_cleaned['ip_address'].apply(
        lambda x: ip_to_int(str(int(x))) if pd.notna(x) else None
    )
    
    # Convert IP address bounds to integers
    ip_country_cleaned['lower_bound_ip_address'] = ip_country_cleaned['lower_bound_ip_address'].astype('int64')
    ip_country_cleaned['upper_bound_ip_address'] = ip_country_cleaned['upper_bound_ip_address'].astype('int64')
    
    # Ensure 'ip_int' is int64
    fraud_data_cleaned['ip_int'] = fraud_data_cleaned['ip_int'].astype('int64')
    
    # Sort the IP range dataset
    ip_country_cleaned.sort_values('lower_bound_ip_address', inplace=True)
    
    # Perform merge based on IP ranges
    fraud_data_with_country = pd.merge_asof(
        fraud_data_cleaned.sort_values('ip_int'),
        ip_country_cleaned[['lower_bound_ip_address', 'upper_bound_ip_address', 'country']],
        left_on='ip_int',
        right_on='lower_bound_ip_address',
        direction='backward'
    )
    
    # Filter results where ip_int falls within IP bounds
    fraud_data_with_country = fraud_data_with_country[
        (fraud_data_with_country['ip_int'] >= fraud_data_with_country['lower_bound_ip_address']) &
        (fraud_data_with_country['ip_int'] <= fraud_data_with_country['upper_bound_ip_address'])
    ]
    
    # Drop unnecessary columns
    fraud_data_with_country.drop(['lower_bound_ip_address', 'upper_bound_ip_address'], axis=1, inplace=True)
    
    return fraud_data_with_country

def create_summary_stats(fraud_data, credit_data):
    # E-commerce statistics
    ecom_stats = {
        'total_transactions': len(fraud_data),
        'fraud_cases': fraud_data['class'].sum(),
        'fraud_percentage': (fraud_data['class'].sum() / len(fraud_data) * 100).round(2)
    }
    
    # Credit card statistics
    credit_stats = {
        'total_transactions': len(credit_data),
        'fraud_cases': credit_data['Class'].sum(),
        'fraud_percentage': (credit_data['Class'].sum() / len(credit_data) * 100).round(2)
    }
    
    return ecom_stats, credit_stats

# Load and process data
fraud_data, credit_data, ip_country = load_data()
fraud_data_processed = process_ecommerce_data(fraud_data, ip_country)
ecom_stats, credit_stats = create_summary_stats(fraud_data_processed, credit_data)

# Create the dashboard layout
app.layout = html.Div([
    # Navigation bar
    html.Div([
        html.H1('Fraud Detection Dashboard', className='nav-title'),
        html.P('Real-time fraud analytics and insights', className='nav-subtitle')
    ], className='navbar'),
    
    # Main content container
    html.Div([
        # Summary Statistics Cards
        html.Div([
            html.Div([
                html.Div([
                    html.I(className='fas fa-shopping-cart stat-icon'),
                    html.Div([
                        html.H3('E-commerce Transactions'),
                        html.Div([
                            html.P([
                                html.Span('Total Transactions', className='stat-label'),
                                html.Span(f"{ecom_stats['total_transactions']:,}", className='stat-value')
                            ]),
                            html.P([
                                html.Span('Fraud Cases', className='stat-label'),
                                html.Span(f"{ecom_stats['fraud_cases']:,}", className='stat-value fraud-value')
                            ]),
                            html.P([
                                html.Span('Fraud Percentage', className='stat-label'),
                                html.Span(f"{ecom_stats['fraud_percentage']}%", className='stat-value fraud-value')
                            ])
                        ], className='stat-details')
                    ])
                ], className='stat-card')
            ], className='col-md-6'),
            
            html.Div([
                html.Div([
                    html.I(className='fas fa-credit-card stat-icon'),
                    html.Div([
                        html.H3('Credit Card Transactions'),
                        html.Div([
                            html.P([
                                html.Span('Total Transactions', className='stat-label'),
                                html.Span(f"{credit_stats['total_transactions']:,}", className='stat-value')
                            ]),
                            html.P([
                                html.Span('Fraud Cases', className='stat-label'),
                                html.Span(f"{credit_stats['fraud_cases']:,}", className='stat-value fraud-value')
                            ]),
                            html.P([
                                html.Span('Fraud Percentage', className='stat-label'),
                                html.Span(f"{credit_stats['fraud_percentage']}%", className='stat-value fraud-value')
                            ])
                        ], className='stat-details')
                    ])
                ], className='stat-card')
            ], className='col-md-6')
        ], className='row stats-container'),
        
        # Charts Section
        html.Div([
            html.Div([
                html.Div([
                    html.H3('Fraud Trends Over Time', className='chart-title'),
                    dcc.Graph(
                        figure=px.line(
                            fraud_data_processed.groupby(fraud_data_processed['purchase_time'].dt.date)['class'].sum().reset_index(),
                            x='purchase_time',
                            y='class',
                            template='plotly_white'
                        ).update_traces(line_color='#e74c3c')
                    )
                ], className='chart-card')
            ], className='mb-4'),
            
            html.Div([
                html.H3('Geographical Distribution of Fraud', className='chart-title'),
                dcc.Graph(
                    figure=px.choropleth(
                        fraud_data_processed[fraud_data_processed['class'] == 1].groupby('country').size().reset_index(name='count'),
                        locations='country',
                        locationmode='country names',
                        color='count',
                        color_continuous_scale='Reds',
                        template='plotly_white'
                    )
                )
            ], className='chart-card mb-4'),
            
            # Device and Browser Analysis in cards
            html.Div([
                html.Div([
                    html.Div([
                        html.H3('Fraud by Device', className='chart-title'),
                        dcc.Graph(
                            figure=px.bar(
                                fraud_data_processed.groupby(['device_id', 'class']).size().unstack().fillna(0),
                                template='plotly_white',
                                color_discrete_sequence=['#2ecc71', '#e74c3c']
                            )
                        )
                    ], className='chart-card')
                ], className='col-md-6'),
                
                html.Div([
                    html.Div([
                        html.H3('Fraud by Browser', className='chart-title'),
                        dcc.Graph(
                            figure=px.bar(
                                fraud_data_processed.groupby(['browser', 'class']).size().unstack().fillna(0),
                                template='plotly_white',
                                color_discrete_sequence=['#2ecc71', '#e74c3c']
                            )
                        )
                    ], className='chart-card')
                ], className='col-md-6')
            ], className='row mb-4'),
            
            # Time Patterns
            html.Div([
                html.Div([
                    html.Div([
                        html.H3('Fraud by Hour of Day', className='chart-title'),
                        dcc.Graph(
                            figure=px.bar(
                                fraud_data_processed[fraud_data_processed['class'] == 1].groupby('purchase_hour').size(),
                                template='plotly_white',
                                color_discrete_sequence=['#e74c3c']
                            ).update_layout(
                                xaxis_title='Hour of Day',
                                yaxis_title='Number of Fraud Cases'
                            )
                        )
                    ], className='chart-card')
                ], className='col-md-6'),
                
                html.Div([
                    html.Div([
                        html.H3('Fraud by Day of Week', className='chart-title'),
                        dcc.Graph(
                            figure=px.bar(
                                fraud_data_processed[fraud_data_processed['class'] == 1].groupby('purchase_day').size(),
                                template='plotly_white',
                                color_discrete_sequence=['#e74c3c']
                            ).update_layout(
                                xaxis_title='Day of Week',
                                yaxis_title='Number of Fraud Cases'
                            )
                        )
                    ], className='chart-card')
                ], className='col-md-6')
            ], className='row mb-4')
        ], className='charts-container')
    ], className='main-content')
])

# Enhanced CSS styles
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Fraud Detection Dashboard</title>
        {%favicon%}
        {%css%}
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
            :root {
                --primary-color: #2c3e50;
                --accent-color: #e74c3c;
                --background-color: #f8f9fa;
                --card-background: #ffffff;
                --text-color: #2c3e50;
                --border-radius: 12px;
                --box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            body {
                background-color: var(--background-color);
                color: var(--text-color);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
            }
            
            .navbar {
                background-color: var(--card-background);
                padding: 1.5rem 2rem;
                box-shadow: var(--box-shadow);
                margin-bottom: 2rem;
            }
            
            .nav-title {
                color: var(--primary-color);
                font-size: 2.5rem;
                font-weight: 600;
                margin: 0;
            }
            
            .nav-subtitle {
                color: #666;
                margin: 0.5rem 0 0 0;
                font-size: 1.1rem;
            }
            
            .main-content {
                padding: 0 2rem;
                max-width: 1400px;
                margin: 0 auto;
            }
            
            .stats-container {
                margin-bottom: 2rem;
            }
            
            /* Enhanced stat card styles */
            .stat-card {
                border-radius: var(--border-radius);
                padding: 1.5rem;
                margin-bottom: 1rem;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
                border: none;
            }
            
            /* E-commerce card gradient */
            .stat-card:nth-child(1) {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            /* Credit card gradient */
            .stat-card:nth-child(2) {
                background: linear-gradient(135deg, #6B73FF 0%, #000DFF 100%);
                color: white;
            }
            
            .stat-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
                opacity: 0;
                transition: opacity 0.3s ease;
            }
            
            .stat-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
            }
            
            .stat-card:hover::before {
                opacity: 1;
            }
            
            .stat-icon {
                font-size: 2rem;
                color: rgba(255, 255, 255, 0.9);
                margin-right: 1.5rem;
                padding: 1rem;
                background-color: rgba(255, 255, 255, 0.15);
                border-radius: 50%;
                backdrop-filter: blur(5px);
            }
            
            .stat-details {
                flex-grow: 1;
            }
            
            .stat-card h3 {
                color: white;
                font-size: 1.5rem;
                margin-bottom: 1rem;
                font-weight: 500;
            }
            
            .stat-label {
                color: rgba(255, 255, 255, 0.8);
                font-size: 0.9rem;
                display: inline-block;
                width: 140px;
                font-weight: 400;
            }
            
            .stat-value {
                font-size: 1.1rem;
                font-weight: 600;
                color: white;
            }
            
            .fraud-value {
                color: #ffd700;
                text-shadow: 0 0 10px rgba(255, 215, 0, 0.3);
            }
            
            /* Rest of the styles remain the same... */
            .chart-card {
                background-color: var(--card-background);
                border-radius: var(--border-radius);
                padding: 1.5rem;
                box-shadow: var(--box-shadow);
                margin-bottom: 1rem;
            }
            
            .chart-title {
                color: var(--primary-color);
                font-size: 1.5rem;
                margin-bottom: 1.5rem;
                font-weight: 600;
            }
            
            .charts-container {
                margin-top: 2rem;
            }
            
            /* Responsive adjustments */
            @media (max-width: 768px) {
                .main-content {
                    padding: 0 1rem;
                }
                
                .nav-title {
                    font-size: 2rem;
                }
                
                .stat-card {
                    flex-direction: column;
                }
                
                .stat-icon {
                    margin-bottom: 1rem;
                    margin-right: 0;
                }
            }
            
            /* Add subtle animation for loading */
            @keyframes cardFadeIn {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .stat-card {
                animation: cardFadeIn 0.5s ease-out forwards;
            }
            
            .stat-card:nth-child(2) {
                animation-delay: 0.2s;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8050)