from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import json

app = Flask(__name__)
CORS(app)

# Water quality data
data = [
    {"id": 16, "temp": 25.5, "pH": 8.9, "cond": 1581, "sat": 10.54, "Pseudomonadota": 0.39, "Actinomycetota": 0.20, "Chordata": 0.15, "Cyanobacteriota": 0.11, "Other": 0.05},
    {"id": 17, "temp": 25.8, "pH": 6.98, "cond": 621, "sat": 10.42, "Pseudomonadota": 0.41, "Actinomycetota": 0.31, "Chordata": 0.19, "Cyanobacteriota": 0.07, "Other": 0.03},
    {"id": 18, "temp": 26.6, "pH": 6.78, "cond": 551, "sat": 14.69, "Pseudomonadota": 0.57, "Actinomycetota": 0.14, "Chordata": 0.16, "Cyanobacteriota": 0.08, "Other": 0.05},
    {"id": 19, "temp": 26.1, "pH": 6.99, "cond": 591, "sat": 11.48, "Pseudomonadota": 0.51, "Actinomycetota": 0.21, "Chordata": 0.12, "Cyanobacteriota": 0.12, "Other": 0.04},
    {"id": 21, "temp": 26, "pH": 7.12, "cond": 855, "sat": 15.98, "Pseudomonadota": 0.49, "Actinomycetota": 0.18, "Chordata": 0.26, "Cyanobacteriota": 0.04, "Other": 0.02},
    {"id": 22, "temp": 26, "pH": 7.12, "cond": 256, "sat": 20.58, "Pseudomonadota": 0.52, "Actinomycetota": 0.18, "Chordata": 0.24, "Cyanobacteriota": 0.01, "Other": 0.06},
    {"id": 23, "temp": 26.1, "pH": 7.06, "cond": 455, "sat": 33.87, "Pseudomonadota": 0.50, "Actinomycetota": 0.16, "Chordata": 0.28, "Cyanobacteriota": 0.03, "Other": 0.03}
]

organisms = ['Pseudomonadota', 'Actinomycetota', 'Chordata', 'Cyanobacteriota', 'Other']
water_params = ['temp', 'pH', 'cond', 'sat']
param_names = {'temp': 'Temperature', 'pH': 'pH', 'cond': 'Conductivity', 'sat': '% Saturated'}

# Convert data to DataFrame
df = pd.DataFrame(data)

def calculate_correlation(x, y):
    """Calculate Pearson correlation coefficient"""
    return np.corrcoef(x, y)[0, 1]

def build_linear_regression_model(organism):
    """Build a proper linear regression model using scikit-learn"""
    # Prepare features (X) and target (y)
    X = df[water_params].values
    y = df[organism].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Calculate model metrics
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Calculate correlations for each parameter
    correlations = {}
    for param in water_params:
        corr = calculate_correlation(df[param].values, y)
        correlations[param] = corr
    
    return {
        'model': model,
        'scaler': scaler,
        'coefficients': model.coef_.tolist(),
        'intercept': float(model.intercept_),
        'r2_score': float(r2),
        'rmse': float(rmse),
        'correlations': correlations
    }

@app.route('/')
def home():
    return render_template('index.html', organisms=organisms, param_names=param_names)

@app.route('/correlations')
def correlations_page():
    return render_template('correlations.html', organisms=organisms, param_names=param_names)

@app.route('/models')
def models_page():
    return render_template('models.html', organisms=organisms)

@app.route('/api/data', methods=['GET'])
def get_data():
    """Get all water quality data"""
    return jsonify({
        'data': data,
        'count': len(data)
    })

@app.route('/api/organisms', methods=['GET'])
def get_organisms():
    """Get list of organisms"""
    return jsonify({
        'organisms': organisms
    })

@app.route('/api/correlations', methods=['GET'])
def get_correlations():
    """Calculate and return correlation matrix"""
    correlation_matrix = []
    
    for org in organisms:
        for param in water_params:
            x_values = df[param].values
            y_values = df[org].values
            corr = calculate_correlation(x_values, y_values)
            
            correlation_matrix.append({
                'organism': org,
                'parameter': param_names[param],
                'correlation': float(corr)
            })
    
    # Create heatmap data
    heatmap_data = []
    for org in organisms:
        row = {'organism': org}
        for param in water_params:
            corr_item = next((c for c in correlation_matrix 
                            if c['organism'] == org and c['parameter'] == param_names[param]), None)
            row[param_names[param]] = corr_item['correlation'] if corr_item else 0
        heatmap_data.append(row)
    
    return jsonify({
        'correlation_matrix': correlation_matrix,
        'heatmap_data': heatmap_data
    })

@app.route('/api/model/<organism>', methods=['GET'])
def get_model_info(organism):
    """Get linear regression model information for specific organism"""
    if organism not in organisms:
        return jsonify({'error': 'Invalid organism'}), 400
    
    model_info = build_linear_regression_model(organism)
    
    # Remove the model and scaler objects before sending response
    response_data = {
        'organism': organism,
        'coefficients': model_info['coefficients'],
        'intercept': model_info['intercept'],
        'r2_score': model_info['r2_score'],
        'rmse': model_info['rmse'],
        'correlations': {param_names[k]: float(v) for k, v in model_info['correlations'].items()},
        'feature_names': [param_names[p] for p in water_params]
    }
    
    return jsonify(response_data)

@app.route('/api/predict', methods=['POST'])
def predict_abundance():
    """Predict organism abundance using linear regression"""
    try:
        request_data = request.get_json()
        
        # Validate inputs
        organism = request_data.get('organism', 'Pseudomonadota')
        if organism not in organisms:
            return jsonify({'error': 'Invalid organism'}), 400
        
        temperature = float(request_data.get('temperature', 26.0))
        pH = float(request_data.get('pH', 7.0))
        conductivity = float(request_data.get('conductivity', 600))
        saturated = float(request_data.get('saturated', 12))
        
        # Build model
        model_data = build_linear_regression_model(organism)
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Prepare input features
        X_new = np.array([[temperature, pH, conductivity, saturated]])
        X_new_scaled = scaler.transform(X_new)
        
        # Make prediction
        prediction = model.predict(X_new_scaled)[0]
        
        # Clamp prediction between 0 and 1
        prediction = max(0, min(1, prediction))
        
        return jsonify({
            'organism': organism,
            'inputs': {
                'temperature': temperature,
                'pH': pH,
                'conductivity': conductivity,
                'saturated': saturated
            },
            'prediction': {
                'abundance': float(prediction),
                'abundance_percentage': float(prediction * 100),
                'r2_score': model_data['r2_score'],
                'rmse': model_data['rmse']
            },
            'model_info': {
                'coefficients': model_data['coefficients'],
                'intercept': model_data['intercept'],
                'correlations': {param_names[k]: float(v) for k, v in model_data['correlations'].items()}
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/linear-regression-demo', methods=['GET'])
def linear_regression_demo():
    """Demo endpoint showing linear regression for all organisms"""
    results = {}
    
    for org in organisms:
        model_info = build_linear_regression_model(org)
        
        results[org] = {
            'coefficients': dict(zip([param_names[p] for p in water_params], model_info['coefficients'])),
            'intercept': model_info['intercept'],
            'r2_score': model_info['r2_score'],
            'rmse': model_info['rmse'],
            'correlations': {param_names[k]: float(v) for k, v in model_info['correlations'].items()}
        }
    
    return jsonify({
        'message': 'Linear Regression Models for All Organisms',
        'models': results,
        'interpretation': {
            'r2_score': 'Coefficient of determination (0-1, higher is better)',
            'rmse': 'Root Mean Squared Error (lower is better)',
            'coefficients': 'Impact of each water parameter on organism abundance',
            'correlations': 'Pearson correlation between parameters and abundance'
        }
    })

if __name__ == '__main__':
    print("Starting Water Quality Analysis API with Linear Regression...")
    print("Available endpoints:")
    print("  GET  /api/data - Get all data")
    print("  GET  /api/organisms - Get organisms list")
    print("  GET  /api/correlations - Get correlation matrix")
    print("  GET  /api/model/<organism> - Get model info")
    print("  POST /api/predict - Predict abundance")
    print("  GET  /api/linear-regression-demo - See all models")
    app.run(debug=True, port=5000)