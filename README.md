# Water Quality Analysis API with Linear Regression

A Flask-based API for analyzing water quality data and predicting organism abundance using linear regression models.

## Features

- **Linear Regression Models**: Uses scikit-learn's LinearRegression for accurate predictions
- **Correlation Analysis**: Calculate Pearson correlations between water parameters and organism abundance
- **Multiple Organisms**: Analyze 5 different organisms (Pseudomonadota, Actinomycetota, Chordata, Cyanobacteriota, Other)
- **Model Metrics**: R² score, RMSE, coefficients, and correlations
- **RESTful API**: Easy-to-use endpoints for data retrieval and predictions

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

The API will start on `http://localhost:5000`

## API Endpoints

### 1. Get All Data
```
GET /api/data
```
Returns all water quality data points.

### 2. Get Organisms List
```
GET /api/organisms
```
Returns list of available organisms.

### 3. Get Correlation Matrix
```
GET /api/correlations
```
Returns correlation matrix between water parameters and organisms.

### 4. Get Model Information
```
GET /api/model/<organism>
```
Returns linear regression model details for a specific organism.

Example: `GET /api/model/Pseudomonadota`

### 5. Predict Organism Abundance
```
POST /api/predict
Content-Type: application/json

{
  "organism": "Pseudomonadota",
  "temperature": 26.0,
  "pH": 7.0,
  "conductivity": 600,
  "saturated": 12
}
```

Returns predicted abundance with model metrics.

### 6. Linear Regression Demo
```
GET /api/linear-regression-demo
```
Shows linear regression models for all organisms with coefficients and metrics.

## Water Parameters

- **Temperature**: Water temperature (°C)
- **pH**: pH level
- **Conductivity**: Electrical conductivity (μS/cm)
- **% Saturated**: Oxygen saturation percentage

## Example Usage

### Using curl:
```bash
# Get all data
curl http://localhost:5000/api/data

# Get correlations
curl http://localhost:5000/api/correlations

# Predict abundance
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"organism\":\"Pseudomonadota\",\"temperature\":26,\"pH\":7,\"conductivity\":600,\"saturated\":12}"

# View all linear regression models
curl http://localhost:5000/api/linear-regression-demo
```

### Using Python:
```python
import requests

# Predict organism abundance
response = requests.post('http://localhost:5000/api/predict', json={
    'organism': 'Pseudomonadota',
    'temperature': 26.5,
    'pH': 7.2,
    'conductivity': 550,
    'saturated': 15.0
})

result = response.json()
print(f"Predicted abundance: {result['prediction']['abundance_percentage']:.2f}%")
print(f"Model R² score: {result['prediction']['r2_score']:.3f}")
```

## Linear Regression Model

The API uses **scikit-learn's LinearRegression** with the following approach:

1. **Feature Standardization**: Uses StandardScaler to normalize water parameters
2. **Multiple Linear Regression**: Fits a model with all 4 water parameters as features
3. **Model Evaluation**: Provides R² score and RMSE metrics
4. **Correlation Analysis**: Calculates Pearson correlations for each parameter

### Model Equation:
```
Abundance = intercept + (β₁ × temp) + (β₂ × pH) + (β₃ × conductivity) + (β₄ × saturated)
```

Where β₁, β₂, β₃, β₄ are the learned coefficients for each parameter.

## Response Format

Prediction response includes:
- **abundance**: Predicted value (0-1)
- **abundance_percentage**: Percentage format
- **r2_score**: Model accuracy (0-1, higher is better)
- **rmse**: Root Mean Squared Error (lower is better)
- **coefficients**: Impact of each water parameter
- **correlations**: Pearson correlations

## Notes

- Predictions are clamped between 0 and 1 (0-100%)
- Models are trained on 7 data points
- StandardScaler ensures fair comparison between different parameter scales
- R² score indicates model fit quality
