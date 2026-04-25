# Correlation Between Water Quality and Abundance

A small API and analysis project that explores relationships between **water quality parameters** (e.g., temperature, pH, conductivity, oxygen saturation) and **organism abundance** using **correlation analysis** and **multiple linear regression**.

> Tech highlights: Flask API · scikit-learn LinearRegression · Pearson correlations

---

## What this repository contains

- A **Flask REST API** that serves:
  - Raw dataset access
  - Correlation matrix results
  - Model summaries for each organism
  - Prediction endpoint for organism abundance from water parameters
  - Linear regression demo for all organisms
- A **multiple linear regression** model (scikit-learn `LinearRegression`)
- Basic **model metrics** (R², RMSE) and **coefficients**
- Support for 5 organism targets:
  - Pseudomonadota
  - Actinomycetota
  - Chordata
  - Cyanobacteriota
  - Other

---

## Features

- **Linear Regression Models**: Uses scikit-learn's `LinearRegression` for accurate predictions
- **Correlation Analysis**: Calculates Pearson correlations between water parameters and organism abundance
- **Multiple Organisms**: Analyzes 5 different organisms
- **Model Metrics**: R² score, RMSE, coefficients, and correlations per organism
- **RESTful API**: Easy-to-use endpoints for data retrieval and predictions

---

## Installation & Run (Local)

### Prerequisites

- Python 3.9+ (recommended)
- pip

### Setup

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Run the API
python app.py
```

The API will start on `http://localhost:5000`.

---

## API Overview

Base URL (local): `http://localhost:5000`

### Endpoints

#### 1. Get all data

```
GET /api/data
```

Returns all water quality data points.

#### 2. List supported organisms

```
GET /api/organisms
```

Returns the list of available organisms.

#### 3. Correlation matrix

```
GET /api/correlations
```

Returns the correlation matrix between water parameters and organisms.

#### 4. Model info (per organism)

```
GET /api/model/<organism>
```

Returns linear regression model details for a specific organism.

Example: `GET /api/model/Pseudomonadota`

#### 5. Predict abundance

```
POST /api/predict
Content-Type: application/json
```

Example request body:

```json
{
  "organism": "Pseudomonadota",
  "temperature": 26.0,
  "pH": 7.0,
  "conductivity": 600,
  "saturated": 12
}
```

Returns predicted abundance with model metrics.

#### 6. Linear regression demo

```
GET /api/linear-regression-demo
```

Shows linear regression models for all organisms with coefficients and metrics.

---

## Water Parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| **Temperature** | Water temperature | °C |
| **pH** | Acidity/alkalinity level | — |
| **Conductivity** | Electrical conductivity | μS/cm |
| **% Saturated** | Oxygen saturation percentage | % |

---

## Example Usage

### cURL

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

### Python

```python
import requests

response = requests.post("http://localhost:5000/api/predict", json={
    "organism": "Pseudomonadota",
    "temperature": 26.5,
    "pH": 7.2,
    "conductivity": 550,
    "saturated": 15.0
})

result = response.json()
print(f"Predicted abundance: {result['prediction']['abundance_percentage']:.2f}%")
print(f"Model R² score:      {result['prediction']['r2_score']:.3f}")
```

---

## Response Format

A prediction response includes:

| Field | Description |
|-------|-------------|
| `abundance` | Predicted value (0–1) |
| `abundance_percentage` | Percentage format |
| `r2_score` | Model accuracy (0–1, higher is better) |
| `rmse` | Root Mean Squared Error (lower is better) |
| `coefficients` | Impact of each water parameter |
| `correlations` | Pearson correlations per parameter |

---

## Modeling Approach

The model uses:

1. **Feature standardization** — `StandardScaler` normalizes parameter scales so each feature contributes fairly
2. **Multiple linear regression** — `LinearRegression` with all 4 water parameters as predictors
3. **Model evaluation** — R² score and RMSE reported per organism
4. **Pearson correlation** — per-parameter correlation reported for interpretability

### Model equation

```
Abundance = intercept + (β₁ × temperature) + (β₂ × pH) + (β₃ × conductivity) + (β₄ × saturated)
```

Where β₁ – β₄ are the learned coefficients for each parameter.

---

## Notes / Limitations

- Predictions are clamped between 0 and 1 (0–100%)
- Models are trained on 7 data points — results should be treated as exploratory
- `StandardScaler` ensures fair comparison between parameters at different scales
- R² score indicates model fit quality but may be misleading on such a small dataset
- Intended as an exploratory analysis and API demonstration, not a production model

---

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for step-by-step instructions on deploying to **Render.com** (free tier available).

Quick summary:
- **Build command**: `pip install -r requirements.txt`
- **Start command**: `gunicorn app:app`
- Free-tier apps sleep after 15 minutes of inactivity (~30 s to wake)

---

## License

TODO: Add a LICENSE file (MIT / Apache-2.0 / etc.) and update this section.
