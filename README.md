# Car Price Prediction Using Machine Learning

This project focuses on predicting used car prices based on key features such as brand, mileage, year of manufacture, fuel type, and more. 
The goal was to apply various regression models, evaluate them, and identify the most accurate approach for price estimation that is the best fit model for this case.

---

## Project Overview

Predicting car prices is considered a classic regression problem and has real-world applications in the auto resale and valuation industry. 
For this project:

1.I used a **realistic like synthetic dataset** of 1,000 used cars.
2. Applied **feature preprocessing** (encoding + scaling),
3.Trained 3 regression models: **Linear Regression, Random Forest, XGBoost**  to compare the result and check for the best fit.
4.Evaluated using **Root Mean Squared Error (RMSE)** and **R² Score**

---

## Short Dataset Summary

The dataset includes the following columns:

| Feature        | Description                              |
|----------------|------------------------------------------|
| `brand`        | Manufacturer (BMW, Toyota, etc.)         |
| `model`        | Car model (A, B, C)                      |
| `year`         | Manufacturing year (2005–2022)           |
| `mileage`      | Total kilometers driven                  |
| `fuel_type`    | Petrol, Diesel, Hybrid, Electric         |
| `transmission` | Manual or Automatic                      |
| `engine_size`  | Size of engine in liters (1.0–3.5)       |
| `price`        | Target variable — resale price (€)       |

---

## Model Training & Evaluation

I used a `Pipeline` to combine preprocessing and model training. Features were processed as follows:

- **Numerical**: scaled using `StandardScaler`
- **Categorical**: encoded using `OneHotEncoder`
- Data was split 80/20 into training and testing sets.

### Models Trained
- **Linear Regression**
- **Random Forest Regressor**
- **XGBoost Regressor**

---

## Results & Findings

The following results were obtained:

| Model             | RMSE (€) | R² Score |
|------------------|----------|----------|
| Linear Regression | 1,589    | 0.92     |
| Random Forest     | 1,201    | 0.95     |
| **XGBoost**        | **1,077**| **0.96** |

> **Conclusion:** XGBoost performed best, explaining 96% of the variance in car prices, with an average error of just ~€1,077.  
>  Feature engineering (like engine size and mileage) had a clear impact on model performance.

---

## Files in This Repo

- `car_prices.csv` — The dataset (1,000 samples)
- `car_price_model.py` — Full pipeline: preprocessing, training, evaluation
- `README.md` — This file
- `requirements.txt` - small list of requirements 
---

## How to Run

1. Install dependencies:
```bash
pip install pandas scikit-learn xgboost


