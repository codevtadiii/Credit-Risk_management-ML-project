from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_regression_models(X_train, y_train):
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Support Vector Regression': SVR()
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"Trained {name}")

    return trained_models

def evaluate_regression_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            'RMSE': rmse,
            'R2 Score': r2
        }
        print(f"{name} - RMSE: {rmse:.2f}, R2 Score: {r2:.4f}")

    return results
# This module contains functions to train and evaluate regression models for credit risk prediction.
# It includes models like Linear Regression, Ridge Regression, Lasso Regression, and Support Vector Regression.