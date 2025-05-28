import numpy as np
import pandas as pd

def compute_expected_loss(class_model, reg_model, X):
    
    if hasattr(class_model, "predict_proba"):
        p_default = class_model.predict_proba(X)[:, 1]
    else:
        p_default = class_model.predict(X)

    
    predicted_loss = reg_model.predict(X)

    
    expected_loss = p_default * predicted_loss

    return pd.DataFrame({
        'P(Default)': p_default,
        'Predicted Loss': predicted_loss,
        'Expected Loss': expected_loss
    })
