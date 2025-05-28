from credit_risk_model.data_loader import load_and_preprocess_data
from credit_risk_model.classification import train_classification_models, evaluate_classification_models
from credit_risk_model.regression import train_regression_models, evaluate_regression_models
from credit_risk_model.evaluation import compute_expected_loss

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


file_path = "C:/Users/adity/OneDrive/Desktop\ecom/Credit-Risk_management-ML-project/credit_risk_model/data/accepted_2007_to_2018Q4.csv"


X_train_c, X_test_c, y_train_c, y_test_c, X_train_r, X_test_r, y_train_r, y_test_r = load_and_preprocess_data(file_path)


print("\n--- Classification Models ---")
class_models = train_classification_models(X_train_c, y_train_c)
evaluate_classification_models(class_models, X_test_c, y_test_c)


print("\n--- Regression Models ---")
reg_models = train_regression_models(X_train_r, y_train_r)
evaluate_regression_models(reg_models, X_test_r, y_test_r)


print("\n--- Expected Loss Calculation (Random Forest + Linear Regression) ---")
class_model = class_models['Random Forest']
reg_model = reg_models['Linear Regression']
expected_loss_df = compute_expected_loss(class_model, reg_model, X_test_c)
print(expected_loss_df.head())


print("\n--- Visualizing Expected Loss Distribution ---")
plt.figure(figsize=(10, 6))
sns.histplot(expected_loss_df['Expected Loss'], bins=50, kde=True, color='orange')
plt.title('Distribution of Expected Loss')
plt.xlabel('Expected Loss')
plt.ylabel('Number of Borrowers')
plt.grid(True)
plt.tight_layout()
plt.show()


print("\nTop 10 Highest Expected Losses:")
print(expected_loss_df.sort_values(by='Expected Loss', ascending=False).head(10))
