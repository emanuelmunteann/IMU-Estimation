import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def calculate_r2(y_true, y_pred):
    r2_scores = {}
    for i, target in enumerate(y_true.columns):
        r2_scores[target] = r2_score(y_true[target], y_pred[:, i])
    return r2_scores

def calculate_rmse(y_true, y_pred):
    rmse_scores = {}
    for i, target in enumerate(y_true.columns):
        rmse_scores[target] = np.sqrt(mean_squared_error(y_true[target], y_pred[:, i]))
    return rmse_scores

# Load training data
X_train = pd.read_csv('X_Train.csv')
y_train = pd.read_csv('Y_Train.csv')

# Load testing data from Excel
test_data = pd.read_excel('DateTestGrafice.xlsx')
X_test = test_data.iloc[:, :3]  # First 3 columns
y_test = test_data.iloc[:, -3:] # Last 3 columns

# Initialize and train model
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=1.0,
    bootstrap=True,
    random_state=0,
    criterion='squared_error'
)

model.fit(X_train, y_train)
joblib.dump(model, 'random_forest_model.joblib')

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
r2_results = calculate_r2(y_test, y_pred)
rmse_results = calculate_rmse(y_test, y_pred)

print("R² scores for each target:")
for target, score in r2_results.items():
    print(f"{target}: {score:.4f}")

print("\nRMSE scores for each target:")
for target, score in rmse_results.items():
    print(f"{target}: {score:.4f}")

# Plot results
plt.figure(figsize=(12, 8))
for i, target in enumerate(y_test.columns):
    plt.subplot(3, 1, i+1)
    plt.plot(y_test[target].values, label='Actual')
    plt.plot(y_pred[:, i], label='Predicted', linestyle='--')
    plt.title(f'Actual vs Predicted for {target}\nR² = {r2_results[target]:.4f}, RMSE = {rmse_results[target]:.4f}')
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.legend()
    
plt.tight_layout()
plt.show()
