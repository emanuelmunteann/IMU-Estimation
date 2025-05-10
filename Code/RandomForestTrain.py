import time
from tqdm import tqdm
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

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

# Load data
file_path = 'DateUnite.csv'
data = pd.read_csv(file_path)

features = ['Shoulder_X', 'Shoulder_Y', 'Shoulder_Z']
targets = ['Biceps_X', 'Biceps_Y', 'Biceps_Z']
X = data[features]
y = data[targets]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save splits to CSV files
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Define parameter ranges
n_estimators_values = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
random_state_values = [0, 1, 21, 42, 77]

results = []
total_iterations = len(n_estimators_values) * len(random_state_values)

with tqdm(total=total_iterations, desc='Training Progress') as pbar:
    start_time = time.time()
    for n_estimators in n_estimators_values:
        for random_state in random_state_values:
            model = RandomForestRegressor(
                n_estimators=n_estimators,  # Number of trees in the forest
                max_depth=None,             # Maximum depth of each tree
                min_samples_split=2,        # Minimum number of samples required to split an internal node
                min_samples_leaf=1,         # Minimum number of samples required at a leaf node
                max_features=1.0,           # Number of features to consider when looking for the best split
                bootstrap=True,             # Whether bootstrap samples are used when building trees
                random_state=random_state,  # Seed for reproducibility
                criterion='squared_error'   # Function to measure split quality ("gini", "entropy", "log_loss")
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2_scores = calculate_r2(y_test, y_pred)
            rmse_scores = calculate_rmse(y_test, y_pred)
            results.append({
                'n_estimators': n_estimators, 
                'random_state': random_state, 
                'R^2 Biceps_X': round(r2_scores['Biceps_X'], 4),
                'R^2 Biceps_Y': round(r2_scores['Biceps_Y'], 4),
                'R^2 Biceps_Z': round(r2_scores['Biceps_Z'], 4),
                'RMSE Biceps_X': round(rmse_scores['Biceps_X'], 4),
                'RMSE Biceps_Y': round(rmse_scores['Biceps_Y'], 4),
                'RMSE Biceps_Z': round(rmse_scores['Biceps_Z'], 4)
            })

            pbar.update(1)
            elapsed_time = time.time() - start_time
            iterations_done = len(results)
            iterations_left = total_iterations - iterations_done
            if iterations_done > 0:
                estimated_time_left = elapsed_time / iterations_done * iterations_left
                pbar.set_postfix({'ETA (s)': f'{estimated_time_left:.1f}'})


results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv('Tree_si_RandomState.csv', index=False)
