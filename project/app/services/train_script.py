import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessor import load_data, preprocess_data

# 1. Load raw data
data_path = "D:\\SHASHWAT\\TECH\\Black Coffer\\chatbot dev\\collections - dummy data.xlsx"
df = load_data(data_path)

# 2. Preprocess
df = preprocess_data(df)

# Ensure expected features exist or create them
if 'Invoice Month' not in df.columns and 'Invoice date' in df.columns:
    df['Invoice Month'] = pd.to_datetime(df['Invoice date'], errors='coerce').dt.month

if 'Cluster' not in df.columns:
    df['Cluster'] = 'Unknown'  # Or apply a clustering algorithm if required

# 3. Feature engineering
features = [
    'Invoice date', 'Payment terms', 'Revenue Type', 'Payment method', 
    'Amount', 'Days_Until_Due', 'Invoice Month', 'Cluster'
]
target = 'Days_To_Pay'

data = df[features + [target]].copy()

# Convert Invoice date to datetime and extract parts
data['Invoice date'] = pd.to_datetime(data['Invoice date'], errors='coerce')
data['Invoice_day'] = data['Invoice date'].dt.day
data['Invoice_month'] = data['Invoice date'].dt.month
data['Invoice_year'] = data['Invoice date'].dt.year
data.drop(columns=['Invoice date', 'Invoice Month'], inplace=True)

# Drop rows with missing values
data.dropna(subset=[target], inplace=True)
data.dropna(inplace=True)

# One-hot encode categorical features
categorical_cols = ['Payment terms', 'Revenue Type', 'Payment method', 'Cluster']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# 4. Define features and target
X = data.drop(columns=[target])
y = data[target]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Define base models and stacking model
base_learners = [
    ('ridge', Ridge()),
    ('dt', DecisionTreeRegressor(max_depth=10, min_samples_split=5))
]

stacking_model = StackingRegressor(
    estimators=base_learners,
    final_estimator=Ridge(),
    passthrough=False
)

# 7. Hyperparameter tuning with GridSearchCV
param_grid = {
    'final_estimator__alpha': [0.1, 1, 10]
}

grid_search = GridSearchCV(
    estimator=stacking_model,
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# 8. Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("Best Hyperparameters:", grid_search.best_params_)
print("Evaluation on Test Set:")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# 9. Save model and columns
os.makedirs("models", exist_ok=True)

with open("models/best_stacking_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("models/stacking_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)
