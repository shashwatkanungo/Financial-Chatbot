from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os
import pickle


MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_stacking_model.pkl')
COLUMNS_PATH = os.path.join(MODEL_DIR, 'stacking_columns.pkl')

def train_model(X, y):
    """Train a stacking model using RandomForestRegressor, XGBRegressor, and a LinearRegression meta-learner."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define base learners
    base_learners = [
        ('rf', RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=200, max_depth=10, learning_rate=0.1, subsample=1, colsample_bytree=1, random_state=42))
    ]

    # Define meta-learner
    meta_learner = LinearRegression()

    # Create stacking model
    stacking_model = StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_learner,
        passthrough=True,  # Optional: Pass original features to meta-learner
        cv=5,
        n_jobs=-1
    )

    # Fit model
    stacking_model.fit(X_train, y_train)

    # Predict
    y_pred_stack = stacking_model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred_stack)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_stack))
    r2 = r2_score(y_test, y_pred_stack)

    print("Stacking Regressor Results:")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R² Score:", r2)

     # Save model and columns
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(stacking_model, f)

    with open(COLUMNS_PATH, 'wb') as f:
        pickle.dump(X.columns.tolist(), f)

    return stacking_model

def load_model():
    """Load the trained stacking regressor model from the disk."""
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(X):
    """Use the trained stacking model to make predictions."""
    model = load_model()
    predictions = model.predict(X)
    return predictions
