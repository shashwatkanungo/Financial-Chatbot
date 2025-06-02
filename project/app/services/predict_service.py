# import os
# import pickle
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import StackingRegressor, RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from xgboost import XGBRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # Import your preprocessing functions
# from app.services.data_preprocessor import load_data, preprocess_data
# from eda import clean_data

# def train_model(X, y):
#     """Train and save the stacking model."""
#     os.makedirs("models", exist_ok=True)

#     # Split data into training and testing
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Define base learners and meta-learner
#     base_learners = [
#         ('rf', RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)),
#         ('xgb', XGBRegressor(n_estimators=200, max_depth=10, learning_rate=0.1,
#                              subsample=1, colsample_bytree=1, random_state=42))
#     ]
#     meta_learner = LinearRegression()

#     # Stacking model
#     stacking_model = StackingRegressor(
#         estimators=base_learners,
#         final_estimator=meta_learner,
#         passthrough=True,
#         cv=5,
#         n_jobs=-1
#     )

#     # Fit the model
#     stacking_model.fit(X_train, y_train)

#     # Evaluate the model
#     y_pred = stacking_model.predict(X_test)
#     print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
#     print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
#     print(f"R²: {r2_score(y_test, y_pred):.2f}")

#     # Save the model
#     with open("models/stacking_model.pkl", "wb") as f:
#         pickle.dump(stacking_model, f)

#     # Save the feature column order
#     with open("models/stacking_columns.pkl", "wb") as f:
#         pickle.dump(X.columns.tolist(), f)

#     print("✅ Model and column order saved successfully.")

# if __name__ == "__main__":
#     # Step 1: Load cleaned and processed data
#     # Assuming `load_cleaned_data` is a function in your `eda.py` that returns cleaned DataFrame
#     file_path = "D:\\SHASHWAT\\TECH\\Black Coffer\\chatbot dev\\collections - dummy data.xlsx"
#     df_raw = load_data(file_path)
#     df_cleaned = clean_data(df_raw)
#     df_processed = preprocess_data(df_cleaned)

#     # Step 2: Prepare features (X) and target (y)
#     # Define features and target
#     features = [
#         'Invoice day', 'Invoice month', 'Invoice year', 'Amount', 'Days_Until_Due', 
#         'Payment terms_1', 'Payment terms_2', 'Payment terms_3', 'Payment terms_4', 'Payment terms_5',
#         'Revenue Type_Expansion', 'Revenue Type_New', 'Revenue Type_Renewal', 'Revenue Type_Services', 'Revenue Type_X',
#         'Payment method_Credit card', 'Payment method_Record transfer',
#         'Cluster_1', 'Cluster_2'
#     ]
#     X = df_processed[features]
#     y = df_processed['Days_To_Pay']

#     # Step 3: Train the stacking model using the processed data
#     train_model(X, y)
