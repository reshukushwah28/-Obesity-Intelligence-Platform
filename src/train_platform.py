import pandas as pd
import numpy as np
import os
import joblib
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier

# Advanced Models
try:
    from xgboost import XGBClassifier, XGBRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    XGBClassifier = XGBRegressor = LGBMClassifier = LGBMRegressor = CatBoostClassifier = CatBoostRegressor = None

import sys
# Add project root to path so 'src' module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features import FeatureEngineer, RegressionFeatureEngineer

# CONFIG
DATA_PATH = os.path.join("data", "ObesityDataSet_raw_and_data_sinthetic.csv")
ARTIFACTS_DIR = "artifacts"
if not os.path.exists(ARTIFACTS_DIR):
    os.makedirs(ARTIFACTS_DIR)

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")
    return pd.read_csv(DATA_PATH)

def get_preprocessor(numerical_cols, categorical_cols):
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

def train_classification(df):
    print("\n=== Training Classification Models (Obesity Level) ===")
    target = 'NObeyesdad'
    X = df.drop(columns=[target])
    y = df[target]
    
    # Encoder Target (Required for XGBoost/LightGBM)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Save Encoder for App
    joblib.dump(le, os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"))
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Columns
    # Note: FeatureEngineer adds columns, so we need to know what they are?
    # Actually, FeatureEngineer runs BEFORE ColumnTransformer.
    # The numerical cols will be original num cols + [BMI, Lifestyle_Score, Hydration_Index, Calorie_Risk_Score, Sedentary_Risk]
    
    orig_num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    orig_cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    new_num_cols = ['BMI', 'Lifestyle_Score', 'Hydration_Index', 'Calorie_Risk_Score', 'Sedentary_Risk']
    
    # Preprocessor
    # We apply FeatureEngineer first, so the output DF has the new columns.
    # Then ColumnTransformer needs to know the names of ALL numerical columns (orig + new)
    
    final_num_cols = orig_num_cols + new_num_cols
    final_cat_cols = orig_cat_cols
    
    preprocessor = get_preprocessor(final_num_cols, final_cat_cols)
    
    # Models to test
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100),
        'GradientBoosting': GradientBoostingClassifier(),
        'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
    }
    
    if XGBClassifier: models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    if LGBMClassifier: models['LightGBM'] = LGBMClassifier(verbose=-1)
    # CatBoost disabled due to sklearn 1.6 compatibility issues
    # if CatBoostClassifier: models['CatBoost'] = CatBoostClassifier(verbose=0)
    
    best_score = 0
    best_model = None
    best_name = ""
    metrics = {}
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('features', FeatureEngineer()),
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"{name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}")
        metrics[name] = {'Accuracy': acc, 'F1': f1}
        
        if f1 > best_score:
            best_score = f1
            best_model = pipeline
            best_name = name
            
    print(f"🏆 Best Classification Model: {best_name} (F1: {best_score:.4f})")
    
    # Save Best Model
    joblib.dump(best_model, os.path.join(ARTIFACTS_DIR, "best_classifier.joblib"))
    
    # Save Metadata
    metadata = {
        'model_name': best_name,
        'metrics': metrics,
        'features': {
            'numerical': final_num_cols,
            'categorical': final_cat_cols,
            'options': {col: df[col].unique().tolist() for col in orig_cat_cols}
        }
    }
    joblib.dump(metadata, os.path.join(ARTIFACTS_DIR, "classifier_metadata.joblib"))
    return best_model

def train_regression(df):
    print("\n=== Training Regression Models (Predicted Weight) ===")
    # Goal: Predict Weight based on Height + Habits (Age, Gender, etc)
    # Exclude: 'Weight', 'NObeyesdad' (leaks)
    
    target = 'Weight'
    exclude = ['Weight', 'NObeyesdad'] # BMI is not in raw data yet
    
    X = df.drop(columns=exclude)
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Identify columns
    orig_num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    orig_cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # FeatureEngineer will run, but it calculates BMI using Weight.
    # We use the specialized RegressionFeatureEngineer from src.features
    
    # New cols: Lifestyle_Score, Sedentary_Risk, Hydration_Index, Calorie_Risk_Score
    # (Skip BMI)

    new_num_cols = ['Lifestyle_Score', 'Hydration_Index', 'Calorie_Risk_Score', 'Sedentary_Risk']
    final_num_cols = orig_num_cols + new_num_cols
    
    preprocessor = get_preprocessor(final_num_cols, orig_cat_cols)
    
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100),
        'GradientBoosting': GradientBoostingRegressor()
    }
    
    if XGBRegressor: models['XGBoost'] = XGBRegressor()
    if LGBMRegressor: models['LightGBM'] = LGBMRegressor(verbose=-1)
    # if CatBoostRegressor: models['CatBoost'] = CatBoostRegressor(verbose=0)
    
    best_score = -float('inf') # R2 can be negative
    best_model = None
    best_name = ""
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('features', RegressionFeatureEngineer()),
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"{name} -> R2: {r2:.4f}, MAE: {mae:.2f}")
        
        if r2 > best_score:
            best_score = r2
            best_model = pipeline
            best_name = name
            
    print(f"🏆 Best Regression Model: {best_name} (R2: {best_score:.4f})")
    
    joblib.dump(best_model, os.path.join(ARTIFACTS_DIR, "best_regressor.joblib"))

if __name__ == "__main__":
    df = load_data()
    train_classification(df)
    train_regression(df)
