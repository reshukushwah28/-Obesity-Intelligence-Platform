import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# 1. Load Data
DATA_PATH = os.path.join("data", "ObesityDataSet_raw_and_data_sinthetic.csv")

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}. Please run download_data.py first.")
    return pd.read_csv(path)

# 2. Feature Engineering
def add_features(df):
    df = df.copy()
    # BMI Calculation: Weight (kg) / Height (m)^2
    # Ensure Height is in meters (Observation: Dataset usually has height in meters e.g., 1.70)
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    
    # Physical Activity Score: FAF / (TUE + 1)
    df['PhysicalActivityScore'] = df['FAF'] / (df['TUE'] + 1)
    return df

def train():
    print("Loading data...")
    df = load_data(DATA_PATH)
    
    print("Engineering features...")
    df = add_features(df)
    
    # 3. Define Features and Target
    target_col = 'NObeyesdad'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify column types
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    # 4. Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # 5. Model Selection & Training
    # We will wrap the classifier in a pipeline with the preprocessor
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        score = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {score:.4f}")
        print(classification_report(y_test, y_pred))
        
        if score > best_score:
            best_score = score
            best_model = pipeline
            best_name = name
            
    print(f"\nBest Model: {best_name} with Accuracy: {best_score:.4f}")
    
    # 6. Save Artifacts
    if best_model:
        model_path = "final_model.joblib"
        joblib.dump(best_model, model_path)
        print(f"Model saved to {model_path}")
        
        # Get unique values for categorical columns (for Streamlit selectboxes)
        cat_options = {col: df[col].unique().tolist() for col in categorical_cols}
        
        # Save column metadata for the app
        meta_data = {
            "numerical_cols": numerical_cols,
            "categorical_cols": categorical_cols,
            "cat_options": cat_options,
            "all_columns": X.columns.tolist() 
        }
        joblib.dump(meta_data, "model_metadata.joblib")
        print("Metadata saved to model_metadata.joblib")

if __name__ == "__main__":
    train()
