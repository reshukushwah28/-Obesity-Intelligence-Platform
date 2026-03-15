import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. BMI
        # Ensure Height is in meters. If max height > 3, likely cm, so divide by 100.
        # But per dataset specs, it's usually meters. We'll assume input is correct or handled before.
        X['BMI'] = X['Weight'] / (X['Height'] ** 2)
        
        # 2. Lifestyle Score
        # Logic: (Physical Activity * 2) + (Veg Intake) - (Tech Use)
        X['Lifestyle_Score'] = (X['FAF'] * 2) + X['FCVC'] - X['TUE']
        
        # 3. Sedentary Risk Flag
        # High Tech Usage (> 1.5) AND Low/No Activity (< 1.0)
        X['Sedentary_Risk'] = ((X['TUE'] > 1.5) & (X['FAF'] < 1.0)).astype(int)
        
        # 4. Hydration Index
        # CH2O is 1-3 (Liters). We normalize it relative to 3.
        X['Hydration_Index'] = X['CH2O'] / 3.0
        
        # 5. Calorie Risk Score
        # FAVC (High Cal Food): yes=1, no=0
        favc_map = {'yes': 1, 'no': 0}
        
        # CAEC (Snacking): no=0, Sometimes=1, Frequently=2, Always=3
        caec_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        
        # SCC (Monitor Calories): yes=0 (good), no=1 (bad risk)
        scc_map = {'yes': 0, 'no': 1}
        
        # Map values safely (handle potential unexpected strings with defaults if needed)
        favc_score = X['FAVC'].map(favc_map).fillna(0)
        caec_score = X['CAEC'].map(caec_map).fillna(1)
        scc_score = X['SCC'].map(scc_map).fillna(1)
        
        X['Calorie_Risk_Score'] = favc_score + caec_score + scc_score
        
        return X

class RegressionFeatureEngineer(FeatureEngineer):
    def transform(self, X):
        X = X.copy()
        # Skip BMI Calculation as it requires Weight (which is the target)
        
        # 1. Lifestyle Score
        X['Lifestyle_Score'] = (X['FAF'] * 2) + X['FCVC'] - X['TUE']
        
        # 2. Sedentary Risk
        X['Sedentary_Risk'] = ((X['TUE'] > 1.5) & (X['FAF'] < 1.0)).astype(int)
        
        # 3. Hydration Index
        X['Hydration_Index'] = X['CH2O'] / 3.0
        
        # 4. Calorie Risk Score
        favc_map = {'yes': 1, 'no': 0}
        caec_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        scc_map = {'yes': 0, 'no': 1}
        
        X['Calorie_Risk_Score'] = (
            X['FAVC'].map(favc_map).fillna(0) + 
            X['CAEC'].map(caec_map).fillna(1) + 
            X['SCC'].map(scc_map).fillna(1)
        )
        return X
