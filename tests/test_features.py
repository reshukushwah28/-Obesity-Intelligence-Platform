import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import FeatureEngineer, RegressionFeatureEngineer

class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'Weight': [80.0],
            'Height': [1.80], # Meters
            'FAF': [1.0],
            'FCVC': [2.0],
            'TUE': [1.0],
            'CH2O': [2.0],
            'FAVC': ['yes'],
            'CAEC': ['Sometimes'],
            'SCC': ['no']
        })

    def test_bmi_calculation(self):
        fe = FeatureEngineer()
        transformed = fe.transform(self.data)
        expected_bmi = 80 / (1.8 ** 2)
        self.assertAlmostEqual(transformed['BMI'].iloc[0], expected_bmi, places=2)

    def test_lifestyle_score(self):
        fe = FeatureEngineer()
        transformed = fe.transform(self.data)
        # (1*2) + 2 - 1 = 3
        self.assertEqual(transformed['Lifestyle_Score'].iloc[0], 3.0)

class TestRegressionFeatureEngineer(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'FAF': [1.0],
            'FCVC': [2.0],
            'TUE': [1.0],
            'CH2O': [2.0],
            'FAVC': ['yes'],
            'CAEC': ['Sometimes'],
            'SCC': ['no']
        })
        
    def test_no_weight_needed(self):
        fe = RegressionFeatureEngineer()
        transformed = fe.transform(self.data)
        self.assertIn('Lifestyle_Score', transformed.columns)
        self.assertNotIn('BMI', transformed.columns)

if __name__ == '__main__':
    unittest.main()
