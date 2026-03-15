import pandas as pd
import numpy as np
import os

def generate_data():
    print("Generating synthetic data...")
    num_rows = 500
    
    data = {
        'Gender': np.random.choice(['Female', 'Male'], num_rows),
        'Age': np.random.randint(18, 60, num_rows),
        'Height': np.random.uniform(1.5, 2.0, num_rows),
        'Weight': np.random.uniform(50, 120, num_rows),
        'family_history_with_overweight': np.random.choice(['yes', 'no'], num_rows),
        'FAVC': np.random.choice(['yes', 'no'], num_rows),
        'FCVC': np.random.randint(1, 4, num_rows),
        'NCP': np.random.randint(1, 5, num_rows),
        'CAEC': np.random.choice(['Sometimes', 'Frequently', 'Always', 'no'], num_rows),
        'SMOKE': np.random.choice(['yes', 'no'], num_rows),
        'CH2O': np.random.uniform(1, 3, num_rows),
        'SCC': np.random.choice(['yes', 'no'], num_rows),
        'FAF': np.random.uniform(0, 3, num_rows),
        'TUE': np.random.uniform(0, 2, num_rows),
        'CALC': np.random.choice(['no', 'Sometimes', 'Frequently', 'Always'], num_rows),
        'MTRANS': np.random.choice(['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'], num_rows),
        'NObeyesdad': np.random.choice([
            'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II',
            'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
        ], num_rows)
    }
    
    df = pd.DataFrame(data)
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, "ObesityDataSet_raw_and_data_sinthetic.csv")
    df.to_csv(output_file, index=False)
    print(f"Synthetic data generated at {output_file}")

if __name__ == "__main__":
    generate_data()
