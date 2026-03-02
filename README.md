# -Obesity-Intelligence-Platform
An End-to-End AI Solution for Health Risk Assessment &amp; Analysis 
📌 Project Overview
This project is a production-grade health analytics platform that uses advanced Machine Learning to assess obesity risks. Unlike basic models, this platform features a Dual-Engine Architecture:

Classification Engine: Categorizes users into 7 specific obesity risk levels.

Regression Engine: Predicts an "Expected Healthy Weight" based on individual lifestyle metrics to quantify the health gap.

🚀 Key Features for Placement Highlights
Automated Model Selection: A custom training pipeline that evaluates XGBoost, LightGBM, CatBoost, and Random Forest, automatically saving the best-performing model based on cross-validation.

Advanced Feature Engineering: Custom logic to calculate BMI, Lifestyle Scores (Activity vs. Technology usage), and Hydration Indices.

Explainable AI (XAI): Integrated SHAP values to explain the "Why" behind every prediction, making the AI transparent and trustworthy.

Interactive Dashboard: A modern UI built with Streamlit and Plotly, featuring real-time "What-If" analysis for weight management.

🛠️ Tech Stack
Language: Python 3.x

Machine Learning: Scikit-Learn, XGBoost, LightGBM, CatBoost

Data Processing: Pandas, NumPy

Visualization: Plotly, Seaborn, Matplotlib

Deployment: Streamlit (Web UI), Joblib (Model Serialization)

📂 Project Architecture
Plaintext
├── app/              # Streamlit Dashboard & Custom CSS
├── artifacts/        # Trained Models (.joblib) and Metadata
├── src/              # Core Logic: Feature Engineering & Training Pipeline
├── requirements.txt  # Project Dependencies
└── run_platform.bat  # One-click automation script
📈 Performance & Methodology
Preprocessing: Handled categorical encoding (OneHot) and numerical scaling (StandardScaler) using Scikit-Learn Pipelines to prevent data leakage.

Optimization: Utilized Gradient Boosting variants to handle non-linear relationships in lifestyle data.

Deployment: Optimized the model size and inference speed for real-time response on the dashboard.
