# 🏥 Obesity Intelligence Platform

**v2.0 - Enterprise Edition**

A production-grade AI platform for Obesity Risk Assessment and Personalized Health Recommendations.
This project uses an ensemble of advanced Machine Learning models (Random Forest, Gradient Boosting, etc.) to provide accurate multiclass classification and regression analysis.

## ✨ New Features in v2.0
- **Multi-Model Intelligence**: Automatically trains and selects the best model from Random Forest, Gradient Boosting, XGBoost, LightGBM, and CatBoost.
- **Dual-Prediction Engines**:
  - **Classification**: Predicts Obesity Risk Level (7 Categories).
  - **Regression**: Estimates "Expected Weight" based on lifestyle to quantify health gaps.
- **Interactive Health Dashboard**: A modern Streamlit UI with animated gauges, probability distributions, and risk meters.
- **Explainable AI (XAI)**: Integrated feature importance analysis to explain *why* a risk level was predicted.
- **Smart Recommendations**: Hybrid Rule-ML engine providing actionable lifestyle advice.

## 📂 Project Structure

```
MLProject/
├── app/
│   ├── streamlit_app.py    # Main Platform Application
│   └── style.css           # Modern Healthcare Theme
├── artifacts/             # Saved Models & Metadata
├── data/
│   └── ObesityDataSet...csv
├── src/
│   ├── features.py         # 🔍 Centralized Feature Engineering
│   ├── train_platform.py   # 🚂 End-to-End Training Pipeline
│   └── __init__.py
├── requirements.txt        # Dependencies
└── README.md
```

## � Quick Start

1. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Platform**
   This determines the best models for your data and saves the artifacts.
   ```bash
   python src/train_platform.py
   ```
   *Note: This script automatically handles cross-validation and selects between XGBoost, LightGBM, etc. based on availability.*

3. **Launch the Dashboard**
   ```bash
   streamlit run app/streamlit_app.py
   ```

## 🧪 Model Details

### Feature Engineering
The system auto-calculates key health indicators:
- **BMI**: Body Mass Index.
- **Lifestyle Score**: Composite metric of Activity vs. Tech Usage.
- **Hydration Index**: Normalized water intake.
- **Calorie Risk**: Aggregation of high-calorie food & snacking habits.

### Supported Models
The training pipeline supports:
- **RF**: Random Forest (Robust baseline)
- **GBM**: Gradient Boosting (High accuracy)
- **XGB**: XGBoost (State-of-the-art tabular)
- **LGBM**: LightGBM (Speed & Performance)
- **Cat**: CatBoost (Categorical optimization)
- **NN**: MLP Neural Network (Complex patterns)

## 📊 Dashboard Overview
- **Diagnosis Tab**: Instant prediction with confidence score.
- **Weight Lab**: "What-if" analysis for weight goals.
- **Metrics**: Visual gauges for BMI and Lifestyle scores.
- **Recs**: Automated, safe lifestyle suggestions.

---
*Developed for Advanced Healthcare AI Portfolio.*
