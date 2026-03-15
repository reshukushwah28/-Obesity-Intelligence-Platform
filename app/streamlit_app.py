import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

# Import custom feature engineer so pickle can find it
import sys
sys.path.append(os.path.abspath('.'))
from src.features import FeatureEngineer

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Obesity Intelligence Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

if os.path.exists("app/style.css"):
    local_css("app/style.css")

# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_models():
    artifacts_dir = "artifacts"
    try:
        clf = joblib.load(os.path.join(artifacts_dir, "best_classifier.joblib"))
        reg = joblib.load(os.path.join(artifacts_dir, "best_regressor.joblib"))
        meta = joblib.load(os.path.join(artifacts_dir, "classifier_metadata.joblib"))
        le = joblib.load(os.path.join(artifacts_dir, "label_encoder.joblib"))
        return clf, reg, meta, le
    except FileNotFoundError:
        return None, None, None, None

clf_pipeline, reg_pipeline, metadata, le = load_models()

if not clf_pipeline:
    st.error("🚨 Models not found! Please run 'src/train_platform.py' first.")
    st.stop()

# --- SIDEBAR INPUTS ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3050/3050525.png", width=100)
st.sidebar.title("Patient Profile")

with st.sidebar.form("profile_form"):
    st.subheader("Demographics")
    age = st.slider("Age", 14, 65, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    height = st.number_input("Height (m)", 1.40, 2.00, 1.70, 0.01)
    weight = st.number_input("Weight (kg)", 40.0, 160.0, 70.0, 0.5)
    
    st.subheader("Genetics & Habits")
    family_hist = st.selectbox("Family History of Obesity", ["yes", "no"])
    smoke = st.selectbox("Smoker", ["yes", "no"])
    calc = st.selectbox("Alcohol Consumption", ["no", "Sometimes", "Frequently", "Always"])
    
    st.subheader("Dietary Habits")
    favc = st.selectbox("High Caloric Food Intake", ["yes", "no"])
    fcvc = st.slider("Vegetable Consumption (1-3)", 1.0, 3.0, 2.0, help="1: Never, 2: Sometimes, 3: Always")
    ncp = st.slider("Meals per Day", 1.0, 4.0, 3.0)
    caec = st.selectbox("Snacking Frequency", ["no", "Sometimes", "Frequently", "Always"])
    ch2o = st.slider("Daily Water Intake (L)", 1.0, 3.0, 2.0)
    scc = st.selectbox("Monitor Calories?", ["yes", "no"])
    
    st.subheader("Physical Condition")
    faf = st.slider("Physical Activity (Days/Week)", 0.0, 3.0, 1.0)
    tue = st.slider("Tech Usage (Hours/Day)", 0.0, 2.0, 1.0)
    mtrans = st.selectbox("Transportation", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])
    
    run_pred = st.form_submit_button("Analyze Health Profile")

# --- MAIN LOGIC ---

# --- MAIN LOGIC ---

# 1. Prepare Data
input_data = {
    'Gender': [gender], 'Age': [age], 'Height': [height], 'Weight': [weight],
    'family_history_with_overweight': [family_hist],
    'FAVC': [favc], 'FCVC': [fcvc], 'NCP': [ncp], 'CAEC': [caec],
    'SMOKE': [smoke], 'CH2O': [ch2o], 'SCC': [scc], 'FAF': [faf],
    'TUE': [tue], 'CALC': [calc], 'MTRANS': [mtrans]
}
df_input = pd.DataFrame(input_data)

# 2. Predictions & Feature Engineering
try:
    # Feature Engineering (Shared Logic)
    # We use the classifier's feature engineer to get derived scores for display
    fe = clf_pipeline.named_steps['features']
    df_transformed = fe.transform(df_input)
    
    # Classification
    pred_class_idx = clf_pipeline.predict(df_input)[0]
    pred_class = le.inverse_transform([pred_class_idx])[0]
    pred_proba = clf_pipeline.predict_proba(df_input)[0]
    classes = le.classes_
    
    # Regression (Expected Weight)
    exp_weight = reg_pipeline.predict(df_input)[0]

except Exception as e:
    st.error(f"Prediction Error: {e}")
    st.stop()

# 3. Derived Metrics for UI
bmi = weight / (height**2)
lifestyle = df_transformed['Lifestyle_Score'].iloc[0]
hydration = df_transformed['Hydration_Index'].iloc[0]
cal_risk = df_transformed['Calorie_Risk_Score'].iloc[0]

# --- TABS LAYOUT ---
t1, t2, t3, t4, t5 = st.tabs(["🩺 Diagnosis", "⚖️ Weight Lab", "📊 Health Metrics", "💡 Advice", "🧠 Explainability"])

with t1:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Top Row: Prediction Card & Stats
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        with st.container(border=True):
            st.subheader("Current Status")
            
            # Color Logic
            if "Obesity" in pred_class:
                status_color = "#FF4B4B" # Red
                status_icon = "🚨"
            elif "Overweight" in pred_class:
                status_color = "#FF914D" # Orange
                status_icon = "⚠️"
            else:
                status_color = "#4ECDC4" # Teal
                status_icon = "✅"
            
            st.markdown(f"""
                <div style="background-color: {status_color}; padding: 1.5rem; border-radius: 12px; text-align: center; color: white;">
                    <h1 style="color: white !important; margin: 0; font-size: 2.5rem;">{status_icon}</h1>
                    <h3 style="color: white !important; margin: 0;">{pred_class.replace('_', ' ')}</h3>
                    <p style="color: rgba(255,255,255,0.9); margin-top: 0.5rem;">Confidence: {max(pred_proba)*100:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
        
    with c2:
        with st.container(border=True):
            st.subheader("Probability Breakdown")
            
            prob_df = pd.DataFrame({"Category": classes, "Probability": pred_proba})
            prob_df = prob_df.sort_values("Probability", ascending=True)
            
            fig = px.bar(prob_df, x="Probability", y="Category", orientation='h',
                         text_auto='.1%', color="Probability", 
                         color_continuous_scale="RdYlGn_r")
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=0, b=0),
                height=250,
                xaxis=dict(showgrid=False),
                yaxis=dict(title=None)
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with t2:
    with st.container(border=True):
        st.subheader("AI Weight Analysis")
        st.info("Based on your lifestyle and demographics (excluding your actual weight), our AI estimates what your weight 'should' be.")
        
        wc1, wc2, wc3 = st.columns(3)
        
        with wc1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Actual Weight</div>
                <div class="metric-value">{weight} kg</div>
            </div>
            """, unsafe_allow_html=True)
            
        with wc2:
            diff = weight - exp_weight
            diff_color = "red" if diff > 5 else "green" if diff < -5 else "orange"
            arrow = "⬇️" if diff > 0 else "⬆️" if diff < 0 else "✅"
            
            st.markdown(f"""
            <div class="metric-box" style="border-left-color: {diff_color};">
                <div class="metric-label">Projected Weight</div>
                <div class="metric-value">{exp_weight:.1f} kg</div>
                <div style="font-size: 0.8rem; color: {diff_color};">{arrow} Gap: {abs(diff):.1f} kg</div>
            </div>
            """, unsafe_allow_html=True)

        with wc3:
             st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Target BMI</div>
                <div class="metric-value">{exp_weight / (height**2):.1f}</div>
            </div>
            """, unsafe_allow_html=True)

with t3:
    st.markdown("### Key Health Indicators")
    
    hc1, hc2 = st.columns(2)
    
    with hc1:
        with st.container(border=True):
            st.subheader("BMI Gauge")
            
            fig_bmi = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = bmi,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Current BMI"},
                delta = {'reference': 25, 'increasing': {'color': "red"}},
                gauge = {
                    'axis': {'range': [10, 50], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#00B4D8"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 18.5], 'color': "#CAF0F8"},
                        {'range': [18.5, 25], 'color': "#90E0EF"},
                        {'range': [25, 30], 'color': "#FFE5D9"},
                        {'range': [30, 50], 'color': "#FFCAD4"}],
                    }))
            fig_bmi.update_layout(paper_bgcolor='white', font={'color': "#2B2D42"}, height=250, margin=dict(l=20,r=20,t=0,b=0))
            st.plotly_chart(fig_bmi, use_container_width=True)
        
    with hc2:
        with st.container(border=True):
            st.subheader("Lifestyle Score")
            
            # Radar Chart for Lifestyle
            categories = ['Physical Activity', 'Hydration', 'Vegetables', 'Tech Usage (Inv)', 'Calorie Awareness']
            
            # Normalize values roughly for the chart (0-1 scale)
            vals = [
                min(faf/3.0, 1), 
                min(ch2o/3.0, 1), 
                min(fcvc/3.0, 1), 
                max(1 - (tue/2.0), 0), # Inverse because less is better
                1 if scc == "yes" else 0.5
            ]
            
            fig_radar = px.line_polar(r=vals, theta=categories, line_close=True)
            fig_radar.update_traces(fill='toself', line_color='#00B4D8')
            fig_radar.update_layout(
                paper_bgcolor='white',
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1]),
                    bgcolor='#F8F9FA'
                ),
                margin=dict(l=20,r=20,t=20,b=20),
                height=250
            )
            st.plotly_chart(fig_radar, use_container_width=True)

with t4:
    st.subheader("Personalized Action Plan")
    
    recs = []
    
    if bmi > 25:
        recs.append(("warning", "⚠️ **Weight Management**: Your BMI implies you are above the recommended range. A slight caloric deficit (300-500 kcal) is advised."))
    if faf < 2:
        recs.append(("info", "🏃‍♂️ **Activity**: Boost your metabolism! Aim for at least 3 days of moderate exercise (brisk walking, cycling) per week."))
    if tue > 3:
        recs.append(("warning", "💻 **Digital Detox**: High screen time correlates with sedentary habits. Try the 20-20-20 rule: Every 20 mins, look 20 ft away for 20 secs."))
    if fcvc < 2:
        recs.append(("success", "🥦 **Nutrition**: Volume eating is key. Add more leafy greens to your main meals to feel full with fewer calories."))
    if ch2o < 2:
        recs.append(("info", "💧 **Hydration**: You're under-hydrated. Drinking 2.5L+ daily helps flush toxins and reduces false hunger signals."))
    if favc == "yes":
        recs.append(("warning", "🍔 **Dietary Choices**: High-calorie foods are dense but not filling. Swap one processed snack a day for a fruit."))
        
    if not recs:
        st.markdown('<div class="rec-box rec-success">🎉 Amazing! You have a very balanced and healthy lifestyle. Keep it up!</div>', unsafe_allow_html=True)
    else:
        for type_, text in recs:
            css_class = f"rec-{type_}"
            st.markdown(f'<div class="rec-box {css_class}">{text}</div>', unsafe_allow_html=True)

with t5:
    st.subheader("Understand the Prediction")
    st.markdown("See which features acted as the strongest drivers for your specific result.")
    
    if st.button("Generate Feature Importance"):
        try:
            model = clf_pipeline.named_steps['classifier']
            preprocessor = clf_pipeline.named_steps['preprocessor']
            
            # Helper to get feature names
            num_cols = metadata['features']['numerical']
            try:
                cat_encoder = preprocessor.named_transformers_['cat']
                cat_cols = cat_encoder.get_feature_names_out(metadata['features']['categorical'])
                all_features = np.concatenate([num_cols, cat_cols])
            except:
                 all_features = [f"Feature {i}" for i in range(50)]
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:10]
                
                top_features = all_features[indices]
                top_scores = importances[indices]
                
                fig_imp = px.bar(x=top_scores, y=top_features, orientation='h', 
                                 title="Top 10 Influential Factors", labels={'x':"Impact", 'y': "Feature"})
                fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor='white', plot_bgcolor='white')
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("The selected model does not support direct feature importance visualization.")
                
        except Exception as e:
            st.error(f"Could not generate explanation: {e}")

st.markdown("---")
st.caption("Obesity Intelligence Platform v2.0 | Enterprise Edition")
