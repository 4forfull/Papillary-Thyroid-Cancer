import streamlit as st
import joblib
import numpy as np

model = joblib.load('LGBM-SHAP10.m')

st.image('image.jpg', width=700)

pdw = st.number_input("PDW (9.0-17.0, %)", min_value=0.0, max_value=100.0, step=0.1)
apt = st.number_input("APT (25.0-31.3, sec)", min_value=0.0, max_value=100.0, step=0.1)
tt = st.number_input("TT (10.0-21.0, sec)", min_value=0.0, max_value=100.0, step=0.1)
pt_inr = st.number_input("PT-INR (0.80-1.50)", min_value=0.0, max_value=10.0, step=0.01)
pt_rati = st.number_input("PT-Rati (0.90-1.20)", min_value=0.0, max_value=10.0, step=0.01)
fib = st.number_input("FIB (2.000-4.000, g/l)", min_value=0.0, max_value=100.0, step=0.1)
pt_act = st.number_input("PT-ACT (70.0-120.0, %)", min_value=0.0, max_value=500.0, step=0.1)
age = st.number_input("Age (year)", min_value=0, max_value=200, step=1)
p_lcr = st.number_input("P-LCR (13.0-43.0)", min_value=0.0, max_value=100.0, step=0.1)
mo = st.number_input("MO% (0.10-0.60, 10^9/L)", min_value=0.0, max_value=10.0, step=0.1)

if st.button("RUN"):
    input_features = np.array([[pdw, apt, tt, pt_inr, pt_rati, fib, pt_act, age, p_lcr, mo]])

    prediction = model.predict(input_features)[0]
    prediction_proba = model.predict_proba(input_features)[0]

    result_text = "Healthy" if prediction == 0 else "Papillary Thyroid Carcinoma"
    proba_text = f"Probability (Healthy): {prediction_proba[0] * 100:.2f}% | Probability (Cancer): {prediction_proba[1] * 100:.2f}%"

    st.markdown(f"<h3 style='color: red;'>Prediction Result: {result_text}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color: red;'>{proba_text}</h4>", unsafe_allow_html=True)