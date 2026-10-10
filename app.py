import streamlit as st
import pandas as pd
import joblib

# ==========================================
# SESSION STATE FOR RESET
# ==========================================

if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ==========================================
# LOAD MODEL FILES
# ==========================================

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("❤️ Heart Disease Predictor")

st.sidebar.markdown("""
### 🧠 Model Details

❤️ Heart Disease Prediction

🤖 Algorithm: K-Nearest Neighbors (KNN)

⚙️ Framework: Streamlit

👨‍💻 Developer: Naman
""")

# ==========================================
# HEADER
# ==========================================

header_col1, header_col2 = st.columns([9, 2])

with header_col1:
    st.title("❤️ Heart Disease Prediction System")

with header_col2:

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "🔄 Reset",
        use_container_width=True
    ):
        st.session_state.reset_counter += 1
        st.rerun()

st.info(
    "🩺 Enter patient details below and click Predict to estimate heart disease risk."
)

# ==========================================
# INPUT SECTION
# ==========================================

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "🎂 Age",
        min_value=18,
        max_value=100,
        value=40,
        step=1,
        key=f"age_{st.session_state.reset_counter}"
    )

    sex_display = st.selectbox(
        "👤 Sex",
        ["Male", "Female"],
        key=f"sex_{st.session_state.reset_counter}"
    )

    sex = "M" if sex_display == "Male" else "F"

    chest_pain_display = st.selectbox(
        "💢 Chest Pain Type",
        [
            "Atypical Angina (ATA)",
            "Non-Anginal Pain (NAP)",
            "Typical Angina (TA)",
            "Asymptomatic (ASY)"
        ],
        key=f"chest_{st.session_state.reset_counter}"
    )

    cp_map = {
        "Atypical Angina (ATA)": "ATA",
        "Non-Anginal Pain (NAP)": "NAP",
        "Typical Angina (TA)": "TA",
        "Asymptomatic (ASY)": "ASY"
    }

    chest_pain = cp_map[chest_pain_display]

    resting_bp = st.number_input(
        "🩸 Resting Blood Pressure (mm Hg)",
        min_value=80,
        max_value=200,
        value=120,
        key=f"bp_{st.session_state.reset_counter}"
    )

    cholesterol = st.number_input(
        "🧪 Cholesterol (mg/dL)",
        min_value=100,
        max_value=600,
        value=200,
        key=f"chol_{st.session_state.reset_counter}"
    )

    slope_display = st.selectbox(
        "📊 ST Slope",
        [
            "Upsloping (Healthy)",
            "Flat (Moderate Risk)",
            "Downsloping (High Risk)"
        ],
        key=f"slope_{st.session_state.reset_counter}"
    )

    slope_map = {
        "Upsloping (Healthy)": "Up",
        "Flat (Moderate Risk)": "Flat",
        "Downsloping (High Risk)": "Down"
    }

    st_slope = slope_map[slope_display]

with col2:

    fasting_bs_display = st.selectbox(
        "🍬 Fasting Blood Sugar",
        [
            "Normal (≤ 120 mg/dL)",
            "High (> 120 mg/dL)"
        ],
        key=f"fasting_{st.session_state.reset_counter}"
    )

    fasting_bs = (
        0 if fasting_bs_display == "Normal (≤ 120 mg/dL)"
        else 1
    )

    ecg_display = st.selectbox(
        "📈 Resting ECG",
        [
            "Normal",
            "ST Abnormality",
            "LVH (Left Ventricular Hypertrophy)"
        ],
        key=f"ecg_{st.session_state.reset_counter}"
    )

    ecg_map = {
        "Normal": "Normal",
        "ST Abnormality": "ST",
        "LVH (Left Ventricular Hypertrophy)": "LVH"
    }

    resting_ecg = ecg_map[ecg_display]

    max_hr = st.number_input(
        "❤️ Maximum Heart Rate",
        min_value=60,
        max_value=220,
        value=150,
        key=f"hr_{st.session_state.reset_counter}"
    )

    exercise_display = st.selectbox(
        "🏃 Exercise-Induced Angina",
        ["Yes", "No"],
        key=f"angina_{st.session_state.reset_counter}"
    )

    exercise_angina = (
        "Y" if exercise_display == "Yes"
        else "N"
    )

    oldpeak = st.number_input(
        "📉 Oldpeak (ST Depression)",
        min_value=0.0,
        max_value=6.0,
        value=1.0,
        step=0.1,
        key=f"oldpeak_{st.session_state.reset_counter}"
    )

# ==========================================
# BUTTONS
# ==========================================

st.markdown("###")

btn_left, btn_center, btn_right = st.columns([3, 4, 3])

with btn_center:
    predict_btn = st.button(
        "❤️ Predict Heart Disease Risk",
        use_container_width=True
    )


# ==========================================
# PREDICTION
# ==========================================

if predict_btn:

    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,

        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    st.markdown("---")

    try:

        probability = model.predict_proba(input_scaled)[0]
        confidence = float(max(probability))

        metric1, metric2 = st.columns(2)

        with metric1:
            st.metric(
                "📊 Confidence Score",
                f"{confidence * 100:.2f}%"
            )

        with metric2:
            if prediction == 1:
                st.metric(
                    "Prediction",
                    "High Risk 🚨"
                )
            else:
                st.metric(
                    "Prediction",
                    "Low Risk ✅"
                )

        st.progress(confidence)

    except:
        confidence = None

    if prediction == 1:

        st.error("🚨 High Risk of Heart Disease")

        st.warning("""
### 💡 Recommendations

✅ Consult a Cardiologist

✅ Monitor Blood Pressure

✅ Monitor Cholesterol

✅ Exercise Regularly

✅ Maintain Healthy Weight

✅ Reduce Stress

✅ Avoid Smoking

✅ Follow a Heart-Healthy Diet
""")

    else:

        st.success("✅ Low Risk of Heart Disease")

        st.balloons()

        st.info("""
### 🎉 Great!

Current inputs indicate a lower risk of heart disease.

Continue maintaining:

✅ Healthy Diet

✅ Regular Exercise

✅ Proper Sleep

✅ Routine Health Checkups
""")

    if cholesterol > 240:
        st.warning(
            "⚠️ Cholesterol level is higher than recommended."
        )

    if resting_bp > 140:
        st.warning(
            "⚠️ Blood Pressure is elevated."
        )

# ==========================================
# ABOUT PROJECT
# ==========================================

with st.expander("ℹ️ About This Project"):

    st.markdown("""
### ❤️ Heart Disease Prediction System

This application uses Machine Learning to predict the likelihood of heart disease.

### Features Used

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- Oldpeak
- ST Slope

### Model Used

✅ K-Nearest Neighbors (KNN)

### Developed By

👨‍💻 Naman
""")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    """
    <div style='text-align:center'>
        👨‍💻 Developed by <b>Naman</b> |
        ❤️ Heart Disease Prediction System
    </div>
    """,
    unsafe_allow_html=True
)