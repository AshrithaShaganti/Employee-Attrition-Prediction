import streamlit as st
import joblib
import numpy as np

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================

model = joblib.load("attrition_model.pkl")
scaler = joblib.load("attrition_scaler.pkl")

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.big-font {
    font-size:40px !important;
    font-weight:bold;
    color:#1f77b4;
}

.metric-card {
    background-color:#f5f7fa;
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 2px 8px rgba(0,0,0,0.1);
}

.high-risk {
    background-color:#ffebee;
    padding:20px;
    border-radius:15px;
    text-align:center;
    color:#c62828;
    font-weight:bold;
    font-size:24px;
}

.low-risk {
    background-color:#e8f5e9;
    padding:20px;
    border-radius:15px;
    text-align:center;
    color:#2e7d32;
    font-weight:bold;
    font-size:24px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================

st.markdown(
    "<p class='big-font'>📊 AI Employee Attrition Prediction System</p>",
    unsafe_allow_html=True
)

st.markdown(
    "Predict whether an employee is likely to leave the organization."
)

st.divider()

# =========================
# INPUT SECTION
# =========================

left, right = st.columns([1,1])

with left:

    st.subheader("👤 Employee Information")

    age = st.slider(
        "Age",
        18,
        60,
        30
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=50000,
        value=5000,
        step=500
    )

    total_working_years = st.slider(
        "Total Working Years",
        0,
        40,
        5
    )

with right:

    st.subheader("💼 Job Information")

    job_satisfaction = st.select_slider(
        "Job Satisfaction",
        options=[1,2,3,4],
        value=2
    )

    overtime = st.selectbox(
        "OverTime",
        ["No","Yes"]
    )

# =========================
# PREDICT BUTTON
# =========================

st.write("")
predict_btn = st.button(
    "🔍 Predict Attrition Risk",
    use_container_width=True
)

# =========================
# PREDICTION
# =========================

if predict_btn:

    overtime_value = 1 if overtime == "Yes" else 0

    data = np.array([[
        age,
        monthly_income,
        job_satisfaction,
        overtime_value,
        total_working_years
    ]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]

    probability = model.predict_proba(data_scaled)[0]

    risk_probability = probability[1] * 100

    st.divider()

    st.subheader("📈 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Age",
            age
        )

    with col2:
        st.metric(
            "Income",
            f"₹{monthly_income:,.0f}"
        )

    with col3:
        st.metric(
            "Experience",
            total_working_years
        )

    st.write("")

    if prediction == 1:

        st.markdown(
            f"""
            <div class='high-risk'>
            ⚠ HIGH ATTRITION RISK
            <br><br>
            Probability: {risk_probability:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class='low-risk'>
            ✅ LOW ATTRITION RISK
            <br><br>
            Probability: {100-risk_probability:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    st.progress(min(int(risk_probability),100))

    st.info(
        f"Estimated Attrition Probability: {risk_probability:.2f}%"
    )

    # HR Suggestions

    st.subheader("💡 HR Recommendations")

    if risk_probability > 70:

        st.error("""
        - Review compensation package
        - Improve employee engagement
        - Discuss career growth opportunities
        - Improve work-life balance
        """)

    elif risk_probability > 40:

        st.warning("""
        - Schedule employee feedback sessions
        - Monitor job satisfaction
        - Offer training opportunities
        """)

    else:

        st.success("""
        - Employee appears stable
        - Continue current engagement practices
        - Encourage professional development
        """)

# =========================
# FOOTER
# =========================

st.divider()

st.caption(
    "AI-Powered Employee Attrition Prediction System | Logistic Regression Model"
)