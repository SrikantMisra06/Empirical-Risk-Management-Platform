import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import warnings


warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Clinical Risk Prediction Platform", page_icon="⚕️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container { max-width: 95% !important; padding-top: 2rem !important; }
    

    
    .stApp { background-color: #ffffff; color: #111827; }
    [data-testid="stSidebar"] { background-color: #0f172a !important; border-right: 1px solid #1e293b; }
    [data-testid="stSidebar"] * { color: #f8fafc !important; }
    .stNumberInput label, .stSelectbox label { color: #374151 !important; font-weight: 600 !important; font-size: 0.95rem !important;}
    
    button[data-baseweb="tab"] {
        font-size: 1rem !important; font-weight: 600 !important; color: #64748b !important; 
        background-color: transparent !important; border: none !important; padding: 0.75rem 2rem !important; 
        border-bottom: 2px solid #e2e8f0 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0d9488 !important; border-bottom: 3px solid #0d9488 !important; background-color: #f0fdfa !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%) !important;
        color: white !important; border: none; padding: 0.6rem 2.5rem;
        border-radius: 6px; font-weight: 600; margin-top: 1.5rem; width: 100%; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover { background: linear-gradient(135deg, #0f766e 0%, #115e59 100%) !important; transform: translateY(-1px); }
    
    .result-box { padding: 1.5rem; border-radius: 8px; margin-top: 2rem; border: 1px solid #e5e7eb;}
    .high-risk { background-color: #fef2f2; border-left: 5px solid #ef4444; color: #991b1b; }
    .low-risk { background-color: #f0fdf4; border-left: 5px solid #10b981; color: #065f46; }
    
    /* 🌟 PREMIUM CLINICAL INFO CARDS 🌟 */
    .clinical-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #0d9488;
        padding: 1.5rem;
        border-radius: 8px;
        height: 100%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .clinical-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .clinical-card h4 { color: #0f766e; margin-top: 0; font-size: 1.15rem; font-weight: 700; }
    .clinical-card p { color: #64748b; font-size: 0.9rem; margin-bottom: 0.8rem; }
    .clinical-card li { color: #475569; font-size: 0.9rem; margin-bottom: 0.4rem; }

    /* 🌟 FILE UPLOADER REDESIGN 🌟 */
    [data-testid="stFileUploadDropzone"] {
        background-color: #f8fafc !important;
        border: 2px dashed #cbd5e1 !important;
        border-radius: 10px !important;
        padding: 2rem !important;
    }
    [data-testid="stFileUploadDropzone"] * { color: #334155 !important; }
    [data-testid="stFileUploadDropzone"] button {
        background: #0d9488 !important;
        color: white !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stFileUploadDropzone"] button:hover {
        background: #0f766e !important;
        transform: scale(1.05) !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    models = {}
    try:
        models["heart_model"] = joblib.load(os.path.join(BASE_DIR, "heart_model.pkl"))
        models["heart_scaler"] = joblib.load(os.path.join(BASE_DIR, "heart_scaler.pkl"))
        models["diabetes_model"] = joblib.load(os.path.join(BASE_DIR, "diabetes_model.pkl"))
        models["diabetes_scaler"] = joblib.load(os.path.join(BASE_DIR, "diabetes_scaler.pkl"))
        models["parkinsons_model"] = joblib.load(os.path.join(BASE_DIR, "parkinsons_model.pkl"))
        models["parkinsons_scaler"] = joblib.load(os.path.join(BASE_DIR, "parkinsons_scaler.pkl"))
    except Exception as e:
        st.sidebar.error(f"Missing .pkl files. Please check folder. {e}")
    return models

models = load_models()

st.sidebar.markdown("## 🧬 Diagnostics Menu")
st.sidebar.markdown("---")
disease = st.sidebar.radio("Select a module:", ["Home / Overview", "🩸 Diabetes Prediction", "❤️ Heart Disease Prediction", "🧠 Parkinson's Prediction"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.info("🌟 **Patient Portal**\n\nYour health data is processed securely. This tool provides instant, preliminary insights to help you manage your wellness.")

def display_prediction(prediction, prob):
    if prediction == 1:
        st.markdown(f'<div class="result-box high-risk"><h3 style="margin:0;">⚠️ High Risk Detected</h3><p style="margin-top:0.5rem; margin-bottom:0;">Model confidence: {prob*100:.2f}%. Please consult a healthcare professional.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-box low-risk"><h3 style="margin:0;">✅ Low Risk</h3><p style="margin-top:0.5rem; margin-bottom:0;">Model confidence: {(1-prob)*100:.2f}%. Patient profile appears normal.</p></div>', unsafe_allow_html=True)


# HOME PAGE (PATIENT-FACING DASHBOARD)

if disease == "Home / Overview":
    # 1. WELCOMING HERO BANNER
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #0d9488 100%); padding: 4rem 3rem; border-radius: 1.5rem; color: white; margin-bottom: 2.5rem; box-shadow: 0 10px 25px -5px rgba(13, 148, 136, 0.4);">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span style="background: rgba(94, 234, 212, 0.2); color: #5eead4; padding: 0.4rem 1rem; border-radius: 2rem; font-weight: 600; font-size: 0.85rem; letter-spacing: 1px;">🌟 PATIENT WELLNESS PORTAL</span>
        </div>
        <h1 style="font-size: 3.8rem; margin: 0; color: white; line-height: 1.1; font-weight: 800;">Take Control of Your<br><span style="color: #5eead4;">Health Journey</span></h1>
        <p style="font-size: 1.2rem; color: #e2e8f0; margin-top: 1.5rem; max-width: 650px; line-height: 1.6;">A smart, secure, and easy-to-use tool to help you understand your risk factors for common health conditions. Select a test from the menu to begin.</p>
    </div>
    """, unsafe_allow_html=True)

    # 2. PATIENT-FOCUSED BENEFITS ROW
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown('<div style="background:#f8fafc; padding:1.5rem; border-radius:10px; border-left:4px solid #0d9488; box-shadow:0 4px 6px rgba(0,0,0,0.05);"> <h3 style="margin:0; color:#0f172a; font-size:2rem;">Instant</h3> <p style="margin:0; color:#64748b; font-weight:600;">Results & Insights</p> </div>', unsafe_allow_html=True)
    m2.markdown('<div style="background:#f8fafc; padding:1.5rem; border-radius:10px; border-left:4px solid #3b82f6; box-shadow:0 4px 6px rgba(0,0,0,0.05);"> <h3 style="margin:0; color:#0f172a; font-size:2rem;">100%</h3> <p style="margin:0; color:#64748b; font-weight:600;">Private & Secure</p> </div>', unsafe_allow_html=True)
    m3.markdown('<div style="background:#f8fafc; padding:1.5rem; border-radius:10px; border-left:4px solid #8b5cf6; box-shadow:0 4px 6px rgba(0,0,0,0.05);"> <h3 style="margin:0; color:#0f172a; font-size:2rem;">3</h3> <p style="margin:0; color:#64748b; font-weight:600;">Health Checks</p> </div>', unsafe_allow_html=True)
    m4.markdown('<div style="background:#f8fafc; padding:1.5rem; border-radius:10px; border-left:4px solid #10b981; box-shadow:0 4px 6px rgba(0,0,0,0.05);"> <h3 style="margin:0; color:#0f172a; font-size:2rem;">24/7</h3> <p style="margin:0; color:#64748b; font-weight:600;">Always Accessible</p> </div>', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # 3. INTERACTIVE MODULE CARDS (Non-Technical Descriptions)
    st.markdown("<h3 style='color: #1e293b; margin-bottom: 1.5rem;'>Available Health Checks</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .home-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 2rem;
        transition: all 0.3s ease; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        height: 100%; position: relative; overflow: hidden;
    }
    .home-card:hover { transform: translateY(-8px); box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1); border-color: #0d9488;}
    .home-card h4 { color: #0f172a; font-size: 1.25rem; margin-top: 1rem; font-weight: 700; }
    .home-card p { color: #64748b; font-size: 0.95rem; line-height: 1.6; }
    .icon-wrapper { font-size: 2.8rem; margin-bottom: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="home-card">
            <div class="icon-wrapper">🩸</div>
            <h4>Diabetes Screening</h4>
            <p>Answer a few simple questions about your health history and vitals to understand your potential risk for diabetes.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="home-card">
            <div class="icon-wrapper">❤️</div>
            <h4>Heart Health Check</h4>
            <p>Upload a routine lab report or enter your vitals manually to get an instant assessment of your cardiovascular health.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="home-card">
            <div class="icon-wrapper">🧠</div>
            <h4>Parkinson's Assessment</h4>
            <p>Upload a standard vocal analysis file to securely screen for early micro-tremors associated with Parkinson's.</p>
        </div>
        """, unsafe_allow_html=True)

# DIABETES PREDICTION

elif disease == "🩸 Diabetes Prediction":
    st.markdown("<h2 style='color: #111827;'>Diabetes Risk Assessment</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; margin-bottom: 1rem;'>Select your preferred data input method below.</p>", unsafe_allow_html=True)

    input_tab1, input_tab2 = st.tabs(["✍️ Manual Entry", "📁 CSV Lab Upload"])

    with input_tab1:
        st.write("")
        col1, col2, col3 = st.columns(3)
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.35, step=0.001)
        with col2:
            glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
            insulin = st.number_input("Insulin (μU/mL)", min_value=0, max_value=900, value=85)
            age = st.number_input("Age", min_value=1, max_value=120, value=45)
        with col3:
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=80)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=26.5, step=0.1)

        if st.button("Predict Diabetes Risk", key="diabetes_manual_btn"):
            try:
                features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
                scaled_features = models["diabetes_scaler"].transform(features)
                pred = models["diabetes_model"].predict(scaled_features)[0]
                prob = models["diabetes_model"].predict_proba(scaled_features)[0][1]
                display_prediction(pred, prob)
            except Exception as e:
                st.error(f"Prediction Error: {e}")

    with input_tab2:
        st.write("")
        st.info("Upload a patient's CSV file. Ensure it contains the 8 required clinical parameters.")
        
        st.markdown("""
        <style>
            [data-testid="stFileUploadDropzone"] {
                background-color: #1e293b !important; 
                border: 2px dashed #0d9488 !important;
                border-radius: 8px !important;
            }
            [data-testid="stFileUploadDropzone"] * {
                color: #ffffff !important; 
            }
            [data-testid="stFileUploader"] button {
                background-color: #0d9488 !important; 
                border: none !important;
                border-radius: 5px !important;
            }
            [data-testid="stFileUploader"] button * {
                color: #ffffff !important; 
                font-weight: bold !important;
            }
        </style>
        """, unsafe_allow_html=True)
       

        uploaded_file = st.file_uploader("Upload Patient Data (.csv)", type=["csv"], key="diabetes_csv")
        
        if uploaded_file is not None:
            patient_data = pd.read_csv(uploaded_file)
            st.dataframe(patient_data.head(3), use_container_width=True)
            
            if st.button("Run Lab Report Analysis", key="diabetes_csv_btn"):
                try:
                    scaled_data = models['diabetes_scaler'].transform(patient_data)
                    pred = models['diabetes_model'].predict(scaled_data)[0]
                    prob = models['diabetes_model'].predict_proba(scaled_data)[0][1]
                    display_prediction(pred, prob)
                except Exception as e:
                    st.error(f"Data format mismatch. Ensure your CSV matches the 8 required columns. Error: {e}")


# HEART DISEASE PREDICTION

elif disease == "❤️ Heart Disease Prediction":
    st.markdown("<h2 style='color: #111827;'>Heart Disease Risk Assessment</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; margin-bottom: 1rem;'>Select your preferred data input method below.</p>", unsafe_allow_html=True)

    input_tab1, input_tab2 = st.tabs(["✍️ Manual Entry", "📁 CSV Lab Upload"])

    with input_tab1:
        st.write("") 
        col1, col2, col3 = st.columns(3)
        with col1:
            male = st.selectbox("Sex", ["Female", "Male"])
            age = st.number_input("Age", min_value=1, max_value=120, value=55)
            education = st.selectbox("Education Level", ["1", "2", "3", "4"])
            cigsPerDay = st.number_input("Cigarettes Per Day", min_value=0, value=0)
            BPMeds = st.selectbox("On BP Medication?", ["No", "Yes"])
        with col2:
            prevalentStroke = st.selectbox("History of Stroke?", ["No", "Yes"])
            prevalentHyp = st.selectbox("Prevalent Hypertension?", ["No", "Yes"])
            diabetes = st.selectbox("Diabetes?", ["No", "Yes"])
            totChol = st.number_input("Total Cholesterol", min_value=0, value=240)
        with col3:
            sysBP = st.number_input("Systolic BP", min_value=0.0, value=130.0, step=0.1)
            diaBP = st.number_input("Diastolic BP", min_value=0.0, value=80.0, step=0.1)
            bmi = st.number_input("BMI", min_value=0.0, value=25.0, step=0.1)
            glucose = st.number_input("Glucose Level", min_value=0, value=85)

        if st.button("Predict Heart Disease Risk", key="heart_manual_btn"):
            try:
                male_val = 1 if male == "Male" else 0
                bp_meds_val = 1 if BPMeds == "Yes" else 0
                stroke_val = 1 if prevalentStroke == "Yes" else 0
                hyp_val = 1 if prevalentHyp == "Yes" else 0
                diab_val = 1 if diabetes == "Yes" else 0

                features = np.array([[male_val, age, int(education), cigsPerDay, bp_meds_val, stroke_val, hyp_val, diab_val, totChol, sysBP, diaBP, bmi, glucose]])
                
                scaled_features = models["heart_scaler"].transform(features)
                pred = models["heart_model"].predict(scaled_features)[0]
                prob = models["heart_model"].predict_proba(scaled_features)[0][1]
                display_prediction(pred, prob)
            except Exception as e:
                st.error(f"Prediction Error: Ensure inputs match model format. Details: {e}")

    with input_tab2:
        st.write("")
        st.info("Upload a patient's CSV file. Ensure it contains the 13 required clinical parameters.")
        
        st.markdown("""
        <style>
            [data-testid="stFileUploadDropzone"] {
                background-color: #1e293b !important; 
                border: 2px dashed #0d9488 !important;
                border-radius: 8px !important;
            }
            [data-testid="stFileUploadDropzone"] * {
                color: #ffffff !important; 
            }
            [data-testid="stFileUploader"] button {
                background-color: #0d9488 !important; 
                border: none !important;
                border-radius: 5px !important;
            }
            [data-testid="stFileUploader"] button * {
                color: #ffffff !important; 
                font-weight: bold !important;
            }
        </style>
        """, unsafe_allow_html=True)
        

        uploaded_file = st.file_uploader("Upload Patient Data (.csv)", type=["csv"], key="heart_csv")
        
        if uploaded_file is not None:
            patient_data = pd.read_csv(uploaded_file)
            st.dataframe(patient_data.head(3), use_container_width=True)
            
            if st.button("Run Lab Report Analysis", key="heart_csv_btn"):
                try:
                    scaled_data = models['heart_scaler'].transform(patient_data)
                    pred = models['heart_model'].predict(scaled_data)[0]
                    prob = models['heart_model'].predict_proba(scaled_data)[0][1]
                    display_prediction(pred, prob)
                except Exception as e:
                    st.error(f"Data format mismatch. Error: {e}")


# PARKINSON'S DISEASE PREDICTION

elif disease == "🧠 Parkinson's Prediction":
    st.markdown("<h2 style='color: #111827;'>Parkinson's Audio Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; margin-bottom: 2rem;'>Due to the high dimensionality of the audio data, manual entry is disabled. Please upload the patient's vocal biomarker CSV report.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="clinical-card">
            <h4>🎙️ Acoustic Biomarkers Analyzed</h4>
            <p>The AI detects micro-tremors in the patient's phonation using these features:</p>
            <ul>
                <li><b>Jitter & Shimmer:</b> Measures microscopic variations in vocal pitch and volume.</li>
                <li><b>NHR & HNR:</b> Evaluates the noise-to-harmonic ratio in the vocal tract.</li>
                <li><b>DFA (Detrended Fluctuation):</b> Detects stochastic changes in vocal patterns.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="clinical-card">
            <h4>🩺 Common Clinical Symptoms</h4>
            <p>Vocal degradation often precedes or accompanies these physical symptoms:</p>
            <ul>
                <li><b>Bradykinesia:</b> Slowness of movement and loss of spontaneous activity.</li>
                <li><b>Resting Tremor:</b> Often begins in a limb, typically the hand or fingers.</li>
                <li><b>Rigidity:</b> Muscle stiffness that can occur in any part of the body.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    

    st.markdown("""
    <style>
        /* 1. Give the dropzone a sleek dark background with the teal border */
        [data-testid="stFileUploadDropzone"] {
            background-color: #1e293b !important; 
            border: 2px dashed #0d9488 !important;
            border-radius: 8px !important;
        }
        /* 2. Force ALL text inside the dropzone to be bright white */
        [data-testid="stFileUploadDropzone"] * {
            color: #ffffff !important; 
        }
        /* 3. Keep the Browse Files Button solid Teal */
        [data-testid="stFileUploader"] button {
            background-color: #0d9488 !important; 
            border: none !important;
            border-radius: 5px !important;
        }
        /* 4. Force the text inside the button to be white */
        [data-testid="stFileUploader"] button * {
            color: #ffffff !important; 
            font-weight: bold !important;
        }
    </style>
    """, unsafe_allow_html=True)


    uploaded_file = st.file_uploader("Upload Patient Audio Data (.csv)", type=["csv"])
    
    if uploaded_file is not None:
        patient_data = pd.read_csv(uploaded_file)
        st.dataframe(patient_data.head(3), use_container_width=True)
        
        if st.button("Run Neurological Analysis"):
            try:
                scaled_patient_data = models['parkinsons_scaler'].transform(patient_data)
                pred = models['parkinsons_model'].predict(scaled_patient_data)[0]
                prob = models['parkinsons_model'].predict_proba(scaled_patient_data)[0][1]
                display_prediction(pred, prob)
            except Exception as e:
                st.error(f"Data format mismatch. Ensure the CSV columns match your training data exactly. Error: {e}")