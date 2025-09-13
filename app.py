import os
import logging
from datetime import datetime

import joblib
import numpy as np
import streamlit as st


#  Logging básico

if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("App started")

 
#  Config Streamlit
 
st.set_page_config(page_title="Predictor de GPA (modelo .pkl)", layout="wide")
st.title("🎓 Predictor de GPA (usando linear_regression_model.pkl)")
st.caption("Este modelo **no utiliza** Gender ni Ethnicity (mitigación de sesgos).")

 
#  Mapeos y utilidades
 
# Orden de features EXACTO esperado por el modelo guardado
FEATURE_ORDER = [
    "Age",
    "ParentalEducation",
    "StudyTimeWeekly",
    "Absences",
    "Tutoring",
    "ParentalSupport",
    "Extracurricular",
    "Sports",
    "Music",
    "Volunteering",
]

# Etiquetas amigables para categorías numéricas
LABELS = {
    "ParentalEducation": {
        0: "None", 1: "High School", 2: "Some College", 3: "Bachelor's", 4: "Higher"
    },
    "Tutoring": {0: "No", 1: "Yes"},
    "ParentalSupport": {0: "None", 1: "Low", 2: "Moderate", 3: "High", 4: "Very High"},
    "Extracurricular": {0: "No", 1: "Yes"},
    "Sports": {0: "No", 1: "Yes"},
    "Music": {0: "No", 1: "Yes"},
    "Volunteering": {0: "No", 1: "Yes"},
}

# Inversos (label → código) para pasar números al modelo
INV_LABELS = {col: {v: k for k, v in mapping.items()} for col, mapping in LABELS.items()}

GRADE_LABELS = {
    0: "A (GPA ≥ 3.5)",
    1: "B (3.0 ≤ GPA < 3.5)",
    2: "C (2.5 ≤ GPA < 3.0)",
    3: "D (2.0 ≤ GPA < 2.5)",
    4: "F (GPA < 2.0)",
}
def gpa_to_gradeclass(gpa: float):
    if gpa >= 3.5:
        return 0, GRADE_LABELS[0]
    elif gpa >= 3.0:
        return 1, GRADE_LABELS[1]
    elif gpa >= 2.5:
        return 2, GRADE_LABELS[2]
    elif gpa >= 2.0:
        return 3, GRADE_LABELS[3]
    else:
        return 4, GRADE_LABELS[4]

 
#  Carga del modelo (.pkl)
 
@st.cache_resource
def load_model(path: str = "linear_regression_model.pkl"):
    try:
        model = joblib.load(path)
        logging.info("[+] Model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"[-] Error loading model: {str(e)}")
        raise e

model = None
try:
    model = load_model()
    st.success("✅ Modelo cargado correctamente (linear_regression_model.pkl)")
except Exception as e:
    st.error(f"❌ Error cargando el modelo: {str(e)}")
    st.stop()

 
#  Sidebar: Inputs
 
st.sidebar.header("✍️ Ingresar características del estudiante")

# Rangos sugeridos según la descripción del dataset
age = st.sidebar.number_input("Age", min_value=15, max_value=18, value=17, step=1)

pe_label = st.sidebar.selectbox(
    "ParentalEducation",
    options=list(LABELS["ParentalEducation"].values()),
    index=3  # "Bachelor's"
)
parental_education = INV_LABELS["ParentalEducation"][pe_label]

study_time = st.sidebar.number_input("StudyTimeWeekly (hours)", min_value=0.0, max_value=20.0, value=8.0, step=0.5)
absences = st.sidebar.number_input("Absences (days)", min_value=0, max_value=30, value=2, step=1)

tut_label = st.sidebar.selectbox("Tutoring", options=list(LABELS["Tutoring"].values()), index=0)
tutoring = INV_LABELS["Tutoring"][tut_label]

ps_label = st.sidebar.selectbox(
    "ParentalSupport",
    options=list(LABELS["ParentalSupport"].values()),
    index=2  # Moderate
)
parental_support = INV_LABELS["ParentalSupport"][ps_label]

ext_label = st.sidebar.selectbox("Extracurricular", options=list(LABELS["Extracurricular"].values()), index=1)
extracurricular = INV_LABELS["Extracurricular"][ext_label]

sports_label = st.sidebar.selectbox("Sports", options=list(LABELS["Sports"].values()), index=0)
sports = INV_LABELS["Sports"][sports_label]

music_label = st.sidebar.selectbox("Music", options=list(LABELS["Music"].values()), index=0)
music = INV_LABELS["Music"][music_label]

vol_label = st.sidebar.selectbox("Volunteering", options=list(LABELS["Volunteering"].values()), index=0)
volunteering = INV_LABELS["Volunteering"][vol_label]

# Empaquetar en el orden esperado por el modelo
def build_input_vector():
    values = {
        "Age": age,
        "ParentalEducation": parental_education,
        "StudyTimeWeekly": study_time,
        "Absences": absences,
        "Tutoring": tutoring,
        "ParentalSupport": parental_support,
        "Extracurricular": extracurricular,
        "Sports": sports,
        "Music": music,
        "Volunteering": volunteering,
    }
    return [values[col] for col in FEATURE_ORDER]

 
#  Acción: Predecir
 
col_left, col_right = st.columns([1, 1])
with col_left:
    if st.button("Predecir GPA", type="primary", use_container_width=True):
        try:
            x = np.array([build_input_vector()], dtype=float)
            t0 = datetime.now()
            gpa_pred = float(model.predict(x)[0])
            dt = (datetime.now() - t0).total_seconds()

            logger.info(f"Prediction made: GPA={gpa_pred:.3f} in {dt:.3f}s")

            # Mostrar resultados
            st.subheader("Resultado")
            st.metric("GPA Predicho", f"{gpa_pred:.3f}")

            cls_id, cls_lbl = gpa_to_gradeclass(gpa_pred)
            st.success(f"GradeClass: **{cls_id} → {cls_lbl}**")

            st.caption(
                "Escala: 0=A (≥3.5), 1=B [3.0–3.5), 2=C [2.5–3.0), 3=D [2.0–2.5), 4=F (<2.0)."
            )

        except Exception as e:
            logger.exception("Prediction error")
            st.error(f"❌ Error haciendo la predicción: {e}")

with col_right:
    st.info(
        "ℹ**Notas del modelo**\n\n"
        "- Entrenado con RidgeCV y guardado en `linear_regression_model.pkl`.\n"
        "- **No** usa `Gender` ni `Ethnicity` para reducir sesgos.\n"
        "- Asegúrate de mantener el orden de features original."
    )


#  Sidebar: Stats

st.sidebar.markdown("---")
st.sidebar.subheader("Estadísticas de la app")

def get_app_stats():
    try:
        with open('logs/app.log', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        preds = [ln for ln in lines if "Prediction made:" in ln]
        # La latencia la registramos por evento, aquí solo contamos
        return {"total_predictions": len(preds)}
    except Exception as e:
        logging.error(f"Error loading app stats: {str(e)}")
        return {"total_predictions": 0}

stats = get_app_stats()
st.sidebar.write(f"Total de predicciones: **{stats['total_predictions']}**")
