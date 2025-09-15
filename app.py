import os
import logging
from datetime import datetime

import joblib
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# =========================
# Logging
# =========================
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/app.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.info("App started")

# =========================
# Config + Estilos (fondo oscuro)
# =========================
st.set_page_config(page_title="Predictor de GPA (.pkl)", page_icon="🎓", layout="wide")

st.markdown("""
<style>
/* Fondo negro y texto claro */
.stApp, .block-container, body {
  background: #0b0f1a !important;  /* casi negro azul */
  color: #e5e7eb !important;        /* gris claro */
}

/* Banner (gradiente sobre dark) */
.hero {
  padding: 1.1rem 1.4rem;
  background: linear-gradient(90deg, #A7B7A7, #CCAAAA);
  color: white;
  border-radius: 14px;
  margin-bottom: 14px;
}

/* Tarjetas dark */
.card {
  border-radius: 14px;
  padding: 1rem 1.2rem;
  background: #111827;          /* gris-azul oscuro */
  border: 1px solid #1f2937;    /* borde sutil */
  box-shadow: 0 8px 22px rgba(0,0,0,0.35);
  color: #e5e7eb;
}

/* Badges adaptadas a dark */
.badge {
  display: inline-block; padding: .25rem .6rem; border-radius: 999px;
  font-size: 0.80rem; margin-right: .35rem;
  background: #0f172a; color: #93c5fd; border: 1px solid #1e3a8a;
}
.badge-crit { background:#7f1d1d; color:#fecaca; border:none; }
.badge-high { background:#7c2d12; color:#fde68a; border:none; }
.badge-med  { background:#064e3b; color:#bbf7d0; border:none; }
.badge-low  { background:#111827; color:#e5e7eb; border:1px solid #334155; }

/* Texto gris sutil */
.muted { color: #94a3b8; font-size: 0.92rem; }

/* Botón primario ancho */
.stButton>button[kind="primary"] {
  width: 100%;
  border-radius: 10px;
}

/* Métricas: texto claro */
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {
  color: #e5e7eb !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h2 style="margin:0;">Predictor de GPA (modelo .pkl)</h2>
  <p style="margin:.25rem 0 0 0;">Predice el GPA y recibe <b>recomendaciones accionables</b> para mejorar. Sin usar Gender/Ethnicity.</p>
</div>
""", unsafe_allow_html=True)

# =========================
# Mapeos y utilidades
# =========================
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

LABELS = {
    "ParentalEducation": {0: "None", 1: "High School", 2: "Some College", 3: "Bachelor's", 4: "Higher"},
    "Tutoring": {0: "No", 1: "Yes"},
    "ParentalSupport": {0: "None", 1: "Low", 2: "Moderate", 3: "High", 4: "Very High"},
    "Extracurricular": {0: "No", 1: "Yes"},
    "Sports": {0: "No", 1: "Yes"},
    "Music": {0: "No", 1: "Yes"},
    "Volunteering": {0: "No", 1: "Yes"},
}
INV_LABELS = {col: {v: k for k, v in mapping.items()} for col, mapping in LABELS.items()}

GRADE_LABELS = {
    0: "A (GPA ≥ 3.5)",
    1: "B (3.0 ≤ GPA < 3.5)",
    2: "C (2.5 ≤ GPA < 3.0)",
    3: "D (2.0 ≤ GPA < 2.5)",
    4: "F (GPA < 2.0)",
}

def gpa_to_gradeclass(gpa: float):
    if gpa >= 3.5: return 0, GRADE_LABELS[0]
    if gpa >= 3.0: return 1, GRADE_LABELS[1]
    if gpa >= 2.5: return 2, GRADE_LABELS[2]
    if gpa >= 2.0: return 3, GRADE_LABELS[3]
    return 4, GRADE_LABELS[4]

def gauge_gpa(gpa_value: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gpa_value,
        number={'suffix': " GPA", 'font': {'size': 28, 'color': '#e5e7eb'}},
        gauge={
            'axis': {'range': [0, 4], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
            'bar': {'color': "#22c55e"},
            'steps': [
                {'range': [0.0, 2.0],  'color': '#450a0a'},
                {'range': [2.0, 2.5],  'color': '#7c2d12'},
                {'range': [2.5, 3.0],  'color': '#373d20'},
                {'range': [3.0, 3.5],  'color': '#14532d'},
                {'range': [3.5, 4.0],  'color': '#064e3b'},
            ],
            'threshold': {'line': {'color': "#e5e7eb", 'width': 3}, 'thickness': 0.75, 'value': gpa_value},
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='#0b0f1a',
        plot_bgcolor='#0b0f1a',
        font={'color': '#e5e7eb'}
    )
    return fig

def make_recommendations(vals: dict, gpa_pred: float):
    recs = []

    # 1) Estudio semanal
    target_study = 10.0
    if vals["StudyTimeWeekly"] < target_study:
        gap = round(target_study - vals["StudyTimeWeekly"], 1)
        recs.append({
            "title": "Incrementa el tiempo de estudio 📚",
            "why": f"Estudias {vals['StudyTimeWeekly']} h/sem; objetivo sugerido: {target_study} h/sem.",
            "action": f"Añade {gap} h/sem en 5 días (+{round(gap/5,1)} h/día) con Pomodoro (25–30 min).",
            "priority": "Alta" if gpa_pred < 3.2 else "Media"
        })

    # 2) Ausencias
    if vals["Absences"] > 5:
        recs.append({
            "title": "Reduce ausencias 🗓️",
            "why": f"{int(vals['Absences'])} ausencias pueden afectar el GPA.",
            "action": "Planifica alarmas, usa recordatorios y comunica imprevistos con antelación.",
            "priority": "Alta" if gpa_pred < 3.2 else "Media"
        })

    # 3) Tutorías
    if vals["Tutoring"] == 0 and gpa_pred < 3.2:
        recs.append({
            "title": "Activa tutorías 1:1 🎯",
            "why": "Apoyo personalizado acelera mejoras en cursos desafiantes.",
            "action": "Agenda 1 sesión semanal en la asignatura con menor rendimiento.",
            "priority": "Alta"
        })

    # 4) Apoyo parental
    if vals["ParentalSupport"] <= 1:
        lvl = "None" if vals["ParentalSupport"] == 0 else "Low"
        recs.append({
            "title": "Coordina apoyo en casa 🤝",
            "why": f"ParentalSupport: {lvl}. Rutinas compartidas mejoran hábitos.",
            "action": "Acuerden horarios de estudio fijos y un check semanal de progreso.",
            "priority": "Media"
        })

    # 5) Actividades co/extra-curriculares
    if vals["Extracurricular"] == 0:
        recs.append({
            "title": "Únete a una actividad extracurricular 🎭",
            "why": "La estructura y el sentido de pertenencia mejoran la motivación.",
            "action": "Inscríbete en 1 club/círculo del campus este mes.",
            "priority": "Media"
        })
    if vals["Sports"] == 0:
        recs.append({
            "title": "Suma actividad física 🏃",
            "why": "El ejercicio favorece concentración y memoria.",
            "action": "3 sesiones/sem de 30–45 min (equipo, gym o correr).",
            "priority": "Media"
        })
    if vals["Music"] == 0:
        recs.append({
            "title": "Canaliza estrés con música 🎶",
            "why": "La práctica musical apoya la regulación emocional.",
            "action": "2 bloques/sem de 30 min (ensayo, coro o instrumento).",
            "priority": "Baja"
        })
    if vals["Volunteering"] == 0:
        recs.append({
            "title": "Haz voluntariados 💚",
            "why": "Desarrolla habilidades socioemocionales y propósito.",
            "action": "Participa 2 h/sem en un programa universitario o local.",
            "priority": "Media"
        })

    # 6) Plan intensivo si está en D/F
    cls_id, _ = gpa_to_gradeclass(gpa_pred)
    if cls_id >= 3:
        recs.insert(0, {
            "title": "Plan intensivo de recuperación 🚀",
            "why": "Predicción en zona de riesgo (D/F).",
            "action": "Bloques diarios dirigidos, tutoría semanal y seguimiento con coordinación académica.",
            "priority": "Crítica"
        })

    # Orden por prioridad visual
    order = {"Crítica": 0, "Alta": 1, "Media": 2, "Baja": 3}
    recs.sort(key=lambda r: order.get(r["priority"], 9))
    return recs

# =========================
# Cargar modelo (.pkl)
# =========================
@st.cache_resource
def load_model(path: str = "linear_regression_model.pkl"):
    model = joblib.load(path)
    logging.info("[+] Model loaded successfully")
    return model

try:
    model = load_model()
    st.success("✅ Modelo cargado correctamente (linear_regression_model.pkl)")
except Exception as e:
    st.error(f"❌ Error cargando el modelo: {e}")
    st.stop()

# =========================
# Sidebar: Inputs
# =========================
st.sidebar.header("✍️ Ingresar características del estudiante")

age = st.sidebar.number_input("Age", min_value=15, max_value=18, value=17, step=1)

pe_label = st.sidebar.selectbox(
    "ParentalEducation",
    options=list(LABELS["ParentalEducation"].values()),
    index=3
)
parental_education = INV_LABELS["ParentalEducation"][pe_label]

study_time = st.sidebar.number_input("StudyTimeWeekly (hours)", min_value=0.0, max_value=20.0, value=8.0, step=0.5)
absences = st.sidebar.number_input("Absences (days)", min_value=0, max_value=30, value=2, step=1)

tut_label = st.sidebar.selectbox("Tutoring", options=list(LABELS["Tutoring"].values()), index=0)
tutoring = INV_LABELS["Tutoring"][tut_label]

ps_label = st.sidebar.selectbox("ParentalSupport", options=list(LABELS["ParentalSupport"].values()), index=2)
parental_support = INV_LABELS["ParentalSupport"][ps_label]

ext_label = st.sidebar.selectbox("Extracurricular", options=list(LABELS["Extracurricular"].values()), index=1)
extracurricular = INV_LABELS["Extracurricular"][ext_label]

sports_label = st.sidebar.selectbox("Sports", options=list(LABELS["Sports"].values()), index=0)
sports = INV_LABELS["Sports"][sports_label]

music_label = st.sidebar.selectbox("Music", options=list(LABELS["Music"].values()), index=0)
music = INV_LABELS["Music"][music_label]

vol_label = st.sidebar.selectbox("Volunteering", options=list(LABELS["Volunteering"].values()), index=0)
volunteering = INV_LABELS["Volunteering"][vol_label]

def build_input_vector():
    vals = {
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
    return [vals[col] for col in FEATURE_ORDER], vals

# =========================
# Acción: Predecir
# =========================
left, right = st.columns([1, 1])

with left:
    if st.button("Predecir GPA", type="primary", use_container_width=True):
        try:
            x_vec, vals = build_input_vector()
            x = np.array([x_vec], dtype=float)
            t0 = datetime.now()
            gpa_pred = float(model.predict(x)[0])
            latency = (datetime.now() - t0).total_seconds()
            logger.info(f"Prediction made: GPA={gpa_pred:.3f} in {latency:.3f}s")

            # Tarjeta de resultado + gauge
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Resultado")
            st.plotly_chart(gauge_gpa(gpa_pred), use_container_width=True)
            cls_id, cls_lbl = gpa_to_gradeclass(gpa_pred)
            st.metric("GPA predicho", f"{gpa_pred:.3f}")
            st.caption(f"GradeClass: **{cls_id} → {cls_lbl}**")
            st.markdown("</div>", unsafe_allow_html=True)

            # Mensaje motivacional
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            if cls_id <= 2:
                st.success("💪 ¡Vas por buen camino! Mantén tus hábitos y busca pequeños incrementos.")
            else:
                st.warning("🌱 Estás a tiempo de mejorar. Aquí tienes un plan concreto y realista.")
            st.markdown("</div>", unsafe_allow_html=True)

            # Recomendaciones
            recs = make_recommendations(vals, gpa_pred)
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🧭 Recomendaciones accionables")
            if not recs:
                st.write("✅ ¡No hay recomendaciones críticas ahora mismo! Mantén tu estrategia actual.")
            else:
                for r in recs:
                    badge = "badge-crit" if r["priority"]=="Crítica" else ("badge-high" if r["priority"]=="Alta" else ("badge-med" if r["priority"]=="Media" else "badge-low"))
                    st.markdown(
                        f"**{r['title']}**  <span class='badge {badge}'>{r['priority']}</span><br>"
                        f"<span class='muted'>Por qué:</span> {r['why']}<br>"
                        f"<span class='muted'>Acción:</span> {r['action']}",
                        unsafe_allow_html=True
                    )
                    st.markdown("---")
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            logger.exception("Prediction error")
            st.error(f"❌ Error haciendo la predicción: {e}")

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("ℹ️ Notas")
    st.write(
        "- Modelo cargado desde `linear_regression_model.pkl` (RidgeCV en entrenamiento).\n"
        "- No se usan `Gender` ni `Ethnicity` (mitigación de sesgos).\n"
        "- Las recomendaciones son orientativas; se sugiere seguimiento académico semanal."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Estadísticas
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Estadísticas de la aplicación")
    try:
        with open('logs/app.log', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        preds = [ln for ln in lines if "Prediction made:" in ln]
        st.write(f"Total de predicciones: **{len(preds)}**")
    except Exception as e:
        logging.error(f"Error loading app stats: {e}")
        st.write("Total de predicciones: **0**")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer breve
st.caption("Escala de clases: 0=A (≥3.5), 1=B [3.0–3.5), 2=C [2.5–3.0), 3=D [2.0–2.5), 4=F (<2.0).")
