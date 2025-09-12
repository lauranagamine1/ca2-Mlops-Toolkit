import logging
import streamlit as st
from datetime import datetime
import pickle
import joblib
import os

if not os.path.exists('logs'):
    os.makedirs('logs')

# -- Estructura base/inicial de logging y Streamlit --

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()  # También en consola
    ]
)
logger = logging.getLogger()
logger.info("App started")

@st.cache_resource
def load_model():
    try:
        model = joblib.load('linear_regression_model.pkl')
        logging.info("[+] Model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"[-] Error loading model: {str(e)}")
        raise e
    
st.title("Predictor de Rendimiento Académico")

model = None
try:
    model = load_model()
    if model is not None:
        st.success("✅ Modelo cargado correctamente")
    else:
        st.error("❌ Modelo no pudo ser cargado")
        st.stop()
except Exception as e:
    st.error(f"❌ Error cargando modelo: {str(e)}")
    st.stop()
