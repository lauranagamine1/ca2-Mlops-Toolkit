<h1 align="center">Predictor de Rendimiento Académico</h1>

<h3 align="center"><em>Grupo 1</em></h3>
<br>

- Laura Nagamine
- Eduardo Tello
- Mauricio Teran
- Adrian Sandoval


## Descripción

Implementamos un modelo de regresión lineal multivariado para predecir el GPA de los alumnos en base a datos de edad, educación de los padres, tiempo de estudio semanal, faltas a clases, acitivades extracurriculares, etc para identificar quienes requieren de apoyo académico.

## Modelos Utilizados

- **Regresión Lineal**: modelo más sencillo y altamente interpretable, permite identificar de manera directa el peso de cada variable en la predicción del GPA, en base a los campos restantes (menos ID).

- **Regresión Polinómica de grado 2**: Este modelo permite capturar relaciones no lineales entre las variables como un enfoque más variable que la regresión lineal lo que eleva complejiad y mejor nivel de interpretabilidad.

## Consideraciones importantes

- En `EDA.ipynb` consite en análisis univariado y bivariado, visualizaciones clave para detectar distribuciones atípicas, correlaciones y comportamiento de variables objetivo.

- La clasificación de `Grade Class` por `GPA` son congruentes en el dataset proporcionado, lo cual la elección del target Y es el ``GPA` ya que es uan variable de tipo decimal para predicciones precisas. La aplicación de Grade Class se incluye en la interfaz.

- En `training.ipynb` se hizo limpieza, tratamiento de faltantes, ingeniería básica de variables (codificación, normalización/estandarización), división de data y aplicación de los modelos.

## Aplicación Streamlit

Presentamos una aplicación en Streamlit útil y accesible con entradas manuales del usuario que ayude a predecir nuestro mejor modelo. Nuestra aplicación implementa un sistema de logging para garantizar la calidad y efectividad a nivel de desarollo de mejorar continuamente el modelo y la interfaz.

**URL**: https://ca2-gpa-predictor-group-1.streamlit.app/ 


## Entorno de ejecución

```sh
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Análisis de resultados

Los resultados salieron bastante parecidos con un R²=0.95, MSE=0.039 y MAE=0.156, lo cual, el modelo lineal es la mas precisa por la mayoría de la variación del GPA analizada por ingeniería. Sin embargo, antes de interpretar esto como una garantía absoluta de generalización, conviene tener en cuenta varios puntos importantes:

- Si el conjunto es pequeño o no representa bien la población objetivo (por ejemplo, viene de una sola institución), las métricas pueden sobrestimar el rendimiento real en despliegue.

- Las métricas similares entre lineal y polinómica apuntan a que la señal es mayormente lineal o bien que el polinómico no aporta complejidad útil.

- Entre las variables sociodemográficas conviene analizar desempeño por subgrupos (p. ej. por nivel socioeconómico) para evitar sesgos antes de usar el modelo para decisiones sensibles.

