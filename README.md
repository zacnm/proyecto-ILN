# Trabajo Final ILN - Detector de Sarcasmo

Desarrollado por Nil Farre Terron e Isaac Marsden.

Un proyecto que entrena modelos de predicción basados en BERT para detectar sarcasmo en frases en inglés, utilizando comentarios de Reddit etiquetados como sarcásticos o no.

---

Para ejecutar el código de entrenamiento, sigue estos pasos:

1. Descarga el archivo `train-balanced-sarcasm.csv` desde [aquí](https://www.kaggle.com/datasets/danofer/sarcasm?select=train-balanced-sarcasm.csv)

2. Extrae y copia el archivo dentro del directorio que contiene el código

3. Ejecuta el código con `python entrenamiento.py`

Una vez que el código se haya ejecutado y se haya guardado un modelo entrenado, puedes probarlo con el front-end de Streamlit, que se ejecuta con `streamlit run app.py`

También se incluye en este repositorio el archivo `V2.py`, que fue un experimento con la técnica de validación cruzada estratificada con 5 particiones. Sin embargo, resultó ser demasiado lento y no lo utilizamos; se incluye únicamente por completitud.