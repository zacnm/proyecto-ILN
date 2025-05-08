import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

# Obtener el directorio del modelo más reciente
def get_latest_model_dir(base_dir="./"):
    model_dirs = [d for d in os.listdir(base_dir) if d.startswith("modelo_entrenado_")]
    if not model_dirs:
        raise FileNotFoundError("No se encontraron directorios de modelos entrenados.")
    latest_model_dir = max(model_dirs, key=lambda d: d.split("_")[-1])
    return os.path.join(base_dir, latest_model_dir)

# Mostrar una barra de carga mientras se carga el modelo
with st.spinner("Cargando el modelo..."):
    try:
        model_dir = get_latest_model_dir()
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        model = BertForSequenceClassification.from_pretrained(model_dir)
        model.eval()
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.stop()

st.success("¡Modelo cargado exitosamente!")

# Configurar la aplicación de Streamlit
st.title("Detección de sarcasmo")
st.write("Escribe un texto a continuación para verificar si es sarcástico o no.")

# Entrada de texto para el usuario
user_input = st.text_input("Escribe tu texto aquí:")

if user_input:
    # Tokenizar y preparar la entrada
    inputs = tokenizer(
        user_input,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    
    # Perform prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
        confidence = int(probabilities[0, prediction].item() * 100)
    
    # Display the result
    if prediction == 1:
        st.write(f"Predicción: Sarcástico 😏 (Confianza: {confidence}%)")
    else:
        st.write(f"Predicción: No sarcástico 😊 (Confianza: {confidence}%)")
