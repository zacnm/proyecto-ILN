import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ---------- Preprocesamiento ----------
def clean_text(text):
    text = text.lower()                              # pasar a minúsculas
    text = re.sub(r"http\S+|www.\S+", "", text)      # quitar URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)          # quitar puntuación y símbolos
    text = re.sub(r"\s+", " ", text).strip()         # quitar espacios múltiples
    return text

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stop_words])

# ---------- Cargar y muestrear datos ----------
df = pd.read_csv("train-balanced-sarcasm.csv")
df = df[['comment', 'label']].dropna()

# Muestreo equilibrado
df_pos = df[df['label'] == 1].sample(5000, random_state=42)
df_neg = df[df['label'] == 0].sample(5000, random_state=42)
df_balanced = pd.concat([df_pos, df_neg]).reset_index(drop=True)

# Tres versiones del texto
df_balanced['text_raw'] = df_balanced['comment']
df_balanced['text_clean'] = df_balanced['comment'].apply(clean_text)
df_balanced['text_clean_nostop'] = df_balanced['text_clean'].apply(remove_stopwords)

# ---------- Cargar y preparar datos completos (con todo el data set) ----------
# df = pd.read_csv("train-balanced-sarcasm.csv")
# df = df[['comment', 'label']].dropna()

# # Filtrar filas donde la columna 'comment' no es una string válida
# df = df[df['comment'].apply(lambda x: isinstance(x, str))]

# # Usar todos los datos etiquetados
# df_balanced = df.copy().reset_index(drop=True)

# # Tres versiones del texto
# df_balanced['text_raw'] = df_balanced['comment']
# df_balanced['text_clean'] = df_balanced['comment'].apply(clean_text)
# df_balanced['text_clean_nostop'] = df_balanced['text_clean'].apply(remove_stopwords)

# ---------- Función de entrenamiento y evaluación ----------
def train_and_evaluate(X_text, y, description):
    print(f"\n------ {description} ------")
    X_train, X_val, y_train, y_val = train_test_split(
        X_text, y, test_size=0.2, stratify=y, random_state=42)

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    model = LinearSVC()
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_val_tfidf)

    acc = accuracy_score(y_val, y_pred)
    print("Accuracy:", round(acc * 100, 2), "%")
    print("Classification report:\n", classification_report(y_val, y_pred))

# ---------- Ejecutar pruebas ----------
y = df_balanced['label']
train_and_evaluate(df_balanced['text_raw'], y, "1. Sin preprocesamiento")
train_and_evaluate(df_balanced['text_clean'], y, "2. Limpieza básica")
train_and_evaluate(df_balanced['text_clean_nostop'], y, "3. Limpieza + stopwords eliminadas")