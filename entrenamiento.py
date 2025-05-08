import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import numpy as np
from datetime import datetime
from transformers import EarlyStoppingCallback
from transformers import Trainer, TrainingArguments
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Mostrar dispositivo (GPU o CPU) y mover el modelo si es posible
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Usando dispositivo (GPU o CPU):", device)

# Definir dataset
class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Quitar la dimensión adicional de batch
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

# Cargar el dataset
df = pd.read_csv("./train-balanced-sarcasm.csv")[['comment','label']].dropna()
df_pos = df[df.label==1].sample(5000, random_state=42)
df_neg = df[df.label==0].sample(5000, random_state=42)
df = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
print("Clases balanceadas:", df.label.value_counts().to_dict())
df = df[['comment', 'label']].dropna()

# Dividir el dataset en entrenamiento y validación
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['comment'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# Inicializar el tokenizador y crear los datasets
tokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-12_H-768_A-12")
train_dataset = SarcasmDataset(train_texts, train_labels, tokenizer, max_length=128)
val_dataset = SarcasmDataset(val_texts, val_labels, tokenizer, max_length=128)

# Cargar el modelo preentrenado BERT para clasificación
model = BertForSequenceClassification.from_pretrained("google/bert_uncased_L-12_H-768_A-12", num_labels=2)
model.to(device)

# Definir una función para la búsqueda de hiperparámetros
def model_init():
    return BertForSequenceClassification.from_pretrained(
        "google/bert_uncased_L-12_H-768_A-12", num_labels=2
    )

# Configurar los argumentos básicos de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    report_to="tensorboard",
    fp16=True,
    seed=42
)

# Función para calcular métricas
def compute_metrics(eval_pred):
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# Data collator para padding dinámico
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Crear el objeto Trainer con model_init para búsqueda
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 5e-5),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-4, 0.1),
    }

# Realizar la búsqueda de hiperparámetros
best_run = trainer.hyperparameter_search(
    direction="maximize",  # Maximizar la métrica (f1...)
    backend="optuna",      # Usar Optuna como backend
    n_trials=10,           # Número de pruebas 10 ya que sino dura mucho
    hp_space=hp_space
)

print("Mejores hiperparámetros:", best_run.hyperparameters)

# Actualizar los argumentos del Trainer con los mejores hiperparámetros
for key, value in best_run.hyperparameters.items():
    setattr(trainer.args, key, value)

# Entrenar el modelo
trainer.train()

# Evaluar el modelo
eval_results = trainer.evaluate()
print("Resultados de evaluación:", eval_results)

# Obtener predicciones en el conjunto de validación
predictions = trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=1)

# Generar la matriz de confusión
cm = confusion_matrix(val_labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Sarcasmo", "Sarcasmo"])

# Mostrar la matriz de confusión
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión")
plt.show()

# Obtener la fecha y hora actual
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_dir = f"./modelo_entrenado_{timestamp}"

# Guardar el modelo entrenado
trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)

print(f"Modelo guardado en: {model_dir}")