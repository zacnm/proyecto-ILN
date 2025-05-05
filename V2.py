import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import numpy as np
from transformers import EarlyStoppingCallback
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
from optuna import Trial, create_study

# 1. Dataset personalizado para PyTorch
class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
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
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

# 2. Lectura de datos y muestreo balanceado
df = pd.read_csv("./train-balanced-sarcasm.csv")[['comment','label']].dropna()
df_pos = df[df.label==1].sample(4999, random_state=42)
df_neg = df[df.label==0].sample(4999, random_state=42)
df = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
print("Clases balanceadas:", df.label.value_counts().to_dict())

# 3. Validación cruzada estratificada
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 4. Tokenizer y dispositivo
tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-12_H-768_A-12")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {device}")

# Métricas y threshold tuning
def tune_threshold(logits, labels):
    probs = torch.softmax(torch.tensor(logits), dim=1)[:,1].numpy()
    best_f1, best_thr = 0, 0.5
    for thr in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1

# Función para inicializar modelo fresco
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        "google/bert_uncased_L-12_H-768_A-12", num_labels=2)

all_metrics = []
# 5. Entrenamiento con scheduler, warmup y threshold tuning por fold
for fold, (train_idx, val_idx) in enumerate(skf.split(df.comment, df.label), 1):
    print(f"\n-- Fold {fold}/{n_splits}")
    train_ds = SarcasmDataset(df.comment.iloc[train_idx], df.label.iloc[train_idx], tokenizer)
    val_ds   = SarcasmDataset(df.comment.iloc[val_idx],   df.label.iloc[val_idx],   tokenizer)

    args = TrainingArguments(
        output_dir=f"./results/f{fold}",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,    # batch efectivo 16
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_strategy="epoch",
        learning_rate=2.84e-5,            # Optuna
        weight_decay=0.078,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda p: {
            'precision': precision_recall_fscore_support(p.label_ids, np.argmax(p.predictions, axis=1), average='binary', zero_division=0)[0],
            'recall': precision_recall_fscore_support(p.label_ids, np.argmax(p.predictions, axis=1), average='binary', zero_division=0)[1],
            'f1': precision_recall_fscore_support(p.label_ids, np.argmax(p.predictions, axis=1), average='binary', zero_division=0)[2],
            'accuracy': accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))
        },
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    

    trainer.train()
    preds_output = trainer.predict(val_ds)
    best_thr, best_f1 = tune_threshold(preds_output.predictions, preds_output.label_ids)
    print(f"Mejor umbral: {best_thr:.2f} -> F1: {best_f1:.4f}")

    probs = torch.softmax(torch.tensor(preds_output.predictions), dim=1)[:,1].numpy()
    final_preds = (probs >= best_thr).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(preds_output.label_ids, final_preds, average='binary')
    acc = accuracy_score(preds_output.label_ids, final_preds)
    print(f"Acc={acc:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")
    all_metrics.append({'acc':acc,'prec':precision,'rec':recall,'f1':f1})

# 6. Búsqueda de hiperparámetros con Optuna (primer fold)
study = create_study(direction="maximize")
# Obtener los índices del primer fold una vez
def get_first_fold_indices():
    splits = list(skf.split(df.comment, df.label))
    return splits[0]
train_ids, val_ids = get_first_fold_indices()

def hp_objective(trial: Trial):
    lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    wd = trial.suggest_float("wd", 0.0, 0.3)
    bs = trial.suggest_categorical("batch_size", [8,16])
    args = TrainingArguments(
        output_dir="./hp_search",
        num_train_epochs=2,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=lr,
        weight_decay=wd,
        warmup_steps=200,
        lr_scheduler_type="cosine",
        report_to="none",
        fp16=torch.cuda.is_available()
    )
    # Trainer for HP search con compute_metrics que devuelve dict
    trainer_hp = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=SarcasmDataset(df.comment.iloc[train_ids], df.label.iloc[train_ids], tokenizer),
        eval_dataset=SarcasmDataset(df.comment.iloc[val_ids],   df.label.iloc[val_ids],   tokenizer),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda p: {
            'accuracy': accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1)),
            'precision': precision_recall_fscore_support(p.label_ids, np.argmax(p.predictions, axis=1), average='binary', zero_division=0)[0],
            'recall': precision_recall_fscore_support(p.label_ids, np.argmax(p.predictions, axis=1), average='binary', zero_division=0)[1],
            'f1': precision_recall_fscore_support(p.label_ids, np.argmax(p.predictions, axis=1), average='binary', zero_division=0)[2]
        }
    )
    trainer_hp.train()
    res = trainer_hp.evaluate()
    return res['eval_f1']

# Iniciar optimización
study.optimize(hp_objective, n_trials=5)
print("Mejores hiperparámetros:", study.best_params)

# 7. Resumen global
def summarize(mets, key): return np.mean([m[key] for m in mets]), np.std([m[key] for m in mets])
print("\nResumen cross-val:")
for k in ['acc','prec','rec','f1']:
    mean, std = summarize(all_metrics, k)
    print(f"{k}: {mean:.4f} ± {std:.4f}")
