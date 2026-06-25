"""
XLM-RoBERTa Duygu Analizi - Eğitim Scripti
Etiketler: mutlu | uzgun | kizgin | saskin | notr
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import os

# ─────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────
INPUT_FILE   = os.getenv("INPUT_FILE", "data/merged_labeled.csv")
MODEL_DIR    = os.getenv("MODEL_DIR", "models/sentiment_model")
MODEL_NAME   = "xlm-roberta-base"

MAX_LEN      = 80
BATCH_SIZE   = 64        # 4060 için 32 rahat gider, sorun çıkarsa 16 yap
EPOCHS       = 3
LR           = 2e-5
TEST_SIZE    = 0.1
SEED         = 42

LABEL2ID = {"mutlu": 0, "uzgun": 1, "kizgin": 2, "saskin": 3, "notr": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Cihaz: {device}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx]
        }

# ─────────────────────────────────────────────
# VERİ YÜKLE
# ─────────────────────────────────────────────
print(f"\n📂 Veri yükleniyor: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE).dropna(subset=["clean_text", "label"])
df["label_id"] = df["label"].map(LABEL2ID)
df = df.dropna(subset=["label_id"])
df["label_id"] = df["label_id"].astype(int)

print(f"✅ {len(df):,} satır hazır")
print("Dağılım:\n", df["label"].value_counts().to_string())

train_df, val_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=SEED, stratify=df["label_id"]
)
print(f"\nEğitim: {len(train_df):,}  |  Validasyon: {len(val_df):,}")

# ─────────────────────────────────────────────
# TOKENİZER & MODEL
# ─────────────────────────────────────────────
print(f"\n⬇️  Model indiriliyor: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL2ID),
    id2label=ID2LABEL,
    label2id=LABEL2ID
).to(device)

train_dataset = TweetDataset(train_df["clean_text"].values, train_df["label_id"].values, tokenizer)
val_dataset   = TweetDataset(val_df["clean_text"].values,   val_df["label_id"].values,   tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ─────────────────────────────────────────────
# OPTİMİZER & SCHEDULER
# ─────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)

# ─────────────────────────────────────────────
# EĞİTİM DÖNGÜSÜ
# ─────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    return avg_loss, all_preds, all_labels

print(f"\n🚀 Eğitim başlıyor — {EPOCHS} epoch\n")
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for batch in loop:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_train = total_train_loss / len(train_loader)
    val_loss, preds, true_labels = evaluate(model, val_loader)

    print(f"\nEpoch {epoch} → Train Loss: {avg_train:.4f} | Val Loss: {val_loss:.4f}")
    print(classification_report(
        true_labels, preds,
        target_names=list(LABEL2ID.keys()),
        digits=3
    ))

    # En iyi modeli kaydet
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        print(f"💾 Model kaydedildi → {MODEL_DIR}\n")

print("✅ Eğitim tamamlandı!")
