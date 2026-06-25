"""
Mevcut modeli Türkçe çeviri verisiyle fine-tune eder
"""

import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report
from tqdm import tqdm

# ─────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────
BASE_MODEL  = os.getenv("BASE_MODEL", "models/sentiment_model")   # eğitilmiş modelimiz
TR_DATA     = os.getenv("TR_DATA", "data/turkish_translated.csv")
OUTPUT_DIR  = os.getenv("OUTPUT_DIR", "models/sentiment_model_v2")

EPOCHS      = 3
BATCH_SIZE  = 64
MAX_LEN     = 80
LR          = 1e-5   # fine-tuning için daha düşük LR

LABEL2ID = {"mutlu": 0, "uzgun": 1, "kizgin": 2, "saskin": 3, "notr": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

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
# MAIN
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Cihaz: {device}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# Veri yükle
print(f"\n📂 Türkçe veri yükleniyor: {TR_DATA}")
df = pd.read_csv(TR_DATA).dropna(subset=["clean_text", "label"])
df = df[df["label"].isin(LABEL2ID.keys())]
df["label_id"] = df["label"].map(LABEL2ID)
print(f"✅ {len(df):,} satır hazır")
print(df["label"].value_counts())

# Train/val split
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label"])
print(f"\nEğitim: {len(train_df):,}  |  Validasyon: {len(val_df):,}")

# Model yükle
print(f"\n⬇️  Model yükleniyor: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL).to(device)

# Dataset & DataLoader
train_ds = TweetDataset(train_df["clean_text"].tolist(), train_df["label_id"].tolist(), tokenizer)
val_ds   = TweetDataset(val_df["clean_text"].tolist(),   val_df["label_id"].tolist(),   tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

optimizer = AdamW(model.parameters(), lr=LR)

# ─────────────────────────────────────────────
# EĞİTİM
# ─────────────────────────────────────────────
print(f"\n🚀 Fine-tuning başlıyor — {EPOCHS} epoch\n")
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    # TRAIN
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=True)

    for batch in pbar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_train_loss = total_loss / len(train_loader)

    # VAL
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()

            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)

    print(f"\nEpoch {epoch} → Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(classification_report(all_labels, all_preds, target_names=list(LABEL2ID.keys())))

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"💾 Model kaydedildi → {OUTPUT_DIR}\n")

print("✅ Fine-tuning tamamlandı!")
print(f"📁 Model: {OUTPUT_DIR}")
print("\npredict.py'da MODEL_DIR'i şu şekilde güncelle:")
print(f'   MODEL_DIR = r"{OUTPUT_DIR}"')
