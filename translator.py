"""
Helsinki-NLP ile İngilizce tweet → Türkçe çeviri
GPU destekli, batch işleme
"""

import pandas as pd
import torch
import os
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

# ─────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────
INPUT_FILE  = os.getenv("INPUT_FILE", "data/merged_labeled.csv")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "data/turkish_translated.csv")
MODEL_NAME  = "Helsinki-NLP/opus-mt-tc-big-en-tr"
SAMPLE_SIZE = 50000
BATCH_SIZE  = 64   # GPU'ya göre ayarla, VRAM dolursa 32 yap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Cihaz: {device}")

# ─────────────────────────────────────────────
# MODELİ YÜKLE
# ─────────────────────────────────────────────
print(f"⬇️  Çeviri modeli yükleniyor: {MODEL_NAME}")
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME).to(device)
model.eval()
print("✅ Model hazır\n")

# ─────────────────────────────────────────────
# VERİ YÜKLE
# ─────────────────────────────────────────────
print(f"📂 Veri yükleniyor: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE).dropna(subset=["clean_text", "label"])

# Türkçe karakteri olmayanları al (İngilizce olanlar)
TR_CHARS = set("şğüöıçŞĞÜÖİÇ")
df_en = df[~df["clean_text"].apply(lambda x: any(c in TR_CHARS for c in str(x)))]

# Her etiketten dengeli örnekle
sample_per_label = SAMPLE_SIZE // df_en["label"].nunique()
df_sample = df_en.groupby("label", group_keys=False).apply(
    lambda x: x.sample(min(len(x), sample_per_label), random_state=42)
).reset_index(drop=True)

print(f"✅ {len(df_sample):,} satır seçildi")
print("Dağılım:\n", df_sample["label"].value_counts().to_string())

# ─────────────────────────────────────────────
# ÇEVİRİ FONKSİYONU
# ─────────────────────────────────────────────
def translate_batch(texts):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=80
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=80)

    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

# ─────────────────────────────────────────────
# BATCH ÇEVİRİ
# ─────────────────────────────────────────────
print(f"\n🔄 Çeviri başlıyor — {len(df_sample):,} satır, batch={BATCH_SIZE}\n")

texts = df_sample["clean_text"].tolist()
translated = []

for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Çevriliyor"):
    batch = texts[i:i+BATCH_SIZE]
    try:
        result = translate_batch(batch)
        translated.extend(result)
    except Exception as e:
        print(f"⚠️ Batch {i} hata: {e}")
        translated.extend(batch)  # hata olursa orijinali koy

# ─────────────────────────────────────────────
# KAYDET
# ─────────────────────────────────────────────
df_sample["clean_text"] = translated
df_sample.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print(f"\n✅ Kaydedildi → {OUTPUT_FILE}")
print(f"🎯 Toplam: {len(df_sample):,} Türkçe tweet")
print("\nÖrnek çeviriler:")
for _, row in df_sample.head(5).iterrows():
    print(f"  [{row['label']}] {row['clean_text'][:80]}")
