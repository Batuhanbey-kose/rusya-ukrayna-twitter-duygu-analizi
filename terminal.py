"""
Duygu Analizi - Tahmin Scripti
Eğitilmiş modeli yükler, tweet gir → duygu al
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# ─────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────
MODEL_DIR = os.getenv("MODEL_DIR", "models/sentiment_model_v2")
MAX_LEN   = 80

ID2LABEL = {0: "mutlu", 1: "uzgun", 2: "kizgin", 3: "saskin", 4: "notr"}

EMOJI = {
    "mutlu":  "😊",
    "uzgun":  "😢",
    "kizgin": "😡",
    "saskin": "😲",
    "notr":   "😐"
}

# ─────────────────────────────────────────────
# MODEL YÜKLE
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Cihaz: {device}")
print(f"⬇️  Model yükleniyor: {MODEL_DIR}\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

# ─────────────────────────────────────────────
# TAHMİN FONKSİYONU
# ─────────────────────────────────────────────
def predict(text: str) -> dict:
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits
        probs   = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_id    = probs.argmax()
    pred_label = ID2LABEL[pred_id]

    return {
        "etiket": pred_label,
        "emoji":  EMOJI[pred_label],
        "güven":  f"{probs[pred_id]*100:.1f}%",
        "tüm_skorlar": {ID2LABEL[i]: f"{p*100:.1f}%" for i, p in enumerate(probs)}
    }

# ─────────────────────────────────────────────
# İNTERAKTİF DÖNGÜ
# ─────────────────────────────────────────────
print("=" * 50)
print("   Duygu Analizi Hazır — çıkmak için 'q'")
print("=" * 50)

while True:
    tweet = input("\n📝 Tweet gir: ").strip()
    if tweet.lower() == "q":
        print("Çıkılıyor...")
        break
    if not tweet:
        continue

    result = predict(tweet)
    print(f"\n  Duygu  : {result['etiket']}  {result['emoji']}")
    print(f"  Güven  : {result['güven']}")
    print(f"  Tüm    : {result['tüm_skorlar']}")
