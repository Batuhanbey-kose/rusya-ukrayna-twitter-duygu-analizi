"""
Flask API — DuyguAI
Modeli yükler, /predict endpoint'i üzerinden tweet alır, duygu döndürür
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# ─────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────
MODEL_DIR = os.getenv("MODEL_DIR", "models/sentiment_model_v2")
MAX_LEN   = 80

ID2LABEL = {0: "mutlu", 1: "uzgun", 2: "kizgin", 3: "saskin", 4: "notr"}
EMOJI    = {"mutlu": "😊", "uzgun": "😢", "kizgin": "😡", "saskin": "😲", "notr": "😐"}

# 🔍 Model klasörü kontrol
if not os.path.exists(MODEL_DIR):
    raise ValueError(f"Model klasörü bulunamadı: {MODEL_DIR}")

print("⬇️ Model yükleniyor...")

# ─────────────────────────────────────────────
# MODEL YÜKLE
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True
).to(device)

model.eval()

print(f"✅ Model hazır — {device}")

# ─────────────────────────────────────────────
# FLASK
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Metin boş"}), 400

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
        probs   = torch.softmax(outputs.logits, dim=1).squeeze().cpu().numpy()

    pred_id    = int(probs.argmax())
    pred_label = ID2LABEL[pred_id]

    return jsonify({
        "etiket": pred_label,
        "emoji":  EMOJI[pred_label],
        "guven":  round(float(probs[pred_id]) * 100, 1),
        "skorlar": {ID2LABEL[i]: round(float(p) * 100, 1) for i, p in enumerate(probs)}
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    print("🚀 Sunucu başlatılıyor: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
