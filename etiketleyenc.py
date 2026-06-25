"""
Twitter Duygu Etiketleme Scripti
Etiketler: mutlu | uzgun | kizgin | saskin | notr
Yöntem: VADER sentiment + anahtar kelime tabanlı hibrit
"""

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
import re
import os

# --- İlk çalıştırmada VADER sözlüğünü indir ---
nltk.download("vader_lexicon", quiet=True)

# ─────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────
INPUT_FILE  = os.getenv("INPUT_FILE", "data/merged_cleaned_dataset.csv")       # giriş dosyası
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "data/merged_labeled.csv")    # çıkış dosyası
TEXT_COL    = "clean_text"     # metin sütunu adı

# ─────────────────────────────────────────────
# ANAHTAR KELİME LİSTELERİ  (genişletebilirsin)
# ─────────────────────────────────────────────
KEYWORDS = {
    "mutlu": [
        "victory", "won", "liberated", "celebrate", "success", "great news",
        "freed", "saved", "hero", "proud", "amazing", "excellent", "happy",
        "congrats", "winning", "breakthrough", "cheer", "good news", "survived",
        "rescued", "triumph", "praise"
    ],
    "uzgun": [
        "killed", "dead", "died", "death", "casualties", "loss", "tragic",
        "mourning", "grief", "sorrow", "crying", "tears", "heartbreak",
        "victims", "civilians died", "massacre", "funeral", "RIP", "devastated",
        "suffering", "pain", "innocent", "children killed", "families"
    ],
    "kizgin": [
        "damn", "war crime", "bastard", "disgusting", "outrage", "furious",
        "unacceptable", "condemn", "shame", "disgrace", "fuck", "shit",
        "evil", "monster", "criminal", "brutal", "atrocity", "invasion",
        "aggression", "terrorist", "murder", "genocide", "boycott", "sanctions"
    ],
    "saskin": [
        "breaking", "shocking", "unbelievable", "suddenly", "unexpected",
        "wow", "wtf", "omg", "what", "really", "seriously", "just happened",
        "can't believe", "no way", "incredible", "stunning", "explosion",
        "reportedly", "confirmed", "just in", "developing", "alert"
    ],
}

# ─────────────────────────────────────────────
# ETİKETLEME FONKSİYONU
# ─────────────────────────────────────────────
sia = SentimentIntensityAnalyzer()

def keyword_hit(text: str) -> dict:
    """Her etiket için kaç anahtar kelime eşleşti sayar."""
    text_lower = text.lower()
    return {
        label: sum(1 for kw in words if kw in text_lower)
        for label, words in KEYWORDS.items()
    }

def etiketle(text: str) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return "notr"

    # 1) VADER skoru
    scores = sia.polarity_scores(text)
    compound = scores["compound"]   # -1 (çok negatif) → +1 (çok pozitif)
    neg      = scores["neg"]
    pos      = scores["pos"]

    # 2) Anahtar kelime sayıları
    hits = keyword_hit(text)
    max_hit_label = max(hits, key=hits.get)
    max_hit_count = hits[max_hit_label]

    # 3) Karar ağacı (hibrit)

    # Anahtar kelime baskınsa → öncelik ver
    if max_hit_count >= 2:
        return max_hit_label
    
    if max_hit_count == 1:
        # VADER ile çelişmiyorsa kabul et
        if max_hit_label == "mutlu"  and compound > 0.05:
            return "mutlu"
        if max_hit_label == "uzgun"  and compound < -0.05:
            return "uzgun"
        if max_hit_label == "kizgin" and neg > 0.15:
            return "kizgin"
        if max_hit_label == "saskin":
            return "saskin"   # şaşkınlık için VADER zayıf, kelimeye güven

    # Sadece VADER skoru ile karar ver
    if compound >= 0.35:
        return "mutlu"
    if compound <= -0.35:
        # Yüksek negatif → kızgın mı üzgün mü?
        # Kızgın: olumsuz kelimeler + düşük üzüntü ifadesi
        if neg > 0.25 and hits["uzgun"] == 0:
            return "kizgin"
        return "uzgun"
    if -0.35 < compound < -0.05:
        return "uzgun"
    if 0.05 < compound < 0.35:
        return "mutlu"

    return "notr"

# ─────────────────────────────────────────────
# ANA AKIŞ
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"📂 Dosya okunuyor: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    print(f"✅ {len(df):,} satır yüklendi. Etiketleme başlıyor...\n")

    tqdm.pandas(desc="Etiketleniyor")
    df["label"] = df[TEXT_COL].progress_apply(etiketle)

    # Dağılımı göster
    print("\n📊 Etiket Dağılımı:")
    dist = df["label"].value_counts()
    for label, count in dist.items():
        bar = "█" * (count * 30 // len(df))
        print(f"  {label:<10} {count:>6} ({count/len(df)*100:.1f}%)  {bar}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Kaydedildi → {OUTPUT_FILE}")
    print("Sütunlar:", list(df.columns))
