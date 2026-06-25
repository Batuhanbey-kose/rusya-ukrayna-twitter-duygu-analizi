# Rusya-Ukrayna Savaşı Twitter Duygu Analizi

Bu proje, Rusya-Ukrayna savaşıyla ilgili Twitter paylaşımlarının duygu sınıflandırmasını yapmak için geliştirilmiş bir Türkçe duygu analizi modelidir. Model, `xlm-roberta-base` mimarisi üzerine eğitilmiş/fine-tune edilmiştir.

## Duygu Sınıfları

- `mutlu`
- `uzgun`
- `kizgin`
- `saskin`
- `notr`

## Proje Dosyaları

- `api.py`: Flask API ile `/predict` endpoint'i üzerinden tahmin yapar.
- `terminal.py`: Terminalden metin girerek duygu tahmini yapar.
- `site.html`: API'ye bağlanan basit web arayüzü.
- `cmodel.py`: İlk model eğitim scripti.
- `fın.py`: Türkçe çeviri verisiyle fine-tuning scripti.
- `translator.py`: İngilizce tweetleri Türkçeye çevirme scripti.
- `etiketleyenc.py`: VADER + anahtar kelime tabanlı etiketleme scripti.

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS için sanal ortam aktivasyonu:

```bash
source .venv/bin/activate
```

## Model Dosyaları

Model ağırlıkları büyük olduğu için GitHub'a eklenmemelidir. Model dosyalarını Hugging Face Model Hub'a yüklemek daha uygundur.

Yerel çalıştırma için model klasörü varsayılan olarak şu konumda beklenir:

```text
models/sentiment_model_v2
```

Alternatif olarak farklı model yolu vermek için:

```bash
set MODEL_DIR=models/sentiment_model_v2
python api.py
```

PowerShell kullanıyorsanız:

```powershell
$env:MODEL_DIR="models/sentiment_model_v2"
python api.py
```

## API Kullanımı

API'yi başlatmak için:

```bash
python api.py
```

Sağlık kontrolü:

```bash
curl http://localhost:5000/health
```

Tahmin isteği örneği:

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"text\":\"Bugün savaşla ilgili çok üzücü haberler var.\"}"
```

## Terminalden Tahmin

```bash
python terminal.py
```

## Eğitim

Etiketlenmiş veriyle ilk eğitimi başlatmak için:

```bash
python cmodel.py
```

Türkçeye çevrilmiş veriyle fine-tuning yapmak için:

```bash
python fın.py
```

Scriptler varsayılan olarak `data/` klasöründeki veri dosyalarını kullanır. Gerekirse `INPUT_FILE`, `OUTPUT_FILE`, `BASE_MODEL`, `TR_DATA` ve `OUTPUT_DIR` ortam değişkenleriyle yollar değiştirilebilir.

## GitHub'a Ne Yüklenmeli?

GitHub'a yüklenmesi önerilenler:

- Kod dosyaları (`.py`, `.html`)
- `README.md`
- `requirements.txt`
- `.gitignore`
- Küçük örnek veri dosyası varsa anonimleştirilmiş örnek veri

GitHub'a yüklenmemesi önerilenler:

- Büyük model ağırlıkları (`model.safetensors`)
- Büyük ham veri dosyaları
- Twitter kullanıcı bilgisi veya kişisel veri içeren dosyalar
- API anahtarı, token veya gizli bilgiler

## Hugging Face'e Ne Yüklenmeli?

Hugging Face Model Hub'a yüklenmesi önerilen model klasörü:

```text
models/sentiment_model_v2/
```

Bu klasörde genelde şu dosyalar bulunur:

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `model.safetensors`

## Not

Veri setleri Twitter/X içerikleri barındırıyorsa paylaşmadan önce veri setinin lisansı, kaynak şartları ve kişisel veri içeriği kontrol edilmelidir. TÜBİTAK proje sunumu için ham veri yerine veri kaynağı, işleme adımları ve anonimleştirilmiş küçük örnek paylaşmak daha güvenlidir.
