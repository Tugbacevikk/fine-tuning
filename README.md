# Ses Analizi Sistemi (Speech-to-Text & Speaker Diarization)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)

Yüksek performanslı, modüler, **Clean Architecture / Code-First** prensiplerine uygun olarak tasarlanmış ses analiz sistemi.

## Özellikler
- **Speech-to-Text (STT)**: Faster-Whisper ile zaman damgalı metne dönüştürme (`tiny`, `small`, `medium` model desteği).
- **%100 Çevrimdışı Speaker Diarization**:
  - **Birincil Motor**: Sherpa-ONNX (%100 Çevrimdışı, Token-Free, Pyannote segmentation-3.0 ONNX + 3DSpeaker ONNX embedding, 4.8s hızlı model yükleme & 632 MB RAM footprint).
  - **Zor Durum Kademesi**: Pyannote Community-1 (`pyannote.audio 4.x`, opsiyonel `DIARIZATION_ESCALATION=unknown_speakers` yükseltme kademesi).
  - **Yedek Motorlar**: SpeechBrain ECAPA-TDNN -> Local Spectral Clustering.
- **SemanticRefiner & Yerel LLM Entegrasyonu**:
  - Alan Odaklı Kurallar (`domain_mode="call_center"` ile müşteri/temsilci geçiş tespiti ve rol sabitleme).
  - Opsiyonel yerel Ollama LLM (`llama3.2` / `qwen2.5`) entegrasyonu ile konuşmacı metinlerinin anlamsal iyileştirilmesi.
- **Fusion Engine**: IoU ve Midpoint çakışma çözümleme algoritması + 1.5s sessizlik eşiği.
- **Overlap Detection (Deneysel)**: Eşzamanlı konuşma ve çakışma tespiti özet alanı (`supported=False` olarak işaretlenir).
- **Voice Activity Detection (VAD)**: Silero VAD ile gürültü ve sessizlik halüsinasyon filtrelemesi.
- **Rust PyO3 Native DSP (Deneysel)**: `experimental/native/` klasöründe yer alan deneysel modül (üretim akışında varsayılan olarak Python/NumPy/SciPy tercih edilir).
- **Asenkron Mesaj Kuyruğu Mimarisi**: Redis Streams consumer group (tavsiye edilen) ve FastAPI BackgroundTasks ile non-blocking HTTP 202 istek işleme. *(Not: BackgroundTasks modunda geçici hata alan işler PENDING durumunda kalır; otomatik RETRY ve DLQ kuyruğu için `USE_REDIS_STREAM=true` önerilir).*
- **Sıcak Yükleme (Warm-Loading)**: Singleton AI Pipeline ile hızlı ve düşük gecikmeli analiz.

## 🏢 Kurumsal Çevrimdışı (Air-Gapped / Token-Free) Yapılandırma

Sistem, internete hiç çıkmadan ve **herhangi bir HuggingFace Token'ına ihtiyaç duymadan (Token-Free)** %100 yerel modda çalışır:

* **Çevrimdışı (Air-Gapped) Çalıştırma:** Modeller yerel diskinizdeki önbellekten veya `storage/models/` klasöründen okunur. Herhangi bir dış API veya HuggingFace token zorunluluğu yoktur.
* **Token-Free Diarization:** **SpeechBrain ECAPA-TDNN** ve **Local Spectral Cluster** diyarizasyon motorları tamamen yerel matematiksel vektör hesaplaması yapar ve internet/token gerektirmez. `TARGET_NUM_SPEAKERS` ortam değişkeni ayarlanmadığında dinamik konuşmacı tespiti devreye girer. Dinamik tespitte kullanılan `DIARIZATION_THRESHOLD` eşik değerinin optimum sonuç için gerçek kayıtlarla ayarlanması (tuning) gerekmektedir.

## Kurulum ve Başlatma

### 1. Bağımlılıkların Yüklenmesi
```bash
# Sanal ortamı aktifleştirme:
# Windows (PowerShell):
.\.venv\Scripts\activate

# Linux / macOS:
source .venv/bin/activate

# Paketi ve tüm bağımlılıkları yükleme:
pip install -e .
```

### 2. API Sunucusunun Başlatılması (Uvicorn)
Aşağıdaki komutlardan herhangi biriyle Web API sunucusunu ve İnteraktif Arayüzü çalıştırabilirsiniz:

```bash
# Seçenek A: Doğrudan Uvicorn ile
uvicorn audio_analyzer.api.main:app --host 0.0.0.0 --port 8000 --reload

# Seçenek B: CLI Giriş Noktası ile (pip install -e . sonrası)
audio-analyzer-api

# Seçenek C: Python Modülü olarak
python -m audio_analyzer.api.main
```

Sunucu başladıktan sonra:
- **Web UI & İnteraktif Arayüz**: `http://localhost:8000/`
- **Swagger API Dokümantasyonu**: `http://localhost:8000/docs`

### 3. CLI Analiz Komutunun Çalıştırılması
```bash
python run_analysis.py --audio storage/raw/ornek_ses.wav
```

### 4. Testlerin Çalıştırılması
```bash
pytest
```

### 5. Rust PyO3 Native DSP Modülü (Deneysel)
*Not: `experimental/native/` klasöründeki Rust modülü deneyseldir ve varsayılan üretim akışında kullanılmamaktadır (NumPy/SciPy polyphase resample ve NumPy vectorization işlemleri standart olarak yürütülür).*

---

## 📊 Gözlemlenebilirlik & Metrikler (Prometheus)

- **Prometheus Metrik Endpoint'i:** `GET /metrics`
  - `audio_queue_depth`: Redis Stream (`audio_analysis_stream`) kuyruk derinliği (`XLEN`).
  - `audio_pending_jobs_count`: Veritabanında bekleyen `PENDING` iş sayısı.
  - `audio_oldest_message_age_seconds`: Akıştaki en eski işlenmemiş mesajın yaşı.
  - `audio_pipeline_stage_duration_seconds`: Aşamaların (decode, denoise, VAD, STT, diarize, refine) çalışma süresi histogramı.
  - `audio_jobs_total`: Yaşam döngüsü durum sayaçları (`COMPLETED`, `FAILED`, `RETRY`).
  - `audio_dlq_jobs_total`: Dead-Letter Stream (DLQ) akışına aktarılan iş sayısı.
  - `audio_webhook_deliveries_total`: Webhook teslim durumları (`SUCCESS`, `FAILED`, `DEAD`).
- **Liveness & Readiness Probes:**
  - Liveness: `GET /health`
  - Readiness: `GET /health/ready` (DB + Redis + Model hazır kontrolü).

---

## 🧹 Veri Saklama Süreleri (Retention) & S3 Lifecycle

### 1. Otomatik Temizlik Betiği
- **`AUDIO_RETENTION_HOURS`** (varsayılan: 24 saat): `COMPLETED` / `FAILED` durumundaki işlerin ham ses dosyaları disktan temizlenir.
- **`RESULT_RETENTION_DAYS`** (varsayılan: 30 gün): Veritabanı sonuç kayıtları temizlenir.

```bash
python scripts/cleanup_retention.py --retention-hours 24 --retention-days 30
```

### 2. AWS S3 Lifecycle Policy (Opsiyonel)
S3 kullanıldığında nesne silme maliyetini ve depolamayı otomatikleştirmek için AWS S3 Bucket Lifecycle kuralı eklenmesi önerilir:
- **Prefix:** `raw/`
- **Expiration:** 1 Gün (24 Saat sonra nesnelerin S3 tarafından otomatik silinmesi).

---

## ⚡ CPU Verimliliği & Kapasite Planlaması

### 1. CPU Thread Bütçesi Kuralları
Sistemde CPU aşırı kullanımı (oversubscription) engellemek için thread bütçesi `WORKER_CPU_THREADS` ortam değişkeni ile yönetilir:

```bash
# Örnek: 8 Çekirdekli bir sunucuda 2 Worker çalıştırma
export WORKER_CPU_THREADS=4
```

**Kural:** `(worker_süreç_sayısı × WORKER_CPU_THREADS) <= toplam_çekirdek_sayısı`

### 2. Pipeline Profilleri (`PIPELINE_PROFILE`)
- **`feedback` Profil:** Kısa ve yüksek hacimli geri bildirim mesajları için optimize edilmiştir.
  - Diarization: Ses < `PIPELINE_MIN_DIARIZE_SEC` (varsayılan 10s) ise atlanır (`SPEAKER_00`).
  - Denoiser: Yalnızca ölçülen SNR < `PIPELINE_MIN_SNR_DB` (varsayılan 15 dB) ise çalışır.
  - SemanticRefiner: Yalnızca `DOMAIN_MODE` tanımlıysa çalışır.
  - Whisper Beam Size: 1.
- **`full` Profil (Varsayılan):** Tüm analiz adımlarını eksiksiz çalıştırır. Whisper Beam Size: 5.

### 3. Çevrimdışı Kapasite Hesaplama Formülü
Kapasite planlaması yapılırken aşağıdaki matematiksel formül esas alınır:

$$\text{Gereken Çekirdek Sayısı} = \frac{\text{Toplam Ses Süresi (sn)} \times k}{\text{Hedef Tamamlanma Süresi (sn)}}$$

> **Not:** $k$ katsayısı (ses saniyesi başına harcanan çekirdek-saniye), `scripts/benchmark_cpu.py` betiği çalıştırılarak donanım üzerinde ampirik olarak ölçülmelidir (tahmin yazılmaz).

### 4. CPU Performans Benchmark Betiği
```bash
python scripts/benchmark_cpu.py --audio-dir ./storage/benchmark_samples --models small --profiles feedback,full --threads 4
```

### 5. CPU Ölçüm Sonuçları ve 4000 Mesaj Kapasite Planlama Tablosu

Aşağıdaki değerler `scripts/benchmark_cpu.py` ile **small** model ve 4 CPU thread bütçesi (`WORKER_CPU_THREADS=4`) kullanılarak ampirik olarak ölçülmüştür:

#### Ölçülen Örnek Performansı (Worker Başına):
| Ses Dosyası | Ses Süresi | İşlem Süresi | Worker RTF | Worker RAM Kullanımı |
|---|---|---|---|---|
| `sample_10s.wav` | 10.0 s | 1.32 s | 0.132 | 1530.8 MB (~1.5 GB) |
| `sample_30s.wav` | 30.0 s | 3.84 s | 0.128 | 1531.0 MB (~1.5 GB) |
| `sample_60s.wav` | 60.0 s | 7.87 s | 0.131 | 1531.2 MB (~1.5 GB) |
| **Ortalama / Toplam** | **100.0 s** | **13.03 s** | **0.130** | **~1.5 GB RAM / Worker** |

#### 4000 Sesli Geri Bildirim Mesajı İçin İşlem Süresi ve Kaynak İhtiyacı Hesabı:

| Mesaj Uzunluğu | Toplam Ses Hacmi | Worker Sayısı (Çekirdek) | Toplam RAM İhtiyacı | Tahmini İşlem Süresi |
|---|---|---|---|---|
| **10 Saniye** (Geri Bildirim) | 40.000s (~11.1 saat) | 1 Worker (4 Çekirdek) | ~1.5 GB RAM | **1.44 Saat** (86.7 dk) |
| **10 Saniye** (Geri Bildirim) | 40.000s (~11.1 saat) | 2 Worker (8 Çekirdek) | ~3.0 GB RAM | **0.72 Saat** (43.3 dk) |
| **10 Saniye** (Geri Bildirim) | 40.000s (~11.1 saat) | 4 Worker (16 Çekirdek) | ~6.0 GB RAM | **0.36 Saat** (21.6 dk) |
| **30 Saniye** (Diyalog/Çağrı) | 120.000s (~33.3 saat) | 1 Worker (4 Çekirdek) | ~1.5 GB RAM | **4.33 Saat** (260 dk) |
| **30 Saniye** (Diyalog/Çağrı) | 120.000s (~33.3 saat) | 2 Worker (8 Çekirdek) | ~3.0 GB RAM | **2.17 Saat** (130 dk) |
| **30 Saniye** (Diyalog/Çağrı) | 120.000s (~33.3 saat) | 4 Worker (16 Çekirdek) | ~6.0 GB RAM | **1.08 Saat** (65 dk) |
| **60 Saniye** (Uzun Mesaj) | 240.000s (~66.7 saat) | 1 Worker (4 Çekirdek) | ~1.5 GB RAM | **8.67 Saat** (520 dk) |
| **60 Saniye** (Uzun Mesaj) | 240.000s (~66.7 saat) | 2 Worker (8 Çekirdek) | ~3.0 GB RAM | **4.33 Saat** (260 dk) |
| **60 Saniye** (Uzun Mesaj) | 240.000s (~66.7 saat) | 4 Worker (16 Çekirdek) | ~6.0 GB RAM | **2.17 Saat** (130 dk) |





