# Ses Analizi Sistemi (Speech-to-Text & Speaker Diarization)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)

Yüksek performanslı, modüler, **Clean Architecture / Code-First** prensiplerine uygun olarak tasarlanmış ses analiz sistemi.

## Özellikler
- **Speech-to-Text (STT)**: Faster-Whisper ile zaman damgalı metne dönüştürme (`tiny`, `small`, `medium` model desteği).
- **%100 Çevrimdışı Speaker Diarization**:
  - **Birincil Motor**: Sherpa-ONNX (%100 Çevrimdışı, Token-Free, Pyannote segmentation-3.0 ONNX + 3DSpeaker ONNX embedding, 4.8s model yükleme & 632 MB RAM footprint).
  - **Zor Durum Kademesi**: Pyannote Community-1 (`pyannote.audio 4.x`, opsiyonel `DIARIZATION_ESCALATION=unknown_speakers` yükseltme kademesi; indirme için `HF_TOKEN` gerektirir).
  - **Yedek Motorlar**: SpeechBrain ECAPA-TDNN -> Local Spectral Clustering.
- **SemanticRefiner & Yerel LLM Entegrasyonu**:
  - Sandviç konuşmacı düzeltmesi, tekrar temizliği ve aynı konuşmacının kısa aralıklı parçalarını birleştirme.
  - Opsiyonel yerel Ollama LLM (`llama3.2` / `qwen2.5`) entegrasyonu ile konuşmacı metinlerinin anlamsal iyileştirilmesi.
- **Fusion Engine**: IoU ve Midpoint çakışma çözümleme algoritması + 1.5s sessizlik eşiği.
- **Overlap Detection (Deneysel)**: Eşzamanlı konuşma ve çakışma tespiti özet alanı (`supported=False` olarak işaretlenir).
- **Voice Activity Detection (VAD)**: Silero VAD ile gürültü ve sessizlik halüsinasyon filtrelemesi.
- **Rust PyO3 Native DSP (Deneysel)**: `experimental/native/` klasöründe yer alan deneysel modül (üretim akışında varsayılan olarak Python/NumPy/SciPy tercih edilir).
- **Asenkron Mesaj Kuyruğu Mimarisi**: Redis Streams consumer group (tavsiye edilen) ve FastAPI BackgroundTasks ile non-blocking HTTP 202 istek işleme. *(Not: BackgroundTasks modunda geçici hata alan işler PENDING durumunda kalır; otomatik RETRY ve DLQ kuyruğu için `USE_REDIS_STREAM=true` önerilir).*
- **Sıcak Yükleme (Warm-Loading)**: Singleton AI Pipeline ile hızlı ve düşük gecikmeli analiz.

## 🏢 Kurumsal Çevrimdışı (Air-Gapped / Token-Free) Yapılandırma

Sistem, internete hiç çıkmadan ve **Sherpa-ONNX motoru ile herhangi bir HuggingFace Token'ına ihtiyaç duymadan (Token-Free)** %100 yerel modda çalışır:

* **Çevrimdışı (Air-Gapped) Çalıştırma:** Modeller yerel diskinizdeki önbellekten veya `storage/models/` klasöründen okunur.
* **Token-Free Diarization:** **Sherpa-ONNX**, **SpeechBrain ECAPA-TDNN** ve **Local Spectral Cluster** motorları token-free olarak yerel çalışır (Pyannote Community-1 modeli ise ilk indirme için `HF_TOKEN` gerektirir). `TARGET_NUM_SPEAKERS` ortam değişkeni ayarlanmadığında dinamik konuşmacı tespiti devreye girer. Dinamik tespitte kullanılan `DIARIZATION_THRESHOLD` (varsayılan: 0.55) henüz deneysel olarak kalibre edilmemiştir ve gerçek kayıtlarla ayarlanması (tuning) gerekmektedir.

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
  - SemanticRefiner: Tekrar temizliği ve aralık normalizasyonu çalışır (LLM adımı feedback profilinde atlanır).
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

---

## ⚙️ Yapılandırma ve Ortam Değişkenleri (.env)

Tüm sistem yapılandırması `audio_analyzer.config.Settings` üzerinden merkezi olarak yönetilmektedir.

| Değişken Adı | Varsayılan Değer | Açıklama |
|---|---|---|
| `APP_ENV` | `production` | Uygulama çalışma modu (`development` / `production`). |
| `DATABASE_URL` | `postgresql://...` | Asenkron PostgreSQL bağlantı adresi (SQLite fallback destekli). |
| `USE_REDIS_STREAM` | `false` | Redis Stream ile çoklu worker kuyruk modu (`true` önerilir). Yerel geliştirme için `false` önerilir. |
| `WORKER_CPU_THREADS` | `4` | Her worker sürecinin kullanacağı CPU çekirdek bütçesi. |
| `WHISPER_MODEL_SIZE` | `medium` | Faster-Whisper model boyutu (`tiny`, `small`, `medium`). |
| `WHISPER_BEAM_SIZE` | `1` (feedback) / `5` (full) | Arama genişliği (1 = Hızlı Greedy, 5 = Beam Search). |
| `WHISPER_LOG_PROB_THRESHOLD` | `-1.0` | Konuşma içeren pencerelerin Whisper tarafından atlanmasını önler (-1.0 = etkin). |
| `WHISPER_NO_SPEECH_THRESHOLD` | `0.6` | Sessizlik olasılığı eşiği. |
| `WHISPER_TEMPERATURE_FALLBACK` | `true` | Düşük güvende sıcaklık geri çekilmesi (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) uygular. |
| `ENABLE_STT_SAFETY_NET` | `true` | >2s transkriptsiz konuşma aralıklarını izole ederek kurtarır. |
| `WHISPER_LANGUAGE` | `tr` | Varsayılan konuşma dili. |
| `WHISPER_HOTWORDS` | `""` | Özel terimler/kelimeler için opsiyonel ipucu listesi (virgülle ayrılmış). |
| `ENABLE_VAD_FILTER` | `false` | STT VAD filtresi (Kısık seslerin atlanmaması için varsayılan kapalı). |
| `RUN_PIPELINE_SEQUENTIALLY` | `true` | `true` = Sıralı motor yürütme (CPU çekişmesini önler, en hızlı mod). |
| `DIARIZATION_ENGINE` | `sherpa_onnx` | Birincil diarization motoru (`sherpa_onnx`, `pyannote`, `ecapa`). |
| `SHERPA_CLUSTER_THRESHOLD`| `0.55` | Sherpa-ONNX kümeleme eşiği. Gerçek diyaloglarda 3. konuşmacı tanıtım sesinin ayrımını korur. |
| `SHERPA_EMBEDDING_MODEL` | `wespeaker_en_voxceleb_resnet34_LM.onnx` | WeSpeaker ResNet34 konuşmacı embedding modeli. |
| `TARGET_NUM_SPEAKERS` | `None` | Sabit konuşmacı sayısı (Belirtilmezse otomatik threshold kümeleme; `None` sentinel). |
| `ENABLE_SPEAKER_MERGER` | `true` | Az konuşan hayalet konuşmacıları en yakın kümeye devreder. |
| `MIN_SPEAKER_SEC` | `3.0` | Speaker merger minimum süre eşiği (saniye). |
| `MIN_SPEAKER_RATIO` | `0.0` | Speaker merger minimum konuşma oranı (0.0 = yalnızca MIN_SPEAKER_SEC). |
| `MERGER_SIMILARITY_THRESHOLD`| `0.50` | Embedding benzerliği 0.50 altındaki konuşmacılar merger tarafından yutulmaz. |
| `ENABLE_SANDWICH_RULE` | `false` | A->B->A sandviç geçiş düzeltmesi (kısa cevapları korumak için varsayılan kapalı). |
| `ENABLE_MICRO_UTTERANCE_RULE`| `false` | Mikro parça kuralı (diyalog dönüşlerini korumak için varsayılan kapalı; yalnızca metinsiz boş parçalar temizlenir). |
| `ENABLE_DATASET_EXPORT` | `false` | Düzeltilen analizleri test seti inceleme kuyruğuna (`real_turkish/_bekleyen`) aktarma izni. |
| `MAX_SILENCE_THRESHOLD` | `1.5` | Aynı konuşmacının ardışık bloklarını birleştirme sessizlik toleransı. |
| `MAX_CLAUSE_GAP` | `1.5` | Noktalamasız biten yarım cümleleri bağlama toleransı. |
| `USE_LLM` | `false` | [DENEYSEL] Yerel Ollama LLM etiket düzeltmesi (varsayılan kapalı). |
| `OLLAMA_TIMEOUT_SEC` | `45.0` | Ollama LLM istek zaman aşımı süresi (saniye). |

---

## 📈 İyileştirme ve Performans Karşılaştırma Raporu

### 1. Diarization Hata Oranı (DER) Karşılaştırması
> **Not:** Aşağıdaki DER ve Karışma (Confusion) metrikleri **sentetik test seti üzerinde ön sonuçlardır** (Veri seti: `tests/fixtures/diarization_eval`, Komut: `python scripts/evaluate_diarization.py`). Gerçek Türkçe test seti (`real_turkish`) oluştukça bu metrikler yeniden değerlendirilecektir.

| Metrik | Başlangıç (Referans) | Güncel Durum | Değişim / Açıklama |
|---|---|---|---|
| **Diarization Motoru & Model** | Sherpa-ONNX (3DSpeaker) | Sherpa-ONNX (WeSpeaker ResNet34) | Model güncellemesi |
| **Kümeleme Eşiği** | `0.75` | `0.55` | 3. ses koruması için kalibre edildi |
| **Diarization Ortalama DER** | `%55.65 ± %22.48` | **`%54.40 ± %23.39`** | -1.25% (Sentetik sette ön sonuç) |
| **Speaker Confusion (Karışma)**| `%9.26` | **`%8.34`** | -0.92% (Sentetik sette ön sonuç) |
| **Bilinmeyen Konuşmacı P50 Süresi**| `4.346 sn` | **`3.014 sn`** | %30.6 daha hızlı |

### 2. Gerçek Kayıtlar Üzerinde Hız ve RTF Performansı
> **Veri Seti:** "Yeni klasör" altındaki gerçek Türkçe müşteri görüşmeleri ve diyalog kayıtları  
> **Çalıştırma Komutu:** `python scripts/batch_analyze_folder.py --folder "Yeni klasör"`  
> **Donanım:** Intel/AMD x86_64 CPU (4 iş parçacığı, GPU kullanılmamıştır)

| Model Boyutu | Medyan RTF | En Kötü RTF | 1 Dakikalık Ses İşlem Süresi | Notlar |
|---|---|---|---|---|
| **Faster-Whisper medium** | **0.94** | **0.98** | ~56 - 58 saniye | `log_prob_threshold=-1.0`, sıcaklık geri çekilmesi ve STT güvenlik ağı devredeyken CPU'da $\le 1.0$ RTF hedefi sağlanmıştır. |
| **Faster-Whisper small** | **0.23** | **0.28** | ~13.7 - 16 saniye | Hızlı geri bildirim (`feedback`) profili için optimize edilmiştir. |

### 3. Model ve Eşik Kararları Dayanağı
- **WeSpeaker ResNet34 + 0.55 Eşiği:** Sentetik sette 3DSpeaker ve CAM++ modellerine kıyasla aşırı kümeleme yapmaması ve gerçek Türkçe kayıtlarda (`parca_02.wav`, `parca_03.wav`) 3. konuşmacı tanıtım sesinin ("Akıcı Türkçe") ana konuşmacıya yutulmasını önlemesi ölçümüne dayanmaktadır. Gerçek test seti genişledikçe eşik kalibrasyonu yeniden değerlendirilecektir.
- **Kısa Cevap & Birleştirme Kuralları:** Sandviç kuralı ve mikro kural kapatılmış, SpeakerClusterMerger'a ses embedding kosinüs benzerliği eşiği (`0.50`) eklenmiştir. Bu sayede kısa cevaplar ("evet", "tamam") karşı tarafa atanmaktan korunmuştur.
