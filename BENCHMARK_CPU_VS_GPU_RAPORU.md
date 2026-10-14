# 🚀 CPU vs GPU (CUDA) Karşılaştırmalı Performans Benchmark Raporu

**Proje:** [fine-tuning (audio_analyzer)](https://github.com/Tugbacevikk/fine-tuning)  
**Tarih:** 14 Ekim 2026  
**Test Edilen Donanım:**
* **İşlemci (CPU):** Intel Core i5-11400H (6 Çekirdek / 12 Thread, 2.70 GHz - 4.50 GHz, 16 GB RAM)
* **Ekran Kartı (GPU):** NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB GDDR6 VRAM, Sürücü Sürümü: 617.42, CUDA: 13.4 / 12.1)
* **İşletim Sistemi:** Windows 11 64-bit
* **STT Modeli:** Whisper `medium` (GPU: float16, CPU: int8)
* **Diarization Motoru:** Sherpa-ONNX (Kod tabanında `provider="cpu"` olarak sabitlenmiştir)
* **Test Koşulları:** 1 Isınma (Warm-up) turu + **3'er Tekrar** (her dosya için CPU ve GPU değerlerinin aritmetik ortalaması alınmıştır).
* **Ham JSON Verisi:** `benchmark_cpu_vs_gpu_results.json`

---

## 1. Dosya Bazında Karşılaştırma Tablosu (`diyalog_parca_*.wav`)

| Dosya Adı | Ses Süresi | CPU Süresi (3 Tekrar Ort.) | GPU Süresi (3 Tekrar Ort.) | Hızlanma Oranı (CPU / GPU) | CPU RTF | GPU RTF |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`diyalog_parca_01.wav`** | 120.20 sn | 77.27 sn | **37.51 sn** | **2.06x** | 0.643 | **0.312** |
| **`diyalog_parca_02.wav`** | 115.75 sn | 75.16 sn | **31.80 sn** | **2.36x** | 0.649 | **0.275** |
| **`diyalog_parca_03.wav`** | 128.00 sn | 82.46 sn | **36.00 sn** | **2.29x** | 0.644 | **0.281** |
| **`diyalog_parca_04.wav`** | 125.85 sn | 74.10 sn | **33.70 sn** | **2.20x** | 0.589 | **0.268** |
| **`diyalog_parca_05.wav`** | 111.90 sn | 71.65 sn | **30.87 sn** | **2.32x** | 0.640 | **0.276** |
| **`diyalog_parca_06.wav`** | 112.90 sn | 64.67 sn | **30.04 sn** | **2.15x** | 0.573 | **0.266** |
| **`diyalog_parca_07.wav`** | 117.50 sn | 72.64 sn | **32.36 sn** | **2.24x** | 0.618 | **0.275** |
| **`diyalog_parca_08.wav`** | 131.98 sn | 83.52 sn | **31.97 sn** | **2.61x** | 0.633 | **0.242** |
| **GENEL ORTALAMA** | **120.51 sn** | **75.18 sn** | **33.03 sn** | **⚡ 2.28x** | **0.624** | **🚀 0.274** |

---

## 2. Aşama Bazında Karşılaştırma (STT vs Diarization)

| İşlem Aşaması | CPU Süresi (Ort.) | GPU Süresi (Ort.) | Aşama Hızlanması | Teknik Durum |
| :--- | :---: | :---: | :---: | :--- |
| **STT (Faster-Whisper)** | **49.53 sn** | **6.88 sn** | 🚀 **7.20x Hızlı** | **Asıl hız artışının kaynağı.** Tensor çekirdekleri devreye girdiğinde STT süresi ~50 saniyeden 6.8 saniyeye düşmektedir. |
| **Diarization (Sherpa-ONNX)** | **25.65 sn** | **26.15 sn** | **1.00x** | Kod tabanında `provider="cpu"` olarak sabitlendiği için her iki cihaz testinde de CPU çekirdeklerini kullanmıştır. |
| **TOPLAM İŞLEM SÜRESİ** | **75.18 sn** | **33.03 sn** | **2.28x** | Genel analiz süresi yarıdan daha aza inmiştir. |

---

## 3. Model Yükleme Süreleri ve Bellek Tüketimi

* **Model Yükleme Süresi (Cold Start):**
  * **CPU:** Ortalama **~11.8 saniye** (Whisper `medium` modeli RAM'e yüklenirken)
  * **GPU:** Ortalama **~7.1 saniye** (Model VRAM'e aktarılırken, CPU'dan daha hızlı yüklenmiştir)
* **GPU VRAM Tüketimi:**
  * **En Yüksek VRAM Kullanımı:** **2801.8 MB (~2.73 GB)**
  * **GPU Donanım Kapasitesi:** 4096 MB (4.00 GB)
  * **Değerlendirme:** Whisper `medium` modeli float16 hassasiyetinde, 4 GB'lık RTX 3050 Ti Laptop GPU'nun belleğine **çok rahat ve kararlı biçimde** sığmıştır (~1.3 GB boş VRAM payı kalmaktadır).

---

## 4. Teknik Değerlendirme & Yorum

### 1. GPU En Çok Hangi Aşamada Fark Yarattı?
* **Açık ara STT (Faster-Whisper) aşamasında devasa bir fark yarattı.**
  * CPU üzerinde ortalama **49.53 saniye** süren konuşmayı metne dökme işlemi, RTX 3050 Ti GPU üzerinde **6.88 saniyeye** inmiştir (**7.2 kat hızlanma**).
  * Kısa ses dosyalarında (`sample1_2spk.wav`) bu hızlanma **19.4 katına** kadar çıkmaktadır.

### 2. "1 Dakikalık Sesi En Fazla 1 Dakikada Analiz Etme" (RTF ≤ 1.0) Hedefi:
* **CPU Durumu:** **Başarılı (RTF: 0.62).** 1 dakikalık ses CPU'da yaklaşık **37 saniyede** analiz edilmektedir.
* **GPU Durumu:** **Olağanüstü Başarılı (RTF: 0.27).** 1 dakikalık ses GPU'da sadece **16 saniyede** analiz edilmektedir! Gerçek zamanın neredeyse **4 katı hızında** (Ultra Fast) çalışmaktadır.

### 3. Gelecek Optimizasyon Önerisi (Diarization):
* Sherpa-ONNX konuşmacı ayrıştırma motoru şu anda `provider="cpu"` olarak sabitlenmiştir. İlerleyen süreçte Sherpa-ONNX de ONNXRuntime CUDA sağlayıcısı ile GPU'ya aktarılırsa, ~26 saniyelik konuşmacı ayrımı da 2-3 saniyeye inebilir ve 2 dakikalık bir sesin uçtan uca tüm analizi **yaklaşık 10 saniyede** tamamlanabilir.

---
*Rapor Dosyası: `BENCHMARK_CPU_VS_GPU_RAPORU.md`*  
*İlişkili Veri: `benchmark_cpu_vs_gpu_results.json`*
