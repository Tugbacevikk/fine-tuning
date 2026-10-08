# ADR 0001: Birincil Konuşmacı Ayrıştırma (Diarization) Motorunun Sherpa-ONNX'e Geçirilmesi

* **Durum:** Kabul Edildi (Accepted)
* **Tarih:** 2026-10-08
* **Karar Vericiler:** Tuğba Çevik, Antigravity AI Team

---

## 1. Bağlam ve Problem Tanımı

Ses Analizi platformu; Türkçe ses kayıtları için Speech-to-Text (STT) ve Konuşmacı Ayrıştırma (Speaker Diarization) hizmeti sunmaktadır. Sistem yalnızca CPU kaynaklarına sahip, kurumsal ve tamamen dış internete kapalı (air-gapped) ortamlarda, anlık yüksek hacimli mesaj patlamalarını (~4000 mesajlık dalgalanmalar) işlemek durumundadır.

Eski mimaride birincil motor olarak planlanan Pyannote 3.1;
1. HuggingFace token ve kapalı ağ erişim bağımlılıklarına sahipti.
2. Yüksek CPU bellek footprint'ine (~1.1 GB - 1.5 GB RAM) ve yavaş model başlatma sürelerine (18+ saniye) yol açıyordu.
3. PyTorch bağımlılığı nedeniyle konteyner boyutlarını ve bağımlılık çakışmalarını artırıyordu.

---

## 2. Değerlendirilen Seçenekler

1. **Seçenek A: SpeechBrain ECAPA-TDNN (Mevcut Varsayılan)**
   - *Artıları:* %100 Token-Free ve çevrimdışı çalışabilme, PyTorch tabanlı nöral embedding çıkarımı.
   - *Eksileri:* Model ilk yükleme süresi yüksek (18.911s), bellek kullanımı yüksek (972.1 MB), yüksek yanlış alarm (FA: %1.87) ve konuşmacı karmaşası (Confusion: %31.72).

2. **Seçenek B: Sherpa-ONNX (Pyannote Segmentation-3.0 ONNX + 3DSpeaker/WeSpeaker ONNX Embedding) — SEÇİLEN**
   - *Artıları:* %100 Token-Free, ONNX Runtime ile C++ düzeyinde optimize CPU çalışması, 4.812 saniye ultra hızlı model yükleme, 632.7 MB düşük bellek footprint'i, %77 daha düşük yanlış alarm oranı (%0.42) ve %19 daha düşük konuşmacı karmaşası (%25.58).
   - *Eksileri:* PyTorch native GPU opsiyonu bulunmaz (yalnızca ONNX Runtime CPU/GPU provider'ları).

3. **Seçenek C: Pyannote.audio 4.x (Community-1)**
   - *Artıları:* SOTA duyarlılık.
   - *Eksileri:* HuggingFace token gereksinimi, yüksek bellek ve CPU yükü. Zor durum kademesi (Escalation) olarak konumlandırılmıştır.

---

## 3. Deneysel Ölçüm Sonuçları (40 Kayıtlık Veri Seti)

Aynı 40 diyalog kaydı üzerinde yapılan eşleştirilmiş Wilcoxon benchmark ölçüm sonuçları:

| Metrik | SpeechBrain ECAPA | Sherpa-ONNX | İyileşme / Fark |
| :--- | :---: | :---: | :---: |
| **Model Yükleme Süresi** | `18.911 sn` | **`4.812 sn`** | ⚡ **%74.5 Daha Hızlı** |
| **Maksimum Bellek (RAM RSS)** | `972.1 MB` | **`632.7 MB`** | 📉 **%35 Bellek Tasarrufu** |
| **Ortalama DER** | `%35.45` | **`%35.44`** | **%-0.01 (Eşit / Stabil)** |
| **Yanlış Alarm Oranı (FA)** | `%1.87` | **`%0.42`** | 🎯 **%77 Daha Düşük Yanlış Alarm** |
| **Konuşmacı Karmaşası (Confusion)** | `%31.72` | **`%25.58`** | 🔍 **%19 Daha Az Karmaşa** |
| **Wilcoxon İyileşen Kayıt Oranı** | - | **27 / 40 (%67.5)** | **Ortalama DER Farkı: %-1.57** |

---

## 4. Karar

Birincil konuşmacı ayrıştırma motoru varsayılan olarak **`DIARIZATION_ENGINE=sherpa_onnx`** olarak ayarlanmıştır.

Mimari zincir (Pipeline Factory):
1. **Birincil Motor:** Sherpa-ONNX (`SherpaOnnxAudioAdapter`)
2. **Zor Durum Yükseltme Kademesi:** Pyannote Community-1 (`DIARIZATION_ESCALATION=unknown_speakers` modunda aktif)
3. **Yedek (Fallback) Kademesi:** SpeechBrain ECAPA-TDNN (`SpeechBrainECAPADiarizer`)
4. **Akustik Yedek:** Local Spectral Clustering (`LocalSpectralClusterDiarizer`)

---

## 5. Geri Alma Yolu (Rollback Strategy)

Olası bir uyumsuzluk veya acil durumda sistemi eski motorlara döndürmek için tek bir ortam değişkeni değiştirmek yeterlidir:

- SpeechBrain ECAPA'ya dönmek için:
  ```env
  DIARIZATION_ENGINE=ecapa
  ```
- Pyannote Community-1'e dönmek için:
  ```env
  DIARIZATION_ENGINE=community1
  ```
