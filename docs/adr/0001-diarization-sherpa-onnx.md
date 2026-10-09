# ADR 0001: Birincil Konuşmacı Ayrıştırma (Diarization) Motorunun Sherpa-ONNX'e Geçirilmesi

* **Durum:** Koşullu Kabul (Conditionally Accepted)
* **Tarih:** 2026-10-08
* **Karar Vericiler:** Tuğba Çevik, Antigravity AI Team

---

## 1. Bağlam ve Problem Tanımı

Ses Analizi platformu; Türkçe ses kayıtları için Speech-to-Text (STT) ve Konuşmacı Ayrıştırma (Speaker Diarization) hizmeti sunmaktadır. Sistem yalnızca CPU kaynaklarına sahip, kurumsal ve tamamen dış internete kapalı (air-gapped) ortamlarda, anlık yüksek hacimli mesaj patlamalarını (~4000 mesajlık dalgalanmalar) işlemek durumundadır.

Eski mimaride birincil motor olarak değerlendirilen Pyannote 3.1;
1. HuggingFace token ve kapalı ağ erişim bağımlılıklarına sahipti.
2. Yüksek CPU bellek footprint'ine (~1.1 GB RAM) ve yavaş model başlatma sürelerine (18+ saniye) yol açıyordu.

---

## 2. Değerlendirilen Seçenekler

1. **Seçenek A: SpeechBrain ECAPA-TDNN (Eski Varsayılan)**
   - *Artıları:* %100 Token-Free ve çevrimdışı çalışabilme, PyTorch tabanlı nöral embedding çıkarımı.
   - *Eksileri:* Model ilk yükleme süresi yüksek (10.926s), bellek kullanımı yüksek (934.7 MB).

2. **Seçenek B: Sherpa-ONNX (Pyannote Segmentation-3.0 ONNX + 3DSpeaker ONNX Embedding) — SEÇİLEN**
   - *Artıları:* %100 Token-Free (İndirmede ve çalıştırmada token gerekmez), ONNX Runtime ile C++ düzeyinde optimize CPU çalışması, 4.812 saniye ultra hızlı model yükleme, 632.7 MB düşük bellek footprint'i (302 MB RAM tasarrufu).
   - *Eksileri:* PyTorch native GPU opsiyonu bulunmaz.

3. **Seçenek C: Pyannote.audio 4.x (Community-1)**
   - *Artıları:* Yüksek duyarlılık.
   - *Eksileri:* Model indirme için `HF_TOKEN` gereksinimi vardır (%100 Token-Free değildir). Zor durum kademesi (Escalation) olarak konumlandırılmıştır.

---

## 3. Deneysel Ölçüm Sonuçları (JSON Karşılaştırma Raporu)

Aşağıdaki tablo, `evaluate_diarization.py --report-md docs/benchmarks/diarization_ecapa_20261008_145546.json docs/benchmarks/diarization_sherpa_onnx_20261008_153539.json` komutu ile doğrudan ölçüm JSON dosyalarından üretilmiştir:

| Metrik / Performans Göstergesi | ECAPA (Eski) | SHERPA_ONNX (Yeni) | Fark / Değişim |
| :--- | :---: | :---: | :---: |
| **Model Yükleme Süresi (`load_time_sec`)** | `10.926s` | `4.812s` | `-6.114s` |
| **Ortalama DER ± Std** | `%37.01 ± %11.16` | `%35.44 ± %11.91` | `%-1.57` |
| **Kaçırılan Konuşma (% Missed)** | `%11.33` | `%9.44` | `%-1.89` |
| **Yanlış Alarm (% False Alarm)** | `%0.11` | `%0.42` | `%+0.31` |
| **Konuşmacı Karmaşası (% Confusion)** | `%25.57` | `%25.58` | `%+0.01` |
| **Konuşmacı Sayısı Eşleşme Oranı** | `%100.00` | `%92.50` | `%-7.50` |
| **Medyan İşlem Süresi (P50)** | `5.375s` | `3.965s` | `-1.410s` |
| **P95 İşlem Süresi (P95)** | `7.551s` | `6.789s` | `-0.762s` |
| **Maksimum RAM Kullanımı (RSS)** | `934.7 MB` | `632.7 MB` | `-302.0 MB` |
| **Wilcoxon p-değeri** | - | `p = 0.1621` | İstatistiksel Olarak Anlamlı Fark Yok (p >= 0.05) |

> **İstatistiksel Not:** Eşleştirilmiş Wilcoxon signed-rank test sonucu `p = 0.1621` (p >= 0.05) çıkmıştır. Bu sonuç, iki motor arasındaki DER başarım farkının **istatistiksel olarak anlamlı OLMADIĞINI** göstermektedir.

---

## 4. Karar

Karar hız ve bellek kazancına dayanır. DER karşılaştırması doğrulanmamış bir veri setiyle yapıldı (storage/raw kaynaklı, kopya dosyalar içeriyor). Doğruluk, Common Voice tabanlı set ile yeniden ölçülecek ve karar teyit edilecek.

**Yapılandırma:**
- Birincil motor varsayılan olarak `DIARIZATION_ENGINE=sherpa_onnx` seçilmiştir.
- `DIARIZATION_THRESHOLD=0.55` varsayılan değeri henüz deneysel olarak kalibre edilmemiştir (kalibre edilmedi).

---

## 5. Geri Alma Yolu (Rollback Strategy)

Sistemi eski motorlara döndürmek için ortam değişkeni değiştirmek yeterlidir:
- SpeechBrain ECAPA'ya dönmek için: `DIARIZATION_ENGINE=ecapa`
- Pyannote Community-1'e dönmek için: `DIARIZATION_ENGINE=community1`
