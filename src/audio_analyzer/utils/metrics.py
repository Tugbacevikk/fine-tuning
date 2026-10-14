"""
Prometheus Metrik Tanımlamaları Modülü.
services, adapters ve workers katmanları tarafından API bağımlılığı olmadan doğrudan kullanılır.
"""

from prometheus_client import Counter, Gauge, Histogram

# 1. Kuyruk ve Görev Metrikleri
QUEUE_DEPTH = Gauge(
    "audio_queue_depth", "Redis Stream (audio_analysis_stream) toplam bekleyen mesaj sayısı (XLEN)"
)
PENDING_JOBS = Gauge(
    "audio_pending_jobs_count", "Veritabanında PENDING durumunda bekleyen analiz görev sayısı"
)
OLDEST_MESSAGE_AGE = Gauge(
    "audio_oldest_message_age_seconds", "Akıştaki en eski işlenmemiş mesajın saniye cinsinden yaşı"
)

# 2. Pipeline ve İşlem Süresi Metrikleri
STAGE_DURATION_HISTOGRAM = Histogram(
    "audio_pipeline_stage_duration_seconds",
    "Pipeline aşamalarının saniye cinsinden çalışma süresi histogramı",
    labelnames=["stage"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

JOB_STATUS_COUNTER = Counter(
    "audio_jobs_total",
    "Yaşam döngüsü durumlarına göre toplam analiz görevi sayısı",
    labelnames=["status"],
)

DLQ_COUNTER = Counter(
    "audio_dlq_jobs_total", "Dead-Letter Stream (DLQ) akışına aktarılan toplam başarısız iş sayısı"
)

WEBHOOK_DELIVERY_COUNTER = Counter(
    "audio_webhook_deliveries_total",
    "Durumlara göre toplam webhook teslimat sayısı",
    labelnames=["status"],
)

DIARIZATION_FALLBACK_COUNTER = Counter(
    "audio_diarization_fallbacks_total",
    "Diarization motoru fallback tetiklenme sayısı",
    labelnames=["from_engine", "to_engine", "reason"],
)

DIARIZATION_ESCALATION_COUNTER = Counter(
    "audio_diarization_escalations_total",
    "Diarization motoru yükseltme (escalation) tetiklenme sayısı",
    labelnames=["reason", "target_engine"],
)
