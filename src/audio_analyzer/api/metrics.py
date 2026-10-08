"""
Prometheus Gözlemlenebilirlik & Metrik Modülü.
Kuyruk derinliği, bekleyen işler, aşama süreleri, DLQ ve webhook metriklerini toplar ve /metrics endpoint'i sunar.
"""

import logging
import os
import time

from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Metrics & Monitoring"])

# 1. Metrik Tanımlamaları
QUEUE_DEPTH = Gauge(
    "audio_queue_depth", "Redis Stream (audio_analysis_stream) toplam bekleyen mesaj sayısı (XLEN)"
)
PENDING_JOBS = Gauge(
    "audio_pending_jobs_count", "Veritabanında PENDING durumunda bekleyen analiz görev sayısı"
)
OLDEST_MESSAGE_AGE = Gauge(
    "audio_oldest_message_age_seconds", "Akıştaki en eski işlenmemiş mesajın saniye cinsinden yaşı"
)

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



async def update_dynamic_gauges():
    """Redis ve DB üzerinden dinamik kuyruk ve bekleyen iş metriklerini günceller."""
    # 1. Redis Stream XLEN & En Eski Mesaj Yaşı
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        import redis.asyncio as aioredis

        from audio_analyzer.adapters.messaging.redis_stream_adapter import (
            RedisStreamAdapter,
        )

        pool = RedisStreamAdapter.get_pool(redis_url)
        client = aioredis.Redis(connection_pool=pool)

        stream_key = os.getenv("STREAM_KEY", "audio_analysis_stream")
        
        total_depth = 0
        try:
            groups = await client.xinfo_groups(stream_key)
            if groups:
                for group in groups:
                    g_dict = group if isinstance(group, dict) else {}
                    lag = g_dict.get("lag", 0) or 0
                    pending = g_dict.get("pending", 0) or 0
                    total_depth += (lag + pending)
            else:
                total_depth = await client.xlen(stream_key)
        except Exception:
            total_depth = await client.xlen(stream_key)

        QUEUE_DEPTH.set(total_depth)

        # En eski mesaj yaşı
        first_items = await client.xrange(stream_key, min="-", max="+", count=1)
        if first_items:
            msg_id = str(first_items[0][0])
            ms_str = msg_id.split("-")[0]
            if ms_str.isdigit():
                age_sec = max(0.0, time.time() - (float(ms_str) / 1000.0))
                OLDEST_MESSAGE_AGE.set(age_sec)
        else:
            OLDEST_MESSAGE_AGE.set(0.0)
    except Exception as ex:
        logger.debug("Prometheus Redis metrik güncelleme notu: %s", ex)

    # 2. Veritabanı PENDING İş Sayısı
    try:
        from audio_analyzer.api.dependencies import get_uow

        async with get_uow() as uow:
            stale_pending = await uow.repository.get_stale_pending_records(stale_seconds=0, limit=10000)
            PENDING_JOBS.set(len(stale_pending))
    except Exception as ex:
        logger.debug("Prometheus DB metrik güncelleme notu: %s", ex)


@router.get("/metrics")
async def get_prometheus_metrics():
    """Prometheus scrapper için /metrics metrik çıktısını döndürür."""
    await update_dynamic_gauges()
    metrics_data = generate_latest()
    return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)
