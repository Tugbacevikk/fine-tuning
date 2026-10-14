"""
Prometheus Gözlemlenebilirlik & Metrik Modülü.
Kuyruk derinliği, bekleyen işler, aşama süreleri, DLQ ve webhook metriklerini toplar ve /metrics endpoint'i sunar.
"""

import logging
import os
import time

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from audio_analyzer.adapters.repository.session import get_uow
from audio_analyzer.utils.metrics import (
    DIARIZATION_ESCALATION_COUNTER,
    DIARIZATION_FALLBACK_COUNTER,
    DLQ_COUNTER,
    JOB_STATUS_COUNTER,
    OLDEST_MESSAGE_AGE,
    PENDING_JOBS,
    QUEUE_DEPTH,
    STAGE_DURATION_HISTOGRAM,
    WEBHOOK_DELIVERY_COUNTER,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Metrics & Monitoring"])

__all__ = [
    "QUEUE_DEPTH",
    "PENDING_JOBS",
    "OLDEST_MESSAGE_AGE",
    "STAGE_DURATION_HISTOGRAM",
    "JOB_STATUS_COUNTER",
    "DLQ_COUNTER",
    "WEBHOOK_DELIVERY_COUNTER",
    "DIARIZATION_FALLBACK_COUNTER",
    "DIARIZATION_ESCALATION_COUNTER",
    "router",
    "get_prometheus_metrics",
    "update_dynamic_gauges",
]


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
                    total_depth += lag + pending
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
        async with get_uow() as uow:
            stale_pending = await uow.repository.get_stale_pending_records(
                stale_seconds=0, limit=10000
            )
            PENDING_JOBS.set(len(stale_pending))
    except Exception as ex:
        logger.debug("Prometheus DB metrik güncelleme notu: %s", ex)


@router.get("/metrics")
async def get_prometheus_metrics():
    """Prometheus scrapper için /metrics metrik çıktısını döndürür."""
    await update_dynamic_gauges()
    metrics_data = generate_latest()
    return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)
