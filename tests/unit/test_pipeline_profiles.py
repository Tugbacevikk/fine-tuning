from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from audio_analyzer.adapters.repository.postgres_repository import PostgresRepository
from audio_analyzer.adapters.storage.in_memory_storage_adapter import (
    InMemoryStorageAdapter,
)
from audio_analyzer.api.main import app
from audio_analyzer.domain.models import AudioRecord, JobStatus
from audio_analyzer.services.retention_service import RetentionService


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.unit
def test_prometheus_metrics_endpoint(client):
    """GET /metrics Prometheus endpoint'inin 200 OK ve metrikleri döndürdüğünü doğrular."""
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "audio_queue_depth" in res.text
    assert "audio_jobs_total" in res.text
    assert "audio_pipeline_stage_duration_seconds" in res.text


@pytest.mark.unit
def test_health_ready_endpoint(client):
    """GET /health/ready readiness probe endpoint'ini doğrular."""
    res = client.get("/health/ready")
    assert res.status_code in (200, 503)
    data = res.json()
    assert "status" in data
    assert "database" in data
    assert "redis" in data
    assert "model_loaded" in data


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retention_service_cleanup(in_memory_db):
    """RetentionService'in AUDIO_RETENTION_HOURS sonrası ses dosyalarını temizlediğini doğrular."""
    storage = InMemoryStorageAdapter()
    repository = PostgresRepository(session=in_memory_db)

    uri_old = storage.save(b"olddata", "old_audio.wav")
    uri_new = storage.save(b"newdata", "new_audio.wav")

    # 48 saat önce oluşturulmuş COMPLETED kayıt
    old_time = datetime.now(timezone.utc) - timedelta(hours=48)
    rec_id = await repository.save_record(
        AudioRecord(
            storage_uri=uri_old,
            file_name="old_audio.wav",
            status=JobStatus.COMPLETED,
            created_at=old_time,
        )
    )

    # 1 saat önce oluşturulmuş yeni kayıt
    recent_time = datetime.now(timezone.utc) - timedelta(hours=1)
    new_rec_id = await repository.save_record(
        AudioRecord(
            storage_uri=uri_new,
            file_name="new_audio.wav",
            status=JobStatus.COMPLETED,
            created_at=recent_time,
        )
    )

    ret_svc = RetentionService(storage=storage, repository=repository)
    cleaned = await ret_svc.cleanup_expired_audio_files(retention_hours=24)

    assert cleaned >= 1
