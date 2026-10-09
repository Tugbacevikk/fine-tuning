import json
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Request

from audio_analyzer.adapters.repository.postgres_repository import PostgresRepository
from audio_analyzer.adapters.storage.in_memory_storage_adapter import (
    InMemoryStorageAdapter,
)
from audio_analyzer.api.rate_limiter import RedisRateLimiter
from audio_analyzer.domain.models import AudioRecord, JobStatus
from audio_analyzer.services.job_service import JobService
from audio_analyzer.workers.webhook_worker import deliver_single_webhook, sign_payload


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rate_limiter_fail_open_on_redis_error():
    """Redis bağlantısı yoksa rate limiter fail-open (isteğe izin verme) çalışır."""
    limiter = RedisRateLimiter(redis_url="redis://non_existent_host:9999/0")
    scope = {"type": "http", "headers": [], "client": ("127.0.0.1", 12345)}
    request = Request(scope)

    # Redis çöktüğünde exception fırlatmamalı, fail-open geçmeli
    await limiter.check_rate_limit(request)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_idempotency_with_external_id(in_memory_db):
    """Aynı external_id ile iki kez istek atıldığında yeni iş açılmaz, aynı job_id döner."""
    storage = InMemoryStorageAdapter()
    repository = PostgresRepository(session=in_memory_db)
    job_service = JobService(storage=storage, repository=repository)

    file_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    ext_id = "tenant-101-order-888"

    # 1. İstek: Yeni kayıt açılır
    job_id_1 = await job_service.create_job(
        file_name="ses1.wav", file_bytes=file_bytes, external_id=ext_id
    )

    # 2. İstek: Aynı external_id ile tekrar çağır
    job_id_2 = await job_service.create_job(
        file_name="ses1_kopya.wav", file_bytes=file_bytes, external_id=ext_id
    )

    assert job_id_1 == job_id_2

    # Veritabanında tek kayıt olduğunu doğrula
    rec = await repository.get_record_by_external_id(ext_id)
    assert rec is not None
    assert rec.id == job_id_1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_webhook_outbox_isolation(in_memory_db):
    """execute_job webhook'u satır içi göndermez, outbox tablosuna yazar."""
    storage = InMemoryStorageAdapter()
    repository = PostgresRepository(session=in_memory_db)

    class FastMockPipeline:
        def process_bytes(self, b):
            return [], "tr", None

    pipeline = FastMockPipeline()
    job_service = JobService(storage=storage, repository=repository, pipeline=pipeline)

    file_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    callback_url = "http://slow-offline-receiver.local/webhook"

    job_id = await job_service.create_job(
        file_name="test.wav", file_bytes=file_bytes, callback_url=callback_url
    )

    # execute_job çalıştır (Yavaş webhook olsa dahi CPU iş parçacığını engellemez)
    status, attempts, err = await job_service.execute_job_detailed(job_id, file_bytes=file_bytes)
    assert status == "COMPLETED"

    # Outbox tablosunu kontrol et: webhook_deliveries tablosunda PENDING satır olmalı
    due = await repository.get_due_webhook_deliveries(limit=10)
    assert len(due) == 1
    assert due[0]["job_id"] == job_id
    assert due[0]["url"] == callback_url
    assert due[0]["status"] == "PENDING"


@pytest.mark.unit
def test_hmac_signature():
    """HMAC SHA256 imza doğrulama fonksiyonu testi."""
    payload = '{"job_id": "123", "status": "COMPLETED"}'
    secret = "my_super_secret_key"
    sig = sign_payload(payload, secret)

    assert sig.startswith("sha256=")
    assert len(sig) == 71  # "sha256=" + 64 hex char


@pytest.mark.unit
@pytest.mark.asyncio
async def test_webhook_worker_delivery_failure_and_dead_status(in_memory_db):
    """Webhook worker kapalı alıcıya ulaşamadığında üstel backoff uygular ve max attempt sonrası DEAD durumuna geçer."""
    repository = PostgresRepository(session=in_memory_db)
    job_id = uuid.uuid4()

    # Sahte bir audio record oluşturalım ki foreign key hatası vermesin
    rec = AudioRecord(
        id=job_id,
        storage_uri="mem://test",
        file_name="test.wav",
        status=JobStatus.COMPLETED,
    )
    await repository.save_record(rec)

    delivery_id = await repository.create_webhook_delivery(
        job_id=job_id,
        url="http://non-existent-server-12345.local/callback",
        payload=json.dumps({"test": "data"}),
    )

    due = await repository.get_due_webhook_deliveries(limit=10)
    assert len(due) == 1
    delivery_item = due[0]

    engine_mock = MagicMock()
    # Mock uow context manager
    uow_mock = MagicMock()
    uow_mock.repository = repository

    class AsyncContextManagerMock:
        async def __aenter__(self):
            return uow_mock

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("audio_analyzer.workers.webhook_worker.get_uow_with_engine", return_value=AsyncContextManagerMock()):
        # Max attempts = 1 verip doğrudan DEAD moduna düşmesini doğrulayalım
        success = await deliver_single_webhook(
            delivery=delivery_item,
            engine=engine_mock,
            max_attempts=1,
            timeout_sec=1.0,
            host_concurrency=2,
            secret="test_secret",
        )

        assert success is False

        # Status DEAD olmalı
        from sqlalchemy import select

        from audio_analyzer.adapters.repository.models import WebhookDeliveryModel

        stmt = select(WebhookDeliveryModel).where(WebhookDeliveryModel.id == delivery_id)
        res = await in_memory_db.execute(stmt)
        orm_del = res.scalar_one()
        assert orm_del.status == "DEAD"
        assert orm_del.attempts == 1
        assert orm_del.error_message is not None
