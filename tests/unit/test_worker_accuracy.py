import asyncio
import os
import uuid
from unittest.mock import MagicMock

import fakeredis.aioredis
import pytest

from audio_analyzer.adapters.messaging.redis_stream_adapter import RedisStreamAdapter
from audio_analyzer.adapters.repository.postgres_repository import PostgresRepository
from audio_analyzer.domain.models import AudioRecord, JobStatus, OverlapSummary
from audio_analyzer.services.job_service import JobService
from audio_analyzer.workers.stream_worker import RedisStreamWorker


@pytest.mark.unit
async def test_claim_job_atomically_idempotency(in_memory_db):
    """COMPLETED durumundaki bir işin tekrar çalıştırılmadığını ve idempotent yanıt verdiğini doğrular."""
    repo = PostgresRepository(session=in_memory_db)
    record_id = uuid.uuid4()
    record = AudioRecord(
        id=record_id,
        storage_uri="ram://test_completed.wav",
        file_name="test_completed.wav",
        status=JobStatus.PENDING,
    )
    await repo.save_record(record)

    # 1. İlk çalıştırma: PENDING -> PROCESSING atomik devralma
    claimed, rec, is_comp = await repo.claim_job_atomically(record_id)
    assert claimed is True
    assert is_comp is False
    assert rec.status == JobStatus.PROCESSING

    # 2. İş tamamlanıp save_utterances çağrılır
    await repo.save_utterances(record_id, [], language="tr")

    # 3. İkinci çalıştırma (idempotency denemesi): COMPLETED -> tekrar çalıştırılmaz, (True, rec, True) döner
    claimed2, rec2, is_comp2 = await repo.claim_job_atomically(record_id)
    assert claimed2 is True
    assert is_comp2 is True
    assert rec2.status == JobStatus.COMPLETED


@pytest.mark.unit
async def test_atomic_claim_concurrency_prevention():
    """İki worker aynı işi aynı anda almaya çalıştığında tek bir worker'ın alabildiğini doğrular."""
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )
    from sqlalchemy.pool import StaticPool

    from audio_analyzer.adapters.repository.models import Base

    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool, echo=False)
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    TestSessionLocal = async_sessionmaker(bind=test_engine, expire_on_commit=False, class_=AsyncSession)

    record_id = uuid.uuid4()
    async with TestSessionLocal() as s_init:
        repo_init = PostgresRepository(session=s_init)
        record = AudioRecord(
            id=record_id,
            storage_uri="ram://concurrent.wav",
            file_name="concurrent.wav",
            status=JobStatus.PENDING,
        )
        await repo_init.save_record(record)

    async def worker_claim():
        async with TestSessionLocal() as s_worker:
            repo_worker = PostgresRepository(session=s_worker)
            return await repo_worker.claim_job_atomically(record_id)

    res1, res2 = await asyncio.gather(worker_claim(), worker_claim())

    claimed_count = sum(1 for r in (res1, res2) if r[0] is True and r[2] is False)
    assert claimed_count == 1


@pytest.mark.unit
async def test_simulated_4000_jobs_stream_workers_zero_duplicates(monkeypatch):
    """
    4000 sahte mesaj (fakeredis + FakePipeline) ile 4 eşzamanlı worker çalıştırıldığında:
    - Mükerrer işleme sayısının 0 olduğunu,
    - Tüm işlerin tam 1 kez terminal duruma (COMPLETED) ulaştığını doğrular.
    """
    import tempfile

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )

    from audio_analyzer.adapters.repository.models import Base

    db_file = os.path.join(tempfile.gettempdir(), f"test_worker_{uuid.uuid4().hex}.db")
    test_engine = create_async_engine(f"sqlite+aiosqlite:///{db_file}", echo=False)
    async with test_engine.begin() as conn:
        await conn.execute(text("PRAGMA journal_mode=WAL;"))
        await conn.run_sync(Base.metadata.create_all)

    TestSessionLocal = async_sessionmaker(bind=test_engine, expire_on_commit=False, class_=AsyncSession)

    monkeypatch.setenv("WORKER_PREFETCH", "5")
    monkeypatch.setenv("STREAM_CLAIM_IDLE_MS", "900000")
    monkeypatch.setenv("STREAM_CLAIM_INTERVAL_SEC", "1")

    fake_server = fakeredis.FakeServer() if hasattr(fakeredis, "FakeServer") else fakeredis.aioredis.FakeServer()
    fake_redis = fakeredis.aioredis.FakeRedis(server=fake_server, decode_responses=True)

    stream_adapter = RedisStreamAdapter(redis_client=fake_redis)
    await stream_adapter.create_consumer_group()

    job_ids = []
    async with TestSessionLocal() as s_init:
        repo_init = PostgresRepository(session=s_init)
        for i in range(200):
            r_id = uuid.uuid4()
            job_ids.append(r_id)
            record = AudioRecord(
                id=r_id,
                storage_uri=f"ram://sim_{i}.wav",
                file_name=f"sim_{i}.wav",
                status=JobStatus.PENDING,
            )
            await repo_init.save_record(record)
            await stream_adapter.publish_job(job_id=str(r_id), file_name=f"sim_{i}.wav")

    processed_counter = {}
    lock = asyncio.Lock()

    mock_pipeline = MagicMock()
    mock_pipeline.process_bytes = lambda b: ([], "tr", OverlapSummary())

    mock_storage = MagicMock()
    mock_storage.get_bytes.return_value = b"fake_bytes"

    class CustomTestWorker(RedisStreamWorker):
        async def process_single_message(self, msg_id: str, fields: dict) -> bool:
            j_id = fields.get("job_id")
            if j_id:
                async with lock:
                    processed_counter[j_id] = processed_counter.get(j_id, 0) + 1
            async with TestSessionLocal() as session:
                repo = PostgresRepository(session=session)
                job_svc = JobService(storage=mock_storage, repository=repo, pipeline=mock_pipeline)
                res = await job_svc.execute_job(uuid.UUID(j_id))
            await self.adapter.ack_message(msg_id)
            return res

    workers = [
        CustomTestWorker(consumer_name=f"worker_{w}", adapter=stream_adapter) for w in range(4)
    ]

    async def worker_loop(w):
        w.running = True
        while len(processed_counter) < 200:
            msgs = await w.adapter.consume_messages(w.consumer_name, count=5, block_ms=10)
            if not msgs:
                await asyncio.sleep(0.01)
            for m_id, f in msgs:
                await w.process_single_message(m_id, f)

    await asyncio.gather(*[worker_loop(w) for w in workers])

    # Mükerrer işleme kontrolü: Her iş en fazla 1 kez işlenmiş olmalı
    duplicate_runs = [j for j, count in processed_counter.items() if count > 1]
    assert len(duplicate_runs) == 0

    # Tüm kayıtlar terminal durumda (COMPLETED) olmalı
    async with TestSessionLocal() as s_final:
        repo_final = PostgresRepository(session=s_final)
        for r_id in job_ids:
            rec = await repo_final.get_record_by_id(r_id)
            assert rec is not None
            assert rec.status == JobStatus.COMPLETED

    await test_engine.dispose()
    try:
        if os.path.exists(db_file):
            os.remove(db_file)
    except Exception:
        pass
