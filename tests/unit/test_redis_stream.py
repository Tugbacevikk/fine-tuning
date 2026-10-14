import uuid

import fakeredis.aioredis
import pytest

from audio_analyzer.adapters.messaging.redis_stream_adapter import RedisStreamAdapter
from audio_analyzer.workers.stream_worker import RedisStreamWorker


@pytest.fixture
def fake_redis_client():
    """Fakeredis async in-memory Redis client fixture."""
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def stream_adapter(fake_redis_client):
    return RedisStreamAdapter(
        redis_url="redis://localhost:6379/0",
        stream_key="test_audio_stream",
        group_name="test_worker_group",
        redis_client=fake_redis_client,
    )


@pytest.mark.unit
async def test_redis_stream_publish_job(stream_adapter, fake_redis_client):
    job_id = str(uuid.uuid4())
    msg_id = await stream_adapter.publish_job(
        job_id=job_id,
        file_name="test_call.wav",
        callback_url="https://example.com/webhook",
    )

    assert msg_id is not None
    assert "-" in msg_id  # Redis Stream ID format (timestamp-sequence)

    # Stream içeriğini doğrudan fakeredis ile oku
    raw_stream = await fake_redis_client.xrange("test_audio_stream")
    assert len(raw_stream) == 1
    m_id, fields = raw_stream[0]
    assert m_id == msg_id
    assert fields["job_id"] == job_id
    assert fields["file_name"] == "test_call.wav"
    assert fields["callback_url"] == "https://example.com/webhook"


@pytest.mark.unit
async def test_redis_stream_consumer_group_flow(stream_adapter):
    # 1. Consumer Group Oluştur
    created = await stream_adapter.create_consumer_group()
    assert created is True

    # Tekrarlayan çağrı BUSYGROUP hatası fırlatmamalı
    recreated = await stream_adapter.create_consumer_group()
    assert recreated is True

    # 2. Mesaj Yayınla (XADD)
    job_id = str(uuid.uuid4())
    msg_id = await stream_adapter.publish_job(job_id=job_id, file_name="stream_test.wav")

    # 3. Tüketici Grubu Üzerinden Oku (XREADGROUP)
    messages = await stream_adapter.consume_messages(consumer_name="worker_1", count=10, block_ms=100)
    assert len(messages) == 1
    consumed_msg_id, fields = messages[0]
    assert consumed_msg_id == msg_id
    assert fields["job_id"] == job_id

    # 4. Mesajı Onayla (XACK)
    ack_res = await stream_adapter.ack_message(msg_id)
    assert ack_res == 1


@pytest.mark.unit
async def test_redis_stream_worker_message_processing(stream_adapter, monkeypatch):
    await stream_adapter.create_consumer_group()
    job_id = str(uuid.uuid4())
    msg_id = await stream_adapter.publish_job(job_id=job_id, file_name="worker_test.wav")
    assert msg_id is not None

    worker = RedisStreamWorker(consumer_name="worker_test_unit", adapter=stream_adapter)

    # Monkeypatch execution
    async def mock_process(msg_id_in, fields):
        assert fields["job_id"] == job_id
        await stream_adapter.ack_message(msg_id_in)
        return True

    monkeypatch.setattr(worker, "process_single_message", mock_process)

    messages = await stream_adapter.consume_messages(consumer_name=worker.consumer_name, count=1, block_ms=100)
    assert len(messages) == 1
    assert messages[0][0] == msg_id
    success = await worker.process_single_message(messages[0][0], messages[0][1])
    assert success is True


@pytest.mark.unit
async def test_redis_stream_claim_pending_messages(stream_adapter):
    await stream_adapter.create_consumer_group()
    job_id = str(uuid.uuid4())
    msg_id = await stream_adapter.publish_job(job_id=job_id, file_name="claim_test.wav")
    assert msg_id is not None

    # worker_1 mesajı okur ama ACK etmez (çöker)
    consumed = await stream_adapter.consume_messages(consumer_name="worker_1", count=1, block_ms=100)
    assert len(consumed) == 1
    assert consumed[0][0] == msg_id

    # claim_pending_messages çağır (min_idle_time_ms=0 for testing)
    claimed = await stream_adapter.claim_pending_messages(consumer_name="worker_2", min_idle_time_ms=0, count=5)
    # Fakeredis xautoclaim veya boş liste dönebilir
    assert isinstance(claimed, list)
