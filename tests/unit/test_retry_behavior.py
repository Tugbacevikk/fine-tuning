import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from audio_analyzer.domain.models import AudioRecord, JobStatus
from audio_analyzer.services.job_service import JobService
from audio_analyzer.workers.stream_worker import RedisStreamWorker


@pytest.mark.asyncio
async def test_transient_error_in_process_bytes_raises_and_triggers_retry():
    mock_storage = MagicMock()
    mock_repo = AsyncMock()
    record_id = uuid.uuid4()

    mock_record = AudioRecord(
        id=record_id,
        storage_uri="disk://test.wav",
        file_name="test.wav",
        status=JobStatus.PENDING,
    )
    mock_repo.claim_job_atomically.return_value = (True, mock_record, False)
    mock_repo.handle_job_failure.return_value = (1, False)  # 1 attempt, not final (is_transient)

    mock_pipeline = MagicMock()
    # process_bytes raises RuntimeError (transient OOM error)
    mock_pipeline.process_bytes.side_effect = RuntimeError("Out of Memory")

    job_service = JobService(
        storage=mock_storage,
        repository=mock_repo,
        pipeline=mock_pipeline,
    )

    status_str, attempts, err_msg = await job_service.execute_job_detailed(record_id, file_bytes=b"dummy")

    assert status_str == "RETRY"
    assert attempts == 1
    # Verify process_bytes was called, but process(path) was NOT called
    mock_pipeline.process_bytes.assert_called_once()
    mock_pipeline.process.assert_not_called()


@pytest.mark.asyncio
async def test_redis_stream_worker_republishes_on_retry():
    adapter_mock = MagicMock()
    adapter_mock.publish_job = AsyncMock()
    adapter_mock.ack_message = AsyncMock()

    worker = RedisStreamWorker(adapter=adapter_mock)

    fake_repo = AsyncMock()
    fake_repo.claim_job_atomically.return_value = (True, AudioRecord(id=uuid.uuid4(), file_name="t.wav", storage_uri="disk://t.wav", status=JobStatus.PENDING), False)
    fake_repo.handle_job_failure.return_value = (1, False)  # RETRY

    fake_pipeline = MagicMock()
    fake_pipeline.process.side_effect = RuntimeError("Transient Error")
    fake_pipeline.process_bytes.side_effect = RuntimeError("Transient Error")

    class DummyUOW:
        def __init__(self, repository):
            self.repository = repository
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("audio_analyzer.workers.stream_worker.SqlAlchemyUnitOfWork", return_value=DummyUOW(fake_repo)), \
         patch("audio_analyzer.workers.stream_worker.get_shared_pipeline", return_value=fake_pipeline), \
         patch("audio_analyzer.workers.stream_worker.asyncio.sleep", AsyncMock()):

        msg_id = "msg-101"
        job_id = str(uuid.uuid4())
        fields = {"job_id": job_id, "file_name": "t.wav"}

        result = await worker.process_single_message(msg_id, fields)

        assert result is False
        if worker._background_tasks:
            await asyncio.gather(*worker._background_tasks)
        # Verify job was republished to stream for RETRY and old message ACKed
        adapter_mock.publish_job.assert_called_once()
        adapter_mock.ack_message.assert_called_once_with(msg_id)
