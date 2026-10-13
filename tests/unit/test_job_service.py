from unittest.mock import MagicMock

from audio_analyzer.adapters.repository.postgres_repository import PostgresRepository
from audio_analyzer.adapters.storage.in_memory_storage_adapter import (
    InMemoryStorageAdapter,
)
from audio_analyzer.domain.models import JobStatus, OverlapSummary
from audio_analyzer.services.job_service import JobService, sanitize_error_message


def test_sanitize_error_message_removes_file_paths():
    ex = FileNotFoundError("No such file: C:\\Users\\testuser\\Desktop\\sesAnalizi\\storage\\raw\\test.wav")
    sanitized = sanitize_error_message(ex)

    assert "C:\\Users\\" not in sanitized
    assert "[FILE_PATH]" in sanitized
    assert "FileNotFoundError" in sanitized


async def test_execute_job_failure_does_not_leak_traceback(in_memory_db):
    storage = InMemoryStorageAdapter()
    repository = PostgresRepository(session=in_memory_db)

    # Pipeline that raises a permanent error
    mock_pipeline = MagicMock()
    err = ValueError("Internal model failure in C:\\Secret\\Path\\model.py")
    mock_pipeline.process.side_effect = err
    mock_pipeline.process_bytes.side_effect = err

    job_service = JobService(storage=storage, repository=repository, pipeline=mock_pipeline)

    record_id = await job_service.create_job("test_audio.wav", b"RIFFfakebytes")
    success = await job_service.execute_job(record_id)

    assert success is False

    record = await repository.get_record_by_id(record_id)
    assert record is not None
    assert record.status == JobStatus.FAILED
    assert record.error_message is not None

    # Verify no raw traceback or local path is present in error_message
    assert "Traceback (most recent call last)" not in record.error_message
    assert "C:\\Secret\\Path" not in record.error_message
    assert "ValueError" in record.error_message


async def test_execute_job_triggers_webhook_callback(in_memory_db):
    storage = InMemoryStorageAdapter()
    repository = PostgresRepository(session=in_memory_db)

    mock_pipeline = MagicMock()
    ret = ([], "tr", OverlapSummary(supported=False))
    mock_pipeline.process.return_value = ret
    mock_pipeline.process_bytes.return_value = ret

    job_service = JobService(storage=storage, repository=repository, pipeline=mock_pipeline)

    record_id = await job_service.create_job("test_audio.wav", b"RIFFfakebytes", callback_url="https://example.com/webhook")
    success = await job_service.execute_job(record_id)

    assert success is True

    # Webhook Outbox kontrolü: satır içi gönderim yerine outbox tablosuna yazılır
    due = await repository.get_due_webhook_deliveries(limit=10)
    assert len(due) == 1
    assert due[0]["job_id"] == record_id
    assert due[0]["url"] == "https://example.com/webhook"
    assert due[0]["status"] == "PENDING"




