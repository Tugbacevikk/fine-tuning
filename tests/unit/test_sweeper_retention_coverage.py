import asyncio
import os
import tempfile
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from audio_analyzer.adapters.audio.audio_converter import AudioConverterProcessor
from audio_analyzer.adapters.repository.models import Base
from audio_analyzer.adapters.repository.postgres_repository import PostgresRepository
from audio_analyzer.domain.models import AudioRecord, JobStatus


@pytest.fixture
async def async_db_session():
    db_file = os.path.join(tempfile.gettempdir(), f"test_madde14_{uuid.uuid4().hex}.db")
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_file}", echo=False)
    async with engine.begin() as conn:
        await conn.execute(text("PRAGMA journal_mode=WAL;"))
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)
    async with session_factory() as session:
        yield session

    await engine.dispose()
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except Exception:
            pass


@pytest.mark.unit
def test_audio_converter_pyav_fallback():
    processor = AudioConverterProcessor()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.wav")
        output_path = os.path.join(tmpdir, "output.wav")
        
        # Create a simple valid wav file
        import wave
        with wave.open(input_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 1600)

        # Force SoundFile to fail so PyAV fallback executes
        with patch.object(processor, "_convert_with_soundfile", side_effect=Exception("Soundfile failure")):
            result_path = processor.normalize_and_resample(input_path, output_path, target_sample_rate=16000)
            assert os.path.exists(result_path)

        # Test convert_bytes_to_ndarray PyAV fallback
        with open(input_path, "rb") as f:
            file_bytes = f.read()

        with patch("soundfile.read", side_effect=Exception("Soundfile bytes failure")):
            arr, sr = processor.convert_bytes_to_ndarray(file_bytes, target_sample_rate=16000)
            assert isinstance(arr, np.ndarray)
            assert sr == 16000


@pytest.mark.unit
async def test_postgres_repository_delete_record(async_db_session):
    repo = PostgresRepository(session=async_db_session)
    rec_id = uuid.uuid4()
    record = AudioRecord(
        id=rec_id,
        storage_uri="ram://test.wav",
        file_name="test.wav",
        status=JobStatus.PENDING,
    )
    await repo.save_record(record)

    fetched = await repo.get_record_by_id(rec_id)
    assert fetched is not None

    deleted = await repo.delete_record(rec_id)
    assert deleted is True

    fetched_after = await repo.get_record_by_id(rec_id)
    assert fetched_after is None

    # Deleting non-existent record returns False
    deleted_nonexistent = await repo.delete_record(uuid.uuid4())
    assert deleted_nonexistent is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stream_worker_main_entrypoint():
    from audio_analyzer.workers import stream_worker

    mock_worker_instance = MagicMock()
    mock_worker_instance.run = AsyncMock()

    with patch.object(stream_worker, "RedisStreamWorker", return_value=mock_worker_instance):
        await stream_worker.start_worker_main()
        mock_worker_instance.run.assert_called_once()
