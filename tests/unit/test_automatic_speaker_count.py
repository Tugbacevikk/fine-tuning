import io
import numpy as np
import pytest
import soundfile as sf

from audio_analyzer.adapters.repository.postgres_repository import PostgresRepository
from audio_analyzer.adapters.storage.in_memory_storage_adapter import InMemoryStorageAdapter
from audio_analyzer.config import get_settings, reset_settings
from audio_analyzer.domain.interfaces import IDiarizer, ISTTEngine
from audio_analyzer.domain.models import DiarizationSegment, WordSegment
from audio_analyzer.services.job_service import JobService
from audio_analyzer.services.pipeline import AudioAnalysisPipeline


class MockTrackingSTT(ISTTEngine):
    def transcribe(self, audio_path: str):
        return [WordSegment(word="Merhaba", start_time=0.0, end_time=1.0)], "tr"


class MockTrackingDiarizer(IDiarizer):
    def __init__(self):
        self.called = False
        self.received_num_speakers = "NOT_CALLED"

    def diarize(self, audio_path: str, num_speakers: int | None = None):
        self.called = True
        self.received_num_speakers = num_speakers
        return [DiarizationSegment(speaker_id="SPEAKER_00", start_time=0.0, end_time=1.0)]


@pytest.mark.asyncio
async def test_automatic_speaker_count_end_to_end(in_memory_db, monkeypatch):
    """
    Bölüm 3.1 & 3.3 Doğrulaması:
    1. TARGET_NUM_SPEAKERS=2 ortam değişkeni ayarlı olsa dahi,
    2. Kullanıcı num_speakers=None (Arayüzde 'Otomatik') seçtiğinde,
    3. Diarizer motoruna kesinlikle num_speakers=None gider (otomatik kümeleme).
    4. Sadece parametresiz çağrılarda TARGET_NUM_SPEAKERS=2 devreye girer.
    """
    monkeypatch.setenv("TARGET_NUM_SPEAKERS", "2")
    reset_settings()
    assert get_settings().target_num_speakers == 2

    storage = InMemoryStorageAdapter()
    repository = PostgresRepository(session=in_memory_db)
    stt_engine = MockTrackingSTT()
    diarizer = MockTrackingDiarizer()
    pipeline = AudioAnalysisPipeline(stt_engine=stt_engine, diarizer=diarizer)
    job_service = JobService(storage=storage, repository=repository, pipeline=pipeline)

    buf = io.BytesIO()
    sf.write(buf, np.zeros(32000, dtype=np.int16), 16000, format="WAV", subtype="PCM_16")
    file_bytes = buf.getvalue()

    # Senaryo 1: num_speakers=None (Otomatik API İsteği)
    job_id = await job_service.create_job(
        file_name="auto_test.wav", file_bytes=file_bytes, num_speakers=None
    )
    status, _, _ = await job_service.execute_job_detailed(job_id, file_bytes=file_bytes)
    assert status == "COMPLETED"
    assert diarizer.called is True
    assert diarizer.received_num_speakers is None  # Kesinlikle None gitmeli!

    # Senaryo 2: Pipeline doğrudan çağrıldığında num_speakers=None verilirse None gitmeli
    diarizer.called = False
    diarizer.received_num_speakers = "NOT_CALLED"
    pipeline.process_bytes(file_bytes, num_speakers=None)
    assert diarizer.called is True
    assert diarizer.received_num_speakers is None

    # Senaryo 3: Pipeline dışarıdan num_speakers belirtilmeden çağrılırsa (API dışı / script) TARGET_NUM_SPEAKERS=2 kullanılmalı
    diarizer.called = False
    diarizer.received_num_speakers = "NOT_CALLED"
    pipeline.process_bytes(file_bytes)  # Unset
    assert diarizer.called is True
    assert diarizer.received_num_speakers == 2

    # Temizlik
    reset_settings()
