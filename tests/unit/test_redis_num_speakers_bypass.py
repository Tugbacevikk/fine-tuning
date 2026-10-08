import io
import uuid
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from audio_analyzer.adapters.messaging.redis_stream_adapter import RedisStreamAdapter
from audio_analyzer.adapters.repository.postgres_repository import PostgresRepository
from audio_analyzer.adapters.storage.in_memory_storage_adapter import InMemoryStorageAdapter
from audio_analyzer.api.main import app
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
        self.received_num_speakers = None

    def diarize(self, audio_path: str, num_speakers: int | None = None):
        self.called = True
        self.received_num_speakers = num_speakers
        return [DiarizationSegment(speaker_id="SPEAKER_00", start_time=0.0, end_time=1.0)]


@pytest.mark.asyncio
async def test_redis_num_speakers_bypass_and_db_persistence(in_memory_db):
    """
    Doğrular:
    1. num_speakers=1 veritabanı kaydında (num_speakers) saklanır.
    2. execute_job_detailed(record_id) parametresiz çağrılsa bile num_speakers veritabanından okunur.
    3. Diarizer motoru HİÇ çağrılmaz (diarizer.called == False).
    """
    storage = InMemoryStorageAdapter()
    repository = PostgresRepository(session=in_memory_db)
    stt_engine = MockTrackingSTT()
    diarizer = MockTrackingDiarizer()
    pipeline = AudioAnalysisPipeline(stt_engine=stt_engine, diarizer=diarizer)

    job_service = JobService(storage=storage, repository=repository, pipeline=pipeline)

    buf = io.BytesIO()
    sf.write(buf, np.zeros(80000, dtype=np.int16), 16000, format="WAV", subtype="PCM_16")
    file_bytes = buf.getvalue()

    # 1. Job oluştur (num_speakers=1 ile)
    job_id = await job_service.create_job(
        file_name="single_speaker.wav", file_bytes=file_bytes, num_speakers=1
    )

    # 2. Veritabanı kaydında num_speakers == 1 olduğunu kontrol et
    record = await repository.get_record_by_id(job_id)
    assert record is not None
    assert record.num_speakers == 1

    # 3. Stream Worker çağrısı gibi parametresiz execute_job_detailed çağır
    status, attempts, err = await job_service.execute_job_detailed(job_id, file_bytes=file_bytes)
    assert status == "COMPLETED"

    # 4. Diarizer motorunun çağrılmadığını doğrula
    assert diarizer.called is False

    # 5. Tamamlanan kaydı oku ve tek konuşmacı (SPEAKER_00) atandığını doğrula
    completed_record = await repository.get_record_by_id(job_id)
    assert len(completed_record.utterances) == 1
    assert completed_record.utterances[0].speaker_id == "SPEAKER_00"


@pytest.mark.asyncio
async def test_num_speakers_2_propagates_to_diarizer(in_memory_db):
    """
    Doğrular: num_speakers=2 verildiğinde diarizer.diarize(audio_path, num_speakers=2) çağrılır.
    """
    storage = InMemoryStorageAdapter()
    repository = PostgresRepository(session=in_memory_db)
    stt_engine = MockTrackingSTT()
    diarizer = MockTrackingDiarizer()
    pipeline = AudioAnalysisPipeline(stt_engine=stt_engine, diarizer=diarizer)

    job_service = JobService(storage=storage, repository=repository, pipeline=pipeline)

    buf = io.BytesIO()
    sf.write(buf, np.zeros(80000, dtype=np.int16), 16000, format="WAV", subtype="PCM_16")
    file_bytes = buf.getvalue()

    job_id = await job_service.create_job(
        file_name="two_speakers.wav", file_bytes=file_bytes, num_speakers=2
    )

    status, _, _ = await job_service.execute_job_detailed(job_id, file_bytes=file_bytes)
    assert status == "COMPLETED"
    assert diarizer.called is True
    assert diarizer.received_num_speakers == 2


def test_api_num_speakers_input_validation():
    """
    Doğrular: num_speakers için 0, -3 veya 500 gibi geçersiz değerler HTTP 422 döner.
    """
    client = TestClient(app)
    fake_wav = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

    # num_speakers=0 (Geçersiz)
    resp = client.post(
        "/api/v1/analyze",
        files={"file": ("test.wav", fake_wav, "audio/wav")},
        data={"num_speakers": "0"},
    )
    assert resp.status_code == 422

    # num_speakers=500 (Geçersiz)
    resp2 = client.post(
        "/api/v1/analyze",
        files={"file": ("test.wav", fake_wav, "audio/wav")},
        data={"num_speakers": "500"},
    )
    assert resp2.status_code == 422
