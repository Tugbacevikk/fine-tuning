
import pytest

from audio_analyzer.adapters.repository.postgres_repository import PostgresRepository
from audio_analyzer.adapters.storage.in_memory_storage_adapter import (
    InMemoryStorageAdapter,
)
from audio_analyzer.domain.interfaces import IDiarizer, ISTTEngine
from audio_analyzer.domain.models import DiarizationSegment, WordSegment
from audio_analyzer.services.job_service import JobService
from audio_analyzer.services.pipeline import AudioAnalysisPipeline


class MockSTTEngine(ISTTEngine):
    """Sistem testleri için taklit STT motoru."""

    def transcribe(self, audio_path: str) -> tuple[list[WordSegment], str | None]:
        words = [
            WordSegment(word="Alo", start_time=0.0, end_time=0.4),
            WordSegment(word="buyurun", start_time=0.5, end_time=0.9),
            WordSegment(word="Nasıl", start_time=1.0, end_time=1.3),
            WordSegment(word="yardımcı", start_time=1.4, end_time=1.8),
            WordSegment(word="olabilirim", start_time=1.9, end_time=2.5),
        ]
        return words, "tr"


class MockDiarizer(IDiarizer):
    """Sistem testleri için taklit Diarization motoru."""

    def diarize(self, audio_path: str, num_speakers: int | None = None) -> list[DiarizationSegment]:
        return [
            DiarizationSegment(speaker_id="SPEAKER_00", start_time=0.0, end_time=0.95),
            DiarizationSegment(speaker_id="SPEAKER_01", start_time=0.98, end_time=2.6),
        ]


@pytest.mark.system
async def test_full_job_service_pipeline_e2e(tmp_path, in_memory_db):
    """
    Tüm sistem bileşenlerinin (Storage + DB Repo + Pipeline + FusionEngine + JobService)
    uçtan uca (E2E) mükemmel bir şekilde bir arada çalıştığını doğrular.
    """
    # 1. Adaptörlerin ve servislerin ayağa kaldırılması (Dependency Injection)
    storage = InMemoryStorageAdapter()
    repository = PostgresRepository(session=in_memory_db)
    stt_engine = MockSTTEngine()
    diarizer = MockDiarizer()
    pipeline = AudioAnalysisPipeline(stt_engine=stt_engine, diarizer=diarizer)

    job_service = JobService(storage=storage, repository=repository, pipeline=pipeline)

    # 2. İstemci Talebi (Ses Dosyası Yükleme & PENDING Görev Oluşturma)
    import io, soundfile as sf, numpy as np
    buf = io.BytesIO()
    sf.write(buf, np.zeros(80000, dtype=np.int16), 16000, format="WAV", subtype="PCM_16")
    mock_audio_bytes = buf.getvalue()
    job_id = await job_service.create_job(file_name="ornek_cagri.wav", file_bytes=mock_audio_bytes)

    record_pending = await repository.get_record_by_id(job_id)
    assert record_pending is not None
    assert record_pending.status.value == "PENDING"

    # 3. Arka Plan Worker Tarafından Görevin İşlenmesi (execute_job)
    success = await job_service.execute_job(job_id)
    assert success is True

    # 4. Veritabanı Sonuçlarının Doğrulanması (COMPLETED)
    record_completed = await repository.get_record_by_id(job_id)
    assert record_completed.status.value == "COMPLETED"
    assert record_completed.language == "tr"
    assert len(record_completed.utterances) == 2

    # Konuşmacı 0 (Müşteri/Arayan): "Alo buyurun"
    assert record_completed.utterances[0].speaker_id == "SPEAKER_00"
    assert record_completed.utterances[0].text == "Alo buyurun"

    # Konuşmacı 1 (Temsilci): "Nasıl yardımcı olabilirim"
    assert record_completed.utterances[1].speaker_id == "SPEAKER_01"
    assert record_completed.utterances[1].text == "Nasıl yardımcı olabilirim"

