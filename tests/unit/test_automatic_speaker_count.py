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


def test_speaker_count_estimator_precedence(monkeypatch):
    """
    Doğrulama: Constructor argümanı > config.py / env > varsayılan (0.18 / 0.22 / 0.14)
    öncelik sırasını test eder.
    """
    from audio_analyzer.services.speaker_count_estimator import SpeakerCountEstimator

    # 1. Varsayılanlar
    monkeypatch.delenv("SPEAKER_EST_CENTROID_DIST_THRESH", raising=False)
    monkeypatch.delenv("SPEAKER_EST_MIN_SILHOUETTE", raising=False)
    monkeypatch.delenv("SPEAKER_EST_DISTINCT_CENTROID_DIST", raising=False)
    reset_settings()
    est_default = SpeakerCountEstimator()
    assert est_default.single_speaker_centroid_dist == 0.18
    assert est_default.min_silhouette_threshold == 0.22
    assert est_default.min_distinct_centroid_dist == 0.14

    # 2. Config / Env önceliği
    monkeypatch.setenv("SPEAKER_EST_CENTROID_DIST_THRESH", "0.25")
    monkeypatch.setenv("SPEAKER_EST_MIN_SILHOUETTE", "0.30")
    monkeypatch.setenv("SPEAKER_EST_DISTINCT_CENTROID_DIST", "0.20")
    reset_settings()
    est_env = SpeakerCountEstimator()
    assert est_env.single_speaker_centroid_dist == 0.25
    assert est_env.min_silhouette_threshold == 0.30
    assert est_env.min_distinct_centroid_dist == 0.20

    # 3. Constructor argümanı env'i ezer (en yüksek öncelik)
    est_override = SpeakerCountEstimator(
        single_speaker_centroid_dist=0.12,
        min_silhouette_threshold=0.15,
        min_distinct_centroid_dist=0.10,
    )
    assert est_override.single_speaker_centroid_dist == 0.12
    assert est_override.min_silhouette_threshold == 0.15
    assert est_override.min_distinct_centroid_dist == 0.10
    reset_settings()

