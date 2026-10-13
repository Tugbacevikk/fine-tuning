import uuid
from pathlib import Path
import pytest
from audio_analyzer.domain.models import AudioRecord, TranscriptUtterance
from audio_analyzer.api.routers.jobs import export_job_as_rttm
from audio_analyzer.api.main import app
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    from audio_analyzer.config import reset_settings
    from audio_analyzer.services.pipeline_factory import reset_pipeline_cache
    monkeypatch.setenv("API_KEY", "")
    reset_settings()
    reset_pipeline_cache()
    with TestClient(app) as c:
        yield c
    reset_settings()
    reset_pipeline_cache()


@pytest.mark.unit
def test_export_job_as_rttm(tmp_path):
    test_dir = tmp_path / "_bekleyen"

    job_id = uuid.uuid4()
    record = AudioRecord(
        id=job_id,
        file_name="dialogue_test.wav",
        storage_uri="mock/uri",
        status="COMPLETED",
        utterances=[
            TranscriptUtterance(
                id=uuid.uuid4(),
                speaker_id="SPEAKER_00",
                start_time=1.5,
                end_time=3.5,
                text="Merhaba nasılsınız",
            ),
            TranscriptUtterance(
                id=uuid.uuid4(),
                speaker_id="SPEAKER_01",
                start_time=3.8,
                end_time=5.0,
                text="İyiyim teşekkürler",
            ),
        ],
    )

    fake_wav_bytes = b"RIFFfakeaudio"
    rttm_path, wav_path = export_job_as_rttm(
        record, audio_bytes=fake_wav_bytes, target_dir=test_dir, reviewer_name="TestReviewer"
    )

    assert Path(rttm_path).exists()
    assert Path(wav_path).exists()
    with open(wav_path, "rb") as f:
        assert f.read() == fake_wav_bytes

    with open(rttm_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    lines = [line for line in all_lines if not line.startswith("#")]
    assert len(lines) == 2
    assert "SPEAKER dialogue_test 1 1.500 2.000 <NA> <NA> SPEAKER_00 <NA> <NA>" in lines[0]
    assert "SPEAKER dialogue_test 1 3.800 1.200 <NA> <NA> SPEAKER_01 <NA> <NA>" in lines[1]
    assert any("TestReviewer" in line for line in all_lines)


@pytest.mark.unit
def test_export_job_as_rttm_collision_protection(tmp_path):
    """Aynı isimde dosya varsa üzerine yazmayıp numara eklediğini doğrular."""
    test_dir = tmp_path / "_bekleyen"

    job_id1 = uuid.uuid4()
    record1 = AudioRecord(
        id=job_id1,
        file_name="meeting.wav",
        storage_uri="mock/uri1",
        status="COMPLETED",
        utterances=[
            TranscriptUtterance(
                id=uuid.uuid4(), speaker_id="SPEAKER_00", start_time=0.0, end_time=2.0, text="İlk kayıt"
            )
        ],
    )
    rttm_1, wav_1 = export_job_as_rttm(record1, audio_bytes=b"wav1", target_dir=test_dir)
    assert Path(rttm_1).name == "meeting.rttm"

    job_id2 = uuid.uuid4()
    record2 = AudioRecord(
        id=job_id2,
        file_name="meeting.wav",
        storage_uri="mock/uri2",
        status="COMPLETED",
        utterances=[
            TranscriptUtterance(
                id=uuid.uuid4(), speaker_id="SPEAKER_01", start_time=1.0, end_time=3.0, text="İkinci kayıt"
            )
        ],
    )
    rttm_2, wav_2 = export_job_as_rttm(record2, audio_bytes=b"wav2", target_dir=test_dir)
    assert Path(rttm_2).name == "meeting_1.rttm"
    assert Path(wav_2).name == "meeting_1.wav"
    assert Path(rttm_1).read_text(encoding="utf-8") != Path(rttm_2).read_text(encoding="utf-8")


@pytest.mark.unit
def test_export_rttm_endpoint_disabled_by_default(client, monkeypatch):
    """ENABLE_DATASET_EXPORT=false (varsayılan) iken 403 Forbidden döndüğünü doğrular."""
    monkeypatch.setenv("ENABLE_DATASET_EXPORT", "false")
    from audio_analyzer.config import reset_settings
    reset_settings()

    job_id = str(uuid.uuid4())
    res = client.post(
        f"/api/v1/jobs/{job_id}/export-rttm",
        json={"reviewer_name": "Adil", "consent_confirmed": True},
    )
    assert res.status_code == 403
    assert "devre dışı" in res.json()["detail"]


@pytest.mark.unit
def test_export_rttm_endpoint_validation(client, monkeypatch):
    """ENABLE_DATASET_EXPORT=true iken eksik reviewer_name veya consent_confirmed=false durumunda 400 döndüğünü doğrular."""
    monkeypatch.setenv("ENABLE_DATASET_EXPORT", "true")
    from audio_analyzer.config import reset_settings
    reset_settings()

    job_id = str(uuid.uuid4())

    # Eksik reviewer_name
    res1 = client.post(
        f"/api/v1/jobs/{job_id}/export-rttm",
        json={"reviewer_name": "", "consent_confirmed": True},
    )
    assert res1.status_code == 400
    assert "reviewer_name" in res1.json()["detail"]

    # Rıza onaylanmamış
    res2 = client.post(
        f"/api/v1/jobs/{job_id}/export-rttm",
        json={"reviewer_name": "Adil", "consent_confirmed": False},
    )
    assert res2.status_code == 400
    assert "rızasının alındığı" in res2.json()["detail"]

