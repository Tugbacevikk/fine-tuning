import io
import math
import struct
import uuid
import wave
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from audio_analyzer.api.main import app
from audio_analyzer.api.routers.jobs import sanitize_filename
from audio_analyzer.config import reset_settings
from audio_analyzer.domain.models import AudioRecord, JobStatus


def make_test_wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        samples = []
        for i in range(1600):
            t = float(i) / 16000
            val = int(32767.0 * 0.3 * math.sin(2.0 * math.pi * 440.0 * t))
            samples.append(struct.pack("<h", val))
        f.writeframes(b"".join(samples))
    return buf.getvalue()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", "")
    reset_settings()
    with TestClient(app) as c:
        yield c
    reset_settings()


@pytest.mark.unit
def test_sanitize_filename_unit():
    """sanitize_filename fonksiyonunun path traversal kısımlarını temizlediğini doğrular."""
    assert sanitize_filename("../x.wav") == "x.wav"
    assert sanitize_filename("..\\x.wav") == "x.wav"
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("..") == "upload.bin"
    assert sanitize_filename(".") == "upload.bin"
    assert sanitize_filename("") == "upload.bin"
    assert sanitize_filename(None) == "upload.bin"
    long_name = "a" * 300 + ".wav"
    assert len(sanitize_filename(long_name)) == 255


@pytest.mark.unit
def test_upload_filename_sanitization(client, monkeypatch):
    """Upload endpoint'inde '../x.wav' dosya adının 'x.wav' olarak sanitize edildiğini doğrular."""
    monkeypatch.setattr("audio_analyzer.api.routers.jobs.run_pipeline_background", AsyncMock())
    wav_bytes = make_test_wav_bytes()
    files = {"file": ("../x.wav", wav_bytes, "audio/wav")}

    res = client.post("/api/v1/analyze", files=files)
    assert res.status_code == 202
    data = res.json()
    assert data["file_name"] == "x.wav"


@pytest.mark.unit
def test_audio_endpoint_path_traversal_prevention(client):
    """
    Storage'da olmayan ve '../../etc/passwd' gibi adlandırılmış bir kayıt için
    /jobs/{job_id}/audio endpoint'inin 404 döndüğünü ve sunucudan dosya sızdırmadığını doğrular.
    """
    fake_id = uuid.uuid4()
    malicious_record = AudioRecord(
        id=fake_id,
        file_name="../../etc/passwd",
        storage_uri="nonexistent://dummy_path",
        status=JobStatus.COMPLETED,
    )

    with patch("audio_analyzer.domain.interfaces.ITranscriptRepository.get_record_by_id", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = malicious_record
        res = client.get(f"/api/v1/jobs/{fake_id}/audio")
        assert res.status_code == 404
        assert "bulunamadı" in res.json()["detail"].lower()


@pytest.mark.unit
def test_reanalyze_endpoint_path_traversal_prevention(client):
    """
    Storage'da olmayan ve '../../etc/passwd' gibi adlandırılmış bir kayıt için
    /jobs/{job_id}/reanalyze endpoint'inin 404 döndüğünü doğrular.
    """
    fake_id = uuid.uuid4()
    malicious_record = AudioRecord(
        id=fake_id,
        file_name="../../etc/passwd",
        storage_uri="nonexistent://dummy_path",
        status=JobStatus.COMPLETED,
    )

    with patch("audio_analyzer.domain.interfaces.ITranscriptRepository.get_record_by_id", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = malicious_record
        res = client.post(f"/api/v1/jobs/{fake_id}/reanalyze", json={"num_speakers": 2})
        assert res.status_code == 404
        assert "bulunamadı" in res.json()["detail"].lower()
