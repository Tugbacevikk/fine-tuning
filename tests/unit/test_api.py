import io
import math
import struct
import wave

import pytest
from fastapi.testclient import TestClient

from audio_analyzer.api.main import app


@pytest.fixture
def client(monkeypatch):
    from audio_analyzer.config import reset_settings
    monkeypatch.setenv("API_KEY", "")
    reset_settings()
    with TestClient(app) as c:
        yield c
    reset_settings()


def make_valid_wav_bytes() -> bytes:
    """Geçerli 16kHz Mono WAV baytları (1 saniyelik sinüs sinyali) üretir."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        samples = []
        for i in range(16000):
            t = float(i) / 16000
            val = int(32767.0 * 0.3 * math.sin(2.0 * math.pi * 440.0 * t))
            samples.append(struct.pack("<h", val))
        f.writeframes(b"".join(samples))
    return buf.getvalue()


@pytest.mark.unit
def test_api_root_endpoint(client):
    """Web Arayüzü (HTML) kök dizin kontrolünü doğrular."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Ses Analizi" in response.text
    assert "numSpeakersInput" in response.text
    assert "Kaç kişi konuşuyor" in response.text


@pytest.mark.unit
def test_api_upload_with_optional_num_speakers(client, monkeypatch):
    """num_speakers parametresinin opsiyonel olarak iletilebildiğini doğrular."""
    from unittest.mock import AsyncMock
    monkeypatch.setattr("audio_analyzer.api.routers.jobs.run_pipeline_background", AsyncMock())
    valid_wav = make_valid_wav_bytes()
    
    # 1. num_speakers ile istek (2 kişi)
    files = {"file": ("test_speakers.wav", valid_wav, "audio/wav")}
    res = client.post("/api/v1/analyze", files=files, data={"num_speakers": 2})
    assert res.status_code == 202
    assert "job_id" in res.json()

    # 2. num_speakers olmadan (otomatik mod) istek
    files2 = {"file": ("test_auto.wav", valid_wav, "audio/wav")}
    res2 = client.post("/api/v1/analyze", files=files2)
    assert res2.status_code == 202
    assert "job_id" in res2.json()


@pytest.mark.unit
def test_api_upload_and_status_flow(client, tmp_path, monkeypatch):
    """POST /api/v1/analyze ve GET /api/v1/jobs/{id} REST akışını doğrular."""
    monkeypatch.setenv("USE_REDIS_QUEUE", "false")
    monkeypatch.setenv("USE_CELERY", "false")
    # 1. POST /api/v1/analyze
    valid_wav = make_valid_wav_bytes()
    files = {"file": ("test_api.wav", valid_wav, "audio/wav")}

    post_res = client.post("/api/v1/analyze", files=files)
    assert post_res.status_code == 202
    post_data = post_res.json()
    assert "job_id" in post_data
    job_id = post_data["job_id"]

    # 2. GET /api/v1/jobs/{job_id}
    get_res = client.get(f"/api/v1/jobs/{job_id}")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert get_data["job_id"] == job_id
    assert get_data["status"] == "COMPLETED"


@pytest.mark.unit
def test_utterance_crud_flow(client):
    """ITranscriptRepository üzerinden PUT, DELETE, POST utterance endpoint akışlarını doğrular."""
    valid_wav = make_valid_wav_bytes()
    files = {"file": ("test_crud.wav", valid_wav, "audio/wav")}
    post_res = client.post("/api/v1/analyze", files=files)
    job_id = post_res.json()["job_id"]

    # 1. POST /api/v1/jobs/{job_id}/utterances (Kutu Ekle)
    add_res = client.post(
        f"/api/v1/jobs/{job_id}/utterances",
        json={
            "speaker_id": "SPEAKER_00",
            "start_time": 0.0,
            "end_time": 1.0,
            "text": "Merhaba dunya",
        },
    )
    assert add_res.status_code == 200
    assert add_res.json()["status"] == "SUCCESS"

    # Fetch job to get utterance UUID
    get_res = client.get(f"/api/v1/jobs/{job_id}")
    assert get_res.status_code == 200
    utt_id = get_res.json()["utterances"][0]["id"]

    # 2. PUT /api/v1/jobs/{job_id}/utterances/{utt_id} (Kutu Guncelle)
    update_res = client.put(
        f"/api/v1/jobs/{job_id}/utterances/{utt_id}",
        json={"speaker_id": "SPEAKER_01", "text": "Guncellenmis metin"},
    )
    assert update_res.status_code == 200
    assert update_res.json()["status"] == "SUCCESS"

    # 3. DELETE /api/v1/jobs/{job_id}/utterances/{utt_id} (Kutu Sil)
    del_res = client.delete(f"/api/v1/jobs/{job_id}/utterances/{utt_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "SUCCESS"


@pytest.mark.unit
def test_list_jobs_endpoint(client):
    """GET /api/v1/jobs sayfalamalı geçmiş işler listeleme endpoint'ini doğrular."""
    list_res = client.get("/api/v1/jobs?skip=0&limit=10")
    assert list_res.status_code == 200
    jobs = list_res.json()
    assert isinstance(jobs, list)
    assert len(jobs) >= 1


@pytest.mark.unit
def test_api_key_authorization_on_read_and_write_endpoints(client, monkeypatch):
    """API_KEY ortam değişkeni ayarlandığında tüm okuma ve yazma endpoint'lerinin 401 döndürdüğünü doğrular."""
    from audio_analyzer.config import reset_settings
    monkeypatch.setenv("API_KEY", "secret-test-key")
    reset_settings()

    try:
        # 1. Okuma Endpoint'leri (GET)
        res_list = client.get("/api/v1/jobs")
        assert res_list.status_code == 401

        res_get = client.get("/api/v1/jobs/00000000-0000-0000-0000-000000000000")
        assert res_get.status_code == 401

        res_audio = client.get("/api/v1/jobs/00000000-0000-0000-0000-000000000000/audio")
        assert res_audio.status_code == 401

        # 2. Yanlış API Key ile erişim
        res_wrong = client.get("/api/v1/jobs", headers={"X-API-Key": "wrong-key"})
        assert res_wrong.status_code == 401

        # 3. Doğru API Key ile erişim
        res_valid = client.get("/api/v1/jobs", headers={"X-API-Key": "secret-test-key"})
        assert res_valid.status_code == 200
    finally:
        reset_settings()


@pytest.mark.unit
def test_health_check_endpoint(client):
    """GET /health liveness probe endpoint'ini doğrular."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert data["database"] == "OK"


@pytest.mark.unit
def test_api_upload_accepts_and_ignores_deprecated_domain_params(client, monkeypatch):
    """Eski istemcilerin gönderdiği domain_mode veya custom_triggers parametrelerinin hata vermediğini doğrular."""
    from unittest.mock import AsyncMock
    monkeypatch.setattr("audio_analyzer.api.routers.jobs.run_pipeline_background", AsyncMock())
    valid_wav = make_valid_wav_bytes()
    files = {"file": ("test_legacy.wav", valid_wav, "audio/wav")}
    res = client.post(
        "/api/v1/analyze",
        files=files,
        data={"domain_mode": "call_center", "custom_triggers": '{"agent_triggers": ["alo"]}'},
    )
    assert res.status_code == 202
    assert "job_id" in res.json()


