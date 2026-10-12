import uuid
from pathlib import Path
import pytest
from audio_analyzer.domain.models import AudioRecord, TranscriptUtterance
from audio_analyzer.api.routers.jobs import export_job_as_rttm


@pytest.mark.unit
def test_export_job_as_rttm(tmp_path, monkeypatch):
    test_dir = tmp_path / "real_turkish"
    monkeypatch.setattr("audio_analyzer.api.routers.jobs.Path", lambda p: test_dir if "real_turkish" in str(p) else Path(p))

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
    rttm_path, wav_path = export_job_as_rttm(record, audio_bytes=fake_wav_bytes)

    assert Path(rttm_path).exists()
    assert Path(wav_path).exists()
    with open(wav_path, "rb") as f:
        assert f.read() == fake_wav_bytes

    with open(rttm_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 2
    assert "SPEAKER dialogue_test 1 1.500 2.000 <NA> <NA> SPEAKER_00 <NA> <NA>" in lines[0]
    assert "SPEAKER dialogue_test 1 3.800 1.200 <NA> <NA> SPEAKER_01 <NA> <NA>" in lines[1]
