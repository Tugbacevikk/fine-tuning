import pytest
from audio_analyzer.config import get_settings, reset_settings


@pytest.mark.unit
@pytest.mark.parametrize(
    "env_key,expected_setting_attr",
    [
        ("APP_ENV", "app_env"),
        ("DATABASE_URL", "database_url"),
        ("S3_ENDPOINT_URL", "s3_endpoint_url"),
        ("MEMORY_STORAGE_MAX_BYTES", "memory_storage_max_bytes"),
        ("SWEEPER_INTERVAL_SEC", "sweeper_interval_sec"),
        ("DIARIZATION_STEP_SEC", "diarization_step_sec"),
        ("DIARIZATION_THRESHOLD", "diarization_threshold"),
    ],
)
def test_adapter_uses_settings_defaults(monkeypatch, env_key, expected_setting_attr):
    monkeypatch.delenv(env_key, raising=False)
    reset_settings()
    settings = get_settings()
    val = getattr(settings, expected_setting_attr)
    assert val is not None
    reset_settings()


@pytest.mark.unit
def test_whisper_beam_size_profile_defaults(monkeypatch):
    # 1. WHISPER_BEAM_SIZE not set, PIPELINE_PROFILE=feedback -> 1
    monkeypatch.delenv("WHISPER_BEAM_SIZE", raising=False)
    monkeypatch.setenv("PIPELINE_PROFILE", "feedback")
    reset_settings()
    assert get_settings().whisper_beam_size == 1

    # 2. WHISPER_BEAM_SIZE not set, PIPELINE_PROFILE=full -> 5
    monkeypatch.setenv("PIPELINE_PROFILE", "full")
    reset_settings()
    assert get_settings().whisper_beam_size == 5

    # 3. Explicit WHISPER_BEAM_SIZE overrides profile
    monkeypatch.setenv("WHISPER_BEAM_SIZE", "3")
    reset_settings()
    assert get_settings().whisper_beam_size == 3
    reset_settings()
