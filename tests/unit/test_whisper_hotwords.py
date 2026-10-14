import pytest
from unittest.mock import MagicMock
from audio_analyzer.config import get_settings, reset_settings
from audio_analyzer.adapters.stt.faster_whisper_adapter import FasterWhisperAdapter
from audio_analyzer.domain.models import DeviceConfig


@pytest.mark.unit
def test_whisper_hotwords_setting_default_and_env(monkeypatch):
    monkeypatch.delenv("WHISPER_HOTWORDS", raising=False)
    reset_settings()
    assert get_settings().whisper_hotwords == ""

    monkeypatch.setenv("WHISPER_HOTWORDS", "beğeni, Türkçe, Akıcı")
    reset_settings()
    assert get_settings().whisper_hotwords == "beğeni, Türkçe, Akıcı"
    reset_settings()


@pytest.mark.unit
def test_whisper_adapter_passes_hotwords_to_transcribe(monkeypatch):
    monkeypatch.setenv("WHISPER_HOTWORDS", "beğeni,Türkçe")
    reset_settings()

    adapter = FasterWhisperAdapter(model_size="small", device_config=DeviceConfig())
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], MagicMock())
    adapter._model = mock_model

    adapter.transcribe("dummy_audio.wav")

    mock_model.transcribe.assert_called_once()
    _, kwargs = mock_model.transcribe.call_args
    assert kwargs.get("hotwords") == "beğeni,Türkçe"
    reset_settings()
