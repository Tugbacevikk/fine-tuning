import os
from unittest.mock import MagicMock, patch
import pytest

from audio_analyzer.adapters.diarization.pyannote_adapter import PyannoteAudioAdapter
from audio_analyzer.config import get_settings


def test_settings_has_model_dir():
    """Verify that Settings class has model_dir attribute and defaults to storage/models."""
    settings = get_settings()
    assert hasattr(settings, "model_dir")
    assert settings.model_dir == os.getenv("MODEL_DIR", "storage/models")


def test_pyannote_adapter_lazy_load_community1(tmp_path):
    """Verify PyannoteAudioAdapter loads pipeline from pyannote-community-1 directory."""
    pyannote_dir = tmp_path / "diarization" / "pyannote-community-1"
    pyannote_dir.mkdir(parents=True, exist_ok=True)
    config_file = pyannote_dir / "config.yaml"
    config_file.write_text("pipeline:\n  name: PyannoteFakePipeline\n", encoding="utf-8")

    fake_pipeline_instance = MagicMock()

    with patch.dict(os.environ, {"MODEL_DIR": str(tmp_path)}), \
         patch("pyannote.audio.Pipeline.from_pretrained", return_value=fake_pipeline_instance) as mock_from_pretrained:
        adapter = PyannoteAudioAdapter()
        adapter._lazy_load_pipeline()

        assert adapter._pipeline is fake_pipeline_instance
        mock_from_pretrained.assert_called_once_with(str(config_file))


def test_pyannote_adapter_lazy_load_failure_handles_exception_cleanly(tmp_path):
    """Verify PyannoteAudioAdapter sets _pipeline to None on error without attribute crashes."""
    with patch.dict(os.environ, {"MODEL_DIR": str(tmp_path)}), \
         patch("pyannote.audio.Pipeline.from_pretrained", side_effect=RuntimeError("HF Hub offline")):
        adapter = PyannoteAudioAdapter()
        adapter._lazy_load_pipeline()

        assert adapter._pipeline is None
        res = adapter.diarize("dummy.wav")
        assert res == []


def test_pyannote_adapter_lazy_load_with_token_no_local_config(tmp_path):
    """Verify PyannoteAudioAdapter calls Pipeline.from_pretrained with token when local config is absent."""
    fake_pipeline_instance = MagicMock()

    with patch.dict(os.environ, {"MODEL_DIR": str(tmp_path), "HF_TOKEN": "hf_dummy_token_123"}), \
         patch("pyannote.audio.Pipeline.from_pretrained", return_value=fake_pipeline_instance) as mock_from_pretrained:
        adapter = PyannoteAudioAdapter()
        adapter._lazy_load_pipeline()

        assert adapter._pipeline is fake_pipeline_instance
        mock_from_pretrained.assert_called_once_with("pyannote/speaker-diarization-3.1", token="hf_dummy_token_123")
