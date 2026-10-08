import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from audio_analyzer.adapters.diarization.sherpa_onnx_adapter import SherpaOnnxAudioAdapter
from audio_analyzer.domain.models import DiarizationSegment


def test_sherpa_onnx_adapter_initialization():
    adapter = SherpaOnnxAudioAdapter(num_speakers=2)
    assert adapter.num_speakers == 2
    assert adapter._initialized is False
    assert adapter._pipeline is None


def test_sherpa_onnx_adapter_lazy_load_real(tmp_path):
    adapter = SherpaOnnxAudioAdapter(num_speakers=2)
    adapter._lazy_load_pipeline()
    if adapter._pipeline is not None:
        assert adapter._initialized is True
        assert adapter._config is not None


def test_sherpa_onnx_adapter_missing_models(tmp_path):
    with patch("audio_analyzer.config.get_settings") as mock_settings:
        mock_settings.return_value.model_dir = str(tmp_path / "non_existent_models")
        adapter = SherpaOnnxAudioAdapter()
        adapter._lazy_load_pipeline()
        assert adapter._pipeline is None
        segments = adapter.diarize("dummy.wav")
        assert segments == []


def test_sherpa_onnx_adapter_diarize_sample():
    eval_wav = Path("tests/fixtures/diarization_eval/large/eval_sample_01_2spk.wav")
    if not eval_wav.exists():
        pytest.skip("Fixture file not found")

    adapter = SherpaOnnxAudioAdapter(num_speakers=2)
    segments = adapter.diarize(str(eval_wav))

    assert len(segments) > 0
    for seg in segments:
        assert isinstance(seg, DiarizationSegment)
        assert seg.speaker_id.startswith("SPEAKER_")
        assert seg.start_time >= 0.0
        assert seg.end_time >= seg.start_time


def test_sherpa_onnx_adapter_numpy_input():
    eval_wav = Path("tests/fixtures/diarization_eval/large/eval_sample_01_2spk.wav")
    if not eval_wav.exists():
        pytest.skip("Fixture file not found")

    import soundfile as sf
    audio, sr = sf.read(str(eval_wav), dtype="float32")

    adapter = SherpaOnnxAudioAdapter(num_speakers=2)
    segments = adapter.diarize(audio, num_speakers=2)

    assert len(segments) > 0
    assert isinstance(segments[0], DiarizationSegment)
